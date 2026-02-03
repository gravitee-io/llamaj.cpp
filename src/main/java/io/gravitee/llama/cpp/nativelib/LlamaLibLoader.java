/*
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.gravitee.llama.cpp.nativelib;

import static java.util.function.Predicate.not;

import io.gravitee.llama.cpp.platform.Platform;
import io.gravitee.llama.cpp.platform.PlatformResolver;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Stream;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaLibLoader {

  static final String LLAMA_CPP_LIB_PATH = "LLAMA_CPP_LIB_PATH";
  static final String LLAMA_CPP_USE_TMP_PATH_LIBS =
    "LLAMA_CPP_USE_TMP_LIB_PATH";

  static final String DYLIB_EXT = ".dylib";
  static final String SO_EXT = ".*\\.so(\\..+)?";
  static final String DLL_EXT = ".dll";

  private static final String LLAMA_CPP_FOLDER = ".llama.cpp";
  private static final String USER_HOME = System.getProperty("user.home");

  private static volatile String loadedPath;

  private LlamaLibLoader() {}

  public static String load() {
    if (loadedPath != null) {
      return loadedPath;
    }
    synchronized (LlamaLibLoader.class) {
      if (loadedPath != null) {
        return loadedPath;
      }
      String envLibPath = System.getenv(LLAMA_CPP_LIB_PATH);
      if (envLibPath != null && !envLibPath.isBlank()) {
        loadedPath = loadFromExternalPath(envLibPath);
      } else {
        loadedPath = loadFromClasspath(PlatformResolver.platform());
      }
      return loadedPath;
    }
  }

  private static String loadFromExternalPath(String envLibPath) {
    try {
      boolean useTmpPath = Boolean.parseBoolean(
        System.getProperty(LLAMA_CPP_USE_TMP_PATH_LIBS)
      );
      if (useTmpPath) {
        var destination = Files.createTempDirectory(LLAMA_CPP_FOLDER);
        Files.copy(
          Path.of(envLibPath),
          destination,
          StandardCopyOption.REPLACE_EXISTING
        );
        load(destination.toString());
      } else {
        load(envLibPath);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return envLibPath;
  }

  public static String load(String path) {
    load(path, PlatformResolver.platform());
    return path;
  }

  /**
   * Regex matching ggml backend plugins that must NOT be loaded via System.load().
   * These are loaded internally by ggml_backend_load_all_from_path() via
   * dlopen(RTLD_NOW | RTLD_LOCAL), which keeps each plugin's symbols isolated.
   *
   * <p>On Linux, pre-loading backend plugins via System.load() would duplicate their
   * symbols and interfere with the backend registry. The llama.cpp plugin system
   * expects to manage these libraries itself via its own dlopen calls.
   *
   * <p>On macOS this is not an issue because libllama.dylib directly links all backends
   * (CPU, Metal, RPC, BLAS) as NEEDED dependencies, so they share the same registry
   * singleton regardless of loading order. The Linux release uses a plugin architecture
   * with multiple CPU variants, which requires RTLD_LOCAL isolation.
   */
  private static final String BACKEND_PLUGIN_PATTERN =
    ".*/libggml-(cpu|rpc|blas|cuda|hip|vulkan|sycl|metal|cann|kompute|amx).*\\.so(\\..*)?";

  /**
   * On Linux, each shared library has a SONAME (e.g. "libggml.so.0") embedded in
   * its ELF header. The directory typically contains three paths per library:
   * <ul>
   *   <li>{@code libfoo.so} — linker name (for -lfoo at compile time)</li>
   *   <li>{@code libfoo.so.0} — soname (what NEEDED deps reference)</li>
   *   <li>{@code libfoo.so.0.9.7} — real file with full version</li>
   * </ul>
   *
   * <p>On Linux, dlopen() deduplicates by path string, not by inode. Loading the same
   * library through different paths (e.g. "libggml.so" vs "libggml.so.0" vs
   * "libggml.so.0.9.7") creates separate library instances, each with its own copy
   * of C++ function-local statics — including the ggml backend registry singleton.
   *
   * <p>We must load ONLY the soname-versioned paths (e.g. "libfoo.so.0") because these
   * match what the dynamic linker resolves for NEEDED dependencies. This ensures that
   * all libraries (libllama.so, libggml.so, libmtmd.so) and the Java FFM bindings
   * share the same backend registry instance.
   *
   * <p>Pattern: matches "libfoo.so.N" where N is one or more digits, but NOT
   * "libfoo.so" (bare) or "libfoo.so.0.9.7" (full version).
   */
  private static final String SONAME_SO_PATTERN = ".*\\.so\\.\\d+";

  public static void load(String path, Platform platform) {
    safeWalk(path)
      .filter(not(p -> p.toString().equals(path)))
      .filter(p -> {
        String file = p.toAbsolutePath().toString();
        return switch (platform.os()) {
          case MAC_OS_X -> file.endsWith(DYLIB_EXT);
          case LINUX -> file.matches(SONAME_SO_PATTERN) &&
          !file.matches(BACKEND_PLUGIN_PATTERN);
          case WINDOWS -> file.endsWith(DLL_EXT);
        };
      })
      .map(Path::toAbsolutePath)
      .map(Path::toString)
      .forEach(System::load);
  }

  private static String loadFromClasspath(Platform platform) {
    try {
      boolean useTmpPath = Boolean.parseBoolean(
        System.getProperty(LLAMA_CPP_USE_TMP_PATH_LIBS)
      );
      Path libDirectory = useTmpPath
        ? Files.createTempDirectory(LLAMA_CPP_FOLDER)
        : getHomeLlamaCpp();

      String libDirStr = libDirectory.toString();
      if (
        !useTmpPath &&
        Files.isDirectory(libDirectory) &&
        !Files.list(libDirectory).toList().isEmpty() &&
        hasValidNativeLibs(libDirectory, platform)
      ) {
        load(libDirStr);
      } else {
        List<String> resources;

        File jarFile = getRunningJarFile();
        if (jarFile != null) {
          System.out.println("Copying llama.cpp native libraries from jar");
          resources = listJarResources(platform.getPackage(), jarFile);
        } else {
          System.out.println(
            "Copying llama.cpp native libraries from classes folder"
          );
          resources = listResourcesFromClassesFolder(platform.getPackage());
        }

        // Extract real files first, then recreate symlinks.
        // This ensures symlink targets exist before symlinks are created.
        List<String> deferredSymlinks = new ArrayList<>();
        for (String resourceName : resources) {
          if (isResourcePackagedSymlink(resourceName)) {
            deferredSymlinks.add(resourceName);
          } else {
            copyFromClasspath(resourceName, libDirectory);
          }
        }
        for (String resourceName : deferredSymlinks) {
          copyFromClasspath(resourceName, libDirectory);
        }

        load(libDirStr);
      }
      return libDirStr;
    } catch (IOException | URISyntaxException e) {
      throw new RuntimeException(e);
    }
  }

  private static List<String> listJarResources(String packageName, File jarFile)
    throws IOException {
    String prefix = packageName.replace('.', '/') + "/";
    try (JarFile jar = new JarFile(jarFile)) {
      return jar
        .stream()
        .map(JarEntry::getName)
        .filter(name -> name.startsWith(prefix) && !name.endsWith("/"))
        .toList();
    }
  }

  private static List<String> listResourcesFromClassesFolder(String packageName)
    throws IOException, URISyntaxException {
    String path = packageName.replace('.', '/');
    Path classesFolder = Path.of(classLoader().getResource(path).toURI());
    List<String> resources = new ArrayList<>();

    try (var walk = Files.walk(classesFolder)) {
      walk
        .filter(Files::isRegularFile)
        .forEach(file -> {
          Path relative = classesFolder.relativize(file);
          resources.add(
            path + "/" + relative.toString().replace(File.separatorChar, '/')
          );
        });
    }
    return resources;
  }

  private static File getRunningJarFile() throws URISyntaxException {
    String path = LlamaLibLoader.class.getProtectionDomain()
      .getCodeSource()
      .getLocation()
      .toURI()
      .getPath();
    File file = new File(path);
    if (file.isFile() && file.getName().endsWith(".jar")) {
      return file;
    }
    return null;
  }

  private static Path getHomeLlamaCpp() throws IOException {
    var homeLlamaCpp = Paths.get(USER_HOME, LLAMA_CPP_FOLDER);
    return Files.exists(homeLlamaCpp)
      ? homeLlamaCpp
      : Files.createDirectory(homeLlamaCpp);
  }

  private static void copyFromClasspath(String name, Path libDirectory) {
    try {
      var resource = classLoader().getResource(name);
      String[] fileSplit = resource.getFile().split(File.separator);
      String fileName = fileSplit[fileSplit.length - 1];

      byte[] content = resource.openStream().readAllBytes();
      Path destination = libDirectory.resolve(fileName);

      // JAR files cannot store symlinks. When symlinks are packaged into a JAR,
      // they become small text files containing the symlink target name.
      // Detect these and recreate them as proper symbolic links so that
      // @rpath resolution works correctly on macOS.
      if (isPackagedSymlink(fileName, content)) {
        String target = new String(content).trim();
        Files.deleteIfExists(destination);
        Files.createSymbolicLink(destination, Path.of(target));
      } else {
        Files.write(destination, content);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Detects whether a resource extracted from a JAR is a packaged symlink.
   * JARs store symlinks as small text files containing the target filename.
   * We detect this by checking if the content is a short string that looks
   * like a library filename (matching the original file's extension pattern).
   */
  private static boolean isPackagedSymlink(String fileName, byte[] content) {
    // Symlink text files are very small (typically < 100 bytes)
    // and contain just the target filename
    if (content.length > 256) {
      return false;
    }
    String text = new String(content).trim();
    // The content should look like a library filename and differ from this file
    return (
      !text.equals(fileName) &&
      !text.isEmpty() &&
      !text.contains("\n") &&
      (text.endsWith(DYLIB_EXT) ||
        text.matches(SO_EXT) ||
        text.endsWith(DLL_EXT))
    );
  }

  /**
   * Checks whether a classpath resource is a packaged symlink without
   * fully extracting it. Used to order extraction so real files come first.
   */
  private static boolean isResourcePackagedSymlink(String name) {
    try {
      var resource = classLoader().getResource(name);
      if (resource == null) return false;
      String[] fileSplit = resource.getFile().split(File.separator);
      String fileName = fileSplit[fileSplit.length - 1];
      byte[] content = resource.openStream().readAllBytes();
      return isPackagedSymlink(fileName, content);
    } catch (IOException e) {
      return false;
    }
  }

  private static Stream<Path> safeWalk(String libPath) {
    try {
      return Files.walk(Path.of(libPath));
    } catch (IOException e) {
      return Stream.empty();
    }
  }

  /**
   * Validates that the cached native library directory contains actual native
   * libraries and not broken symlink text files from a previous JAR extraction.
   * A native library on macOS/Linux must be larger than 256 bytes (a symlink
   * text file stored in a JAR is typically under 100 bytes).
   */
  private static boolean hasValidNativeLibs(
    Path libDirectory,
    Platform platform
  ) {
    try (var walk = Files.walk(libDirectory)) {
      return walk
        .filter(Files::isRegularFile)
        .filter(not(Files::isSymbolicLink))
        .filter(p -> {
          String name = p.toString();
          return switch (platform.os()) {
            case MAC_OS_X -> name.endsWith(DYLIB_EXT);
            case LINUX -> name.matches(SO_EXT);
            case WINDOWS -> name.endsWith(DLL_EXT);
          };
        })
        .allMatch(p -> {
          try {
            return Files.size(p) > 256;
          } catch (IOException e) {
            return false;
          }
        });
    } catch (IOException e) {
      return false;
    }
  }

  private static ClassLoader classLoader() {
    return LlamaLibLoader.class.getClassLoader();
  }
}
