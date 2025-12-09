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
  static final String LLAMA_CPP_USE_TMP_PATH_LIBS = "LLAMA_CPP_USE_TMP_LIB_PATH";

  static final String DYLIB_EXT = ".dylib";
  static final String SO_EXT = ".*\\.so(\\..+)?";
  static final String DLL_EXT = ".dll";

  private static final String LLAMA_CPP_FOLDER = ".llama.cpp";
  private static final String USER_HOME = System.getProperty("user.home");

  private LlamaLibLoader() {}

  public static String load() {
    String envLibPath = System.getenv(LLAMA_CPP_LIB_PATH);
    if (envLibPath != null && !envLibPath.isBlank()) {
      return loadFromExternalPath(envLibPath);
    } else {
      return loadFromClasspath(PlatformResolver.platform());
    }
  }

  private static String loadFromExternalPath(String envLibPath) {
    try {
      boolean useTmpPath = Boolean.parseBoolean(System.getProperty(LLAMA_CPP_USE_TMP_PATH_LIBS));
      if (useTmpPath) {
        var destination = Files.createTempDirectory(LLAMA_CPP_FOLDER);
        Files.copy(Path.of(envLibPath), destination, StandardCopyOption.REPLACE_EXISTING);
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

  public static void load(String path, Platform platform) {
    safeWalk(path)
      .map(Path::toAbsolutePath)
      .map(Path::toString)
      .filter(not(path::equals))
      .filter(file ->
        switch (platform.os()) {
          case MAC_OS_X -> file.endsWith(DYLIB_EXT);
          case LINUX -> file.matches(SO_EXT);
          case WINDOWS -> file.endsWith(DLL_EXT);
        }
      )
      .forEach(System::load);
  }

  private static String loadFromClasspath(Platform platform) {
    try {
      boolean useTmpPath = Boolean.parseBoolean(System.getProperty(LLAMA_CPP_USE_TMP_PATH_LIBS));
      Path libDirectory = useTmpPath ? Files.createTempDirectory(LLAMA_CPP_FOLDER) : getHomeLlamaCpp();

      String libDirStr = libDirectory.toString();
      if (!useTmpPath && Files.isDirectory(libDirectory) && !Files.list(libDirectory).toList().isEmpty()) {
        load(libDirStr);
      } else {
        List<String> resources;

        File jarFile = getRunningJarFile();
        if (jarFile != null) {
          System.out.println("Copying llama.cpp native libraries from jar");
          resources = listJarResources(platform.getPackage(), jarFile);
        } else {
          System.out.println("Copying llama.cpp native libraries from classes folder");
          resources = listResourcesFromClassesFolder(platform.getPackage());
        }

        for (String resourceName : resources) {
          copyFromClasspath(resourceName, libDirectory);
        }

        load(libDirStr);
      }
      return libDirStr;
    } catch (IOException | URISyntaxException e) {
      throw new RuntimeException(e);
    }
  }

  private static List<String> listJarResources(String packageName, File jarFile) throws IOException {
    String prefix = packageName.replace('.', '/') + "/";
    try (JarFile jar = new JarFile(jarFile)) {
      return jar.stream().map(JarEntry::getName).filter(name -> name.startsWith(prefix) && !name.endsWith("/")).toList();
    }
  }

  private static List<String> listResourcesFromClassesFolder(String packageName) throws IOException, URISyntaxException {
    String path = packageName.replace('.', '/');
    Path classesFolder = Path.of(Thread.currentThread().getContextClassLoader().getResource(path).toURI());
    List<String> resources = new ArrayList<>();

    try (var walk = Files.walk(classesFolder)) {
      walk
        .filter(Files::isRegularFile)
        .forEach(file -> {
          Path relative = classesFolder.relativize(file);
          resources.add(path + "/" + relative.toString().replace(File.separatorChar, '/'));
        });
    }
    return resources;
  }

  private static File getRunningJarFile() throws URISyntaxException {
    String path = LlamaLibLoader.class.getProtectionDomain().getCodeSource().getLocation().toURI().getPath();
    File file = new File(path);
    if (file.isFile() && file.getName().endsWith(".jar")) {
      return file;
    }
    return null;
  }

  private static Path getHomeLlamaCpp() throws IOException {
    var homeLlamaCpp = Paths.get(USER_HOME, LLAMA_CPP_FOLDER);
    return Files.exists(homeLlamaCpp) ? homeLlamaCpp : Files.createDirectory(homeLlamaCpp);
  }

  private static void copyFromClasspath(String name, Path libDirectory) {
    try {
      var classLoader = Thread.currentThread().getContextClassLoader();

      var resource = classLoader.getResource(name);
      String[] fileSplit = resource.getFile().split(File.separator);
      String fileName = fileSplit[fileSplit.length - 1];

      Files.write(libDirectory.resolve(fileName), resource.openStream().readAllBytes());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static Stream<Path> safeWalk(String libPath) {
    try {
      return Files.walk(Path.of(libPath));
    } catch (IOException e) {
      return Stream.empty();
    }
  }
}
