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

import io.gravitee.llama.cpp.platform.Platform;
import io.gravitee.llama.cpp.platform.PlatformResolver;
import org.reflections.Reflections;
import org.reflections.scanners.Scanners;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.stream.Stream;

import static io.gravitee.llama.cpp.LlamaRuntime.ggml_backend_load_all;
import static java.util.function.Predicate.not;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public final class LlamaLibLoader {

    static final String LLAMA_CPP_LIB_PATH = "LLAMA_CPP_LIB_PATH";
    static final String LLAMA_CPP_USE_TMP_PATH_LIBS = "LLAMA_CPP_USE_TMP_LIB_PATH";

    static final String DYLIB_EXT = ".dylib";
    static final String SO_EXT = ".so";

    private static final String LLAMA_CPP_FOLDER = ".llama.cpp";
    public static final String USER_HOME = System.getProperty("user.home");

    private LlamaLibLoader() {
    }

    public static void load() {
        String envLibPath = System.getenv(LLAMA_CPP_LIB_PATH);
        if (envLibPath != null && !envLibPath.isBlank()) {
            loadFromExternalPath(envLibPath);
        } else {
            loadFromClasspath(PlatformResolver.platform());
        }
    }

    private static void loadFromExternalPath(String envLibPath) {
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
    }

    public static void load(String path) {
        load(path, PlatformResolver.platform());
    }

    public static void load(String path, Platform platform) {
        safeWalk(path)
                .map(Path::toAbsolutePath)
                .map(Path::toString)
                .filter(not(path::equals))
                .filter(file -> switch (platform.os()) {
                    case MAC_OS_X -> file.endsWith(DYLIB_EXT);
                    case LINUX -> file.endsWith(SO_EXT);
                })
                .forEach(System::load);
    }

    private static void loadFromClasspath(Platform platform) {
        try {
            boolean useTmpPath = Boolean.parseBoolean(System.getProperty(LLAMA_CPP_USE_TMP_PATH_LIBS));
            var libDirectory = useTmpPath ? Files.createTempDirectory(LLAMA_CPP_FOLDER) : getHomeLlamaCpp();

            if (!useTmpPath && Files.isDirectory(libDirectory) && !Files.list(libDirectory).toList().isEmpty()) {
                load(libDirectory.toString());
            } else {
                var reflections = new Reflections(platform.getPackage(), Scanners.Resources);
                reflections
                        .getResources(".+")
                        .forEach(name -> copyFromClasspath(name, libDirectory));

                load(libDirectory.toString());
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
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
