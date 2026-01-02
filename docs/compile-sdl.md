# SDL2 Library Compilation Guide

This guide contains instructions for compiling SDL2/SDL3 libraries from source code, with additional compilation methods for the SDL extension library SDL_ttf. Other SDL extension libraries such as SDL_mixer can refer to the compilation configuration method of the ttf library.

## Download Source Code

Download the latest Release version source code from [SDL's GitHub repository](https://github.com/libsdl-org/SDL). Note that the latest Release is SDL3. If you need SDL2, please search in the Release list.

## Extract Source Code

Create a new folder containing build, install, and source folders, and extract the source code to the source directory:

```text
SDL/
    build/
    install/
    source/
        ...
        CMakeLists.txt
```

## Configure CMake Project

MSVC is a multi-configuration compiler that does not specify Debug/Release during configuration, but needs to specify it during compilation and installation; MinGW/GCC are single-configuration compilers that specify Debug/Release during configuration and do not need to specify it during compilation and installation.

### Windows

Open a command line in the build directory and use the configuration command:

```cmd
cmake ../source -G "Visual Studio 17 2022" -DCMAKE_INSTALL_PREFIX="../install"
```

Compile:

```cmd
cmake --build . --config Release --parallel
or
cmake --build . --config Release -j CPU_thread_count
```

Install:

```cmd
cmake --install . --config Release
```

### Linux

Configure in the build directory:

```bash
cmake ../source -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="../install" -DCMAKE_BUILD_TYPE=Release
```

Compile:

```bash
cmake --build . --parallel
```

Install:

```bash
cmake --install .
```

- If using CMake GUI, set CMAKE_INSTALL_PREFIX to the install folder and CMAKE_BUILD_TYPE to Release, keeping other options at their defaults.

## Using Installation Files in Current Project

After installation is complete, in the install folder, you can see the compiled SDL library files:

```text
install/
    bin/
        SDL2.dll
    include/
        SDL.h
        ...
    lib/
        cmake/
        ...
    ...
```

In the current project's CMakeLists.txt, set the variable SDL2_DIR to the path of lib/cmake and automatically find the package:

```cmake
set(SDL2_DIR "/path/to/install/lib/cmake")
find_package(SDL2 REQUIRED)
```

Reconfigure the project's CMake. If there are no errors, the configuration is successful.

- Note: On Windows, you need to manually copy the dynamic library file SDL2.dll to the project executable's generation directory, i.e., in the same directory as the executable, otherwise the dynamic library cannot be correctly linked at runtime

# SDL Extension Library Compilation Guide, Using SDL_ttf as Example

## Download Source Code

Same as the SDL library, extract the downloaded source code to the same directory structure:

```text
SDL_ttf/
    build/
    install/
    source/
        ...
        CMakeLists.txt
```

## Configure CMake Project

Since SDL_ttf depends on SDL, you need to specify the SDL library path during configuration, otherwise the project cannot be generated. The ttf source code contains sample programs, and compiling these sample programs requires linking to SDL.lib and SDLmain.lib.

### Windows

Configure:

```cmd
cmake ../source -G "Visual Studio 17 2022" -DCMAKE_INSTALL_PREFIX=../install -DSDL2_DIR="" -DSDL2_LIBRARY="" -DSDL2_INCLUDE_DIR="" -DSDL2_MAIN_LIBRARY=""
```

Where:

- SDL2_DIR is the SDL installation path: /path/to/SDL/install
- SDL2_LIBRARY is the path to SDL.lib: /path/to/SDL/install/lib/SDL2.lib
- SDL2_INCLUDE_DIR is the SDL header file path: /path/to/SDL/install/include/SDL2
- SDL2_MAIN_LIBRARY is the path to SDLmain.lib: /path/to/SDL/install/lib/SDL2main.lib

Compile and install:

```cmd
cmake --build . --parallel --config Release
cmake --install . --config Release
```

### Linux

## Using Installation Files in Current Project

Same method as the SDL library, just replace SDL2_DIR with SDL2_ttf_DIR; replace SDL2 with SDL2_ttf.