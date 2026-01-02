# VTK Library Compilation Guide

This document guides you through compiling a minimal VTK library, including core data structures and basic IO modules, to meet the needs of reading VTK files.

## Download and Extract Source Code

Visit the [VTK official website](https://vtk.org/download) to download the latest version of the source code (.tar.gz) and extract it.

- The built-in decompression tool in Windows is relatively slow. For VTK source code, it is recommended to install dedicated decompression software such as [BandiZip](https://en.bandisoft.com/bandizip/). tar.gz files on Windows usually need to be extracted twice.
- On Linux, use the system's built-in tar command for extraction.

You can create a directory structure similar to the SDL library:

```text
VTK/
    build/
    install/
    source/
        CMakeLists.txt
        ...
```

## Install Build Tools and CMake

Refer to the first half of the [Build Guide](build-guide.md) to install the latest version of the compiler and CMake according to your operating system.

## Configure CMake Project

1. Open CMake GUI, set the source code directory and build directory
2. Click Configure, select the generator corresponding to your system (for Windows: `Visual Studio 17 2022`, for Linux: `Unix Makefiles`), keep other settings at default
3. Wait for configuration to complete, red options will appear

- On Linux, Configure may report an error: .clang-tidy file not found, causing configuration to fail. In this case, you can delete the corresponding command in CMakeLists.txt based on the error location (usually the second to last command), or create an empty .clang-tidy file in the source code root directory. This will not affect compilation.

4. Uncheck all checkboxes, set all dropdown items to "DONT_WANT", select or keep only Release for build type, set the installation path to the install folder configured in the above directory structure
5. Check the "BUILD_SHARED_LIBS" option, set the "StandAlone" item to "WANT"
6. Keep other settings unchanged from step 4. If you need to compile other VTK functional modules simultaneously, you can selectively enable them.

- Compiling some modules requires corresponding libraries already installed on the system, such as the QT module. If not installed, configuration will be invalid. For the RendererOptiX renderer, only the "StandAlone" module needs to be compiled, which includes VTK core data structures and basic IO functions

- If configuration is invalid or other errors occur, you need to delete all files in the build directory and click Files -> Delete Cache in the upper left corner of the GUI to clear all caches, then re-execute the configuration steps

7. Click Generate to generate the project. If there are no errors, it is recommended to backup the build folder at this point, so that if an error occurs during compilation and you need to recompile, you don't need to re-execute the configuration steps

## Compilation

### Windows

Click Open Project directly to open the project in Visual Studio. In the "Solution Explorer", find the "ALL_BUILD" project, right-click and select "Build", then wait for compilation to complete.

- Compilation time depends on the number of CPU cores and performance. On mainstream consumer-grade CPUs, it takes about 20 minutes, which is slower than compilation on Linux.

### Linux

Close CMake GUI, open a console in the build directory, and use the following command to compile:

```bash
cmake --build . -j CPU_thread_count
```

- On Linux, if CPU usage is too high, the system may kill some compilation processes, causing some targets to fail compilation. Therefore, it is strongly recommended not to use all CPU threads for compilation. You can set it to 80% of the CPU thread count.

## Installation

### Windows
In the "Solution Explorer", find the "INSTALL" project and right-click to build.

- If compilation has no errors but installation reports an error, check the set CMAKE_INSTALL_PREFIX. If a backslash path is used, it will cause installation to fail. In this case, there is no need to recompile. Find the CMakeLists.txt of the INSTALL project and modify the installation path to use forward slashes.

### Linux

Execute the installation command in the build directory:

```bash
cmake --install .
```

## Integration into Project

Same as the SDL library, set the lib path.

- On Windows, you need to copy all dynamic library files from the bin directory to the executable file directory
- On Linux, environment variables are automatically set through CMake, so copying is not necessary.