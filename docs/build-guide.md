# Build Guide

## System Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU, Turing architecture or above (with RT Cores), supporting CUDA and OptiX.
  - Minimum: CUDA Compute Capability 7.5+ (sm_75)
  - Recommended: GeForce RTX 30 series or higher
  - Both Game Ready and Studio drivers are supported
- **Memory**: At least 8GB RAM (16GB or more recommended, depending on the number of cache read/write threads)
- **VRAM**: At least 8GB (12GB or more recommended, depending on VTK particle scale)

### Software Requirements

- **[CMake](https://cmake.org)**: 4.0 or higher
- **[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)**: 12.0+
- **[OptiX SDK](https://developer.nvidia.com/rtx/ray-tracing/optix)**: 9.0.0+
- **[Vulkan SDK](https://www.vulkan.org/)**: 1.1 and above

#### Windows

- **Operating System**: Windows 10/11
- **Compiler**: [Visual Studio 2022 / 2026](https://visualstudio.microsoft.com) (supporting C++20), with "Desktop development with C++" workload installed

#### Linux

- **Operating System**: Ubuntu 22.04/24.04+ or other distributions
- **Compiler**: GCC 10+ or Clang 12+ (supporting C++20)

```bash
sudo apt update && sudo apt install gcc g++ make
# You can check the version with gcc -v
```

- **CMake from system package managers is usually not the latest version, typically 3.2.x**. To install the latest version, Ubuntu users can use the built-in App Center and select the latest version for installation. Other distributions can also [add the CMake repository](install-latest-camke.md) and then install via package manager, or manually download precompiled packages or source code from the [official website](https://cmake.org/download).

```bash
sudo apt install cmake cmake-qt-gui
```

- This document provides build instructions for Debian-based distributions. The source code has been verified to compile and run on Ubuntu 24.04. If using other distributions, you need to use the corresponding system commands for that distribution. If you encounter errors after switching to other distributions, you may need to modify parts of the source code.
- All code in this project is written with cross-platform principles in mind. Except for a small amount of graphics API call code, there is no platform-specific code.

## Dependency Installation

- Before installation, ensure your graphics driver is updated to the latest version.

### CUDA Toolkit

1. Download and install CUDA Toolkit from [NVIDIA official website](https://developer.nvidia.com/cuda-downloads)
2. Ensure `nvcc.exe`/`nvcc` is in the system PATH

#### Windows

Simply run the installer. It is recommended to install to the default directory.

#### Linux

It is recommended to use the "deb(local)" method for installation.

### OptiX SDK

Download OptiX SDK 9.0.0 from [NVIDIA official website](https://developer.nvidia.com/designworks/optix/download).

#### Windows

Simply run the installer. It is recommended to install to the default directory.

#### Linux

It is recommended to extract to a user directory that can be read and written without sudo, for example:  
`~/Public/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64`

### SDL2

#### Windows

1. Download the source code from [SDL2's GitHub repository](https://github.com/libsdl-org/SDL) and compile manually, or download the precompiled "devel-...-VC" version

- If using SDL3, you need to modify the SDL header file inclusion in include/Global/HostFunctions.cuh.
- If compiling from source code, you can refer to the [SDL Build Guide](compile-sdl.md).

2. Extract the downloaded precompiled package or extract the SDL installation folder obtained from manual compilation

#### Linux

You can compile from source code, but it is recommended to install directly using the system package manager:

```bash
sudo apt install libsdl2-dev
```

### Vulkan SDK

1. Download and install Vulkan SDK from [Vulkan official website](https://vulkan.lunarg.com/)
2. Ensure environment variables are correctly set

#### Windows

It is recommended to keep the default installation location. After installation, you can find and run the Vulkan Cube example from the Start menu. If it runs normally, the configuration is successful.

#### Linux

Please download the tarball version of the SDK, and extract it to get the Vulkan installation folder.

### VTK

1. Download source code from [VTK official website](https://vtk.org/download)
2. You can refer to the [VTK Build Guide](compile-vtk.md) for compilation
3. Extract the VTK installation folder obtained from manual compilation

## Building

The following introduces the method using CMake GUI. You can also directly use the command line or modify the default paths at the beginning of CMakeLists.txt.

- If you have generated a CMake project before, after modifying CMakeLists.txt, you need to delete the generated files and reconfigure.

- When configuring, if an error occurs indicating a library cannot be found, modify the library path to the correct path, and you can directly click Configure again without deleting the cache.

### Windows

- Generate .sln solution file:

1. Open CMake GUI, set the source code directory to the project root directory (containing CMakeLists.txt)
2. Set the build directory. You can create a new folder at any other location, or create a build folder in the project root directory
3. No need to modify CMAKE_INSTALL_PREFIX, this variable will not be used
4. Click Configure, select `Visual Studio 17 2022` as the generator, keep other settings unchanged
5. Wait for reading to complete, set the paths for all dependencies. Note that the paths for SDL and VTK need to be specified to `install/cmake/` and `install/lib/cmake/vtk-x.x` respectively
6. Click Generate to generate the CMake project
7. Click Open Project to open the project in Visual Studio. In "Solution Explorer", find the "RendererOptiX" project, right-click and select "Build" to start compilation
8. Return to the source code directory bin/ folder, move the executable file and two library files from the bin/Release folder to the bin directory
9. Confirm that the SDL and VTK dynamic libraries have been correctly copied to the bin directory
10. Enable `cache=true` to start generating cache files and proceed with subsequent rendering

- Generate CMake configuration files, the project does not depend on Visual Studio .sln
- You can use CLion IDE for compilation

### Linux

1. Open CMake GUI, set the source code directory to the project root directory (containing CMakeLists.txt)
2. Set the build directory. You can create a new folder at any other location, or create a build folder in the project root directory
3. No need to modify CMAKE_INSTALL_PREFIX, this variable will not be used
4. Click Configure, select `Unix Makefiles` as the generator, keep other settings unchanged
5. Wait for reading to complete, set the paths for all dependencies
6. Click Generate to generate the CMake project
7. Close the GUI, open the command line at the current location, and enter the build command

```bash
cmake --build . --parallel
```

- You can also use other IDEs such as CLion for compilation. If using CLion, on Windows you need to select the Visual Studio toolchain architecture set to x86_amd64. If unavailable, select the amd64 architecture; on Linux, use the default toolchain.

## Next Steps

After successful build, please refer to:

- [Configuration Reference](configuration.md) - Learn about configuration options
- [Usage Guide](usage.md) - Using the program
- [Technical Details](technical-details.md) - Deep dive into implementation