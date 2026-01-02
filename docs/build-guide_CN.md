# 构建指南

## 系统要求

### 硬件要求

- **GPU**：NVIDIA GPU，Turing架构及以上（拥有RT Core），支持CUDA 和OptiX。
  - 最低：CUDA Compute Capability 7.5+（sm_75）
  - 推荐：GeForce RTX 30 系列或更高
  - Game Ready / Studio 驱动均可
- **内存**：至少8GB RAM（推荐16GB或更多，取决于缓存读写线程数）
- **显存**：至少8GB（推荐12GB或更多，取决于VTK粒子规模）

### 软件要求

- **[CMake](https://cmake.org)**：4.0或更高版本
- **[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)**：12.0+
- **[OptiX SDK](https://developer.nvidia.com/rtx/ray-tracing/optix)**：9.0.0+
- **[Vulkan SDK](https://www.vulkan.org/)**: 1.1及以上

#### Windows

- **操作系统**：Windows 10/11
- **编译器**：[Visual Studio 2022 / 2026](https://visualstudio.microsoft.com)（支持 C++20），并安装“使用C++的桌面开发”模块

#### Linux

- **操作系统**：Ubuntu 22.04/24.04+ 或其他发行版
- **编译器**：GCC 10+ 或 Clang 12+（支持 C++20）

```bash
sudo apt update && sudo apt install gcc g++ make
#可以使用 gcc -v 查看版本号
```

- **系统包管理器的CMake通常不是最新版本，通常情况下为3.2.x**。要安装最新版本，Ubuntu系统可以使用自带的App Center，选择最新版进行安装。其他发行版也可以[添加CMake仓库](install-latest-camke_CN.md)后使用包管理器安装，亦或是前往[官网](https://cmake.org/download)手动下载预编译包或源码。

```bash
sudo apt install cmake cmake-qt-gui
```

- 此文档为Debian系发行版提供构建指南，源代码在Ubuntu 24.04上已验证可编译运行。若使用其他发行版，需要使用发行版对应的系统命令。若切换到其他发行版后遇到报错，可能需要修改部分源代码。
- 本项目的所有代码以跨平台为编写原则，除少量图形API调用代码外，没有平台特定代码。

## 依赖库安装

- 安装前确保将显卡驱动更新到最新版。

### CUDA Toolkit

1. 从[NVIDIA官网](https://developer.nvidia.com/cuda-downloads)下载并安装 CUDA Toolkit
2. 确保 `nvcc.exe`/`nvcc` 在系统 PATH 中

#### Windows

直接运行安装程序进行安装即可，建议安装到默认目录。

#### Linux

建议使用“deb(local)”方式进行安装。

### OptiX SDK

从[NVIDIA官网](https://developer.nvidia.com/designworks/optix/download)下载 OptiX SDK 9.0.0。

#### Windows

直接运行安装程序，建议安装到默认目录。

#### Linux

建议解压到无需sudo即可读写的用户目录，例如：  
`~/Public/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64`

### SDL2

#### Windows

1. 从[SDL2的GitHub仓库](https://github.com/libsdl-org/SDL)下载源代码手动编译或下载 “devel-...-VC” 的预编译版本

- 若使用SDL3，则需要修改include/Global/HostFunctions.cuh中的SDL头文件包含。
- 若从源代码编译，可以参考[SDL编译指南](compile-sdl_CN.md)。

2. 解压下载的预编译包或提取手动编译得到SDL安装文件夹

#### Linux

可以从源代码编译，但推荐直接使用系统包管理器安装：

```bash
sudo apt install libsdl2-dev
```

### Vulkan SDK

1. 从[Vulkan官网](https://vulkan.lunarg.com/)下载并安装 Vulkan SDK
2. 确保环境变量正确设置

#### Windows

建议保持默认安装位置。安装完成后可以从开始菜单中找到Vulkan Cude示例并运行，若正常运行则表示配置成功。

#### Linux

请下载tarball版本的SDK，解压后得到Vulkan安装文件夹。

### VTK

1. 从[VTK官网](https://vtk.org/download)下载源代码
2. 可以参考[VTK编译指南](compile-vtk_CN.md)进行编译
3. 提取手动编译得到的VTK安装文件夹

## 编译

以下介绍使用CMake GUI的方式。也可以直接使用命令行或修改CMakeLists.txt开头的默认路径。

- 如果生成过CMake项目，在修改CMakeLists.txt后，需要删除生成的文件并重新配置。

- Configure时，若报错找不到库，则修改库路径为正确的路径后，无需删除缓存，直接再次点击Configure即可。

### Windows

- 生成.sln解决方案文件：

1. 打开CMake GUI，设置源代码目录为项目根目录（包含CMakeLists.txt）
2. 设置构建目录，可以在任意其他位置新建一个文件夹，或在项目根目录新建build文件夹
3. 无需修改CMAKE_INSTALL_PREFIX，不会使用这个变量
4. 点击Configure，生成器选择`Visual Studio 17 2022`，其他设置保持不变
5. 等待读取完成，设置所有依赖库的路径，注意SDL和VTK的路径需要分别指定到`install/cmake/`和`install/lib/cmake/vtk-x.x`
6. 点击Generate生成CMake项目
7. 点击Open Project在Visual Studio中打开项目，在“解决方案资源管理器”中找到“RendererOptiX”项目，右键“生成”，开始编译。
8. 回到源代码目录bin/文件夹，将bin/Release文件夹下的可执行文件和两个库文件移动到bin目录下
9. 确认SDL和VTK的动态库已经正确拷贝到bin目录下
10. 开启`cache=true`，开始生成缓存文件并进行后续渲染

- 生成CMake配置文件，项目不依赖于Visual Studio .sln
- 可以使用CLion IDE进行编译

### Linux

1. 打开CMake GUI，设置源代码目录为项目根目录（包含CMakeLists.txt）
2. 设置构建目录，可以在任意其他位置新建一个文件夹，或在项目根目录新建build文件夹
3. 无需修改CMAKE_INSTALL_PREFIX，不会使用这个变量
4. 点击Configure，生成器选择`Unix Makefiles`，其他设置保持不变
5. 等待读取完成，设置所有依赖库的路径
6. 点击Generate生成CMake项目
7. 关闭GUI，在当前位置打开命令行，输入编译命令

```bash
cmake --build . --parallel
```

- 也可以使用其他IDE如CLion进行编译。若使用CLion，在Windows下需选择使用Visual Studio工具链架构设置为x86_amd64。若不可用，则选择amd64架构；Linux下使用默认工具链即可。

## 下一步

构建成功后，请参考：

- [配置参考](configuration_CN.md) - 了解配置选项
- [使用指南](usage_CN.md) - 使用程序
- [技术细节](technical-details_CN.md) - 深入了解实现
