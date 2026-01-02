# VTK库编译指南

本文档指导编译最小化的VTK库，包含核心数据结构和基础IO模块，满足读取VTK文件的需要。

## 下载并解压源代码

前往[VTK官网](https://vtk.org/download)下载最新版本的源代码（.tar.gz）并解压。

- Windows系统自带的解压缩速度较慢，对于VTK源代码，建议安装专门的解压缩软件，如[BandiZip](https://en.bandisoft.com/bandizip/)。tar.gz在Windows下通常需要解压两次。
- Linux直接使用系统自带的tar解压即可。

可参考SDL库的方式创建目录结构：

```text
VTK/
    build/
    install/
    source/
        CMakeLists.txt
        ...
```

## 安装构建工具和CMake

可参考[构建指南](build-guide.md)的前半部分，根据操作系统安装最新版本的编译器和CMake。

## 配置CMake项目

1. 打开CMake GUI，设置源代码目录和构建目录
2. 点击Configure，选择系统对应的生成器（Windows为`Visual Studio 17 2022`，Linux为`Unix Makefiles`），其他设置保持默认
3. 等待配置完成，出现红色选项

- Linux下，Configure时可能会报错：找不到.clang-tidy文件，导致配置失败。此时可以根据报错的位置，删除CMakeLists.txt的对应命令（通常为倒数第二条命令），或在源代码根目录创建一个空的.clang-tidy文件，这不会影响编译。

4. 取消勾选所有复选框，将所有下拉列表项设置为“DONT_WANT”，编译类型选择或仅保留Release，设置安装路径为以上目录配置过程中的install文件夹
5. 勾选“BUILD_SHARED_LIBS”选项，将“StandAlone”一项设置为“WANT”
6. 其他设置保持第4步不变。若需要同时编译其他VTK功能模块，可以选择性开启。

- 编译部分模块需要系统中已经安装有对应库，如QT模块，没有安装则会导致配置无效。对于RendererOptiX渲染器，仅需要编译“StandAlone”模块，其中包含VTK核心数据结构和基础IO函数

- 若配置无效或出现其他错误，需要删除build目录下所有文件并点击GUI左上角的Files -> Delete Cache清空所有缓存，并重新执行配置步骤

7. 点击Generate生成项目。若无报错，建议备份此时的build文件夹，这样在编译过程中出错，需要重新编译时，无需重新执行配置步骤

## 编译

### Windows

直接点击Open Project，在Visual Studio中打开项目，在“解决方案资源管理器”中找到“ALL_BUILD”项目，右键点击“生成”，等待编译完成。

- 编译时间取决于CPU核心数和性能，在主流消费级CPU上需要约20分钟，慢于在Linux上的编译速度。

### Linux

关闭CMake GUI，在build目录下打开控制台，使用命令进行编译：

```bash
cmake --build . -j CPU线程数
```

- Linux下，若CPU占用率过高，系统可能会Kill掉一些编译进程导致部分目标编译失败，因此强烈建议不要使用所有CPU线程进行编译，可以设置为CPU线程数的80%。

## 安装

### Windows
在“解决方案资源管理器”中找到“INSTALL”项目，右键生成即可。

- 若编译无报错且生成报错，可以检查设置的CMAKE_INSTALL_PREFIX，若使用了反斜杠路径，则会导致安装失败。此时无需重新编译，找到INSTALL项目的CMakeLists.txt，修改安装路径为正斜杠分割即可。

### Linux

在build目录下执行安装命令：

```bash
cmake --install .
```

## 集成到项目

同SDL库，设置lib路径即可。

- 在Windows下，需要拷贝bin目录下的所有动态库文件到可执行文件目录
- Linux下通过CMake自动设置环境变量，可以不拷贝。
