# SDL2库编译指南

本指南包含了SDL2/SDL3库从源代码编译的方法，并附加SDL扩展库SDL_ttf的编译方法。其他SDL扩展库如SDL_mixer，可以参看ttf库的编译配置方式。

## 下载源代码

从[SDL的GitHub仓库](https://github.com/libsdl-org/SDL)下载最新的Release版本源代码。注意最新Release为SDL3。若需要SDL2，请在Release列表中查找。

## 解压源代码

创建一个新文件夹，包含build、install、source文件夹，并解压缩源代码到source目录：

```text
SDL/
    build/
    install/
    source/
        ...
        CMakeLists.txt
```

## 配置CMake项目

MSVC为多配置编译器，在配置时不指定Debug/Release，在编译和安装时均需要指定；MinGW/GCC为单配置编译器，在配置时指定Debug/Release，编译和安装时无需指定。

### Windows

在build目录下打开命令行，使用配置命令：

```cmd
cmake ../source -G "Visual Studio 17 2022" -DCMAKE_INSTALL_PREFIX="../install"
```

编译：

```cmd
cmake --build . --config Release --parallel
或
cmake --build . --config Release -j CPU线程数
```

安装：

```cmd
cmake --install . --config Release
```

### Linux

在build目录下配置：

```bash
cmake ../source -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="../install" -DCMAKE_BUILD_TYPE=Release
```

编译：

```bash
cmake --build . --parallel
```

安装：

```bash
cmake --install .
```

- 若使用CMake GUI，除了CMAKE_INSTALL_PREFIX设置为install文件夹、CMAKE_BUILD_TYPE设置为Release，其他选项保持默认即可。

## 在当前项目中使用安装文件

安装完成后，在install文件夹下，可以看到SDL库编译后的文件：

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

在当前项目的CMakeLists.txt中，设置变量SDL2_DIR为lib/cmake的路径，并自动查找包：

```cmake
set(SDL2_DIR "/path/to/install/lib/cmake")
find_package(SDL2 REQUIRED)
```

重新配置项目的CMake，若无报错，则说明配置成功。

- 注意：在Windows下，需要手动将动态库文件SDL2.dll复制到项目可执行文件的生成目录，即和可执行文件在同一目录，否则运行时无法正确链接到动态库

# SDL扩展库编译指南，以SDL_ttf为例

## 下载源代码

同SDL库，将下载好的源代码解压到相同的目录结构中：

```text
SDL_ttf/
    build/
    install/
    source/
        ...
        CMakeLists.txt
```

## 配置CMake项目

由于SDL_ttf依赖于SDL，需要在配置时指定SDL库的路径，否则无法生成项目。ttf源代码中包含示例程序，编译这些示例程序需要链接到SDL.lib和SDLmain.lib。

### Windows

配置：

```cmd
cmake ../source -G "Visual Studio 17 2022" -DCMAKE_INSTALL_PREFIX=../install -DSDL2_DIR="" -DSDL2_LIBRARY="" -DSDL2_INCLUDE_DIR="" -DSDL2_MAIN_LIBRARY=""
```

其中：

- SDL2_DIR为SDL的安装路径：/path/to/SDL/install
- SDL2_LIBRARY为SDL.lib的路径：/path/to/SDL/install/lib/SDL2.lib
- SDL2_INCLUDE_DIR为SDL头文件路径：/path/to/SDL/install/include/SDL2
- SDL2_MAIN_LIBRARY为SDLmain.lib的路径：/path/to/SDL/install/lib/SDL2main.lib

编译并安装：

```cmd
cmake --build . --parallel --config Release
cmake --install . --config Release
```

### Linux

## 在当前项目中使用安装文件

方法同SDL库，只是将SDL2_DIR换成SDL2_ttf_DIR；SDL2换成SDL2_ttf即可。