cmake_minimum_required(VERSION 4.0)
set(EXECUTABLE_NAME "RendererOptiX")

#CUDA
if (UNIX)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
else ()
    #Windows使用CUDA安装时自动设置的环境变量
endif ()

#自动检测GPU的计算能力标准
set(CMAKE_CUDA_ARCHITECTURES native)
project(${EXECUTABLE_NAME} LANGUAGES C CXX CUDA)

#设置语言标准
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

#包含项目头文件
include_directories("${CMAKE_SOURCE_DIR}/include")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin")

#OptiX
set(OPTIX_VERSION "9.0.0")
if (UNIX)
    #需要将整个OptiX安装文件夹复制到用户桌面
    set(OPTIX_ROOT "~/Desktop/CUDA/NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64-x86_64")
else ()
    #使用默认安装目录
    set(OPTIX_ROOT "C:/ProgramData/NVIDIA Corporation/OptiX SDK ${OPTIX_VERSION}")
endif ()
include_directories("${OPTIX_ROOT}/include")

#SDL
if (WIN32)
    #需要将SDL.dll复制到可执行文件目录下
    set(SDL2_DIR "${CMAKE_SOURCE_DIR}/lib/SDL2/cmake")
else ()
    #Linux使用系统包管理器安装libsdl2-dev
endif ()
find_package(SDL2 REQUIRED)

#VK
find_package(Vulkan REQUIRED)
include_directories("${Vulkan_INCLUDE_DIRS}")

#VTK
set(VTK_DIR "${CMAKE_SOURCE_DIR}/lib/VTK9.5.2/lib/cmake/vtk-9.5")
find_package(VTK REQUIRED)

#OptiX着色器文件列表
set(OPTIX_SHADER_SOURCES
        shader/Shader.cu
)
#编译着色器文件
foreach(SHADER_SOURCE ${OPTIX_SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
    set(PTX_OUTPUT_FILE "${CMAKE_SOURCE_DIR}/shader/${SHADER_NAME}.ptx")

    #构造编译命令
    set(PTX_COMMAND ${CMAKE_CUDA_COMPILER} -ptx)
    if (WIN32)
        list(APPEND PTX_COMMAND "-Xcompiler=/utf-8")
    endif ()
    list(APPEND PTX_COMMAND -o "${PTX_OUTPUT_FILE}")
    list(APPEND PTX_COMMAND -I "${CMAKE_SOURCE_DIR}/include")
    list(APPEND PTX_COMMAND -I "${OPTIX_ROOT}/include")
    list(APPEND PTX_COMMAND -std=c++${CMAKE_CXX_STANDARD})
    list(APPEND PTX_COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}")

    #执行编译命令
    add_custom_command(
            OUTPUT ${PTX_OUTPUT_FILE}
            COMMAND ${PTX_COMMAND}
            DEPENDS ${SHADER_SOURCE}
            COMMENT "Compiling PTX shader: ${SHADER_NAME}.cu..."
            #此处不能添加-arch参数，否则会触发ptxas汇编器的验证步骤导致编译失败
    )
    list(APPEND GENERATED_PTX_FILES ${PTX_OUTPUT_FILE})
endforeach()
add_custom_target(OPTIX_SHADERS ALL DEPENDS ${GENERATED_PTX_FILES})

#添加构建目标
add_executable(${EXECUTABLE_NAME}
        src/Global/Main.cu
        src/OGL/glad.c
        include/Global/HostFunctions.cuh
        src/Global/HostFunctions.cu
        include/Global/DeviceFunctions.cuh
        include/GraphicsAPI/SDL_D3D11Window.cuh
        include/GraphicsAPI/SDL_D3D12Window.cuh
        include/GraphicsAPI/SDL_GLWindow.cuh
        include/GraphicsAPI/SDL_GraphicsWindow.cuh
        include/GraphicsAPI/SDL_VKWindow.cuh
        src/GraphicsAPI/SDL_D3D11Window.cu
        src/GraphicsAPI/SDL_D3D12Window.cu
        src/GraphicsAPI/SDL_GLWindow.cu
        src/GraphicsAPI/SDL_GraphicsWindow.cu
        src/GraphicsAPI/SDL_VKWindow.cu
        include/Global/RendererImpl.cuh
        src/Global/RendererImpl.cu
        include/Global/Shader.cuh
        include/Util/VTKReader.cuh
        src/Util/VTKReader.cu
        include/Util/ColorRamp.cuh
        include/Global/Renderer.cuh
        src/Global/Renderer.cu
)

#确保在链接主程序之前，所有PTX文件都已生成
add_dependencies(${EXECUTABLE_NAME} OPTIX_SHADERS)

#启用分离编译
set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if (WIN32)
    target_compile_options(${EXECUTABLE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif ()

#D3D
if (WIN32)
    target_link_libraries(${EXECUTABLE_NAME} PUBLIC d3d11 d3d12 dxgi d3dcompiler winmm dxguid)
endif ()

#SDL
target_link_libraries(${EXECUTABLE_NAME} PUBLIC SDL2::SDL2 SDL2::SDL2main)

#VK
target_link_libraries(${EXECUTABLE_NAME} PUBLIC Vulkan::Vulkan)

#VTK
target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${VTK_LIBRARIES})
if(VTK_VERSION VERSION_GREATER_EQUAL "9.0")
    vtk_module_autoinit(TARGETS ${EXECUTABLE_NAME} MODULES ${VTK_LIBRARIES})
endif()
