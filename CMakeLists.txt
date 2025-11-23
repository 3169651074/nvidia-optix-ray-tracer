cmake_minimum_required(VERSION 4.0)
set(EXECUTABLE_NAME "RendererOptiX")

#CACHE PATH会将路径写入缓存文件，修改后需要删除所有CMake缓存并重新生成

#CUDA
if (WIN32)
    set(CMAKE_CUDA_COMPILER_DEFAULT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe")
    set(CUDA_INCLUDE_DIR_DEFAULT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include")
else ()
    set(CMAKE_CUDA_COMPILER_DEFAULT "/usr/local/cuda/bin/nvcc")
    set(CUDA_INCLUDE_DIR_DEFAULT "/usr/local/cuda/include")
endif ()

#OptiX
if (WIN32)
    set(OPTIX_ROOT_DEFAULT "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0")
else ()
    set(OPTIX_ROOT_DEFAULT "~/Public/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64")
endif ()

#SDL
if (WIN32)
    set(CONFIG_SDL2_DIR_DEFAULT "C:/SDL2/cmake")
else ()
    #若手动编译SDL库，则在此处设置路径，否则无需更改
    set(CONFIG_SDL2_DIR_DEFAULT "")
endif ()

#VK
if (WIN32)
    set(CONFIG_VULKAN_DIR_DEFAULT "C:/VulkanSDK/1.4.321.1")
else ()
    set(CONFIG_VULKAN_DIR_DEFAULT "~/Public/vulkan-1.4.321.1/x86_64")
endif ()

#VTK
if (WIN32)
    set(CONFIG_VTK_DIR_DEFAULT "C:/VTK-9.5.0/lib/cmake/vtk-9.5")
else ()
    set(CONFIG_VTK_DIR_DEFAULT "~/Public/VTK-9.5.2/lib/cmake/vtk-9.5")
endif ()

#=======================================================================================

#CUDA
set(CMAKE_CUDA_COMPILER ${CMAKE_CUDA_COMPILER_DEFAULT} CACHE PATH "Absolute path of cuda compiler (nvcc)")
set(CUDA_INCLUDE_DIR ${CUDA_INCLUDE_DIR_DEFAULT} CACHE PATH "Absolute path of cuda include directory")
include_directories(${CUDA_INCLUDE_DIR})
message(STATUS "NVCC Path: " ${CMAKE_CUDA_COMPILER})

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
set(OPTIX_ROOT ${OPTIX_ROOT_DEFAULT} CACHE PATH "Absolute path of OptiX SDK")
message(STATUS "OptiX Root Directory: " ${OPTIX_ROOT})
include_directories("${OPTIX_ROOT}/include")

#SDL
if (WIN32)
    set(CONFIG_SDL2_DIR ${CONFIG_SDL2_DIR_DEFAULT} CACHE PATH "Absolute path of SDL")
else ()
    #Linux使用系统包管理器安装libsdl2-dev
    #仍然提供一个可选的根目录设置，以防用户手动编译
    #set(CONFIG_SDL2_DIR "" CACHE PATH
    #        "Absolute path to the root directory of the SDL2 library (not needed if using system package)")

    #如果用户提供了路径，则使用它来辅助find_package
    if (CONFIG_SDL2_DIR_DEFAULT)
        set(CONFIG_SDL2_DIR "${CONFIG_SDL2_DIR_DEFAULT}")
    endif()
endif ()
set(SDL2_DIR ${CONFIG_SDL2_DIR})
find_package(SDL2 REQUIRED)

#VK
set(CONFIG_VULKAN_SDK_PATH ${CONFIG_VULKAN_DIR_DEFAULT} CACHE PATH "Absolute path of vulkan SDK")
if (UNIX)
    #需要额外设置运行时环境变量
    set(ENV{VULKAN_SDK} ${CONFIG_VULKAN_SDK_PATH})
    set(ENV{VK_LAYER_PATH} "${CONFIG_VULKAN_SDK_PATH}/etc/vulkan/explicit_layer.d")
endif ()
find_package(Vulkan REQUIRED)
include_directories("${Vulkan_INCLUDE_DIRS}")

#VTK
set(CONFIG_VTK_DIR ${CONFIG_VTK_DIR_DEFAULT} CACHE PATH "Absolute path of VTK")
set(VTK_DIR ${CONFIG_VTK_DIR})
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
    if (DISABLE_COMPILE_ERROR)
        #禁用着色器文件extern警告
        #list(APPEND PTX_COMMAND -diag-suppress=20044)
    endif ()

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
file(GLOB_RECURSE SRCS
        "${CMAKE_SOURCE_DIR}/src/*.c"
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/*.cu"
)
add_executable(${EXECUTABLE_NAME} ${SRCS})

#确保在链接主程序之前，所有PTX文件都已生成
add_dependencies(${EXECUTABLE_NAME} OPTIX_SHADERS)

#启用分离编译
set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if (WIN32)
    #为MSVC添加UTF-8编码支持
    target_compile_options(${EXECUTABLE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    target_compile_options(${EXECUTABLE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)

    if (DISABLE_COMPILE_ERROR)
        #禁用D3D头文件编译警告
        #add_compile_options(-diag-suppress=68,174,1394)
    endif ()
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
