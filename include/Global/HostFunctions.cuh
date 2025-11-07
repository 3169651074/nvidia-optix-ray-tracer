#ifndef RENDEREROPTIX_HOSTFUNCTIONS_CUH
#define RENDEREROPTIX_HOSTFUNCTIONS_CUH

/*
 * 主机端头文件，不能用于着色器文件
 * 包含库错误检查函数以及随机数生成、文件读写等函数
 */

// ====== 库头文件包含 ======

#include <Global/DeviceFunctions.cuh>

//Windows
#ifdef _WIN32
#include <Windows.h>
#undef min
#undef max
#endif

//SDL
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>

//CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>

//OPTIX
#include <optix.h>
#include <optix_device.h>
#include <optix_function_table.h>
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

//OGL
#include <OGL/glad.h>
#include <OGL/khrplatform.h>
#include <cuda_gl_interop.h>

//VK
#include <vulkan/vulkan.h>
#include <SDL2/SDL_vulkan.h>
#ifdef _WIN32
#include <vulkan/vulkan_win32.h>
#else
#include <unistd.h>
#endif

//D3D
#ifdef _WIN32
//D3D11
#include <d3d11.h>
#include <dxgi1_5.h>
#include <wrl.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <cuda_d3d11_interop.h>

//D3D12
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <wrl.h>
#include <D3D12/directx/d3d12.h>
#include <D3D12/directx/d3dx12.h>

using namespace DirectX;
using namespace Microsoft::WRL;
#endif

//C++
#include <algorithm>
#include <string>
#include <array>
#include <vector>
#include <stack>

#include <cmath>
#include <cstring>
#include <cstdlib>

#include <random>
#include <chrono>
#include <thread>

namespace project {
    // ====== 随机数生成函数 ======

    class RandomGenerator {
    private:
        static inline std::random_device rd;
        static inline std::mt19937 generator{rd()};
        static inline std::uniform_real_distribution<float> distribution{0.0f, 1.0f};

    public:
        //初始化指定数量的curandState，阻塞直到初始化完成
        static void initDeviceRandomGenerators(curandState * & dev_stateArray, size_t x, size_t y, cudaStream_t stream = nullptr);
        static void freeDeviceRandomGenerators(curandState * & dev_stateArray, cudaStream_t stream = nullptr);

        //生成一个[0, 1)的浮点随机数
        static float randomDouble() {
            return distribution(generator);
        }

        //生成一个[min, max)之间的浮点随机数
        static float randomDouble(float min, float max) {
            return min + (max - min) * randomDouble();
        }

        //生成一个[min, max]之间的整数随机数
        template<typename T>
        static T randomInteger(T min, T max) {
            std::uniform_int_distribution<T> _distribution(min, max);
            return _distribution(generator);
        }
    };

    // ====== 文件读写函数 ======
    class FileHelper {
    public:
        //读取整个文本文件并存入字符串
        static std::string readTextFile(const std::string & filePath);

        //将字符串中所有内容以纯文本写入到指定文件
        static void writeTextFile(const std::string & filePath, const std::string & content);
    };

    // ====== 库函数错误检查 ======

    //SDL
#define SDL_ERROR_EXIT_CODE (-100)
    extern void _SDL_CheckErrorIntImpl(int val, const char * file, const char * function, int line);
    extern void _SDL_CheckErrorPtrImpl(const void * val, const char * file, const char * function, int line);
    extern void _SDL_CheckErrorPtrBool(SDL_bool val, const char * file, const char * function, int line);
#define SDL_CheckErrorInt(call) _SDL_CheckErrorIntImpl(call, __FILE__, __func__, __LINE__)
#define SDL_CheckErrorPtr(call) _SDL_CheckErrorPtrImpl(call, __FILE__, __func__, __LINE__)
#define SDL_CheckErrorBool(call) _SDL_CheckErrorPtrBool(call, __FILE__, __func__, __LINE__)

    //CUDA
#define CUDA_ERROR_EXIT_CODE (-200)
    extern void _cudaCheckError(cudaError_t err, const char * file, const char * function, int line);
#define cudaCheckError(call) _cudaCheckError(call, __FILE__, __func__, __LINE__)

    //OPTIX
#define OPTIX_ERROR_EXIT_CODE (-300)
    extern char optixPerCallLogBuffer[2048];
    extern size_t * optixPerCallLogSize;
    extern void optixCheckPerCallLog();
    extern void _optixCheckError(OptixResult result, const char * file, const char * function, int line);
#define optixCheckError(call) _optixCheckError(call, __FILE__, __func__, __LINE__)
    extern void optixLogCallBackFunction(Uint32 level, const char * tag, const char * message, void * callbackData);

    //VK
#define VULKAN_ERROR_EXIT_CODE (-400)
    extern void _vkCheckError(VkResult result, const char * file, const char * function, int line);
#define vkCheckError(result) _vkCheckError(result, __FILE__, __func__, __LINE__)
    extern VKAPI_ATTR VkBool32 VKAPI_CALL vkDebugCallBackFunction(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void * pUserData);

    //D3D
    #define D3D_ERROR_EXIT_CODE (-500)
    extern void _D3DCheckError(HRESULT result, const char * file, const char * function, int line);
#define D3DCheckError(result) _D3DCheckError(result, __FILE__, __func__, __LINE__)
}

#endif //RENDEREROPTIX_HOSTFUNCTIONS_CUH
