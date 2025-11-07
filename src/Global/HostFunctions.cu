#include <Global/HostFunctions.cuh>
#include <filesystem>
#include <fstream>
#include <optix_function_table_definition.h>

namespace project {
    // ====== 库函数错误检查 ======

    //SDL
    void _SDL_CheckError(bool isError, const char * file, const char * function, int line) {
        if (isError) {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                         "[SDL] Error: %s. At file %s in function %s. (Line %d)",
                         SDL_GetError(), file, function, line);
            SDL_LogError(SDL_LOG_CATEGORY_ERROR, "Aborting due to SDL error!");
            exit(SDL_ERROR_EXIT_CODE);
        }
    }
    void _SDL_CheckErrorIntImpl(int val, const char * file, const char * function, int line) {
        _SDL_CheckError(val < 0, file, function, line);
    }
    void _SDL_CheckErrorPtrImpl(const void * val, const char * file, const char * function, int line) {
        _SDL_CheckError(val == nullptr, file, function, line);
    }
    void _SDL_CheckErrorPtrBool(SDL_bool val, const char * file, const char * function, int line) {
        _SDL_CheckError(val == SDL_FALSE, file, function, line);
    }

    //CUDA
    void _cudaCheckError(cudaError_t val, const char * file, const char * function, int line) {
        if (val != cudaSuccess) {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                         "[CUDA] Error: %s. At file %s in function %s. (Line %d)",
                         cudaGetErrorString(val), function, file, line);
            exit(CUDA_ERROR_EXIT_CODE);
        }
    }

    //OPTIX
    char optixPerCallLogBuffer[2048] = { 0 };
    size_t _optixPerCallLogSize = sizeof(optixPerCallLogBuffer);
    size_t * optixPerCallLogSize = &_optixPerCallLogSize;
    void _optixCheckError(OptixResult result, const char * file, const char * function, int line) {
        if (result != OPTIX_SUCCESS) {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                         "[OptiX] Error: %s. At file %s in function %s. (Line %d)",
                         optixGetErrorString(result), function, file, line);
            exit(OPTIX_ERROR_EXIT_CODE);
        }
    }
    void optixLogCallBackFunction(Uint32 level, const char * tag, const char * message, void * callbackData) {
        SDL_Log("[OptiX] [%u][%s] --> %s", level, tag, message);
    }
    void optixCheckPerCallLog() {
        if (_optixPerCallLogSize > 1) {
            SDL_Log("[OptiX] Per call log: %s", optixPerCallLogBuffer);
        }
    }

    //VK
    VKAPI_ATTR VkBool32 VKAPI_CALL vkDebugCallBackFunction(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void * pUserData)
    {
        SDL_LogPriority priority;
        switch (messageSeverity) {
            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
                priority = SDL_LOG_PRIORITY_VERBOSE;
                break;
            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
                priority = SDL_LOG_PRIORITY_INFO;
                break;
            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
                priority = SDL_LOG_PRIORITY_WARN;
                break;
            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
                priority = SDL_LOG_PRIORITY_ERROR;
                break;
            default:
                priority = SDL_LOG_PRIORITY_INFO;
        }
        SDL_LogMessage(SDL_LOG_CATEGORY_APPLICATION, priority,
                       "[VK] [Validation]: %s", pCallbackData->pMessage);
        return VK_FALSE;
    }
    void _vkCheckError(VkResult result, const char * file, const char * function, int line) {
        if (result != VK_SUCCESS) {
            SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                         "[VK] Error: [%d]. At file %s in function %s. (Line %d)",
                         result, function, file, line);
            exit(VULKAN_ERROR_EXIT_CODE);
        }
    }

    //D3D
    void _D3DCheckError(HRESULT result, const char * file, const char * function, int line) {
        if (FAILED(result)) {
            WCHAR * errorText = nullptr;
            FormatMessageW(
                    FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER,
                    nullptr,
                    result,
                    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                    (LPWSTR)&errorText,
                    0,
                    nullptr);
            std::wstring resultStr;
            if (errorText != nullptr) {
                resultStr = errorText;
                LocalFree(errorText);
            }
            SDL_Log("[D3D] Error: %ls. At file %s in function %s. (Line %d)", resultStr.c_str(), file, function, line);
            exit(D3D_ERROR_EXIT_CODE);
        }
    }

    // ====== 随机数生成函数 ======

    __global__ void __initializeDeviceRandomGenerators(curandState * dev_stateArray) {
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        const unsigned int tid = gridDim.x * blockDim.x * y + x;
        curand_init(tid ^ clock64(), tid, 0, dev_stateArray + tid);
    }
    void RandomGenerator::initDeviceRandomGenerators(curandState * & dev_stateArray, size_t x, size_t y, cudaStream_t stream) {
        const dim3 blocks(x % 16 == 0 ? x / 16 : x / 16 + 1,
                          y % 16 == 0 ? y / 16 : y / 16 + 1, 1);
        constexpr dim3 threads(16, 16, 1);

        cudaCheckError(cudaMallocAsync(&dev_stateArray, x * y * sizeof(curandState), stream));
        __initializeDeviceRandomGenerators<<<blocks, threads, 0, stream>>>(dev_stateArray);
        cudaCheckError(cudaStreamSynchronize(stream));
    }
    void RandomGenerator::freeDeviceRandomGenerators(curandState * & dev_stateArray, cudaStream_t stream) {
        cudaCheckError(cudaFreeAsync(dev_stateArray, stream));
        dev_stateArray = nullptr;
    }

    // ====== 文件读写函数 ======

    std::string FileHelper::readTextFile(const std::string & filePath) {
        const std::ifstream file(filePath);
        if (!std::filesystem::exists(filePath)) {
            throw std::runtime_error("File " + filePath + " does not exist!");
        }
        if(!file.is_open() || file.fail()) {
            throw std::runtime_error("Failed to open file: " + filePath);
        }
        std::ostringstream ss;
        ss << file.rdbuf();
        return ss.str();
    }
    void FileHelper::writeTextFile(const std::string & filePath, const std::string & content) {
        std::filesystem::path pathObj(filePath);
        std::filesystem::path parentPath = pathObj.parent_path();
        if (!parentPath.empty()) {
            try {
                if (!std::filesystem::exists(parentPath)) {
                    if (!std::filesystem::create_directories(parentPath)) {
                        throw std::runtime_error("Failed to create directories to put new file!");
                    }
                }
            } catch (const std::filesystem::filesystem_error& e) {
                throw std::runtime_error("Filesystem error when creating directories!");
            }
        }
        std::ofstream outfile(filePath);
        if (outfile.is_open()) {
            outfile << content;
            outfile.close();
        } else {
            throw std::runtime_error("Failed to write text to file: " + content);
        }
    }
}