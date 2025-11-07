#ifndef OPTIXTEST_SDL_GRAPHICSWINDOW_CUH
#define OPTIXTEST_SDL_GRAPHICSWINDOW_CUH

#ifdef _WIN32
#include <Global/SDL_D3D11Window.cuh>
#include <Global/SDL_D3D12Window.cuh>
#endif

#include <Global/SDL_GLWindow.cuh>
#include <Global/SDL_VKWindow.cuh>
#include <Global/DeviceFunctions.cuh>

//SDL图形API包装类，简化访问
namespace project {
    //相机
    typedef struct SDL_GraphicsWindowCamera {
        float3 upDirection;
        float3 cameraCenter;
        float3 cameraTarget;
        float3 cameraU, cameraV, cameraW;
    } SDL_GraphicsWindowCamera;

    //图形API类型
    typedef enum class SDL_GraphicsWindowAPIType {
        OPENGL, VULKAN,
#ifdef _WIN32
        DIRECT3D11, DIRECT3D12
#endif
    } SDL_GraphicsWindowAPIType;

    //窗口参数
    typedef struct SDL_GraphicsWindowArgs {
        SDL_Window * window = nullptr;
        SDL_GraphicsWindowAPIType type;

        OpenGLArgs glArgs = {};
        SDL_GLContext glContext = nullptr;
        VulkanArgs vkArgs = {};
        Uint32 vkImageIndex = 0;

#ifdef _WIN32
        Direct3D11Args d3d11Args = {};
        Direct3D12Args d3d12Args = {};
        std::pair<ComPtr<ID3D12Resource>, ComPtr<ID3D12Resource>> d3d12Pair = {};
#endif

        size_t fpsLimit = 60;
        float mouseSensitivity = 0.001f;
        float pitchLimitDegree = 80.0f;
        float cameraMoveSpeedStride = 0.02f;
        size_t initialSpeedNTimesStride = 2;
        float cameraMoveSpeed = 0.0f;

        std::chrono::time_point<std::chrono::steady_clock> frameStartTime;
        std::chrono::microseconds targetFrameDuration;
        std::chrono::microseconds sleepMargin;

        cudaGraphicsResource_t cudaGraphicsResource;
        cudaSurfaceObject_t cudaSurfaceObject;
    } SDL_GraphicsWindowArgs;

    //按键参数
    typedef struct SDL_GraphicsWindowKeyMouseInput {
        bool keyW = false, keyA = false;
        bool keyS = false, keyD = false;
        bool keySpace = false, keyLShift = false;
        bool keyQuit = false;

        int mouseX = 0, mouseY = 0; //鼠标变量需要每一帧重置
    } SDL_GraphicsWindowKeyMouseInput;

    //根据基础信息计算相机视口向量
    SDL_GraphicsWindowCamera SDL_GraphicsWindowConfigureCamera(
            const float3 & center, const float3 & target,
            const float3 & up, SDL_GraphicsWindowAPIType type);

    //创建窗口
    SDL_GraphicsWindowArgs SDL_CreateGraphicsWindow(
            const char * title, int width, int height, SDL_GraphicsWindowAPIType type,
            size_t fpsLimit = 60, float mouseSensitivity = 0.001f, float pitchLimitDegree = 80.0f,
            float cameraMoveSpeedStride = 0.02f, size_t initialSpeedNTimesStride = 2);

    //记录帧开始
    void SDL_GraphicsWindowFrameStart(SDL_GraphicsWindowArgs & args);

    //根据相机和输入更新相机
    void SDL_GraphicsWindowUpdateCamera(
            SDL_Event & event, SDL_GraphicsWindowKeyMouseInput & input,
            SDL_GraphicsWindowArgs & args, SDL_GraphicsWindowCamera & camera);

    //准备用于写入的资源
    cudaSurfaceObject_t SDL_GraphicsWindowPrepareFrame(SDL_GraphicsWindowArgs & args);

    //呈现画面
    void SDL_GraphicsWindowPresentFrame(SDL_GraphicsWindowArgs & args);

    //记录帧结束并选择性延迟
    void SDL_GraphicsWindowFrameFinish(const SDL_GraphicsWindowArgs & args);

    //销毁窗口
    void SDL_DestroyGraphicsWindow(SDL_GraphicsWindowArgs & args);
}

#endif //OPTIXTEST_SDL_GRAPHICSWINDOW_CUH
