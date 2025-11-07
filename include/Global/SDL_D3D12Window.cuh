#ifndef OPTIXTEST_SDL_D3D12WINDOW_CUH
#define OPTIXTEST_SDL_D3D12WINDOW_CUH

#ifdef _WIN32
#include <Global/HostFunctions.cuh>

namespace project {
    typedef struct Direct3D12Args {
        ComPtr<ID3D12Debug> debugInterface;
        ComPtr<ID3D12Device2> device;
        ComPtr<IDXGIAdapter1> dxgiAdapter;
        ComPtr<IDXGIFactory4> dxgiFactory4;
        ComPtr<IDXGIFactory5> dxgiFactory5;
        ComPtr<ID3D12CommandQueue> commandQueue;
        ComPtr<IDXGISwapChain4> swapChain;
        std::array<ComPtr<ID3D12Resource>, 2> backBuffers;
        Uint32 rtvDescriptorSize;
        std::array<ComPtr<ID3D12CommandAllocator>, 2> commandAllocators;
        ComPtr<ID3D12GraphicsCommandList> commandList;
        ComPtr<ID3D12DescriptorHeap> rtvDescriptorHeap;
        ComPtr<ID3D12Fence> fence;
        HANDLE fenceEvent;
        std::array<Uint32, 2> fenceValues;
        Uint32 frameIndex;

        std::array<ComPtr<ID3D12Resource>, 2> sharedResources;
        std::array<cudaExternalMemory_t, 2> cudaExtMemories;
        std::array<cudaMipmappedArray_t, 2> cudaMipArrays;
        std::array<cudaArray_t, 2> cudaArrays;
        std::array<cudaSurfaceObject_t, 2> cudaSurfaces;
    } Direct3D12Args;

    //创建窗口
    SDL_Window * SDL_D3D12CreateWindow(const char * title, int windowWidth, int windowHeight);
    void SDL_D3D12DestroyWindow(SDL_Window * & window);

    //初始化D3D12资源
    Direct3D12Args SDL_D3D12InitializeResource(SDL_Window * window);
    void SDL_D3D12CleanupResource(Direct3D12Args & args);

    //渲染一帧
    std::pair<ComPtr<ID3D12Resource>, ComPtr<ID3D12Resource>> SDL_D3D12PrepareFrame(Direct3D12Args & args);
    void SDL_D3D12PresentFrame(Direct3D12Args & args, std::pair<ComPtr<ID3D12Resource>, ComPtr<ID3D12Resource>> & pair);
}
#endif //_WIN32
#endif //OPTIXTEST_SDL_D3D12WINDOW_CUH
