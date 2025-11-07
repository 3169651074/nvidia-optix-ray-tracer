#ifndef OPTIXTEST_SDL_D3D11WINDOW_CUH
#define OPTIXTEST_SDL_D3D11WINDOW_CUH

#ifdef _WIN32
#include <Global/HostFunctions.cuh>

namespace project {
    typedef struct Direct3D11Args {
        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> deviceContext;
        ComPtr<IDXGIFactory2> dxgiFactory;
        ComPtr<IDXGIAdapter> adapter;
        ComPtr<IDXGISwapChain1> swapChain;
        ComPtr<ID3D11Texture2D> backBuffer;
        ComPtr<ID3D11RenderTargetView> renderTargetView;
        ComPtr<ID3DBlob> vsBlob;
        ComPtr<ID3DBlob> psBlob;
        ComPtr<ID3D11VertexShader> vertexShader;
        ComPtr<ID3D11PixelShader> pixelShader;
        ComPtr<ID3D11InputLayout> inputLayout;
        ComPtr<ID3D11Buffer> vertexBuffer;

        ComPtr<ID3D11Texture2D> sharedTexture;
        cudaGraphicsResource_t cudaResource;
    } Direct3D11Args;

    //创建窗口
    SDL_Window * SDL_D3D11CreateWindow(const char * title, int windowWidth, int windowHeight);
    void SDL_D3D11DestroyWindow(SDL_Window * & window);

    //初始化D3D11资源
    Direct3D11Args SDL_D3D11InitializeResource(SDL_Window * window);
    void SDL_D3D11CleanupResource(Direct3D11Args & d3d11Args);

    //映射纹理资源
    cudaSurfaceObject_t SDL_D3D11MapCudaResource(Direct3D11Args & args, cudaStream_t stream = nullptr);
    void SDL_D3D11UnmapCudaResource(Direct3D11Args & args, cudaSurfaceObject_t & object, cudaStream_t stream = nullptr);

    //绘制画面
    void SDL_D3D11PresentFrame(const Direct3D11Args & args);
}
#endif //_WIN32
#endif //OPTIXTEST_SDL_D3D11WINDOW_CUH
