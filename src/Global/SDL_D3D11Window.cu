#ifdef _WIN32
#include <Global/SDL_D3D11Window.cuh>
using namespace project;

namespace {
    ComPtr<IDXGIFactory2> createDXGIFactory();
    ComPtr<IDXGIAdapter> findAdapter(ComPtr<IDXGIFactory2> & dxgiFactory2);
    std::pair<ComPtr<ID3D11Device>, ComPtr<ID3D11DeviceContext>> createDevice(ComPtr<IDXGIAdapter> & adapter);
    ComPtr<IDXGISwapChain1> createSwapchain(SDL_Window * window, ComPtr<IDXGIFactory2> & dxgiFactory2, ComPtr<ID3D11Device> & device);
    std::pair<ComPtr<ID3D11Texture2D>, ComPtr<ID3D11RenderTargetView>> getBackBufferAndRTV(
            ComPtr<IDXGISwapChain1> & swapChain, ComPtr<ID3D11Device> & device);
    std::pair<
            std::pair<ComPtr<ID3D11VertexShader>, ComPtr<ID3D11PixelShader>>,
            std::pair<ComPtr<ID3DBlob>, ComPtr<ID3DBlob>>> compileShader(ComPtr<ID3D11Device> & device);
    ComPtr<ID3D11InputLayout> createInputLayout(ComPtr<ID3D11Device> & device, ComPtr<ID3DBlob> & vsBlob, ComPtr<ID3DBlob> & psBlob);
    ComPtr<ID3D11Buffer> createVertexBuffer(ComPtr<ID3D11Device> & device);
    std::pair<ComPtr<ID3D11Texture2D>, cudaGraphicsResource_t> createSharedTexture(SDL_Window * window, ComPtr<ID3D11Device> & device);
}

namespace project {
    SDL_Window * SDL_D3D11CreateWindow(const char * title, int windowWidth, int windowHeight) {
        SDL_Log("[SDL] Creating SDL window...");
        SDL_CheckErrorInt(SDL_Init(SDL_INIT_EVERYTHING));

        SDL_Window * window;
        SDL_CheckErrorPtr(window = SDL_CreateWindow(
                title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                windowWidth, windowHeight, 0));
        SDL_Log("[SDL] SDL window created.");
        return window;
    }

    void SDL_D3D11DestroyWindow(SDL_Window * & window) {
        SDL_Log("[SDL] Destroying SDL window...");
        SDL_DestroyWindow(window);
        window = nullptr;
        SDL_Quit();
        SDL_Log("[SDL] SDL window destroyed.");
    }

    Direct3D11Args SDL_D3D11InitializeResource(SDL_Window * window) {
        SDL_Log("[D3D11] Initializing D3D11...");

        auto dxgiFactory = createDXGIFactory();
        auto adapter = findAdapter(dxgiFactory);
        auto [device, deviceContext] = createDevice(adapter);
        auto swapChain = createSwapchain(window, dxgiFactory, device);
        auto [backBuffer, renderTargetView] = getBackBufferAndRTV(swapChain, device);
        auto [shader, blob] = compileShader(device);
        auto inputLayout = createInputLayout(device, blob.first, blob.second);
        auto vertexBuffer = createVertexBuffer(device);
        auto sharedTex = createSharedTexture(window, device);

        SDL_Log("[D3D11] D3D11 initialized.");
        return {
            .device = device, .deviceContext = deviceContext, .dxgiFactory = dxgiFactory, .adapter = adapter,
            .swapChain = swapChain, .backBuffer = backBuffer, .renderTargetView = renderTargetView,
            .vsBlob = blob.first, .psBlob = blob.second,
            .vertexShader = shader.first, .pixelShader = shader.second,
            .inputLayout = inputLayout, .vertexBuffer = vertexBuffer,
            .sharedTexture = sharedTex.first, .cudaResource = sharedTex.second
        };
    }

    void SDL_D3D11CleanupResource(Direct3D11Args & d3d11Args) {
        SDL_Log("[D3D11] Cleaning up resources...");

        //释放CUDA资源，所有D3D资源通过智能指针自动管理
        cudaCheckError(cudaGraphicsUnregisterResource(d3d11Args.cudaResource));
        d3d11Args = {};

        SDL_Log("[D3D11] Resources cleaned.");
    }

    cudaSurfaceObject_t SDL_D3D11MapCudaResource(Direct3D11Args & args, cudaStream_t stream) {
        cudaCheckError(cudaGraphicsMapResources(1, &args.cudaResource, stream));
        cudaArray_t cudaArray;
        cudaCheckError(cudaGraphicsSubResourceGetMappedArray(&cudaArray, args.cudaResource, 0, 0));
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cudaArray;

        cudaSurfaceObject_t surface;
        cudaCheckError(cudaCreateSurfaceObject(&surface, &resDesc));
        return surface;
    }

    void SDL_D3D11UnmapCudaResource(Direct3D11Args & args, cudaSurfaceObject_t & object, cudaStream_t stream) {
        cudaCheckError(cudaDestroySurfaceObject(object));
        cudaCheckError(cudaGraphicsUnmapResources(1, &args.cudaResource, stream));
    }

    void SDL_D3D11PresentFrame(const Direct3D11Args & args) {
        //将CUDA计算结果从sharedTexture复制到后备缓冲区
        args.deviceContext->CopyResource(args.backBuffer.Get(), args.sharedTexture.Get());

        //呈现画面
        args.swapChain->Present(0, DXGI_PRESENT_ALLOW_TEARING);
    }
}

namespace {
    //创建DXGI工厂
    ComPtr<IDXGIFactory2> createDXGIFactory() {
        SDL_Log("[D3D11] Creating DXGI factory...");

        ComPtr<IDXGIFactory2> dxgiFactory2 = nullptr;
        D3DCheckError(CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgiFactory2)));

        //启用撕裂支持：尝试将 factory 转换为第5版
        ComPtr<IDXGIFactory5> dxgiFactory5 = nullptr;
        D3DCheckError(dxgiFactory2.As(&dxgiFactory5));
        BOOL allowTearing = FALSE;
        if (SUCCEEDED(dxgiFactory5->CheckFeatureSupport(
                DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing,
                sizeof(allowTearing))))
        {
            if (allowTearing) {
                SDL_Log("[D3D11] Variable refresh rate and tearing is supported.");
            } else {
                SDL_Log("[D3D11] Variable refresh rate and tearing is NOT supported!");
            }
        }
        SDL_Log("[D3D11] DXGI factory created.");
        return dxgiFactory2;
    }

    //查找GPU，要求包含"NVIDIA"
    ComPtr<IDXGIAdapter> findAdapter(ComPtr<IDXGIFactory2> & dxgiFactory2) {
        SDL_Log("[D3D11] Querying GPU devices...");

        ComPtr<IDXGIAdapter> adapter = nullptr;
        bool isFound = false;
        for (Uint32 i = 0; dxgiFactory2->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
            //获取适配器的描述
            DXGI_ADAPTER_DESC desc{};
            adapter->GetDesc(&desc);
            const auto descString = desc.Description;
            SDL_Log("[D3D11] Found adapter: %ls", descString);

            //desc.Description为WChar数组，使用wcsstr检查描述字符串中是否包含 "NVIDIA"
            if (wcsstr(descString, L"NVIDIA") != nullptr) {
                SDL_Log("[D3D11] Found NVIDIA adapter: %ls", descString);
                isFound = true;
                break;
            }
            adapter->Release(); //释放不是NVIDIA的适配器
        }
        if (!isFound) {
            SDL_Log("[D3D11] Failed to find NVIDIA adapter");
            exit(D3D_ERROR_EXIT_CODE);
        }
        return adapter;
    }

    //创建D3D设备
    std::pair<ComPtr<ID3D11Device>, ComPtr<ID3D11DeviceContext>> createDevice(ComPtr<IDXGIAdapter> & adapter) {
        SDL_Log("[D3D11] Creating D3D device...");

        ComPtr<ID3D11Device> device = nullptr;
        ComPtr<ID3D11DeviceContext> deviceContext = nullptr;
        const D3D_FEATURE_LEVEL featureLevelIn = D3D_FEATURE_LEVEL_11_0;
        D3D_FEATURE_LEVEL featureLevelOut;
        D3DCheckError(D3D11CreateDevice(
                adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN,
                nullptr, D3D11_CREATE_DEVICE_SINGLETHREADED,
                &featureLevelIn, 1,
                D3D11_SDK_VERSION, &device,
                &featureLevelOut, &deviceContext
        ));
        if (featureLevelIn == featureLevelOut) {
            SDL_Log("[D3D11] Device feature level: 0x%x, which matches the input level", featureLevelOut);
        } else {
            SDL_Log("[D3D11] Device feature level: 0x%x, which NOT matches the input level: 0x%x", featureLevelOut, featureLevelIn);
        }
        return {device, deviceContext};
    }

    //创建交换链
    ComPtr<IDXGISwapChain1> createSwapchain(SDL_Window * window, ComPtr<IDXGIFactory2> & dxgiFactory2, ComPtr<ID3D11Device> & device) {
        SDL_Log("[D3D11] Creating swap chain...");

        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        const DXGI_SWAP_CHAIN_DESC1 swapChainDescriptor = {
                .Width = static_cast<Uint32>(width),
                .Height = static_cast<Uint32>(height),
                .Format = DXGI_FORMAT_R8G8B8A8_UNORM, //格式同VK，8位RGBA
                .SampleDesc = {.Count = 1, .Quality = 0},
                .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
                .BufferCount = 2, //双缓冲
                .Scaling = DXGI_SCALING_STRETCH,
                .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
                .AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED,
                .Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        };

        //窗口化
        const DXGI_SWAP_CHAIN_FULLSCREEN_DESC swapChainFullscreenDescriptor = {
                .Windowed = true
        };

        //获取SDL窗口原始指针
        SDL_SysWMinfo wmInfo;
        SDL_VERSION(&wmInfo.version)
        if (SDL_GetWindowWMInfo(window, &wmInfo) != SDL_TRUE || wmInfo.subsystem != SDL_SYSWM_WINDOWS) {
            SDL_Log("[SDL] Failed to get window handler: %s.", SDL_GetError());
            exit(SDL_ERROR_EXIT_CODE);
        }
        HWND windowHandler = wmInfo.info.win.window;

        ComPtr<IDXGISwapChain1> swapChain = nullptr;
        D3DCheckError(dxgiFactory2->CreateSwapChainForHwnd(
                device.Get(), windowHandler, &swapChainDescriptor, &swapChainFullscreenDescriptor,
                nullptr, &swapChain
        ));
        SDL_Log("[D3D11] Swap chain created.");
        return swapChain;
    }

    //创建缓冲区和渲染目标视图
    std::pair<ComPtr<ID3D11Texture2D>, ComPtr<ID3D11RenderTargetView>> getBackBufferAndRTV(
            ComPtr<IDXGISwapChain1> & swapChain, ComPtr<ID3D11Device> & device)
    {
        SDL_Log("[D3D11] Getting back buffer and render target view...");

        ComPtr<ID3D11Texture2D> backBuffer = nullptr;
        ComPtr<ID3D11RenderTargetView> renderTargetView = nullptr;
        D3DCheckError(swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)));
        D3DCheckError(device->CreateRenderTargetView(
                backBuffer.Get(), nullptr, &renderTargetView));

        SDL_Log("[D3D11] Back buffer and render target view set up.");
        return {backBuffer, renderTargetView};
    }

    //编译HLSL着色器
    std::pair<
            std::pair<ComPtr<ID3D11VertexShader>, ComPtr<ID3D11PixelShader>>,
            std::pair<ComPtr<ID3DBlob>, ComPtr<ID3DBlob>>> compileShader(ComPtr<ID3D11Device> & device)
    {
        SDL_Log("[D3D11] Compiling shaders...");

        ComPtr<ID3DBlob> errorBlob = nullptr;
        ComPtr<ID3D11VertexShader> vertexShader = nullptr;
        ComPtr<ID3D11PixelShader> pixelShader = nullptr;
        ComPtr<ID3DBlob> vertexBlob = nullptr;
        ComPtr<ID3DBlob> pixelBlob = nullptr;

        //顶点着色器
        D3DCheckError(D3DCompileFromFile(
                L"../src/D3D11/D3D11VertexShader.hlsl", nullptr,
                D3D_COMPILE_STANDARD_FILE_INCLUDE,
                "Main", "vs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0,
                &vertexBlob, &errorBlob
        ));
        if (errorBlob) {
            SDL_Log("[D3D11] Vertex shader compile error: %s.", (char*)errorBlob->GetBufferPointer());
            exit(D3D_ERROR_EXIT_CODE);
        } else {
            SDL_Log("[D3D11] Vertex shader complication success.");
        }
        D3DCheckError(device->CreateVertexShader(
                vertexBlob->GetBufferPointer(), vertexBlob->GetBufferSize(),
                nullptr, &vertexShader
        ));

        //像素着色器
        D3DCheckError(D3DCompileFromFile(
                L"../src/D3D11/D3D11PixelShader.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
                "Main", "ps_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0,
                &pixelBlob, &errorBlob
        ));
        if (errorBlob) {
            SDL_Log("[D3D11] Pixel shader compile error: %s.", (char*)errorBlob->GetBufferPointer());
            exit(D3D_ERROR_EXIT_CODE);
        } else {
            SDL_Log("[D3D11] Pixel shader complication success.");
        }
        D3DCheckError(device->CreatePixelShader(
                pixelBlob->GetBufferPointer(), pixelBlob->GetBufferSize(),
                nullptr, &pixelShader
        ));

        SDL_Log("[D3D11] Shader complication success.");
        return {
                {vertexShader, pixelShader},
                {vertexBlob, pixelBlob}
        };
    }

    //创建输入布局
    typedef struct VertexPositionColor {
        XMFLOAT3 position;
        XMFLOAT3 color;
    } VertexPositionColor;

    ComPtr<ID3D11InputLayout> createInputLayout(ComPtr<ID3D11Device> & device, ComPtr<ID3DBlob> & vsBlob, ComPtr<ID3DBlob> & psBlob) {
        SDL_Log("[D3D11] Creating input layout...");

        const D3D11_INPUT_ELEMENT_DESC vertexInputLayoutInfo[] = {
                {
                        "POSITION", //HLSL中冒号后的特定字段名称
                        0, //字段索引
                        DXGI_FORMAT::DXGI_FORMAT_R32G32B32_FLOAT,
                        0,
                        offsetof(VertexPositionColor, position),
                        D3D11_INPUT_PER_VERTEX_DATA,
                        0
                },
                {
                        "COLOR",
                        0,
                        DXGI_FORMAT::DXGI_FORMAT_R32G32B32_FLOAT,
                        0,
                        offsetof(VertexPositionColor, color),
                        D3D11_INPUT_PER_VERTEX_DATA,
                        0
                },
        };

        ComPtr<ID3D11InputLayout> inputLayout = nullptr;
        D3DCheckError(device->CreateInputLayout(
                vertexInputLayoutInfo, sizeof(vertexInputLayoutInfo) / sizeof(vertexInputLayoutInfo[0]),
                vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &inputLayout
        ));

        SDL_Log("[D3D11] Input layout created");
        return inputLayout;
    }

    //创建顶点缓冲区
    ComPtr<ID3D11Buffer> createVertexBuffer(ComPtr<ID3D11Device> & device) {
        SDL_Log("[D3D11] Creating vertex buffer...");

        const VertexPositionColor vertices[] = {
                { XMFLOAT3{  0.0f,  0.5f, 0.0f }, XMFLOAT3{ 0.25f, 0.39f, 0.19f } },
                { XMFLOAT3{  0.5f, -0.5f, 0.0f }, XMFLOAT3{ 0.44f, 0.75f, 0.35f } },
                { XMFLOAT3{ -0.5f, -0.5f, 0.0f }, XMFLOAT3{ 0.38f, 0.55f, 0.20f } },
        };

        const D3D11_BUFFER_DESC bufferInfo = {
                .ByteWidth = sizeof(vertices),
                .Usage = D3D11_USAGE_IMMUTABLE, //缓冲区不可修改
                .BindFlags = D3D11_BIND_VERTEX_BUFFER //此缓冲区为顶点缓冲区
        };

        const D3D11_SUBRESOURCE_DATA resourceData = {
                .pSysMem = vertices //指向当前在系统内存中的顶点数组的指针
        };

        ComPtr<ID3D11Buffer> vertexBuffer = nullptr;
        D3DCheckError(device->CreateBuffer(
                &bufferInfo, &resourceData, &vertexBuffer
        ));

        SDL_Log("[D3D11] Vertex buffer created");
        return vertexBuffer;
    }

    //创建CUDA共享资源
    std::pair<ComPtr<ID3D11Texture2D>, cudaGraphicsResource_t> createSharedTexture(SDL_Window * window, ComPtr<ID3D11Device> & device) {
        SDL_Log("[D3D11] Creating shared texture...");

        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        //创建共享纹理
        const D3D11_TEXTURE2D_DESC texDesc = {
                .Width = static_cast<Uint32>(width),
                .Height = static_cast<Uint32>(height),
                .MipLevels = 1,
                .ArraySize = 1,
                .Format = DXGI_FORMAT_R8G8B8A8_UNORM, //与交换链格式一致
                .SampleDesc = {.Count = 1, .Quality = 0},
                .Usage = D3D11_USAGE_DEFAULT,
                .BindFlags = D3D11_BIND_SHADER_RESOURCE,
                .CPUAccessFlags = 0,
                .MiscFlags = 0
        };

        ComPtr<ID3D11Texture2D> sharedTexture = nullptr;
        D3DCheckError(device->CreateTexture2D(
                &texDesc, nullptr, &sharedTexture
        ));

        //将D3D纹理资源注册到CUDA（需要手动销毁）
        cudaGraphicsResource_t cudaResource = nullptr;
        cudaCheckError(cudaGraphicsD3D11RegisterResource(
                &cudaResource, sharedTexture.Get(), cudaGraphicsMapFlagsNone
        ));

        SDL_Log("[D3D11] Shared texture created");
        return {sharedTexture, cudaResource};
    }
}
#endif //_WIN32