#ifdef _WIN32
#include <Global/SDL_D3D12Window.cuh>
using namespace project;

namespace {
    ComPtr<ID3D12Debug> enableDebugLayer();
    std::pair<ComPtr<IDXGIFactory4>, ComPtr<IDXGIFactory5>> createDXGIFactory();
    ComPtr<IDXGIAdapter1> queryAdapter(
            ComPtr<IDXGIFactory4> & dxgiFactory4);
    ComPtr<ID3D12Device> createDevice(
            ComPtr<IDXGIAdapter1> & dxgiAdapter);
    ComPtr<ID3D12CommandQueue> createCommandQueue(
            ComPtr<ID3D12Device> & device);
    ComPtr<IDXGISwapChain4> createSwapChain(
            SDL_Window * window, ComPtr<IDXGIFactory4> & dxgiFactory4, ComPtr<ID3D12CommandQueue> & commandQueue);
    std::pair<std::array<ComPtr<ID3D12Resource>, 2>, Uint32> createRenderTargetView(
            ComPtr<ID3D12Device> & device, ComPtr<IDXGISwapChain4> & swapchain,
            ComPtr<ID3D12DescriptorHeap> & rtvDescriptorHeap);
    ComPtr<ID3D12DescriptorHeap> createDescriptorHeap(
            ComPtr<ID3D12Device> & device);
    std::array<ComPtr<ID3D12CommandAllocator>, 2> createCommandAllocator(
            ComPtr<ID3D12Device> & device);
    ComPtr<ID3D12GraphicsCommandList> createCommandList(
            ComPtr<ID3D12Device> & device, ComPtr<ID3D12CommandAllocator> & commandAllocator);
    ComPtr<ID3D12Fence> createFence(
            ComPtr<ID3D12Device> & device);
    HANDLE createFenceEvent();
    Uint32 signal(
            ComPtr<ID3D12CommandQueue> & commandQueue, ComPtr<ID3D12Fence> & fence, Uint32 fenceValue);
    void waitForFenceValue(
            ComPtr<ID3D12Fence> & fence, Uint32 targetFenceValue, HANDLE fenceEvent);
    void flushCommandQueue(
            ComPtr<ID3D12CommandQueue> & commandQueue, ComPtr<ID3D12Fence> & fence, Uint32 fenceValue, HANDLE fenceEvent);
    std::tuple<
            std::array<ComPtr<ID3D12Resource>, 2>, std::array<cudaExternalMemory_t, 2>,
            std::array<cudaMipmappedArray_t, 2>, std::array<cudaArray_t, 2>, std::array<cudaSurfaceObject_t, 2>
    > createCUDAInteropResources(
            SDL_Window * window, ComPtr<ID3D12Device> & device, std::array<ComPtr<ID3D12Resource>, 2> & backBuffers);
}

namespace project {
    SDL_Window * SDL_D3D12CreateWindow(const char * title, int windowWidth, int windowHeight) {
        SDL_Log("[SDL] Creating SDL window...");
        SDL_CheckErrorInt(SDL_Init(SDL_INIT_EVERYTHING));

        SDL_Window * window;
        SDL_CheckErrorPtr(window = SDL_CreateWindow(
                title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                windowWidth, windowHeight, 0));
        SDL_Log("[SDL] SDL window created.");
        return window;
    }

    void SDL_D3D12DestroyWindow(SDL_Window * & window) {
        SDL_Log("[SDL] Destroying SDL window...");
        SDL_DestroyWindow(window);
        window = nullptr;
        SDL_Quit();
        SDL_Log("[SDL] SDL window destroyed.");
    }

    Direct3D12Args SDL_D3D12InitializeResource(SDL_Window * window) {
        SDL_Log("[D3D12] Initializing D3D12...");

        auto debugInterface = enableDebugLayer();
        auto [dxgiFactory4, dxgiFactory5] = createDXGIFactory();
        auto adapter = queryAdapter(dxgiFactory4);
        auto device = createDevice(adapter);
        auto commandQueue = createCommandQueue(device);
        auto swapchain = createSwapChain(window, dxgiFactory4, commandQueue);
        auto desHeap = createDescriptorHeap(device);
        auto [backBuffers, rtvDescSize] = createRenderTargetView(device, swapchain, desHeap);
        auto commandAllocators = createCommandAllocator(device);
        auto commandList = createCommandList(device, commandAllocators[0]);
        auto fence = createFence(device);
        auto fenceEvent = createFenceEvent();
        auto frameIndex = swapchain->GetCurrentBackBufferIndex();
        auto [sharedResources, cudaExtMemories, cudaMipArrays, cudaArrays, cudaSurfaces]
            = createCUDAInteropResources(window, device, backBuffers);

        SDL_Log("[D3D12] D3D11 initialized.");
        return {
            .debugInterface = debugInterface,
            .dxgiFactory4 = dxgiFactory4,
            .dxgiFactory5 = dxgiFactory5,
            .commandQueue = commandQueue,
            .swapChain = swapchain,
            .backBuffers = backBuffers,
            .rtvDescriptorSize = rtvDescSize,
            .commandAllocators = commandAllocators,
            .commandList = commandList,
            .rtvDescriptorHeap = desHeap,
            .fence = fence,
            .fenceEvent = fenceEvent,
            .frameIndex = frameIndex,
            .sharedResources = sharedResources,
            .cudaExtMemories = cudaExtMemories,
            .cudaMipArrays = cudaMipArrays,
            .cudaArrays = cudaArrays,
            .cudaSurfaces = cudaSurfaces
        };
    }

    void SDL_D3D12CleanupResource(Direct3D12Args & args) {
        SDL_Log("[D3D12] Cleaning up resources...");
        
        //等待GPU完成所有任务
        flushCommandQueue(args.commandQueue, args.fence, args.fenceValues[args.frameIndex], args.fenceEvent);
        //CUDA
        for (Uint32 i = 0; i < 2; i++) {
            cudaCheckError(cudaDestroySurfaceObject(args.cudaSurfaces[i]));
            cudaCheckError(cudaFreeMipmappedArray(args.cudaMipArrays[i]));
            cudaCheckError(cudaDestroyExternalMemory(args.cudaExtMemories[i]));
        }
        //关闭Windows事件句柄
        CloseHandle(args.fenceEvent);

        args = {};
        SDL_Log("[D3D12] Resources cleaned.");
    }

    std::pair<ComPtr<ID3D12Resource>, ComPtr<ID3D12Resource>> SDL_D3D12PrepareFrame(Direct3D12Args & args) {
        //等待上一帧完成,确保要使用的命令分配器已经不再被GPU占用
        waitForFenceValue(args.fence, args.fenceValues[args.frameIndex], args.fenceEvent);

        //重置命令分配器和命令列表
        auto commandAllocator = args.commandAllocators[args.frameIndex];
        auto backBuffer = args.backBuffers[args.frameIndex];
        auto sharedResource = args.sharedResources[args.frameIndex];
        D3DCheckError(commandAllocator->Reset());
        D3DCheckError(args.commandList->Reset(commandAllocator.Get(), nullptr));

        return {backBuffer, sharedResource};
    }

    void SDL_D3D12PresentFrame(Direct3D12Args & args, std::pair<ComPtr<ID3D12Resource>, ComPtr<ID3D12Resource>> & pair) {
        auto sharedResource = pair.second;
        auto backBuffer = pair.first;

        //开始记录命令
        //1. 转换共享资源到复制源状态
        CD3DX12_RESOURCE_BARRIER barriers[2] = {
                CD3DX12_RESOURCE_BARRIER::Transition(
                        sharedResource.Get(),
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_SOURCE
                ),
                CD3DX12_RESOURCE_BARRIER::Transition(
                        backBuffer.Get(),
                        D3D12_RESOURCE_STATE_PRESENT,
                        D3D12_RESOURCE_STATE_COPY_DEST
                )
        };
        args.commandList->ResourceBarrier(2, barriers);
        //2. 复制资源
        args.commandList->CopyResource(backBuffer.Get(), sharedResource.Get());
        //3. 转换回原状态
        barriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
                sharedResource.Get(),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON
        );
        barriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
                backBuffer.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PRESENT
        );
        args.commandList->ResourceBarrier(2, barriers);
        //结束记录命令
        D3DCheckError(args.commandList->Close());
        //提交任务
        ID3D12CommandList * commandLists[1] = {args.commandList.Get()};
        args.commandQueue->ExecuteCommandLists(1, commandLists);
        //设置围栏
        args.fenceValues[args.frameIndex] = signal(args.commandQueue, args.fence, args.fenceValues[args.frameIndex]);
        //呈现画面
        D3DCheckError(args.swapChain->Present(0, DXGI_PRESENT_ALLOW_TEARING));
        //更新帧索引
        args.frameIndex = args.swapChain->GetCurrentBackBufferIndex();
    }
}

namespace {
    //启用调试层
    ComPtr<ID3D12Debug> enableDebugLayer() {
        SDL_Log("[D3D12] Enabling debug layer...");

        ComPtr<ID3D12Debug> debugInterface = nullptr;
        D3DCheckError(D3D12GetDebugInterface(IID_PPV_ARGS(&debugInterface)));
        debugInterface->EnableDebugLayer();

        SDL_Log("[D3D12] Debug layer enabled.");
        return debugInterface;
    }

    //创建DXGI工厂
    std::pair<ComPtr<IDXGIFactory4>, ComPtr<IDXGIFactory5>> createDXGIFactory() {
        SDL_Log("[D3D12] Creating DXGI factory...");

        ComPtr<IDXGIFactory4> dxgiFactory4 = nullptr;
        ComPtr<IDXGIFactory5> dxgiFactory5 = nullptr;
        //创建Factory4
        D3DCheckError(CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, IID_PPV_ARGS(&dxgiFactory4)));

        //转换为Factory5以检查撕裂支持
        if (SUCCEEDED(dxgiFactory4.As(&dxgiFactory5))) {
            BOOL tearingSupport = FALSE;
            if (SUCCEEDED(dxgiFactory5->CheckFeatureSupport(
                    DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                    &tearingSupport,
                    sizeof(tearingSupport))))
            {
                if (tearingSupport) {
                    SDL_Log("[D3D12] Tearing enabled.");
                } else {
                    SDL_Log("[D3D12] Tearing is not available!");
                }
            }
        } else {
            SDL_Log("[D3D12] IDXGIFactory5 not available, tearing is disabled!");
        }

        SDL_Log("[D3D12] DXGI factory created.");
        return {dxgiFactory4, dxgiFactory5};
    }

    //查找GPU设备，要求名称包含"NVIDIA"
    ComPtr<IDXGIAdapter1> queryAdapter(ComPtr<IDXGIFactory4> & dxgiFactory4) {
        SDL_Log("[D3D12] Querying GPU adapter...");

        ComPtr<IDXGIAdapter1> dxgiAdapter = nullptr;
        bool isFound = false;
        for (Uint32 i = 0; dxgiFactory4->EnumAdapters1(i, &dxgiAdapter) != DXGI_ERROR_NOT_FOUND; i++) {
            DXGI_ADAPTER_DESC1 dxgiAdapterDesc{};
            D3DCheckError(dxgiAdapter->GetDesc1(&dxgiAdapterDesc));
            const auto descString = dxgiAdapterDesc.Description;
            SDL_Log("[D3D12] Found adapter: %ls.", descString);

            //要求为NVIDIA设备
            if (wcsstr(descString, L"NVIDIA") != nullptr) {
                SDL_Log("[D3D12] Found NVIDIA adapter: %ls.", descString);
                isFound = true;
                break;
            }
            dxgiAdapter->Release();
        }
        if (!isFound) {
            SDL_Log("[D3D12] Failed to find NVIDIA adapter!");
            exit(D3D_ERROR_EXIT_CODE);
        }
        return dxgiAdapter;
    }

    //创建D3D设备
    ComPtr<ID3D12Device> createDevice(ComPtr<IDXGIAdapter1> & dxgiAdapter) {
        SDL_Log("[D3D12] Creating D3D12 device...");

        ComPtr<ID3D12Device> device = nullptr;
        D3DCheckError(D3D12CreateDevice(
                dxgiAdapter.Get(),
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&device)
        ));

        SDL_Log("[D3D12] D3D12 device created.");
        return device;
    }

    //创建命令队列
    ComPtr<ID3D12CommandQueue> createCommandQueue(ComPtr<ID3D12Device> & device) {
        SDL_Log("[D3D12] Creating command queue...");
        const D3D12_COMMAND_QUEUE_DESC desc = {
                .Type = D3D12_COMMAND_LIST_TYPE_DIRECT,
                .Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
                .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
                .NodeMask = 0 //单GPU时设置为0
        };
        ComPtr<ID3D12CommandQueue> commandQueue = nullptr;
        D3DCheckError(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&commandQueue)));

        SDL_Log("[D3D12] Command queue created.");
        return commandQueue;
    }

    //创建交换链
    ComPtr<IDXGISwapChain4> createSwapChain(SDL_Window * window, ComPtr<IDXGIFactory4> & dxgiFactory4, ComPtr<ID3D12CommandQueue> & commandQueue) {
        SDL_Log("[D3D12] Creating swap chain...");

        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        const DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {
                .Width = static_cast<Uint32>(width),
                .Height = static_cast<Uint32>(height),
                .Format = DXGI_FORMAT_R8G8B8A8_UNORM,
                .Stereo = FALSE,
                .SampleDesc = {.Count = 1, .Quality = 0},
                .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
                .BufferCount = 2,
                .Scaling = DXGI_SCALING_STRETCH,
                .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
                .AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED,
                .Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        };
        const DXGI_SWAP_CHAIN_FULLSCREEN_DESC swapChainFullscreenDesc = {
                .Windowed = true
        };

        SDL_SysWMinfo wmInfo;
        SDL_VERSION(&wmInfo.version)
        if (SDL_GetWindowWMInfo(window, &wmInfo) != SDL_TRUE || wmInfo.subsystem != SDL_SYSWM_WINDOWS) {
            SDL_Log("[SDL] Failed to get window handler: %s", SDL_GetError());
            exit(SDL_ERROR_EXIT_CODE);
        }
        HWND windowHandler = wmInfo.info.win.window;

        //先创建swapChain1,再转换为swapChain4
        ComPtr<IDXGISwapChain1> swapchain1 = nullptr;
        ComPtr<IDXGISwapChain4> swapchain4 = nullptr;
        D3DCheckError(dxgiFactory4->CreateSwapChainForHwnd(
                commandQueue.Get(), windowHandler, &swapChainDesc,
                &swapChainFullscreenDesc, nullptr, &swapchain1
        ));
        D3DCheckError(swapchain1.As(&swapchain4));

        SDL_Log("[D3D12] Swap chain created.");
        return swapchain4;
    }

    //创建描述符堆
    ComPtr<ID3D12DescriptorHeap> createDescriptorHeap(ComPtr<ID3D12Device> & device) {
        SDL_Log("[D3D12] Creating descriptor heap...");

        const D3D12_DESCRIPTOR_HEAP_DESC desc = {
                .Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                .NumDescriptors = 2,
                .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                .NodeMask = 0
        };
        ComPtr<ID3D12DescriptorHeap> rtvDescriptorHeap = nullptr;
        D3DCheckError(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&rtvDescriptorHeap)));

        SDL_Log("[D3D12] Descriptor heap created.");
        return rtvDescriptorHeap;
    }

    //创建渲染目标视图
    std::pair<std::array<ComPtr<ID3D12Resource>, 2>, Uint32> createRenderTargetView(
            ComPtr<ID3D12Device> & device, ComPtr<IDXGISwapChain4> & swapchain,
            ComPtr<ID3D12DescriptorHeap> & rtvDescriptorHeap)
    {
        SDL_Log("[D3D12] Creating render target view...");

        //获取单个描述符大小
        const Uint32 rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        //检索堆中第一个描述符的句柄
        CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(rtvDescriptorHeap->GetCPUDescriptorHandleForHeapStart());

        //为每个后台缓冲区创建RTV
        std::array<ComPtr<ID3D12Resource>, 2> backBuffers = {};
        for (Uint32 i = 0; i < 2; i++) {
            ComPtr<ID3D12Resource> backBuffer = nullptr;
            D3DCheckError(swapchain->GetBuffer(i, IID_PPV_ARGS(&backBuffer)));
            device->CreateRenderTargetView(backBuffer.Get(), nullptr, rtvHandle);
            backBuffers[i] = backBuffer;
            rtvHandle.Offset((int)rtvDescriptorSize);
        }

        SDL_Log("[D3D12] Render target view created.");
        return {backBuffers, rtvDescriptorSize};
    }

    //创建命令分配器
    std::array<ComPtr<ID3D12CommandAllocator>, 2> createCommandAllocator(ComPtr<ID3D12Device> & device) {
        SDL_Log("[D3D12] Creating command allocator...");

        std::array<ComPtr<ID3D12CommandAllocator>, 2> commandAllocators = {};
        for (Uint32 i = 0; i < 2; i++) {
            D3DCheckError(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocators[i])));
        }

        SDL_Log("[D3D12] Command allocator created.");
        return commandAllocators;
    }

    //创建命令列表（传入一个命令分配器）
    ComPtr<ID3D12GraphicsCommandList> createCommandList(ComPtr<ID3D12Device> & device, ComPtr<ID3D12CommandAllocator> & commandAllocator) {
        //SDL_Log("[D3D12] Creating command list...");

        ComPtr<ID3D12GraphicsCommandList> commandList = nullptr;
        D3DCheckError(device->CreateCommandList(
                0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                commandAllocator.Get(), nullptr, IID_PPV_ARGS(&commandList)
        ));
        D3DCheckError(commandList->Close());

        //SDL_Log("[D3D12] Command list created.");
        return commandList;
    }

    //创建围栏
    ComPtr<ID3D12Fence> createFence(ComPtr<ID3D12Device> & device) {
        SDL_Log("Creating fence...");

        ComPtr<ID3D12Fence> fence = nullptr;
        D3DCheckError(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));

        SDL_Log("Fence created.");
        return fence;
    }

    //创建围栏事件
    HANDLE createFenceEvent() {
        SDL_Log("Creating fence event...");

        HANDLE fenceEvent = nullptr;
        if ((fenceEvent = CreateEvent(
                nullptr, FALSE,
                FALSE, nullptr)) == nullptr)
        {
            SDL_Log("[D3D12] Failed to create fence event!");
            exit(D3D_ERROR_EXIT_CODE);
        }

        SDL_Log("Fence event created.");
        return fenceEvent;
    }

    //向命令队列设置信号
    Uint32 signal(ComPtr<ID3D12CommandQueue> & commandQueue, ComPtr<ID3D12Fence> & fence, Uint32 fenceValue) {
        const Uint32 newFenceValue = fenceValue + 1;
        D3DCheckError(commandQueue->Signal(fence.Get(), newFenceValue));
        return newFenceValue;
    }

    //等待信号
    void waitForFenceValue(ComPtr<ID3D12Fence> & fence, Uint32 targetFenceValue, HANDLE fenceEvent) {
        constexpr static std::chrono::milliseconds maxDuration = std::chrono::milliseconds::max();

        if (fence->GetCompletedValue() < targetFenceValue) {
            //设置要等待的围栏值
            D3DCheckError(fence->SetEventOnCompletion(targetFenceValue, fenceEvent));
            //等待值到达
            WaitForSingleObject(fenceEvent, static_cast<DWORD>(maxDuration.count()));
        }
    }

    //刷新GPU任务队列
    void flushCommandQueue(ComPtr<ID3D12CommandQueue> & commandQueue, ComPtr<ID3D12Fence> & fence, Uint32 fenceValue, HANDLE fenceEvent) {
        const Uint32 fenceValueForSignal = signal(commandQueue, fence, fenceValue);
        waitForFenceValue(fence, fenceValueForSignal, fenceEvent);
    }

    //为每个后台缓冲区创建CUDA互操作资源
    std::tuple<
            std::array<ComPtr<ID3D12Resource>, 2>, std::array<cudaExternalMemory_t, 2>,
            std::array<cudaMipmappedArray_t, 2>, std::array<cudaArray_t, 2>, std::array<cudaSurfaceObject_t, 2>
    > createCUDAInteropResources(
                SDL_Window * window, ComPtr<ID3D12Device> & device, std::array<ComPtr<ID3D12Resource>, 2> & backBuffers)
    {
        SDL_Log("[D3D12] Creating CUDA interop resources...");

        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        std::array<ComPtr<ID3D12Resource>, 2> sharedResources = {};
        std::array<cudaExternalMemory_t, 2> cudaExtMemories = {};
        std::array<cudaMipmappedArray_t, 2> cudaMipArrays = {};
        std::array<cudaArray_t, 2> cudaArrays = {};
        std::array<cudaSurfaceObject_t, 2> cudaSurfaces = {};

        for (Uint32 i = 0; i < 2; i++) {
            //创建共享资源描述
            const D3D12_RESOURCE_DESC resourceDesc = backBuffers[i]->GetDesc();
            const auto allocInfo = device->GetResourceAllocationInfo(0, 1, &resourceDesc);

            //创建堆属性
            const D3D12_HEAP_PROPERTIES heapProps = {
                    .Type = D3D12_HEAP_TYPE_DEFAULT,
                    .CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                    .MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN
            };
            //创建可共享的资源并保存引用
            D3DCheckError(device->CreateCommittedResource(
                    &heapProps,
                    D3D12_HEAP_FLAG_SHARED,
                    &resourceDesc,
                    D3D12_RESOURCE_STATE_COMMON,
                    nullptr,
                    IID_PPV_ARGS(&sharedResources[i])
            ));
            //获取共享句柄
            HANDLE sharedHandle = nullptr;
            D3DCheckError(device->CreateSharedHandle(
                    sharedResources[i].Get(),
                    nullptr,
                    GENERIC_ALL,
                    nullptr,
                    &sharedHandle
            ));

            //导入到CUDA
            const cudaExternalMemoryHandleDesc externalMemHandleDesc = {
                    .type = cudaExternalMemoryHandleTypeD3D12Resource,
                    .handle = {
                            .win32 = {.handle = sharedHandle}
                    },
                    .size = allocInfo.SizeInBytes,
                    .flags = cudaExternalMemoryDedicated
            };
            cudaCheckError(cudaImportExternalMemory(&cudaExtMemories[i], &externalMemHandleDesc));
            const cudaExternalMemoryMipmappedArrayDesc mipmapDesc = {
                    .offset = 0,
                    .formatDesc = cudaCreateChannelDesc<uchar4>(),
                    .extent = make_cudaExtent(
                            static_cast<size_t>(resourceDesc.Width),
                            static_cast<size_t>(resourceDesc.Height),
                            resourceDesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? resourceDesc.DepthOrArraySize : 0),
                    .flags = cudaArraySurfaceLoadStore,
                    .numLevels = resourceDesc.MipLevels
            };

            //转换为Surface Object
            cudaCheckError(cudaExternalMemoryGetMappedMipmappedArray(&cudaMipArrays[i], cudaExtMemories[i], &mipmapDesc));
            cudaCheckError(cudaGetMipmappedArrayLevel(&cudaArrays[i], cudaMipArrays[i], 0));
            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cudaArrays[i];
            cudaCheckError(cudaCreateSurfaceObject(&cudaSurfaces[i], &resDesc));

            CloseHandle(sharedHandle);
        }
        SDL_Log("[D3D12] CUDA interop resources created.");

        return {sharedResources, cudaExtMemories, cudaMipArrays, cudaArrays, cudaSurfaces};
    }
}
#endif