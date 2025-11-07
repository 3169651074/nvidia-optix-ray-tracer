#include <Global/SDL_VKWindow.cuh>
using namespace project;

namespace {
    std::vector<const char *> checkExtensions(
            SDL_Window * window);
    std::vector<const char *> checkDebugLayer();
    VkInstance createInstance(
            SDL_Window * window, const std::vector<const char *> & requiredExtensions, const std::vector<const char *> & requiredLayers);
    VkDebugUtilsMessengerEXT createDebugLayer(
            VkInstance & instance);
    std::tuple<VkPhysicalDevice, VkPhysicalDeviceFeatures, VkPhysicalDeviceProperties> queryPhysicalDevice(
            VkInstance & instance);
    std::vector<const char *> checkDeviceExtensions(
            VkPhysicalDevice & physicalDevice);
    std::tuple<VkSurfaceKHR, VkSurfaceCapabilitiesKHR, std::vector<VkSurfaceFormatKHR>, std::vector<VkPresentModeKHR>> createSurface(
            SDL_Window * window, VkInstance & instance, VkPhysicalDevice & physicalDevice);
    std::tuple<VkExtent2D, VkSurfaceFormatKHR, VkPresentModeKHR> configureSurface(
            SDL_Window * window, const std::tuple<VkSurfaceKHR, VkSurfaceCapabilitiesKHR,
            std::vector<VkSurfaceFormatKHR>, std::vector<VkPresentModeKHR>> & tuple);
    std::vector<Uint32> getQueueInfo(
            VkPhysicalDevice & physicalDevice, VkSurfaceKHR & surface);
    std::tuple<std::vector<Uint32>, VkDevice, std::vector<VkQueue>> createLogicalDeviceAndQueues(
            const VkPhysicalDevice & physicalDevice,
            const VkPhysicalDeviceFeatures & deviceFeatures, std::vector<Uint32> queueIndices,
            const std::vector<const char *> & deviceExtensions);
    std::tuple<VkSwapchainKHR, std::vector<VkImage>, VkFormat, VkExtent2D> createSwapChain(
            VkSurfaceKHR & surface, VkDevice & logicalDevice,
            const std::vector<Uint32> & queueIndices, const VkSurfaceCapabilitiesKHR & capabilities,
            const std::tuple<VkExtent2D, VkSurfaceFormatKHR, VkPresentModeKHR> & tuple);
    std::vector<VkImageView> createImageViews(
            const std::vector<VkImage> & swapChainImages, const VkFormat & swapChainImageFormat,
            VkDevice & logicalDevice);
    VkRenderPass createRenderPass(
            const VkFormat & swapChainImageFormat, VkDevice & logicalDevice);
    std::vector<VkFramebuffer> createFramebuffers(
            const std::vector<VkImageView> & swapChainImageViews, VkRenderPass & renderPass,
            const VkExtent2D & swapChainExtent, VkDevice & logicalDevice);
    VkCommandPool createCommandPool(
            const std::vector<Uint32> & queueIndices, VkDevice & logicalDevice);
    std::vector<VkCommandBuffer> createCommandBuffers(
            VkCommandPool & commandPool, VkDevice & logicalDevice);
    std::tuple<VkImage, VkDeviceMemory, VkImageView, VkMemoryRequirements> createComputeImage(
            SDL_Window * window, VkPhysicalDevice & physicalDevice, VkDevice & logicalDevice);
    std::tuple<cudaMipmappedArray_t, cudaArray_t, cudaSurfaceObject_t, cudaExternalMemory_t, cudaStream_t> initCUDAInteropResources(
            SDL_Window * window, VkDeviceMemory & computeImageMemory,
            VkDevice & logicalDevice, VkMemoryRequirements & memoryRequirements);
    std::tuple<VkSemaphore, VkSemaphore, cudaExternalSemaphore_t, cudaExternalSemaphore_t> createSyncSemaphores(
            VkDevice & logicalDevice);
    void recordCommandBufferImpl(
            SDL_Window * window, VkImage & computeImage,
            const std::vector<VkImage> & swapChainImages,
            const VkCommandBuffer & commandBuffer, Uint32 imageIndex);
    std::tuple<std::vector<VkSemaphore>, std::vector<VkSemaphore>, std::vector<VkFence>> createSyncObjects(
            VkDevice & logicalDevice);
    void cleanupResources(
            VulkanArgs & args);
}

namespace project {
    SDL_Window * SDL_VKCreateWindow(const char * title, int windowWidth, int windowHeight, bool isVsync) {
        SDL_Log("[SDL] Creating SDL window...");
        SDL_CheckErrorInt(SDL_Init(SDL_INIT_EVERYTHING));

        SDL_Window * window;
        SDL_CheckErrorPtr(window = SDL_CreateWindow(
                title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                windowWidth, windowHeight, SDL_WINDOW_VULKAN));
        SDL_Log("[SDL] SDL window created.");
        return window;
    }

    void SDL_VKDestroyWindow(SDL_Window * & window) {
        SDL_Log("[SDL] Destroying SDL window...");
        SDL_DestroyWindow(window);
        window = nullptr;
        SDL_Quit();
        SDL_Log("[SDL] SDL window destroyed.");
    }

    VulkanArgs SDL_VKInitializeResource(SDL_Window * window) {
        VulkanArgs args{};

        //1. 检查扩展和验证层
        auto extensions = checkExtensions(window);
        auto layers = checkDebugLayer();

        //2. 创建实例和调试层
        args.instance = createInstance(window, extensions, layers);
        args.debugMessenger = createDebugLayer(args.instance);

        //3. 查询物理设备
        auto [physicalDevice, features, properties] = queryPhysicalDevice(args.instance);
        args.physicalDevice = physicalDevice;

        //4. 检查设备扩展
        auto deviceExtensions = checkDeviceExtensions(args.physicalDevice);

        //5. 创建 Surface
        auto surfaceInfoTuple = createSurface(window, args.instance, args.physicalDevice);
        args.surface = std::get<VkSurfaceKHR>(surfaceInfoTuple);
        auto capabilities = std::get<VkSurfaceCapabilitiesKHR>(surfaceInfoTuple);

        //6. 配置 Surface
        auto surfaceConfigTuple = configureSurface(window, surfaceInfoTuple);
        args.swapChainExtent = std::get<VkExtent2D>(surfaceConfigTuple);

        //7. 获取队列信息
        auto queueIndices = getQueueInfo(args.physicalDevice, args.surface);

        //8. 创建逻辑设备和队列
        auto [finalQueueIndices, logicalDevice, queues] = createLogicalDeviceAndQueues(
                physicalDevice, features, queueIndices, deviceExtensions);
        args.logicalDevice = logicalDevice;
        args.graphicsQueue = queues[0];
        args.presentQueue = queues[1];

        //9. 创建交换链
        auto [swapChain, swapChainImages, swapChainImageFormat, swapChainExtent] = createSwapChain(
                args.surface, args.logicalDevice, finalQueueIndices,
                capabilities, surfaceConfigTuple);
        args.swapChain = swapChain;
        args.swapChainImages = swapChainImages;
        args.swapChainImageFormat = swapChainImageFormat;
        args.swapChainExtent = swapChainExtent;

        //10. 创建图像视图
        args.swapChainImageViews = createImageViews(
                args.swapChainImages, swapChainImageFormat, args.logicalDevice);

        //11. 创建渲染通道
        args.renderPass = createRenderPass(
                swapChainImageFormat, args.logicalDevice);

        //12. 创建帧缓冲
        args.swapChainFramebuffers = createFramebuffers(
                args.swapChainImageViews, args.renderPass,
                args.swapChainExtent, args.logicalDevice);

        //13. 创建命令池
        args.commandPool = createCommandPool(finalQueueIndices, args.logicalDevice);

        //14. 创建命令缓冲区
        args.commandBuffers = createCommandBuffers(args.commandPool, args.logicalDevice);

        //15. 创建计算图像 (用于 CUDA)
        auto [computeImage, computeImageMemory, computeImageView, memoryRequirements] = createComputeImage(
                window, args.physicalDevice, args.logicalDevice);
        args.computeImage = computeImage;
        args.computeImageMemory = computeImageMemory;
        args.computeImageView = computeImageView;

        //16. 初始化 CUDA 互操作资源
        auto [cudaMipmappedArray, cudaArray,
              cudaSurface, cudaExtMemory, cudaStream]
              = initCUDAInteropResources(window, args.computeImageMemory, args.logicalDevice, memoryRequirements);
        args.cudaMipmappedArray = cudaMipmappedArray;
        args.cudaArray = cudaArray;
        args.cudaSurface = cudaSurface;
        args.cudaExtMemory = cudaExtMemory;
        args.cudaStream = cudaStream;

        //17. 创建同步信号量 (CUDA-VK)
        auto [cudaUpdateSemaphore, cudaReadySemaphore,
              cudaExtSemaphoreUpdate, cudaExtSemaphoreReady]
              = createSyncSemaphores(args.logicalDevice);
        args.cudaUpdateSemaphore = cudaUpdateSemaphore;
        args.cudaReadySemaphore = cudaReadySemaphore;
        args.cudaExtSemaphoreUpdate = cudaExtSemaphoreUpdate;
        args.cudaExtSemaphoreReady = cudaExtSemaphoreReady;

        //18. 创建同步对象
        auto [imageAvailableSemaphores, renderFinishedSemaphores, inFlightFences] = createSyncObjects(
                args.logicalDevice);
        args.imageAvailableSemaphores = imageAvailableSemaphores;
        args.renderFinishedSemaphores = renderFinishedSemaphores;
        args.inFlightFences = inFlightFences;

        //19. 初始化帧索引
        args.currentBufferFrameIndex = 0;

        //20. 返回填充完毕的VulkanArgs
        SDL_Log("[VK] Vulkan and CUDA resources initialized.");
        return args;
    }

    void SDL_VKCleanupResource(VulkanArgs & args) {
        SDL_Log("[VK] Synchronizing vulkan and cuda...");
        cudaCheckError(cudaStreamSynchronize(args.cudaStream));
        vkCheckError(vkQueueWaitIdle(args.graphicsQueue));
        if (args.presentQueue != args.graphicsQueue) {
            vkCheckError(vkQueueWaitIdle(args.presentQueue));
        }
        vkCheckError(vkDeviceWaitIdle(args.logicalDevice));

        SDL_Log("[VK] Cleaning up vulkan and cuda resources...");
        cleanupResources(args);
    }

    Uint32 SDL_VKPrepareFrame(VulkanArgs & args) {
        vkCheckError(vkWaitForFences(
                args.logicalDevice, 1, &args.inFlightFences[args.currentBufferFrameIndex], VK_TRUE, UINT64_MAX
        ));
        Uint32 imageIndex = 0;
        vkCheckError(vkAcquireNextImageKHR(
                args.logicalDevice, args.swapChain, UINT64_MAX,
                args.imageAvailableSemaphores[args.currentBufferFrameIndex], VK_NULL_HANDLE, &imageIndex
        ));
        vkCheckError(vkResetFences(
                args.logicalDevice, 1, &args.inFlightFences[args.currentBufferFrameIndex]
        ));

        //设置信号量以通知CUDA更新完成
        const VkSubmitInfo signalSubmitInfo = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &args.cudaUpdateSemaphore
        };
        vkCheckError(vkQueueSubmit(
                args.graphicsQueue, 1, &signalSubmitInfo, VK_NULL_HANDLE
        ));

        //等待CUDA设置信号量
        const cudaExternalSemaphoreWaitParams waitParams = {
                .flags = 0
        };
        cudaCheckError(cudaWaitExternalSemaphoresAsync(
                &args.cudaExtSemaphoreUpdate, &waitParams, 1, args.cudaStream
        ));

        return imageIndex;
    }
    
    void SDL_VKPresentFrame(SDL_Window * window, VulkanArgs & args, Uint32 imageIndex) {
        //CUDA设置完成信号量以通知Vulkan渲染已完成
        const cudaExternalSemaphoreSignalParams signalParams = {
                .flags = 0
        };
        cudaCheckError(cudaSignalExternalSemaphoresAsync(
                &args.cudaExtSemaphoreReady, &signalParams,
                1, args.cudaStream
        ));

        //录制命令缓冲区
        vkCheckError(vkResetCommandBuffer(
                args.commandBuffers[args.currentBufferFrameIndex], 0
        ));
        recordCommandBufferImpl(
                window, args.computeImage,
                args.swapChainImages,
                args.commandBuffers[args.currentBufferFrameIndex],
                imageIndex);

        //提交命令缓冲区
        const VkSemaphore waitSemaphores[] = {args.imageAvailableSemaphores[args.currentBufferFrameIndex], args.cudaReadySemaphore};
        const VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
        const VkSemaphore signalSemaphores[] = {args.renderFinishedSemaphores[args.currentBufferFrameIndex]};
        const VkSubmitInfo submitInfo = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .waitSemaphoreCount = 2,
                .pWaitSemaphores = waitSemaphores,
                .pWaitDstStageMask = waitStages,
                .commandBufferCount = 1,
                .pCommandBuffers = &args.commandBuffers[args.currentBufferFrameIndex],
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = signalSemaphores
        };
        vkCheckError(vkQueueSubmit(
                args.graphicsQueue, 1, &submitInfo, args.inFlightFences[args.currentBufferFrameIndex]
        ));

        const VkSwapchainKHR swapChains[] = {args.swapChain};
        const VkPresentInfoKHR presentInfo = {
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = signalSemaphores,
                .swapchainCount = 1,
                .pSwapchains = swapChains,
                .pImageIndices = &imageIndex,
                .pResults = nullptr
        };
        vkQueuePresentKHR(args.presentQueue, &presentInfo);
        args.currentBufferFrameIndex = (args.currentBufferFrameIndex + 1) % 2;
    }
}

namespace {
    //检查扩展支持
    std::vector<const char *> checkExtensions(SDL_Window * window) {
        SDL_Log("[VK] Checking global extensions...");

        //列出所有支持的扩展
        Uint32 globalExtensionCount = 0;
        vkCheckError(vkEnumerateInstanceExtensionProperties(
                nullptr, &globalExtensionCount, nullptr
        ));
        std::vector<VkExtensionProperties> globalExtensions(globalExtensionCount);
        vkCheckError(vkEnumerateInstanceExtensionProperties(
                nullptr, &globalExtensionCount, globalExtensions.data()
        ));
        SDL_Log("[VK] Global extension count: %u.", globalExtensionCount);

        //列出所有需要的扩展
        Uint32 requiredExtensionCount = 0;
        SDL_CheckErrorBool(SDL_Vulkan_GetInstanceExtensions(window, &requiredExtensionCount, nullptr));
        std::vector<const char *> requiredExtensions(requiredExtensionCount);
        SDL_CheckErrorBool(SDL_Vulkan_GetInstanceExtensions(window, &requiredExtensionCount, requiredExtensions.data()));
        //添加调试层扩展
        requiredExtensions.push_back("VK_EXT_debug_utils");
        requiredExtensionCount = requiredExtensions.size();
        SDL_Log("[VK] Required extension count: %u.", requiredExtensionCount);

        //检查扩展支持
        for (Uint32 i = 0; i < requiredExtensionCount; i++) {
            const auto item = requiredExtensions[i];
            SDL_Log("[VK] \t[%u]: %s.", i, item);

            bool isFound = false;
            for (const auto & globalExtension: globalExtensions) {
                if (strcmp(globalExtension.extensionName, item) == 0) {
                    isFound = true;
                    break;
                }
            }
            if (!isFound) {
                SDL_Log("[VK] %s extension is required but does not support!", item);
                exit(VULKAN_ERROR_EXIT_CODE);
            } else {
                SDL_Log("[VK] Required extension %s check success.", item);
            }
        }
        SDL_Log("[VK] All extension check success.");
        return requiredExtensions;
    }

    //检查验证层支持
    std::vector<const char *> checkDebugLayer() {
        SDL_Log("[VK] Checking layers...");

        //列出所有支持的层
        Uint32 availableLayerCount = 0;
        vkCheckError(vkEnumerateInstanceLayerProperties(
                &availableLayerCount, nullptr
        ));
        std::vector<VkLayerProperties> availableLayers(availableLayerCount);
        vkCheckError(vkEnumerateInstanceLayerProperties(
                             &availableLayerCount, availableLayers.data()
        ));
        SDL_Log("[VK] Available layer count: %u.", availableLayerCount);

        //将验证层添加到需要列表
        std::vector<const char *> requiredLayers;
        requiredLayers.push_back("VK_LAYER_KHRONOS_validation");
        const Uint32 requiredLayerCount = requiredLayers.size();
        SDL_Log("[VK] Required layer count: %u.", requiredLayerCount);

        //检查层支持
        for (Uint32 i = 0; i < requiredLayerCount; i++) {
            const auto item = requiredLayers[i];
            SDL_Log("[VK] \t[%u]: %s", i, item);

            bool isFound = false;
            for (const auto & availableLayer: availableLayers) {
                if (strcmp(availableLayer.layerName, item) == 0) {
                    isFound = true;
                    break;
                }
            }
            if (!isFound) {
                SDL_Log("[VK] Requires %s layer but does not support!", item);
                exit(VULKAN_ERROR_EXIT_CODE);
            } else {
                SDL_Log("[VK] Layer %s check success.", item);
            }
        }
        SDL_Log("[VK] All layer check success.");
        return requiredLayers;
    }
    
    //创建实例
    VkInstance createInstance(SDL_Window * window, const std::vector<const char *> & requiredExtensions, const std::vector<const char *> & requiredLayers) {
        SDL_Log("[VK] Creating VK instance...");

        const VkApplicationInfo appInfo = {
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .pApplicationName = SDL_GetWindowTitle(window),
                .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                .pEngineName = "No Engine",
                .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                .apiVersion = VK_API_VERSION_1_0
        };
        const VkInstanceCreateInfo createInfo = {
                .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                .pApplicationInfo = &appInfo,
                .enabledLayerCount = static_cast<Uint32>(requiredLayers.size()),
                .ppEnabledLayerNames = requiredLayers.data(),
                .enabledExtensionCount = static_cast<Uint32>(requiredExtensions.size()),
                .ppEnabledExtensionNames = requiredExtensions.data()
        };
        VkInstance instance = nullptr;
        vkCheckError(vkCreateInstance(
                &createInfo, nullptr, &instance
        ));
        SDL_Log("[VK] VK instance created.");
        return instance;
    }

    //创建验证层
    VkDebugUtilsMessengerEXT createDebugLayer(VkInstance & instance) {
        SDL_Log("[VK] Creating debug layer...");

        const VkDebugUtilsMessengerCreateInfoEXT debugLayerCreateInfo = {
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                //接收所有严重程度的调试信息
                .messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                .messageType =
                //接收所有类型的调试信息
                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                .pfnUserCallback = &vkDebugCallBackFunction,
        };

        //动态加载创建调试层的函数
        const auto createDebugLayerFunc = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        VkDebugUtilsMessengerEXT debugMessenger = nullptr;
        if (createDebugLayerFunc == nullptr) {
            SDL_Log("[VK] Failed to create debug layer.");
            exit(VULKAN_ERROR_EXIT_CODE);
        } else {
            //使用动态函数指针调用函数
            (*createDebugLayerFunc)(instance, &debugLayerCreateInfo, nullptr, &debugMessenger);
        }
        SDL_Log("[VK] Debug layer created.");
        return debugMessenger;
    }

    //查询物理显卡设备，要求设备名称中包含“NVIDIA”
    std::tuple<VkPhysicalDevice, VkPhysicalDeviceFeatures, VkPhysicalDeviceProperties> queryPhysicalDevice(VkInstance & instance) {
        SDL_Log("[VK] Checking physical devices...");

        //列出所有物理设备
        Uint32 deviceCount = 0;
        vkCheckError(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));
        if (deviceCount == 0) {
            SDL_Log("[VK] Failed to find GPU with vulkan support!");
            exit(VULKAN_ERROR_EXIT_CODE);
        }
        std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
        vkCheckError(vkEnumeratePhysicalDevices(
                instance, &deviceCount, physicalDevices.data()
        ));

        //查找并选择所需设备
        bool isFound = false;
        VkPhysicalDeviceProperties deviceProperties{};
        VkPhysicalDeviceFeatures deviceFeatures{};
        VkPhysicalDevice physicalDevice = nullptr;
        for (const auto & device: physicalDevices) {
            vkGetPhysicalDeviceProperties(device, &deviceProperties);
            vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

            //选择独立显卡（DISCRETE_GPU）
            if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                isFound = true;
                physicalDevice = device;
                break;
            }
        }
        if (!isFound) {
            SDL_Log("[VK] Failed to find any discrete GPU!");
            exit(VULKAN_ERROR_EXIT_CODE);
        }
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

        //需要NVIDIA显卡
        if (std::string(deviceProperties.deviceName).find("NVIDIA") == std::string::npos) {
            SDL_Log("[VK] Selected discrete GPU is not an NVIDIA GPU!");
            exit(VULKAN_ERROR_EXIT_CODE);
        }
        SDL_Log("[VK] Found discrete NVIDIA GPU: %s.", deviceProperties.deviceName);
        return {physicalDevice, deviceFeatures, deviceProperties};
    }

    //创建Surface，用于和窗口管理器（SDL）交互
    std::tuple<VkSurfaceKHR, VkSurfaceCapabilitiesKHR, std::vector<VkSurfaceFormatKHR>, std::vector<VkPresentModeKHR>> createSurface(
            SDL_Window * window, VkInstance & instance, VkPhysicalDevice & physicalDevice)
    {
        SDL_Log("[VK] Creating surface...");
        VkSurfaceKHR surface = nullptr;
        SDL_CheckErrorBool(SDL_Vulkan_CreateSurface(window, instance, &surface));

        //检查Surface兼容性
        VkSurfaceCapabilitiesKHR capabilities{};
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
        vkCheckError(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                physicalDevice, surface, &capabilities
        ));

        Uint32 formatCount = 0;
        vkCheckError(vkGetPhysicalDeviceSurfaceFormatsKHR(
                physicalDevice, surface, &formatCount, nullptr
        ));
        if (formatCount != 0) {
            formats.resize(formatCount);
            vkCheckError(vkGetPhysicalDeviceSurfaceFormatsKHR(
                    physicalDevice, surface, &formatCount, formats.data()
            ));
            SDL_Log("[VK] Surface format count: %u.", formatCount);
        } else {
            SDL_Log("[VK] No format supported by surface!");
            exit(VULKAN_ERROR_EXIT_CODE);
        }

        Uint32 presentModeCount = 0;
        vkCheckError(vkGetPhysicalDeviceSurfacePresentModesKHR(
                physicalDevice, surface, &presentModeCount, nullptr
        ));
        if (presentModeCount != 0) {
            presentModes.resize(presentModeCount);
            vkCheckError(vkGetPhysicalDeviceSurfacePresentModesKHR(
                    physicalDevice, surface, &presentModeCount, presentModes.data()
            ));
            SDL_Log("[VK] Surface present mode count: %u.", presentModeCount);
        } else {
            SDL_Log("[VK] No present mode is supported by surface!");
            exit(VULKAN_ERROR_EXIT_CODE);
        }
        SDL_Log("[VK] VK surface created.");
        return {surface, capabilities, formats, presentModes};
    }

    //检查物理设备兼容性
    std::vector<const char *> checkDeviceExtensions(VkPhysicalDevice & physicalDevice) {
        SDL_Log("[VK] Checking device extensions...");

        //列出所有支持的设备扩展
        Uint32 physicalDeviceExtensionCount = 0;
        vkCheckError(vkEnumerateDeviceExtensionProperties(
                physicalDevice, nullptr, &physicalDeviceExtensionCount, nullptr
        ));
        std::vector<VkExtensionProperties> physicalDeviceExtensions(physicalDeviceExtensionCount);
        vkCheckError(vkEnumerateDeviceExtensionProperties(
                physicalDevice, nullptr, &physicalDeviceExtensionCount, physicalDeviceExtensions.data()
        ));
        SDL_Log("[VK] Device extension count: %u.", physicalDeviceExtensionCount);

        //列出所有需要的设备扩展
        std::vector<const char *> requiredDeviceExtensions;
        requiredDeviceExtensions.push_back("VK_KHR_swapchain");
        requiredDeviceExtensions.push_back("VK_KHR_external_memory");
        requiredDeviceExtensions.push_back("VK_KHR_external_semaphore");
#ifdef _WIN32
        requiredDeviceExtensions.push_back("VK_KHR_external_memory_win32");
        requiredDeviceExtensions.push_back("VK_KHR_external_semaphore_win32");
#else
        requiredDeviceExtensions.push_back("VK_KHR_external_memory_fd");
        requiredDeviceExtensions.push_back("VK_KHR_external_semaphore_fd");
#endif
        const Uint32 requiredDeviceExtensionCount = requiredDeviceExtensions.size();
        SDL_Log("[VK] Required device extension count: %u", requiredDeviceExtensionCount);

        //检查设备扩展支持
        for (Uint32 i = 0; i < requiredDeviceExtensionCount; i++) {
            const auto item = requiredDeviceExtensions[i];
            SDL_Log("[VK] \t[%u]: %s", i, item);

            bool isFound = false;
            for (const auto & deviceExtension : physicalDeviceExtensions) {
                if (strcmp(deviceExtension.extensionName, item) == 0) {
                    isFound = true;
                    break;
                }
            }
            if (!isFound) {
                SDL_Log("[VK] %s device extension is required but VK does not support!", item);
                exit(92);
            } else {
                SDL_Log("[VK] Required device extension %s check success.", item);
            }
        }
        SDL_Log("[VK] All device extension check success.");
        return requiredDeviceExtensions;
    }

    //配置Surface
    std::tuple<VkExtent2D, VkSurfaceFormatKHR, VkPresentModeKHR> configureSurface(
            SDL_Window * window, const std::tuple<VkSurfaceKHR, VkSurfaceCapabilitiesKHR,
            std::vector<VkSurfaceFormatKHR>, std::vector<VkPresentModeKHR>> & tuple)
    {
        SDL_Log("[VK] Configuring surface...");

        const auto capabilities = get<VkSurfaceCapabilitiesKHR>(tuple);
        const auto formats = get<std::vector<VkSurfaceFormatKHR>>(tuple);
        const auto presentModes = get<std::vector<VkPresentModeKHR>>(tuple);

        VkSurfaceFormatKHR chosenFormat{};
        VkPresentModeKHR chosenPresentMode{};
        VkExtent2D extent2D{};
        if (capabilities.currentExtent.width != std::numeric_limits<Uint32>::max()) {
            extent2D = capabilities.currentExtent;
        } else {
            int width, height;
            SDL_Vulkan_GetDrawableSize(window, &width, &height);
            extent2D = {
                    .width = std::clamp(static_cast<Uint32>(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                    .height = std::clamp(static_cast<Uint32>(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
            };
        }
        bool isChooseFormat = false;
        for (const auto & item : formats) {
            //使用VK_FORMAT_R8G8B8A8_UNORM格式以和OGL统一
            if (item.format == VK_FORMAT_R8G8B8A8_UNORM && item.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosenFormat = item;
                isChooseFormat = true;
                break;
            }
        }
        if (!isChooseFormat) {
            chosenFormat = formats[0];
        }
        //选择呈现模式
        bool isChoosePresentMode = false;
        for (const auto & item : presentModes) {
            //允许撕裂，无垂直同步
            if (item == VK_PRESENT_MODE_IMMEDIATE_KHR) {
                chosenPresentMode = item;
                isChoosePresentMode = true;
                SDL_Log("[VK] Using IMMEDIATE present mode (tearing allowed)");
                break;
            }
        }
        if (!isChoosePresentMode) {
            //三缓冲，优化版本的垂直同步
            for (const auto & item : presentModes) {
                if (item == VK_PRESENT_MODE_MAILBOX_KHR) {
                    chosenPresentMode = item;
                    isChoosePresentMode = true;
                    SDL_Log("[VK] Using MAILBOX present mode");
                    break;
                }
            }
        }
        if (!isChoosePresentMode) {
            //双缓冲，垂直同步
            for (const auto & item : presentModes) {
                if (item == VK_PRESENT_MODE_FIFO_KHR) {
                    chosenPresentMode = item;
                    isChoosePresentMode = true;
                    SDL_Log("[VK] Using FIFO present mode (V-Sync)");
                    break;
                }
            }
        }
        if (!isChoosePresentMode) {
            chosenPresentMode = presentModes[0];
            SDL_Log("[VK] Using default present mode");
        }
        return {extent2D, chosenFormat, chosenPresentMode};
    }

    //获取设备队列信息
    std::vector<Uint32> getQueueInfo(VkPhysicalDevice & physicalDevice, VkSurfaceKHR & surface) {
        SDL_Log("[VK] Getting queue family info...");

        //列出所有队列家族
        Uint32 queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
        std::vector<Uint32> queueFamiliesIndices(2);
        SDL_Log("[VK] Available queue count: %u.", queueFamilyCount);

        //查找图形队列和呈现队列
        bool foundGraphics = false;
        bool foundPresent = false;
        for (Uint32 i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                queueFamiliesIndices[0] = i;
                foundGraphics = true;
            }
            VkBool32 presentSupport = VK_FALSE;
            vkCheckError(vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport));
            if (presentSupport) {
                queueFamiliesIndices[1] = i;
                foundPresent = true;
            }
            if (foundGraphics && foundPresent) {
                break;
            }
        }
        if (foundGraphics && foundPresent) {
            SDL_Log("[VK] Found graphics queue: [%u], present queue: [%u].", queueFamiliesIndices[0], queueFamiliesIndices[1]);
            return queueFamiliesIndices;
        } else {
            SDL_Log("[VK] Failed to find graphics queue and present queue!");
            exit(VULKAN_ERROR_EXIT_CODE);
        }
        SDL_Log("[VK] Queue family info obtained.");
        return queueFamiliesIndices;
    }

    //创建队列
    std::tuple<std::vector<Uint32>, VkDevice, std::vector<VkQueue>> createLogicalDeviceAndQueues(
            const VkPhysicalDevice & physicalDevice,
            const VkPhysicalDeviceFeatures & deviceFeatures, std::vector<Uint32> queueIndices,
            const std::vector<const char *> & deviceExtensions)
    {
        SDL_Log("[VK] Creating logical device and queues...");

        //检查队列下标
        constexpr float queuePriority = 1.0f;
        const bool isSame = queueIndices[0] == queueIndices[1];
        if (isSame) {
            queueIndices.erase(queueIndices.begin());
            SDL_Log("[VK] Same queue index found, only one queue will be created.");
        }

        //创建队列
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        for (const auto & item: queueIndices) {
            const VkDeviceQueueCreateInfo queueCreateInfo = {
                    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    .queueFamilyIndex = item,
                    .queueCount = 1,
                    .pQueuePriorities = &queuePriority
            };
            queueCreateInfos.push_back(queueCreateInfo);
            SDL_Log("[VK] Creating queue [%u]...", item);
        }

        //创建逻辑设备
        const VkDeviceCreateInfo createInfo = {
                .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                .queueCreateInfoCount = static_cast<Uint32>(queueCreateInfos.size()),
                .pQueueCreateInfos = queueCreateInfos.data(),
                .enabledExtensionCount = static_cast<Uint32>(deviceExtensions.size()),
                .ppEnabledExtensionNames = deviceExtensions.data(),
                .pEnabledFeatures = &deviceFeatures
        };
        VkDevice logicalDevice = nullptr;
        vkCheckError(vkCreateDevice(
                physicalDevice, &createInfo, nullptr, &logicalDevice
        ));

        //使用逻辑设备获取队列
        VkQueue graphicsQueue = nullptr;
        VkQueue presentQueue = nullptr;
        std::vector<Uint32> indices = {queueIndices[0]};
        vkGetDeviceQueue(logicalDevice, queueIndices[0], 0, &graphicsQueue);
        if (!isSame) {
            vkGetDeviceQueue(logicalDevice, queueIndices[1], 0, &presentQueue);
            indices.push_back(queueIndices[1]);
        } else {
            presentQueue = graphicsQueue;
        }
        SDL_Log("[VK] Logical device and queues created.");
        return {indices, logicalDevice, {graphicsQueue, presentQueue}};
    }

    //创建交换链
    std::tuple<VkSwapchainKHR, std::vector<VkImage>, VkFormat, VkExtent2D> createSwapChain(
            VkSurfaceKHR & surface, VkDevice & logicalDevice,
            const std::vector<Uint32> & queueIndices, const VkSurfaceCapabilitiesKHR & capabilities,
            const std::tuple<VkExtent2D, VkSurfaceFormatKHR, VkPresentModeKHR> & tuple)
    {
        SDL_Log("[VK] Creating swap chain...");

        const auto extent2D = get<VkExtent2D>(tuple);
        const auto format = get<VkSurfaceFormatKHR>(tuple);
        const auto presentMode = get<VkPresentModeKHR>(tuple);

        //获取交换链缓冲区数量
        Uint32 bufferCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && bufferCount > capabilities.maxImageCount) {
            bufferCount = capabilities.maxImageCount;
        }
        const bool isSame = queueIndices.size() == 1;
        const VkSwapchainCreateInfoKHR createInfo = {
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                .surface = surface,
                .minImageCount = bufferCount,
                .imageFormat = format.format,
                .imageColorSpace = format.colorSpace,
                .imageExtent = extent2D,
                .imageArrayLayers = 1,
                .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                .imageSharingMode = isSame ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
                .queueFamilyIndexCount = static_cast<Uint32>(isSame ? 0 : 2),
                .pQueueFamilyIndices = isSame ? nullptr : queueIndices.data(),
                .preTransform = capabilities.currentTransform,
                .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                .presentMode = presentMode,
                .clipped = VK_TRUE,
                .oldSwapchain = nullptr
        };

        //创建交换链
        VkSwapchainKHR swapChain = nullptr;
        vkCheckError(vkCreateSwapchainKHR(
                logicalDevice, &createInfo, nullptr, &swapChain
        ));
        SDL_Log("[VK] Swap chain created");

        //获取交换链附加信息
        vkCheckError(vkGetSwapchainImagesKHR(
                logicalDevice, swapChain, &bufferCount, nullptr
        ));
        std::vector<VkImage> swapChainImages;
        swapChainImages.resize(bufferCount);
        vkCheckError(vkGetSwapchainImagesKHR(
                logicalDevice, swapChain, &bufferCount, swapChainImages.data()
        ));
        VkFormat swapChainImageFormat{};
        VkExtent2D swapChainExtent{};
        swapChainImageFormat = format.format;
        swapChainExtent = extent2D;
        SDL_Log("[VK] Swap chain images obtained, image count: %u.", bufferCount);
        return {swapChain, swapChainImages, swapChainImageFormat, swapChainExtent};
    }

    //创建图像视图
    std::vector<VkImageView> createImageViews(
            const std::vector<VkImage> & swapChainImages, const VkFormat & swapChainImageFormat,
            VkDevice & logicalDevice)
    {
        SDL_Log("[VK] Creating image views...");

        std::vector<VkImageView> swapChainImageViews;
        swapChainImageViews.resize(swapChainImages.size());
        for (Uint32 i = 0; i < swapChainImages.size(); i++) {
            const VkImageViewCreateInfo createInfo = {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .image = swapChainImages[i],
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = swapChainImageFormat,
                    .components = {
                            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .a = VK_COMPONENT_SWIZZLE_IDENTITY
                    },
                    .subresourceRange = {
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1
                    }
            };
            vkCheckError(vkCreateImageView(
                    logicalDevice, &createInfo, nullptr, &swapChainImageViews[i]
            ));
        }
        SDL_Log("[VK] Image views created");
        return swapChainImageViews;
    }

    //创建渲染通道
    VkRenderPass createRenderPass(const VkFormat & swapChainImageFormat, VkDevice & logicalDevice) {
        SDL_Log("[VK] Creating render pass...");

        const VkAttachmentDescription colorAttachment = {
                .format = swapChainImageFormat,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        };
        const VkAttachmentReference colorAttachmentRef = {
                .attachment = 0,
                .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        };
        const VkSubpassDescription subpass = {
                .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                .colorAttachmentCount = 1,
                .pColorAttachments = &colorAttachmentRef
        };
        const VkSubpassDependency dependency = {
                .srcSubpass = VK_SUBPASS_EXTERNAL,
                .dstSubpass = 0,
                .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        };
        const VkRenderPassCreateInfo createInfo = {
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                .attachmentCount = 1,
                .pAttachments = &colorAttachment,
                .subpassCount = 1,
                .pSubpasses = &subpass,
                .dependencyCount = 1,
                .pDependencies = &dependency
        };
        VkRenderPass renderPass = nullptr;
        vkCheckError(vkCreateRenderPass(
                logicalDevice, &createInfo, nullptr, &renderPass
        ));
        SDL_Log("[VK] Render pass created");
        return renderPass;
    }

    //创建帧缓冲
    std::vector<VkFramebuffer> createFramebuffers(
            const std::vector<VkImageView> & swapChainImageViews, VkRenderPass & renderPass,
            const VkExtent2D & swapChainExtent, VkDevice & logicalDevice)
    {
        SDL_Log("[VK] Creating framebuffers...");

        std::vector<VkFramebuffer> swapChainFramebuffers;
        swapChainFramebuffers.resize(swapChainImageViews.size());
        for (Uint32 i = 0; i < swapChainImageViews.size(); i++) {
            VkImageView attachments[] = {swapChainImageViews[i]};
            const VkFramebufferCreateInfo createInfo = {
                    .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                    .renderPass = renderPass,
                    .attachmentCount = 1,
                    .pAttachments = attachments,
                    .width = swapChainExtent.width,
                    .height = swapChainExtent.height,
                    .layers = 1
            };
            vkCheckError(vkCreateFramebuffer(
                    logicalDevice, &createInfo, nullptr, &swapChainFramebuffers[i]
            ));
        }
        SDL_Log("[VK] Framebuffers created");
        return swapChainFramebuffers;
    }

    //创建命令池
    VkCommandPool createCommandPool(const std::vector<Uint32> & queueIndices, VkDevice & logicalDevice) {
        SDL_Log("[VK] Creating command pool...");

        const VkCommandPoolCreateInfo createInfo = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = queueIndices[0]
        };
        VkCommandPool commandPool = nullptr;
        vkCheckError(vkCreateCommandPool(
                logicalDevice, &createInfo, nullptr, &commandPool
        ));
        SDL_Log("[VK] Command pool created");
        return commandPool;
    }

    //创建命令缓冲区
    std::vector<VkCommandBuffer> createCommandBuffers(VkCommandPool & commandPool, VkDevice & logicalDevice) {
        SDL_Log("[VK] Creating command buffers...");

        //创建与交换链图像数量相同的命令缓冲区
        std::vector<VkCommandBuffer> commandBuffers;
        commandBuffers.resize(2);
        const VkCommandBufferAllocateInfo allocInfo = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = commandPool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = static_cast<Uint32>(commandBuffers.size())
        };
        vkCheckError(vkAllocateCommandBuffers(
                logicalDevice, &allocInfo, commandBuffers.data()
        ));
        SDL_Log("[VK] Command buffers created");
        return commandBuffers;
    }

    //创建计算图像用于CUDA写入
    std::tuple<VkImage, VkDeviceMemory, VkImageView, VkMemoryRequirements> createComputeImage(
            SDL_Window * window, VkPhysicalDevice & physicalDevice, VkDevice & logicalDevice)
    {
        SDL_Log("[VK] Creating compute image ...");

        //创建可导出的VK计算图像资源
        const VkExternalMemoryImageCreateInfo externalMemoryImageCreateInfo = {
                .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
#ifdef _WIN32
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
#else
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
#endif
        };
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        const VkImageCreateInfo imageInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = &externalMemoryImageCreateInfo,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .extent = {
                        .width = static_cast<Uint32>(width),
                        .height = static_cast<Uint32>(height),
                        .depth = 1
                },
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
        };
        VkImage computeImage = nullptr;
        vkCheckError(vkCreateImage(
                logicalDevice, &imageInfo, nullptr, &computeImage
        ));

        //检查内存需求并为计算图像分配内存
        VkMemoryRequirements memRequirements{};
        vkGetImageMemoryRequirements(logicalDevice, computeImage, &memRequirements);
        const VkExportMemoryAllocateInfo exportMemoryAllocateInfo = {
                .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
#ifdef _WIN32
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
#else
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
#endif
        };

        SDL_Log("[VK] Finding suitable memory type...");
        Uint32 memoryIndex = 0;
        bool isFound = false;
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (Uint32 i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((memRequirements.memoryTypeBits & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                memoryIndex = i;
                isFound = true;
            }
        }
        if (!isFound) {
            SDL_Log("[VK] Failed to find suitable memory type for compute image!");
            exit(VULKAN_ERROR_EXIT_CODE);
        }
        const VkMemoryAllocateInfo allocInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = &exportMemoryAllocateInfo,
                .allocationSize = memRequirements.size,
                .memoryTypeIndex = memoryIndex
        };
        VkDeviceMemory computeImageMemory = nullptr;
        vkCheckError(vkAllocateMemory(
                logicalDevice, &allocInfo, nullptr, &computeImageMemory
        ));
        vkCheckError(vkBindImageMemory(
                logicalDevice, computeImage, computeImageMemory, 0
        ));

        //创建计算图像的视图
        const VkImageViewCreateInfo viewInfo = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = computeImage,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                }
        };
        VkImageView computeImageView = nullptr;
        vkCheckError(vkCreateImageView(
                logicalDevice, &viewInfo, nullptr, &computeImageView
        ));
        SDL_Log("[VK] Compute image created");
        return {computeImage, computeImageMemory, computeImageView, memRequirements};
    }

    //初始化CUDA互操作资源
    std::tuple<cudaMipmappedArray_t, cudaArray_t, cudaSurfaceObject_t, cudaExternalMemory_t, cudaStream_t> initCUDAInteropResources(
            SDL_Window * window, VkDeviceMemory & computeImageMemory,
            VkDevice & logicalDevice, VkMemoryRequirements & memoryRequirements)
    {
        SDL_Log("[VK] Initializing CUDA interop resources...");

        //获取CUDA内存句柄
#ifdef _WIN32
        const VkMemoryGetWin32HandleInfoKHR memoryGetHandleInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
                .memory = computeImageMemory,
                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
        };

        const auto vkGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(logicalDevice, "vkGetMemoryWin32HandleKHR");
        HANDLE handle;
        vkCheckError(vkGetMemoryWin32HandleKHR(
                logicalDevice, &memoryGetHandleInfo, &handle
        ));
#else
        const VkMemoryGetFdInfoKHR memoryGetFdInfo = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                .memory = computeImageMemory,
                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
        };

        const auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(logicalDevice, "vkGetMemoryFdKHR");
        int fd;
        vkCheckError(vkGetMemoryFdKHR(
                logicalDevice, &memoryGetFdInfo, &fd
        ));
#endif

        //导入到CUDA上下文
        cudaExternalMemoryHandleDesc externalMemoryHandleDesc{};
#ifdef _WIN32
        externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        externalMemoryHandleDesc.handle.win32.handle = handle;
#else
        externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        externalMemoryHandleDesc.handle.fd = fd;
#endif
        externalMemoryHandleDesc.size = memoryRequirements.size;

        cudaExternalMemory_t cudaExtMemory = nullptr;
        cudaCheckError(cudaImportExternalMemory(
                &cudaExtMemory, &externalMemoryHandleDesc
        ));

        //创建CUDA映射数组并转换为cudaArray
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        cudaExternalMemoryMipmappedArrayDesc mipDesc{};
        mipDesc.extent = make_cudaExtent(width, height, 0);
        mipDesc.formatDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        mipDesc.numLevels = 1;
        mipDesc.flags = cudaArraySurfaceLoadStore;

        cudaMipmappedArray_t cudaMipmappedArray = nullptr;
        cudaArray_t cudaArray = nullptr;
        cudaCheckError(cudaExternalMemoryGetMappedMipmappedArray(
                &cudaMipmappedArray, cudaExtMemory, &mipDesc
        ));
        cudaCheckError(cudaGetMipmappedArrayLevel(&cudaArray, cudaMipmappedArray, 0));

        //创建CUDA表面对象
        cudaSurfaceObject_t cudaSurface = 0;
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cudaArray;
        cudaCheckError(cudaCreateSurfaceObject(&cudaSurface, &resDesc));

        //创建CUDA流
        cudaStream_t cudaStream = nullptr;
        cudaCheckError(cudaStreamCreate(&cudaStream));

        SDL_Log("[VK] CUDA interop resources initialized.");
        return {cudaMipmappedArray, cudaArray, cudaSurface, cudaExtMemory, cudaStream};
    }

    //创建同步信号量
    std::tuple<VkSemaphore, VkSemaphore, cudaExternalSemaphore_t, cudaExternalSemaphore_t> createSyncSemaphores(VkDevice & logicalDevice) {
        SDL_Log("[VK] Creating sync semaphores...");

        //创建用于互操作的信号量
        const VkExportSemaphoreCreateInfo exportSemaphoreCreateInfo = {
                .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
#ifdef _WIN32
                .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
#else
                .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
#endif
        };
        const VkSemaphoreCreateInfo semaphoreInfo = {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = &exportSemaphoreCreateInfo
        };
        VkSemaphore cudaUpdateSemaphore = nullptr;
        VkSemaphore cudaReadySemaphore = nullptr;
        vkCheckError(vkCreateSemaphore(
                logicalDevice, &semaphoreInfo, nullptr, &cudaUpdateSemaphore
        ));
        vkCheckError(vkCreateSemaphore(
                logicalDevice, &semaphoreInfo, nullptr, &cudaReadySemaphore
        ));

        //导出信号量句柄
#ifdef _WIN32
        const VkSemaphoreGetWin32HandleInfoKHR semaphoreGetHandleInfo1 = {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
                .semaphore = cudaUpdateSemaphore,
                .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
        };
        const VkSemaphoreGetWin32HandleInfoKHR semaphoreGetHandleInfo2 = {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR,
                .semaphore = cudaReadySemaphore,
                .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
        };

        const auto vkGetSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(logicalDevice, "vkGetSemaphoreWin32HandleKHR");
        HANDLE semHandle1, semHandle2;
        vkCheckError(vkGetSemaphoreWin32HandleKHR(
                logicalDevice, &semaphoreGetHandleInfo1, &semHandle1
        ));
        vkCheckError(vkGetSemaphoreWin32HandleKHR(
                logicalDevice, &semaphoreGetHandleInfo2, &semHandle2
        ));
#else
        const VkSemaphoreGetFdInfoKHR semaphoreGetFdInfo1 = {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
                .semaphore = cudaUpdateSemaphore,
                .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
        };
        const VkSemaphoreGetFdInfoKHR semaphoreGetFdInfo2 = {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
                .semaphore = cudaReadySemaphore,
                .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
        };

        const auto vkGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(logicalDevice, "vkGetSemaphoreFdKHR");
        int semFd1, semFd2;
        vkCheckError(vkGetSemaphoreWin32HandleKHR(
                logicalDevice, &semaphoreGetHandleInfo1, &semFd1
        ));
        vkCheckError(vkGetSemaphoreWin32HandleKHR(
                logicalDevice, &semaphoreGetHandleInfo2, &semFd2
        ));
#endif

        //将信号量导入到CUDA
        cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc{};
#ifdef _WIN32
        externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
        externalSemaphoreHandleDesc.handle.win32.handle = semHandle1;
#else
        externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
        externalSemaphoreHandleDesc.handle.fd = semFd1;
#endif
        cudaExternalSemaphore_t cudaExtSemaphoreUpdate = nullptr;
        cudaExternalSemaphore_t cudaExtSemaphoreReady = nullptr;
        cudaCheckError(cudaImportExternalSemaphore(
                &cudaExtSemaphoreUpdate, &externalSemaphoreHandleDesc
        ));
#ifdef _WIN32
        externalSemaphoreHandleDesc.handle.win32.handle = semHandle2;
#else
        externalSemaphoreHandleDesc.handle.fd = semFd2;
#endif
        cudaCheckError(cudaImportExternalSemaphore(
                &cudaExtSemaphoreReady, &externalSemaphoreHandleDesc
        ));

        SDL_Log("[VK] Sync semaphores created");
        return {cudaUpdateSemaphore, cudaReadySemaphore, cudaExtSemaphoreUpdate, cudaExtSemaphoreReady};
    }

    //录制命令缓冲区
    void recordCommandBufferImpl(SDL_Window * window, VkImage & computeImage,
                             const std::vector<VkImage> & swapChainImages,
                             const VkCommandBuffer & commandBuffer, Uint32 imageIndex)
    {
        //SDL_Log("[VK] Recording command buffer for image index %u...", imageIndex);

        const VkCommandBufferBeginInfo beginInfo = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = 0,
                .pInheritanceInfo = nullptr
        };
        vkCheckError(vkBeginCommandBuffer(commandBuffer, &beginInfo));

        //等待CUDA写入完成，转换计算图像布局以供传输
        VkImageMemoryBarrier computeBarrier = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = computeImage,
                .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                }
        };
        vkCmdPipelineBarrier(
                commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0,
                nullptr, 1, &computeBarrier);

        //转换交换链图像布局以供传输
        VkImageMemoryBarrier swapchainBarrier = {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapChainImages[imageIndex],
                .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                }
        };
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &swapchainBarrier);

        //执行图像拷贝
        int width, height;
        SDL_GetWindowSize(window, &width, &height);
        const VkImageCopy copyRegion = {
                .srcSubresource = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                },
                .srcOffset = {0, 0, 0},
                .dstSubresource = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                },
                .dstOffset = {0, 0, 0},
                .extent = {
                        .width = static_cast<Uint32>(width),
                        .height = static_cast<Uint32>(height),
                        .depth = 1
                }
        };
        vkCmdCopyImage(commandBuffer,
                       computeImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       swapChainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &copyRegion);

        //转换交换链图像布局以供呈现
        swapchainBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        swapchainBarrier.dstAccessMask = 0;
        swapchainBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        swapchainBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier(
                commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0, 0, nullptr, 0, nullptr,
                1, &swapchainBarrier);

        //转换计算图像布局以供下一次CUDA写入
        computeBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        computeBarrier.dstAccessMask = 0;
        computeBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        computeBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(
                commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0, 0, nullptr, 0,
                nullptr, 1, &computeBarrier);

        //结束命令缓冲区录制
        vkCheckError(vkEndCommandBuffer(commandBuffer));

        //SDL_Log("[VK] Command buffer recorded for image index %u.", imageIndex);
    }

    //创建同步对象
    std::tuple<std::vector<VkSemaphore>, std::vector<VkSemaphore>, std::vector<VkFence>> createSyncObjects(VkDevice & logicalDevice) {
        SDL_Log("[VK] Creating sync objects...");

        std::vector<VkSemaphore> imageAvailableSemaphores;
        std::vector<VkSemaphore> renderFinishedSemaphores;
        std::vector<VkFence> inFlightFences;
        imageAvailableSemaphores.resize(2);
        renderFinishedSemaphores.resize(2);
        inFlightFences.resize(2);

        const VkSemaphoreCreateInfo semaphoreInfo = {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
        };
        const VkFenceCreateInfo fenceInfo = {
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .flags = VK_FENCE_CREATE_SIGNALED_BIT
        };

        //为每个缓冲区创建信号量和栅栏
        for (Uint32 i = 0; i < 2; i++) {
            vkCheckError(vkCreateSemaphore(
                    logicalDevice, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]
            ));
            vkCheckError(vkCreateSemaphore(
                    logicalDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]
            ));
            vkCheckError(vkCreateFence(
                    logicalDevice, &fenceInfo, nullptr, &inFlightFences[i]
            ));
        }

        SDL_Log("[VK] Sync objects created");
        return {imageAvailableSemaphores, renderFinishedSemaphores, inFlightFences};
    }

    //清理资源
    void cleanupResources(VulkanArgs & args) {
        SDL_Log("[VK] Cleaning up Vulkan and CUDA resources...");

        //CUDA
        cudaCheckError(cudaDestroySurfaceObject(args.cudaSurface));
        cudaCheckError(cudaFreeMipmappedArray(args.cudaMipmappedArray));
        cudaCheckError(cudaDestroyExternalMemory(args.cudaExtMemory));
        cudaCheckError(cudaDestroyExternalSemaphore(args.cudaExtSemaphoreUpdate));
        cudaCheckError(cudaDestroyExternalSemaphore(args.cudaExtSemaphoreReady));
        cudaCheckError(cudaStreamDestroy(args.cudaStream));

        //Vulkan
        vkDestroySemaphore(args.logicalDevice, args.cudaUpdateSemaphore, nullptr);
        vkDestroySemaphore(args.logicalDevice, args.cudaReadySemaphore, nullptr);

        for (Uint32 i = 0; i < 2; i++) {
            vkDestroySemaphore(args.logicalDevice, args.imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(args.logicalDevice, args.renderFinishedSemaphores[i], nullptr);
            vkDestroyFence(args.logicalDevice, args.inFlightFences[i], nullptr);
        }
        vkDestroyCommandPool(args.logicalDevice, args.commandPool, nullptr);
        for (auto framebuffer : args.swapChainFramebuffers) {
            vkDestroyFramebuffer(args.logicalDevice, framebuffer, nullptr);
        }
        vkDestroyRenderPass(args.logicalDevice, args.renderPass, nullptr);
        for (auto imageView : args.swapChainImageViews) {
            vkDestroyImageView(args.logicalDevice, imageView, nullptr);
        }
        vkDestroyImageView(args.logicalDevice, args.computeImageView, nullptr);
        vkDestroyImage(args.logicalDevice, args.computeImage, nullptr);
        vkFreeMemory(args.logicalDevice, args.computeImageMemory, nullptr);

        vkDestroySwapchainKHR(args.logicalDevice, args.swapChain, nullptr);
        vkDestroySurfaceKHR(args.instance, args.surface, nullptr);
        vkDestroyDevice(args.logicalDevice, nullptr);

        const auto destroyDebugLayerFunc = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(args.instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroyDebugLayerFunc == nullptr) {
            SDL_Log("[VK] Failed to destroy debug layer");
        } else {
            (*destroyDebugLayerFunc)(args.instance, args.debugMessenger, nullptr);
            SDL_Log("[VK] Debug layer destroyed");
        }
        vkDestroyInstance(args.instance, nullptr);

        args = {};
        SDL_Log("[VK] Vulkan resources cleaned up");
    }
}