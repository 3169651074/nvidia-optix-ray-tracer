#ifndef OPTIXTEST_SDL_VKWINDOW_CUH
#define OPTIXTEST_SDL_VKWINDOW_CUH

#include <Global/HostFunctions.cuh>

namespace project {
    typedef struct VulkanArgs {
        VkInstance instance;
        VkDebugUtilsMessengerEXT debugMessenger;
        VkPhysicalDevice physicalDevice;
        VkDevice logicalDevice;
        VkQueue graphicsQueue;
        VkQueue presentQueue;
        VkSurfaceKHR surface;
        VkSwapchainKHR swapChain;
        std::vector<VkImage> swapChainImages;
        VkFormat swapChainImageFormat;
        VkExtent2D swapChainExtent;
        std::vector<VkImageView> swapChainImageViews;
        VkRenderPass renderPass;
        std::vector<VkFramebuffer> swapChainFramebuffers;
        VkCommandPool commandPool;
        std::vector<VkCommandBuffer> commandBuffers;
        std::vector<VkSemaphore> imageAvailableSemaphores;
        std::vector<VkSemaphore> renderFinishedSemaphores;
        std::vector<VkFence> inFlightFences;
        Uint32 currentBufferFrameIndex;

        VkImage computeImage;
        VkDeviceMemory computeImageMemory;
        VkImageView computeImageView;
        cudaMipmappedArray_t cudaMipmappedArray;
        cudaArray_t cudaArray;
        cudaSurfaceObject_t cudaSurface;
        cudaExternalMemory_t cudaExtMemory;
        VkSemaphore cudaUpdateSemaphore;
        VkSemaphore cudaReadySemaphore;
        cudaExternalSemaphore_t cudaExtSemaphoreUpdate;
        cudaExternalSemaphore_t cudaExtSemaphoreReady;
        cudaStream_t cudaStream;
    } VulkanArgs;

    //创建窗口
    SDL_Window * SDL_VKCreateWindow(const char * title, int windowWidth, int windowHeight, bool isVsync = false);
    void SDL_VKDestroyWindow(SDL_Window * & window);

    //初始化VK资源
    VulkanArgs SDL_VKInitializeResource(SDL_Window * window);
    void SDL_VKCleanupResource(VulkanArgs & vkArgs);

    //呈现画面
    Uint32 SDL_VKPrepareFrame(VulkanArgs & args);
    void SDL_VKPresentFrame(SDL_Window * window, VulkanArgs & args, Uint32 imageIndex);
}
#endif //OPTIXTEST_SDL_VKWINDOW_CUH
