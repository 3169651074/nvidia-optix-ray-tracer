#ifndef OPTIXTEST_SDL_GLWINDOW_CUH
#define OPTIXTEST_SDL_GLWINDOW_CUH

#include <Global/HostFunctions.cuh>

namespace project {
    typedef struct OpenGLArgs {
        GLuint textureID;
        GLuint VAO, VBO, EBO;
        GLuint shaderProgram;
    } OpenGLArgs;

    //创建窗口
    std::pair<SDL_Window *, SDL_GLContext> SDL_GLCreateWindow(const char * title, int windowWidth, int windowHeight, bool isVsync = false);
    void SDL_GLDestroyWindow(std::pair<SDL_Window *, SDL_GLContext> & windowPtr);

    //初始化OGL资源
    OpenGLArgs SDL_GLInitializeResource(const std::pair<SDL_Window *, SDL_GLContext> & windowPtr);
    void SDL_GLCleanupResource(OpenGLArgs & openGlArgs);

    //获取可写入CUDA对象
    cudaGraphicsResource_t SDL_GLGetWritableCudaObject(const std::pair<SDL_Window *, SDL_GLContext> & windowPtr, const OpenGLArgs & openGlArgs);
    void SDL_GLFreeWritableCudaObject(cudaGraphicsResource_t & resource);

    //映射纹理资源
    cudaSurfaceObject_t SDL_GLMapCudaResource(cudaGraphicsResource_t & resource, cudaStream_t stream = nullptr);
    void SDL_GLUnmapCudaResource(cudaGraphicsResource_t & resource, cudaSurfaceObject_t & object, cudaStream_t stream = nullptr);

    //绘制画面
    void SDL_GLPresentFrame(const std::pair<SDL_Window *, SDL_GLContext> & windowPtr, const OpenGLArgs & openGlArgs);
}
#endif //OPTIXTEST_SDL_GLWINDOW_CUH
