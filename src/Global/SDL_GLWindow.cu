#include <Global/SDL_GLWindow.cuh>

namespace project {
    std::pair<SDL_Window *, SDL_GLContext> SDL_GLCreateWindow(
            const char * title, int windowWidth, int windowHeight, bool isVsync)
    {
        SDL_Log("[SDL] Creating SDL window...");
        SDL_CheckErrorInt(SDL_Init(SDL_INIT_EVERYTHING));

        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);

        SDL_Window * window;
        SDL_CheckErrorPtr(window = SDL_CreateWindow(
                title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                windowWidth, windowHeight, SDL_WINDOW_OPENGL));
        SDL_GLContext context;
        SDL_CheckErrorPtr(context = SDL_GL_CreateContext(window));
        SDL_CheckErrorInt(SDL_GL_SetSwapInterval(isVsync ? 1 : 0));

        SDL_Log("[SDL] SDL window created.");
        return {window, context};
    }

    void SDL_GLDestroyWindow(std::pair<SDL_Window *, SDL_GLContext> & windowPtr) {
        SDL_Log("[SDL] Destroying SDL window...");
        SDL_GL_DeleteContext(windowPtr.second);
        SDL_DestroyWindow(windowPtr.first);
        SDL_Quit();
        windowPtr = {nullptr, nullptr};
        SDL_Log("[SDL] SDL window destroyed.");
    }

    OpenGLArgs SDL_GLInitializeResource(const std::pair<SDL_Window *, SDL_GLContext> &windowPtr) {
        SDL_Log("[OGL] Initializing OpenGL...");
        int cudaDevice = -1;
        Uint32 deviceCount = 0;
        cudaCheckError(cudaGLGetDevices(&deviceCount, &cudaDevice, 1, cudaGLDeviceListAll));
        if (cudaDevice == -1) {
            SDL_Log("[OGL] No CUDA device found supporting OpenGL interop");
            return {};
        }
        cudaCheckError(cudaSetDevice(cudaDevice));

        int w, h;
        SDL_GetWindowSize(windowPtr.first, &w, &h);
        if (!gladLoadGLLoader((GLADloadproc)SDL_GL_GetProcAddress)) {
            SDL_Log("[OGL] Failed to init GLAD!"); return {};
        }
        glViewport(0, 0, w, h);
        GLuint textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
        constexpr float vertices[] = {
                -1.0f,  1.0f,  0.0f, 1.0f,
                -1.0f, -1.0f,  0.0f, 0.0f,
                1.0f, -1.0f,  1.0f, 0.0f,
                1.0f,  1.0f,  1.0f, 1.0f
        };
        constexpr GLuint indices[] = {
                0, 1, 2,
                0, 2, 3
        };
        GLuint VAO, VBO, EBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void *>(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        constexpr const char * vertexShaderSource = R"(
            #version 330 core
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            out vec2 TexCoord;
            void main() {
                gl_Position = vec4(aPos, 0.0, 1.0);
                TexCoord = aTexCoord;
            }
        )";
        constexpr const char * fragmentShaderSource = R"(
            #version 330 core
            out vec4 FragColor;
            in vec2 TexCoord;
            uniform sampler2D ourTexture;
            void main() {
                FragColor = texture(ourTexture, TexCoord);
            }
        )";
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
        glCompileShader(vertexShader);
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
        glCompileShader(fragmentShader);
        GLuint shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        SDL_Log("[OGL] OpenGL initialized.");
        return {
                .textureID = textureID,
                .VAO = VAO,
                .VBO = VBO,
                .EBO = EBO,
                .shaderProgram = shaderProgram
        };
    }

    void SDL_GLCleanupResource(OpenGLArgs & openGlArgs) {
        SDL_Log("[OGL] Cleaning up OpenGL...");

        glDeleteVertexArrays(1, &openGlArgs.VAO);
        glDeleteBuffers(1, &openGlArgs.VBO);
        glDeleteBuffers(1, &openGlArgs.EBO);
        glDeleteProgram(openGlArgs.shaderProgram);
        glDeleteTextures(1, &openGlArgs.textureID);
        openGlArgs = {};

        SDL_Log("[OGL] OpenGL resource cleaned.");
    }

    cudaGraphicsResource_t SDL_GLGetWritableCudaObject(
            const std::pair<SDL_Window *, SDL_GLContext> & windowPtr, const OpenGLArgs & openGlArgs)
    {
        SDL_Log("[OGL] Getting Cuda resources...");
        int w, h;
        SDL_GetWindowSize(windowPtr.first, &w, &h);

        cudaGraphicsResource_t resource = nullptr;
        cudaCheckError(cudaGraphicsGLRegisterImage(
                &resource, openGlArgs.textureID, GL_TEXTURE_2D,
                cudaGraphicsRegisterFlagsWriteDiscard));
        return resource;
    }

    void SDL_GLFreeWritableCudaObject(cudaGraphicsResource_t & resource) {
        SDL_Log("[OGL] Releasing Cuda resources...");
        cudaCheckError(cudaGraphicsUnregisterResource(resource));
        resource = nullptr;
    }

    cudaSurfaceObject_t SDL_GLMapCudaResource(cudaGraphicsResource_t & resource, cudaStream_t stream) {
        cudaCheckError(cudaGraphicsMapResources(1, &resource, stream));
        cudaArray_t cudaTextureArray;
        cudaCheckError(cudaGraphicsSubResourceGetMappedArray(&cudaTextureArray, resource, 0, 0));
        cudaResourceDesc resDesc = {
                .resType = cudaResourceTypeArray,
                .res = {.array = {.array = cudaTextureArray}}
        };
        cudaSurfaceObject_t object = 0;
        cudaCheckError(cudaCreateSurfaceObject(&object, &resDesc));
        return object;
    }

    void SDL_GLUnmapCudaResource(cudaGraphicsResource_t & resource, cudaSurfaceObject_t & object, cudaStream_t stream) {
        cudaCheckError(cudaDestroySurfaceObject(object));
        cudaCheckError(cudaGraphicsUnmapResources(1, &resource, stream));
        object = 0;
    }

    void SDL_GLPresentFrame(
            const std::pair<SDL_Window *, SDL_GLContext> & windowPtr, const OpenGLArgs & openGlArgs)
    {
        glUseProgram(openGlArgs.shaderProgram);
        glBindTexture(GL_TEXTURE_2D, openGlArgs.textureID);
        glBindVertexArray(openGlArgs.VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        SDL_GL_SwapWindow(windowPtr.first);
    }
}
