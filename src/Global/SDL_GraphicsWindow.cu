#include <Global/SDL_GraphicsWindow.cuh>

namespace project {
    SDL_GraphicsWindowCamera SDL_GraphicsWindowConfigureCamera(const float3 & center, const float3 & target, const float3 & up, SDL_GraphicsWindowAPIType type) {
        SDL_GraphicsWindowCamera camera = {.upDirection = up, .cameraCenter = center, .cameraTarget = target};
        //非OpenGL均需要反转相机正方向Y轴
        if (type != SDL_GraphicsWindowAPIType::OPENGL) {
            camera.upDirection.y = -camera.upDirection.y;
        }
        camera.cameraW = camera.cameraTarget - camera.cameraCenter;
        camera.cameraU = normalize(cross(camera.cameraW, camera.upDirection));
        camera.cameraV = normalize(cross(camera.cameraU, camera.cameraW));
        return camera;
    }

    SDL_GraphicsWindowArgs SDL_CreateGraphicsWindow(
            const char * title, int width, int height, SDL_GraphicsWindowAPIType type,
            size_t fpsLimit, float mouseSensitivity, float pitchLimitDegree,
            float cameraMoveSpeedStride, size_t initialSpeedNTimesStride)
    {
        SDL_Log("[SDL] Creating SDL window...");
        SDL_CheckErrorInt(SDL_Init(SDL_INIT_EVERYTHING));

        SDL_GraphicsWindowArgs args{};
        switch (type) {
            case SDL_GraphicsWindowAPIType::OPENGL:
                //OpenGL 4.6
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
                SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
                SDL_CheckErrorPtr(args.window = SDL_CreateWindow(
                        title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                        width, height, SDL_WINDOW_OPENGL));
                SDL_CheckErrorPtr(args.glContext = SDL_GL_CreateContext(args.window));
                SDL_CheckErrorInt(SDL_GL_SetSwapInterval(0));
                args.glArgs = SDL_GLInitializeResource({args.window, args.glContext});
                args.cudaGraphicsResource = SDL_GLGetWritableCudaObject({args.window, args.glContext}, args.glArgs);
                break;
            case SDL_GraphicsWindowAPIType::VULKAN:
                SDL_CheckErrorPtr(args.window = SDL_CreateWindow(
                        title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                        width, height, SDL_WINDOW_VULKAN));
                args.vkArgs = SDL_VKInitializeResource(args.window);
                break;
#if defined(_WIN32) && defined(SDL) && defined(D3D11)
            case SDL_GraphicsWindowAPIType::DIRECT3D11:
                SDL_CheckErrorPtr(args.window = SDL_CreateWindow(
                        title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                        width, height, 0));
                args.d3d11Args = SDL_D3D11InitializeResource(args.window);
                break;
#endif
#if defined(_WIN32) && defined(SDL) && defined(D3D12)
            case SDL_GraphicsWindowAPIType::DIRECT3D12:
                SDL_CheckErrorPtr(args.window = SDL_CreateWindow(
                        title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                        width, height, 0));
                args.d3d12Args = SDL_D3D12InitializeResource(args.window);
                break;
#endif
            default:;
        }
        args.type = type;
        args.cameraMoveSpeed = cameraMoveSpeedStride * initialSpeedNTimesStride;
        args.targetFrameDuration = std::chrono::microseconds(static_cast<Sint64>(1000000.0f / static_cast<float>(fpsLimit)));
        args.sleepMargin = std::chrono::milliseconds(2);
        args.mouseSensitivity = mouseSensitivity;
        args.pitchLimitDegree = pitchLimitDegree;

        //启用鼠标锁定
        SDL_CheckErrorInt(SDL_SetRelativeMouseMode(SDL_TRUE));

        SDL_Log("[SDL] SDL window created.");
        return args;
    }

    void SDL_GraphicsWindowFrameStart(SDL_GraphicsWindowArgs & args) {
        args.frameStartTime = std::chrono::steady_clock::now();
    }

    void SDL_GraphicsWindowUpdateCamera(
            SDL_Event & event, SDL_GraphicsWindowKeyMouseInput & input,
            SDL_GraphicsWindowArgs & args, SDL_GraphicsWindowCamera & camera) {
        //每帧重置鼠标移动量
        input.mouseX = 0;
        input.mouseY = 0;

        //接收输入
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
                input.keyQuit = true;
            }
            switch (event.type) {
                case SDL_KEYDOWN:
                    switch (event.key.keysym.sym) {
                        case SDLK_w:        input.keyW = true;      break;
                        case SDLK_a:        input.keyA = true;      break;
                        case SDLK_s:        input.keyS = true;      break;
                        case SDLK_d:        input.keyD = true;      break;
                        case SDLK_SPACE:    input.keySpace = true;  break;
                        case SDLK_LSHIFT:   input.keyLShift = true; break;
                    }
                    break;
                case SDL_KEYUP:
                    switch (event.key.keysym.sym) {
                        case SDLK_w:        input.keyW = false;       break;
                        case SDLK_a:        input.keyA = false;       break;
                        case SDLK_s:        input.keyS = false;       break;
                        case SDLK_d:        input.keyD = false;       break;
                        case SDLK_SPACE:    input.keySpace = false;   break;
                        case SDLK_LSHIFT:   input.keyLShift = false;  break;
                    }
                    break;
                case SDL_MOUSEBUTTONDOWN:
                    SDL_SetRelativeMouseMode(SDL_GetRelativeMouseMode() == SDL_TRUE ? SDL_FALSE : SDL_TRUE);
                    break;
                case SDL_MOUSEMOTION:
                    if (SDL_GetRelativeMouseMode() == SDL_TRUE) {
                        input.mouseX += event.motion.xrel;
                        input.mouseY += event.motion.yrel;
                    }
                    break;
                case SDL_MOUSEWHEEL: {
                    //SDL_Event.wheel.y：正值：滚轮向上滚动；负值：滚轮向下滚动
                    if (event.wheel.y > 0) {
                        args.cameraMoveSpeed += args.cameraMoveSpeedStride;
                    } else {
                        args.cameraMoveSpeed = args.cameraMoveSpeed < args.cameraMoveSpeedStride ? 0.0f : args.cameraMoveSpeed - args.cameraMoveSpeedStride;
                    }
                    break;
                }
                default:;
            }
        }
        //更新相机位置
        float3 newCenter = camera.cameraCenter;
        float3 newTarget = camera.cameraTarget;

        //鼠标移动
        int dx = input.mouseX;
        int dy = input.mouseY;

        //VK、D3D11/12需要反转相机正方向的Y轴、鼠标移动Y轴和键盘Y轴
        if (dx != 0 || dy != 0) {
            if (args.type != SDL_GraphicsWindowAPIType::OPENGL) {
                dy = -dy;
            }
            const float3 viewDirection = camera.cameraTarget - camera.cameraCenter;

            float3 U = normalize(camera.cameraU);
            float3 V = normalize(camera.cameraV);
            float3 W = normalize(camera.cameraW);

            const float yawAngle = -dx * args.mouseSensitivity;
            W = rotate(W, V, yawAngle);
            float pitchAngle = -dy * args.mouseSensitivity;
            W = rotate(W, U, pitchAngle);

            const float pitchLimitRadian = MathHelper::degreeToRadian(args.pitchLimitDegree);
            float newPitch = std::asin(W.y);
            bool needsCorrection = false;
            if (newPitch > pitchLimitRadian) {
                newPitch = pitchLimitRadian;
                needsCorrection = true;
            } else if (newPitch < -pitchLimitRadian) {
                newPitch = -pitchLimitRadian;
                needsCorrection = true;
            }
            if (needsCorrection) {
                const float3 horizontalDir = normalize(float3{W.x, 0.0f, W.z});
                const float horizontalMagnitude = std::cos(newPitch);
                W = horizontalDir * horizontalMagnitude + float3{0.0f, std::sin(newPitch), 0.0f};
            }
            newTarget = newCenter + W * std::sqrt(lengthSquared(viewDirection));
        }

        //键盘按键
        float3 movementDirection = {};
        const float3 forwardHorizontal = normalize(float3{camera.cameraW.x, 0.0f, camera.cameraW.z});
        if (input.keyW) movementDirection += forwardHorizontal;
        if (input.keyS) movementDirection -= forwardHorizontal;
        if (input.keyD) movementDirection += camera.cameraU;
        if (input.keyA) movementDirection -= camera.cameraU;

        if (args.type != SDL_GraphicsWindowAPIType::OPENGL) {
            if (input.keySpace) movementDirection -= camera.upDirection;
            if (input.keyLShift) movementDirection += camera.upDirection;
        } else {
            if (input.keySpace) movementDirection += camera.upDirection;
            if (input.keyLShift) movementDirection -= camera.upDirection;
        }

        if (lengthSquared(movementDirection) > 0.0f) {
            const float3 translation = normalize(movementDirection) * args.cameraMoveSpeed;
            newCenter += translation;
            newTarget += translation;
        }

        //更新相机
        camera.cameraCenter = newCenter;
        camera.cameraTarget = newTarget;
        camera.cameraW = camera.cameraTarget - camera.cameraCenter;
        camera.cameraU = normalize(cross(camera.cameraW, camera.upDirection));
        camera.cameraV = normalize(cross(camera.cameraU, camera.cameraW));
    }

    cudaSurfaceObject_t SDL_GraphicsWindowPrepareFrame(SDL_GraphicsWindowArgs & args) {
        switch (args.type) {
            case SDL_GraphicsWindowAPIType::OPENGL:
                args.cudaSurfaceObject = SDL_GLMapCudaResource(args.cudaGraphicsResource);
                break;
            case SDL_GraphicsWindowAPIType::VULKAN:
                args.vkImageIndex = SDL_VKPrepareFrame(args.vkArgs);
                args.cudaSurfaceObject = args.vkArgs.cudaSurface;
                break;
            case SDL_GraphicsWindowAPIType::DIRECT3D11:
                args.cudaSurfaceObject = SDL_D3D11MapCudaResource(args.d3d11Args);
                break;
            case SDL_GraphicsWindowAPIType::DIRECT3D12:
                args.d3d12Pair = SDL_D3D12PrepareFrame(args.d3d12Args);
                args.cudaSurfaceObject = args.d3d12Args.cudaSurfaces[args.d3d12Args.frameIndex];
                break;
            default:;
        }
        return args.cudaSurfaceObject;
    }

    void SDL_GraphicsWindowPresentFrame(SDL_GraphicsWindowArgs & args) {
        switch (args.type) {
            case SDL_GraphicsWindowAPIType::OPENGL:
                SDL_GLUnmapCudaResource(args.cudaGraphicsResource, args.cudaSurfaceObject);
                SDL_GLPresentFrame({args.window, args.glContext}, args.glArgs);
                break;
            case SDL_GraphicsWindowAPIType::VULKAN:
                SDL_VKPresentFrame(args.window, args.vkArgs, args.vkImageIndex);
                break;
#if defined(_WIN32) && defined(SDL) && defined(D3D11)
            case SDL_GraphicsWindowAPIType::DIRECT3D11:
                SDL_D3D11UnmapCudaResource(args.d3d11Args, args.cudaSurfaceObject);
                SDL_D3D11PresentFrame(args.d3d11Args);
                break;
#endif
#if defined(_WIN32) && defined(SDL) && defined(D3D12)
            case SDL_GraphicsWindowAPIType::DIRECT3D12:
                SDL_D3D12PresentFrame(args.d3d12Args, args.d3d12Pair);
                break;
#endif
            default:;
        }
    }

    void SDL_GraphicsWindowFrameFinish(const SDL_GraphicsWindowArgs & args) {
        const auto workTime = std::chrono::steady_clock::now() - args.frameStartTime;
        if (workTime < args.targetFrameDuration) {
            const auto timeToWait = args.targetFrameDuration - workTime;
            if (timeToWait > args.sleepMargin) {
                std::this_thread::sleep_for(timeToWait - args.sleepMargin);
            }
            while (std::chrono::steady_clock::now() - args.frameStartTime < args.targetFrameDuration) {}
        }
    }

    void SDL_DestroyGraphicsWindow(SDL_GraphicsWindowArgs & args) {
        SDL_Log("[SDL] Destroying SDL window...");
        switch (args.type) {
            case SDL_GraphicsWindowAPIType::OPENGL:
                SDL_GLFreeWritableCudaObject(args.cudaGraphicsResource);
                SDL_GLCleanupResource(args.glArgs);
                SDL_GL_DeleteContext(args.glContext);
                break;
            case SDL_GraphicsWindowAPIType::VULKAN:
                SDL_VKCleanupResource(args.vkArgs);
                break;
#if defined(_WIN32) && defined(SDL) && defined(D3D11)
            case SDL_GraphicsWindowAPIType::DIRECT3D11:
                SDL_D3D11CleanupResource(args.d3d11Args);
                break;
#endif
#if defined(_WIN32) && defined(SDL) && defined(D3D12)
            case SDL_GraphicsWindowAPIType::DIRECT3D12:
                SDL_D3D12CleanupResource(args.d3d12Args);
                break;
#endif
            default:;
        }
        SDL_DestroyWindow(args.window);
        SDL_Quit();
        SDL_Log("[SDL] SDL window destroyed.");
        args = {};

        //关闭鼠标锁定
        if (SDL_GetRelativeMouseMode() == SDL_TRUE) {
            SDL_CheckErrorInt(SDL_SetRelativeMouseMode(SDL_FALSE));
        }
    }
}