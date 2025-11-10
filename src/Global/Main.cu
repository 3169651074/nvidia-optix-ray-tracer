#include <Global/VTKReader.cuh>
#include <GraphicsAPI/SDL_GraphicsWindow.cuh>
using namespace project;

namespace {
    void updateInstancesTransforms(std::vector<OptixInstance> & instances, unsigned long long frameCount) {
        const static float3 initialCenter = {0.0, 2.0, 0.0};
        const float radius = 2.0f;
        const float speed = 0.02f;
        const float angle = static_cast<float>(frameCount) * speed;
        const float3 newCenter = {
                initialCenter.x + radius * std::cos(angle) * 1.5f,
                initialCenter.y + radius * std::sin(angle) * 1.5f,
                initialCenter.z
        };

        //更新顺序需要对应实例在GAS数组中的顺序：第一个为球体，后续为粒子
        float sphereTransform[12];
        MathHelper::constructTransformMatrix(
                {0.0f, 0.0f, -1010.0f},
                {0.0f, 0.0f, 0.0f},
                {1.0f, 1.0f, 1.0f}, sphereTransform);
        memcpy(instances[0].transform, sphereTransform, 12 * sizeof(float));

        float transform[12];
        MathHelper::constructTransformMatrix(
                newCenter,
                newCenter * 10.0f,
                {3.0f, 3.0f, 3.0f}, transform);
        for (size_t i = 1; i < instances.size(); i++) {
            memcpy(instances[i].transform, transform, 12 * sizeof(float));
        }
    }
}

#undef main
int main(int argc, char * argv[]) {
    //初始化optix
    auto context = createContext();

    // ====== 初始化几何体信息 ======
    //球体
    const std::vector<Sphere> spheres = {
            {MaterialType::ROUGH, 3, float3{0.0f, 0.0f, 0.0f}, 1000.0f},
    };
    //粒子：读取VTK文件，并将文件中的所有粒子信息转换为粒子数组
    const auto vtkParticles = VTKReader::readVTKFile("../files/particle_mesh/particle_000000000040000.vtk");
    const auto particles = VTKReader::convertToRendererData(vtkParticles);

    // ====== 构建加速结构 ======
    //构建顺序需要和SBT记录创建顺序相同（球体 -> 三角形 -> 粒子）
    std::vector<GAS> gasArray;

    //球体
    gasArray.push_back(buildGASForSpheres(context, spheres));
    //粒子，每个粒子一个GAS
    for (size_t i = 0; i < particles.size(); i++) {
        gasArray.push_back(buildGASForTriangles(context, particles[i].triangles));
    }

    //使用GAS信息构建实例列表
    //实例的sbtOffset为实例在GAS中的下标
    auto instances = createInstances(gasArray);

    //构建IAS
    auto ias = buildIAS(context, instances);

    // ====== 创建模块，管线和着色器绑定表 ======
    const OptixPipelineCompileOptions pipelineCompileOptions = {
            .usesMotionBlur = 0,
            .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
            .numPayloadValues = 4,      //3个颜色分量+1个当前追踪深度
            .numAttributeValues = 3,
            .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
            .pipelineLaunchParamsVariableName = "params",
            .usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE)
    };
    auto modules = createModules(context, pipelineCompileOptions);
    auto programs = createProgramGroups(context, modules);
    auto pipeline = linkPipeline(context, programs, pipelineCompileOptions);

    //几何体输入和材质输入
    const GeometryData geoData = {
            .spheres = spheres,
            .particles = particles,
    };
    const MaterialData matData = {
            .roughs = {
                    {.65, .05, .05},
                    {.73, .73, .73},
                    {.12, .45, .15},
                    {.70, .60, .50},
            },
            .metals = {
                    {0.8, 0.85, 0.88, 0.0},
            }
    };
    //每个实例对应一条sbt记录
    auto sbt = createShaderBindingTable(programs, geoData, matData);

    // ====== 初始化窗口和全局资源 ======
    const auto type = SDL_GraphicsWindowAPIType::OPENGL;
    const int w = 1200, h = 800;
    auto camera = SDL_GraphicsWindowConfigureCamera(
            {3.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f}, type
    );
    auto args = SDL_CreateGraphicsWindow(
            "Test", w, h, type, 60);

    //初始化随机数生成器
    curandState * dev_stateArray = nullptr;
    RandomGenerator::initDeviceRandomGenerators(dev_stateArray, w, h);

    //设置全局参数
    const GlobalParams params = {
            .handle = std::get<0>(ias),
            .stateArray = dev_stateArray
    };
    CUdeviceptr dev_params = 0;
    cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_params), sizeof(GlobalParams)));

    // ====== 启动渲染 ======
    SDL_Log("Starting...");
    SDL_Event event;
    SDL_GraphicsWindowKeyMouseInput input;
    unsigned long long frameCount = 0;

    while (!input.keyQuit) {
        SDL_GraphicsWindowFrameStart(args);
        SDL_GraphicsWindowUpdateCamera(event, input, args, camera);

        //更新实例和IAS
        updateInstancesTransforms(instances, frameCount);
        updateIAS(context, ias, instances);

        //更新raygen
        const RayGenParams rgData = {
                .width = w,
                .height = h,
                .surfaceObject = SDL_GraphicsWindowPrepareFrame(args),
                .cameraCenter = camera.cameraCenter,
                .cameraU = camera.cameraU,
                .cameraV = camera.cameraV,
                .cameraW = camera.cameraW
        };
        cudaCheckError(cudaMemcpy(
                reinterpret_cast<void *>(sbt.second[0] + OPTIX_SBT_RECORD_HEADER_SIZE), //SBT记录的头部是optix的header字段
                &rgData, sizeof(RayGenParams), cudaMemcpyHostToDevice));

        //更新全局参数
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_params),&params, sizeof(GlobalParams),cudaMemcpyHostToDevice));

        //启动
        optixCheckError(optixLaunch(
                pipeline, nullptr, dev_params, sizeof(GlobalParams),
                &sbt.first, w, h, 1));
        cudaCheckError(cudaDeviceSynchronize());

        //显示
        SDL_GraphicsWindowPresentFrame(args);
        SDL_GraphicsWindowFrameFinish(args);

        //更新帧计数
        frameCount++;
    }

    // ====== 清理窗口资源和全局资源 ======
    SDL_Log("Finished.");
    SDL_DestroyGraphicsWindow(args);
    cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_params)));
    RandomGenerator::freeDeviceRandomGenerators(dev_stateArray);

    // ====== 清理资源 ======
    SDL_Log("Render finished, cleaning up resources...");
    freeShaderBindingTable(sbt);
    unlinkPipeline(pipeline);
    destroyProgramGroups(programs);
    destroyModules(modules);
    cleanupAccelerationStructure(ias);
    cleanupAccelerationStructure(gasArray);
    destroyContext(context);

    SDL_Log("Cleanup completed.");
    return 0;
}
