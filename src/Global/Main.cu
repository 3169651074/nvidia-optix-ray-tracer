#include <Global/VTKReader.cuh>
#include <GraphicsAPI/SDL_GraphicsWindow.cuh>
#include <thread>
#include <atomic>
#include <mutex>
#include <functional>
#include <algorithm>
#include <cctype>
#include <unordered_map>
using namespace project;
#undef main

namespace {
    enum class ColorRampPreset {
        Viridis,
        Plasma,
        Spectral,
        Terrain,
        Heatmap,
        Grayscale
    };

    struct ColorStop {
        float position;
        float3 color;
    };

    static float3 lerpColor(const float3 &a, const float3 &b, float t) {
        return float3{
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t
        };
    }

    static std::vector<ColorStop> colorStopsForPreset(ColorRampPreset preset) {
        switch (preset) {
            case ColorRampPreset::Plasma:
                return {
                        {0.00f, {0.050f, 0.030f, 0.527f}},
                        {0.25f, {0.537f, 0.062f, 0.549f}},
                        {0.50f, {0.871f, 0.191f, 0.494f}},
                        {0.75f, {0.992f, 0.580f, 0.288f}},
                        {1.00f, {0.940f, 0.975f, 0.131f}}
                };
            case ColorRampPreset::Spectral:
                return {
                        {0.00f, {0.619f, 0.003f, 0.258f}},
                        {0.20f, {0.835f, 0.243f, 0.310f}},
                        {0.40f, {0.957f, 0.427f, 0.263f}},
                        {0.60f, {0.993f, 0.681f, 0.380f}},
                        {0.80f, {0.741f, 0.858f, 0.407f}},
                        {1.00f, {0.400f, 0.761f, 0.647f}}
                };
            case ColorRampPreset::Terrain:
                return {
                        {0.00f, {0.149f, 0.149f, 0.149f}},
                        {0.25f, {0.114f, 0.451f, 0.208f}},
                        {0.50f, {0.639f, 0.784f, 0.325f}},
                        {0.75f, {0.988f, 0.972f, 0.745f}},
                        {1.00f, {0.996f, 0.922f, 0.545f}}
                };
            case ColorRampPreset::Heatmap:
                return {
                        {0.00f, {0.050f, 0.050f, 0.300f}},
                        {0.25f, {0.000f, 0.000f, 1.000f}},
                        {0.50f, {0.000f, 1.000f, 1.000f}},
                        {0.75f, {1.000f, 1.000f, 0.000f}},
                        {1.00f, {1.000f, 0.000f, 0.000f}}
                };
            case ColorRampPreset::Grayscale:
                return {
                        {0.00f, {0.050f, 0.050f, 0.050f}},
                        {1.00f, {0.950f, 0.950f, 0.950f}}
                };
            case ColorRampPreset::Viridis:
            default:
                return {
                        {0.00f, {0.267f, 0.004f, 0.329f}},
                        {0.25f, {0.283f, 0.141f, 0.458f}},
                        {0.50f, {0.254f, 0.265f, 0.530f}},
                        {0.75f, {0.196f, 0.509f, 0.364f}},
                        {1.00f, {0.993f, 0.906f, 0.144f}}
                };
        }
    }

    static std::vector<float3> bakeColorRamp(const std::vector<ColorStop> &stops, size_t count) {
        std::vector<float3> colors;
        if (count == 0 || stops.empty()) return colors;

        colors.resize(count);
        if (count == 1) {
            colors[0] = stops.back().color;
            return colors;
        }

        for (size_t i = 0; i < count; ++i) {
            const float u = static_cast<float>(i) / static_cast<float>(count - 1);
            const ColorStop *lower = &stops.front();
            const ColorStop *upper = &stops.back();

            for (size_t s = 1; s < stops.size(); ++s) {
                if (u <= stops[s].position) {
                    upper = &stops[s];
                    lower = &stops[s - 1];
                    break;
                }
                lower = &stops[s];
            }

            const float span = upper->position - lower->position;
            const float t = span > 0.0f ? (u - lower->position) / span : 0.0f;
            colors[i] = lerpColor(lower->color, upper->color, std::clamp(t, 0.0f, 1.0f));
        }
        return colors;
    }

    static std::string toLower(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    static const char *presetName(ColorRampPreset preset) {
        switch (preset) {
            case ColorRampPreset::Plasma:    return "plasma";
            case ColorRampPreset::Spectral:  return "spectral";
            case ColorRampPreset::Terrain:   return "terrain";
            case ColorRampPreset::Heatmap:   return "heatmap";
            case ColorRampPreset::Grayscale: return "grayscale";
            case ColorRampPreset::Viridis:
            default:                         return "viridis";
        }
    }

    static ColorRampPreset resolveColorRampPreset(int argc, char *argv[]) {
        const std::unordered_map<std::string, ColorRampPreset> mapping = {
                {"viridis",   ColorRampPreset::Viridis},
                {"plasma",    ColorRampPreset::Plasma},
                {"spectral",  ColorRampPreset::Spectral},
                {"terrain",   ColorRampPreset::Terrain},
                {"heatmap",   ColorRampPreset::Heatmap},
                {"grayscale", ColorRampPreset::Grayscale}
        };

        const std::string prefix = "--color-ramp=";
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg.rfind(prefix, 0) != 0) continue;

            const auto iter = mapping.find(toLower(arg.substr(prefix.size())));
            if (iter != mapping.end()) {
                return iter->second;
            }
            SDL_Log("Unknown color ramp preset: %s. Falling back to default.", arg.c_str());
        }

        return ColorRampPreset::Viridis;
    }
}

//#define GENERATE_CACHE_AND_EXIT
static constexpr const char * seriesFilePath = "../files/";
static constexpr const char * seriesFileName = "particle_mesh.vtk-short.series";
static constexpr const char * cacheFilePath = "../cache/";

static void updateInstancesTransforms(std::vector<OptixInstance> & instances, unsigned long long frameCount) {
    //设置球体变换矩阵
    float sphereTransform[12];
    MathHelper::constructTransformMatrix(
            {0.0f, 0.0f, -1000.5f},
            {0.0f, 0.0f, 0.0f},
            {1.0f, 1.0f, 1.0f}, sphereTransform);
    memcpy(instances[0].transform, sphereTransform, 12 * sizeof(float));
}

//子线程函数，操作加速结构数组的不同索引。threadID为文件索引
static void readVTKFileCache(
        size_t threadID, size_t totalFileCount,
        const std::string & cacheFilePathName, std::atomic<size_t> & processedFileCount,
        OptixDeviceContext & context, const std::vector<GAS> & additionalGeometryGAS,
        std::mutex & gasMutex, std::mutex & instanceMutex, std::mutex & iasMutex,
        std::vector<std::vector<GAS>> & gasForAllFiles, std::vector<std::vector<OptixInstance>> & instancesForAllFiles,
        std::vector<IAS> & iasForAllFiles, std::vector<std::vector<Particle>> & particlesForAllFiles)
{
    //读取缓存文件
    const auto particleForThisFile = VTKReader::readVTKDataCache(cacheFilePathName);
    const auto particleCountForThisFile = particleForThisFile.size();
    particlesForAllFiles[threadID] = particleForThisFile;

    //为粒子数组中每一个粒子构造GAS和实例
    std::vector<GAS> gasForThisFile;
    gasForThisFile.reserve(additionalGeometryGAS.size() + particleCountForThisFile);

    //添加独立几何体GAS，添加顺序需要和SBT记录创建顺序相同（球体 -> 三角形 -> 粒子）
    gasForThisFile.insert(gasForThisFile.begin(), additionalGeometryGAS.begin(), additionalGeometryGAS.end());

    //为当前线程创建独立的CUDA流，用于使用同一个optix上下文构建加速结构
    cudaStream_t stream = nullptr;
    cudaCheckError(cudaStreamCreate(&stream));

    //构建粒子GAS
    for (size_t j = 0; j < particleCountForThisFile; j++) {
        gasForThisFile.push_back(buildGASForTriangles(context, particleForThisFile[j].triangles, stream));
    }
    //添加到全局GAS数组，使用锁保护。锁在构造时锁定，析构时释放
    {
        std::lock_guard<std::mutex> gasLock(gasMutex);
        gasForAllFiles[threadID] = gasForThisFile;
    }

    //添加到全局实例数组
    const auto instancesForThisFile = createInstances(gasForThisFile);
    {
        std::lock_guard<std::mutex> instanceLock(instanceMutex);
        instancesForAllFiles[threadID] = instancesForThisFile;
    }

    //将当前文件所有粒子放到IAS中
    const auto iasForThisFile = buildIAS(context, instancesForThisFile, stream);
    {
        std::lock_guard<std::mutex> iasLock(iasMutex);
        iasForAllFiles[threadID] = iasForThisFile;
    }

    cudaCheckError(cudaStreamDestroy(stream));
    processedFileCount++;
    SDL_Log("[%zd/%zd] (%.2f%%) Created cache for VTK file %s.",
            processedFileCount.load(std::memory_order_relaxed), totalFileCount,
            static_cast<float>(processedFileCount) / static_cast<float>(totalFileCount) * 100.0f, cacheFilePathName.c_str());
}

int main(int argc, char * argv[]) {
    //额外几何体
    const std::vector<Sphere> spheres = {
            {MaterialType::ROUGH, 3, float3{0.0f, 0.0f, 0.0f}, 1000.0f},
    };
    const size_t additionalGeometryCount = spheres.size(); //额外几何体的总数，用于偏移粒子的SBT下标

    //额外材质
    const std::vector<Rough> roughs = {
            {.65, .05, .05},
            {.73, .73, .73},
            {.12, .45, .15},
            {.70, .60, .50},
    };
    const std::vector<Metal> metals = {
            {0.8, 0.85, 0.88, 0.0},
    };

    //生成VTK数据缓存并退出程序
#ifdef GENERATE_CACHE_AND_EXIT
    VTKReader::writeVTKDataCache(seriesFilePath, seriesFileName, cacheFilePath, roughs.size());
#endif

    //初始化optix
    auto context = createContext(cacheFilePath);

    //构建额外几何体加速结构
    std::vector<GAS> additionalGeometryGas = {
            buildGASForSpheres(context, spheres)
    };

    //粒子材质
    //读取series文件，获取每个文件持续时间和文件个数
    const auto [vtkFiles, durations, fileCount] = VTKReader::readSeriesFile(seriesFilePath, seriesFileName);

    //生成材质，所有VTK文件的所有粒子使用同一个材质数组，和额外材质数组组合为全局材质数组
    const auto maxCellCountSingleFile = VTKReader::maxCellCountSingleVTKFile(cacheFilePath);
    //const auto rampPreset = resolveColorRampPreset(argc, argv);
    const auto rampPreset = ColorRampPreset::Terrain;
    SDL_Log("Using color ramp preset: %s", presetName(rampPreset));

    const auto rampColors = bakeColorRamp(colorStopsForPreset(rampPreset), maxCellCountSingleFile);
    std::vector<Rough> allRoughs;
    allRoughs.reserve(roughs.size() + rampColors.size());
    for (const auto &color : rampColors) {
        allRoughs.push_back(Rough{color});
    }
    //将额外材质数组插入到粒子材质数组的头部
    allRoughs.insert(allRoughs.begin(), roughs.begin(), roughs.end());

    //粒子几何数据
    /*
     * 在渲染前需要将所有VTK文件包含的几何体信息，连同额外添加的几何体一同加载到设备内存中以备随时访问
     * 则需要创建多组加速结构。一个文件一个IAS，一组实例和一组GAS
     * 渲染时需要保证可以取到需要的IAS，通过IAS自动索引实例列表和GAS列表
     *
     * 构造加速结构时，将每个文件的所有粒子和额外添加的几何实体共同构建例列表和IAS
     * 在加速结构二维数组中，根据文件编号索引IAS
     */
    std::vector<std::vector<GAS>> gasForAllFiles(fileCount);
    std::vector<std::vector<OptixInstance>> instancesForAllFiles(fileCount);
    std::vector<IAS> iasForAllFiles(fileCount);
    std::vector<std::vector<Particle>> particlesForAllFiles(fileCount);

    std::mutex gasMutex, instanceMutex, iasMutex;
    std::atomic<size_t> processedFileCount(0);

    //为每个VTK缓存文件启动一个子线程，读取缓存数据并构建加速结构
    std::vector<std::thread> threads;
    for (size_t i = 0; i < fileCount; i++) {
        threads.emplace_back(
                readVTKFileCache,
                i, fileCount,
                std::string(cacheFilePath) + "particle" + std::to_string(i) + ".cache",
                std::ref(processedFileCount),
                std::ref(context), std::cref(additionalGeometryGas),
                std::ref(gasMutex), std::ref(instanceMutex), std::ref(iasMutex),
                std::ref(gasForAllFiles), std::ref(instancesForAllFiles), std::ref(iasForAllFiles),
                std::ref(particlesForAllFiles));
    }

    //等待缓存和加速结构构建完成
    for (auto & t: threads) {
        t.join();
    }

    //创建模块和管线，所有文件都相同
    const OptixPipelineCompileOptions pipelineCompileOptions = {
            .usesMotionBlur = 0,
            .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
            .numPayloadValues = 4,      //3个颜色分量+1个当前追踪深度
            .numAttributeValues = 3,
            .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
            .pipelineLaunchParamsVariableName = "params",
            .usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE)
    };
    auto modules = createModules(context, pipelineCompileOptions);
    auto programs = createProgramGroups(context, modules);
    auto pipeline = linkPipeline(context, programs, pipelineCompileOptions);

    //初始化窗口和全局资源，所有文件都相同
    const auto type = SDL_GraphicsWindowAPIType::OPENGL;
    const int w = 1200, h = 800;
    const size_t fps = 60;
    auto camera = SDL_GraphicsWindowConfigureCamera(
            {5.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f}, type
    );
    auto args = SDL_CreateGraphicsWindow(
            "Test", w, h, type, fps);
    //初始化随机数生成器
    curandState * dev_stateArray = nullptr;
    RandomGenerator::initDeviceRandomGenerators(dev_stateArray, w, h);

    //设置全局参数
    GlobalParams params = {
            .handle = 0,
            .stateArray = dev_stateArray
    };
    CUdeviceptr dev_params = 0;
    cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_params), sizeof(GlobalParams)));

    //当前使用的文件下标，即加速结构索引
    size_t currentFileIndex = 0;

    //一个文件渲染的帧数倍率，越大则粒子运动的越慢，1为原速度
    const size_t renderSpeedRatio = 4;
    //所有粒子的位置偏移（相对于原始位置）和缩放
    const float3 particleOffset = {0.0f, 0.0f, 0.0f};
    const float3 particleScale = {1.0f, 1.0f, 1.0f};

    //循环每个文件
    SDL_Event event;
    SDL_GraphicsWindowKeyMouseInput input; //定义在循环之外，更换文件时键盘移动不中断
    SDL_Log("Starting...");

    while (true) {
        //准备当前文件资源
        auto & particlesForThisFile = particlesForAllFiles[currentFileIndex];
        auto & instancesForThisFile = instancesForAllFiles[currentFileIndex];
        auto & iasForThisFile = iasForAllFiles[currentFileIndex];

        //设置几何体输入和材质输入，创建着色器绑定表
        const GeometryData geoData = {
                .spheres = spheres,
                .particles = particlesForThisFile
        };
        const MaterialData matData = {
                .roughs = allRoughs,
                .metals = metals
        };
        auto sbt = createShaderBindingTable(programs, geoData, matData);

        //启动渲染
        unsigned long long frameCount = 0;
        const auto frameCountPerFile = static_cast<size_t>(durations[currentFileIndex] * static_cast<float>(fps) * renderSpeedRatio);

        while (frameCount < frameCountPerFile) {
            SDL_GraphicsWindowFrameStart(args);
            SDL_GraphicsWindowUpdateCamera(event, input, args, camera);

            //更新粒子变换矩阵
            float transform[12];
            for (size_t i = additionalGeometryCount; i < instancesForThisFile.size(); i++) {
                //计算该粒子运动总位移向量
                const auto & velocity = particlesForThisFile[i - additionalGeometryCount].velocity;
                const float3 totalShift = velocity * durations[currentFileIndex];

                //获取当前帧运动的位移，加上偏移量得到当前帧的位置
                const float3 shift = totalShift / static_cast<float>(frameCountPerFile);
                MathHelper::constructTransformMatrix(
                        {particleOffset + shift * static_cast<float>(frameCount)},
                        {0.0f, 0.0f, 0.0f}, particleScale, transform);
                memcpy(instancesForThisFile[i].transform, transform, 12 * sizeof(float));
            }
            //更新额外几何体变换矩阵
            updateInstancesTransforms(instancesForThisFile, frameCount);
            //更新IAS
            updateIAS(context, iasForThisFile, instancesForThisFile);

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
            params.handle = std::get<0>(iasForThisFile);
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
        //清理此文件资源
        freeShaderBindingTable(sbt);

        //下一个文件
        if (input.keyQuit) break;
        currentFileIndex++;

        //循环动画
        if (currentFileIndex >= fileCount) {
            currentFileIndex = 0;
        }
    }

    //清理窗口资源和全局资源
    SDL_Log("Finished.");
    SDL_DestroyGraphicsWindow(args);
    cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_params)));
    RandomGenerator::freeDeviceRandomGenerators(dev_stateArray);

    //清理optix资源
    SDL_Log("Render finished, cleaning up resources...");
    unlinkPipeline(pipeline);
    destroyProgramGroups(programs);
    destroyModules(modules);

    //释放加速结构设备内存，球体GAS需要单独释放
    cleanupAccelerationStructure(iasForAllFiles);
    for (auto & gasForThisFile : gasForAllFiles) {
        for (size_t i = additionalGeometryCount; i < gasForThisFile.size(); i++) {
            cleanupAccelerationStructure(gasForThisFile[i]);
        }
    }
    cleanupAccelerationStructure(additionalGeometryGas);

    destroyContext(context);
    SDL_Log("Cleanup completed.");
    return 0;
}
