#include <Global/RendererMesh.cuh>

namespace project {
    //转换额外输入为渲染器格式：将输入数据拷贝至设备内存并释放输入数组内存
    void mapAddGeoAndMatData(RendererMeshData & data, GeometryData & addGeoData, MaterialData & addMatData) {
        //输入数组中一个二维数组元素对应一个或多个几何体，对应一个渲染器输入数据对象
        //球体
        std::vector<RendererSphere> addSpheres; addSpheres.reserve(addGeoData.spheres.size());
        for (size_t i = 0; i < addGeoData.spheres.size(); i++) {
            const auto & currentSpheres = addGeoData.spheres[i];
            const size_t currentSphereCount = currentSpheres.size();

            float3 * dev_centers = nullptr;
            float * dev_radii = nullptr;
            cudaCheckError(cudaMalloc(&dev_centers, currentSphereCount * sizeof(float3)));
            cudaCheckError(cudaMalloc(&dev_radii, currentSphereCount * sizeof(float)));

            //逐个球体拷贝到设备
            for (size_t j = 0; j < currentSphereCount; j++) {
                cudaCheckError(cudaMemcpy(dev_centers + j, &currentSpheres[j].center, sizeof(float3), cudaMemcpyHostToDevice));
                cudaCheckError(cudaMemcpy(dev_radii + j, &currentSpheres[j].radius, sizeof(float), cudaMemcpyHostToDevice));
            }

            addSpheres.push_back({
                .materialType = addGeoData.sphereMaterialIndices[i].materialType,
                .materialIndex = addGeoData.sphereMaterialIndices[i].materialIndex,
                .dev_centers = dev_centers,
                .dev_radii = dev_radii,
                .count = currentSphereCount
            });
        }
        freeVector(addGeoData.spheres);
        freeVector(addGeoData.sphereMaterialIndices);

        //三角形
        std::vector<RendererTriangle> addTriangles; addTriangles.reserve(addGeoData.triangles.size());
        for (size_t i = 0; i < addGeoData.triangles.size(); i++) {
            const auto & currentTriangles = addGeoData.triangles[i];
            const size_t currentTriangleCount = currentTriangles.size();

            float3 * dev_vertices = nullptr;
            float3 * dev_normals = nullptr;
            cudaCheckError(cudaMalloc(&dev_vertices, currentTriangleCount * 3 * sizeof(float3)));
            cudaCheckError(cudaMalloc(&dev_normals, currentTriangleCount * 3 * sizeof(float3)));

            //每个三角形对应3个float3/float3
            for (size_t j = 0; j < currentTriangleCount; j++) {
                cudaCheckError(cudaMemcpy(
                        dev_vertices + j * 3, currentTriangles[i].vertices.data(),
                        3 * sizeof(float3), cudaMemcpyHostToDevice));
                cudaCheckError(cudaMemcpy(
                        dev_normals + j * 3, currentTriangles[i].normals.data(),
                        3 * sizeof(float3), cudaMemcpyHostToDevice));
            }

            addTriangles.push_back({
                .materialType = addGeoData.triangleMaterialIndices[i].materialType,
                .materialIndex = addGeoData.triangleMaterialIndices[i].materialIndex,
                .dev_vertices = dev_vertices,
                .dev_normals = dev_normals,
                .count = currentTriangleCount
            });
        }
        freeVector(addGeoData.triangles);
        freeVector(addGeoData.triangleMaterialIndices);

        //赋值到全局参数
        data.addSpheres = std::move(addSpheres);
        data.addTriangles = std::move(addTriangles);

        //当前只包含额外材质，VTK材质在加载缓存后拼接到materialAllFiles中
        data.materialAllFiles = {
                .roughs = std::move(addMatData.roughs),
                .metals = std::move(addMatData.metals)
        };
    }

    //为额外几何体构建GAS。此GAS和VTK粒子GAS合并，无需单独释放
    std::vector<GAS> buildAddDataGAS(RendererMeshData & data) {
        std::vector<GAS> addGAS;

        //此处将额外球体和三角形视为单图元物体，每个球体/三角形构建一个GAS
        for (size_t i = 0; i < data.addSpheres.size(); i++) {
            addGAS.push_back(buildGASForSpheres(data.context, data.addSpheres[i]));
        }
        for (size_t i = 0; i < data.addTriangles.size(); i++) {
            addGAS.push_back(buildGASForTriangles(data.context, data.addTriangles[i]));
        }
        return addGAS;
    }

    //子线程函数
    void readVTKFileCache(
            size_t currentFileIndex, size_t totalFileCount,
            const std::string & currentCacheFilePathName, std::atomic<size_t> & processedFileCount,
            const std::vector<GAS> & addGAS, RendererMeshData & data)
    {
        //为当前线程创建独立的CUDA流，用于使用同一个optix上下文构建加速结构
        cudaStream_t stream = nullptr;
        cudaCheckError(cudaStreamCreate(&stream));

        //读取缓存文件并直接移动到全局数组
        data.vtkParticleAllFiles[currentFileIndex] = VTKMeshReader::readVTKDataCache(currentCacheFilePathName);
        const auto & particlesThisFile = data.vtkParticleAllFiles[currentFileIndex];

        //为粒子数组中每一个粒子构造GAS和实例
        std::vector<GAS> gasThisFile; gasThisFile.reserve(particlesThisFile.size());
        auto & asThisFile = data.asAllFiles[currentFileIndex];

        //添加独立几何体GAS，添加顺序需要和SBT记录创建顺序相同（球体 -> 三角形 -> 粒子）
        gasThisFile.insert(gasThisFile.begin(), addGAS.begin(), addGAS.end());
        //构建粒子GAS
        for (const auto & i : particlesThisFile) {
            gasThisFile.push_back(buildGASForParticle(data.context, i, stream));
            //释放当前粒子的顶点数据
            cudaCheckError(cudaFree(i.dev_vertices));
        }
        const size_t instanceCount = gasThisFile.size();
        asThisFile.gas = std::move(gasThisFile);

        //构建实例，直接写入页面锁定内存中，默认实例变换矩阵为单位矩阵
        constexpr float defaultTransform[12] = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
        };
        cudaCheckError(cudaMallocHost(
                &asThisFile.pin_instances, instanceCount * sizeof(OptixInstance)));

        for (size_t i = 0; i < instanceCount; i++) {
            /*
             * sbtOffset = i：
             *   实例顺序与后续生成的SBT记录顺序完全一致：gasThisFile先插入额外几何体的GAS，再追加VTK粒子的GAS
             *   对应地，在创建SBT时也先把额外几何体的记录插到sbtRecords开头，再接上当前文件的粒子记录
             * 因此，实例索引 i 正好对齐同序的SBT记录数组
             */
            OptixInstance instance = {
                    .instanceId = 0,
                    .sbtOffset = static_cast<unsigned int>(i),
                    .visibilityMask = 1,
                    .flags = 0,
                    .traversableHandle = asThisFile.gas[i].first
            };
            memcpy(instance.transform, defaultTransform, 12 * sizeof(float));

            //拷贝到页面锁定内存
            memcpy(asThisFile.pin_instances + i, &instance, sizeof(OptixInstance));
        }

        //分配设备内存并拷贝数据
        cudaCheckError(cudaMalloc(
                &asThisFile.dev_instances, instanceCount * sizeof(OptixInstance)));
        cudaCheckError(cudaMemcpy(
                asThisFile.dev_instances, asThisFile.pin_instances,
                instanceCount * sizeof(OptixInstance), cudaMemcpyHostToDevice));
        asThisFile.instanceCount = instanceCount;

        //构建粒子+额外几何体IAS
        asThisFile.ias = buildIAS(data.context, asThisFile.dev_instances, instanceCount, stream);

        //打印进度
        processedFileCount++;
        SDL_Log("[%zd/%zd] (%.2f%%) Read VTK file cache %s.",
                processedFileCount.load(std::memory_order_relaxed), totalFileCount,
                static_cast<float>(processedFileCount) / static_cast<float>(totalFileCount) * 100.0f, currentCacheFilePathName.c_str());
        cudaCheckError(cudaStreamDestroy(stream));
    }

    RendererMeshData RendererMesh::commitRendererData(
            GeometryData & addGeoData, MaterialData & addMatData, const std::string particleMaterials,
            const std::string & seriesFilePath, const std::string & seriesFileName,
            const std::string & cacheFilePath, bool isDebugMode, size_t maxCacheLoadThreadCount)
    {
        SDL_Log("Loading renderer data...");
        RendererMeshData data = {
                .seriesFilePath = seriesFilePath,
                .seriesFileName = seriesFileName,
                .cacheFilePath = cacheFilePath
        };

        //初始化OptiX上下文
        data.context = createContext(cacheFilePath, isDebugMode);

        //转换输入并为额外几何体构建GAS
        SDL_Log("Loading additional geometry and material data...");
        mapAddGeoAndMatData(data, addGeoData, addMatData);
        data.addGAS = buildAddDataGAS(data);

        //读取series文件，获取每个文件持续时间和文件个数
        auto [vtkFiles, durations, fileCount] = VTKMeshReader::readSeriesFile(seriesFilePath, seriesFileName);
        data.durations = std::move(durations);
        data.fileCount = fileCount;

        data.vtkParticleAllFiles.resize(fileCount);
        data.asAllFiles.resize(fileCount);
        data.sbtAllFiles.reserve(fileCount);

        //多线程读取VTK缓存文件
        std::atomic<size_t> processedFileCount(0);

        //启动线程数不超过设定的最大值，防止CPU占用过高
        const size_t threadCount = std::min<size_t>(std::thread::hardware_concurrency(), maxCacheLoadThreadCount);
        SDL_Log("Reading VTK cache file using %zd concurrent threads...", threadCount);

        std::deque<std::thread> workers;
        for (size_t i = 0; i < fileCount; i++) {
            workers.emplace_back(
                    readVTKFileCache,
                    i, fileCount,
                    cacheFilePath + "particle" + std::to_string(i) + ".cache",
                    std::ref(processedFileCount),
                    std::cref(data.addGAS),
                    std::ref(data));
            //当启动的线程数达到限制值时等待队列前端（最先启动的线程）完成后，将其析构
            if (workers.size() == threadCount) {
                workers.front().join();
                workers.pop_front();
            }
        }

        //在等待线程执行时主线程生成材质。所有VTK文件的所有粒子使用同一个材质数组，和额外材质数组组合为全局材质数组
        //使用颜色映射器和最大粒子数生成VTK材质数组
        SDL_Log("Generating materials...");
        const auto rampPreset = resolveColorRampPreset(particleMaterials);
        SDL_Log("Using color ramp preset: %s", presetName(rampPreset));
        const auto maxCellCountSingleFile = VTKMeshReader::readMaxCellCountAllVTKFile(cacheFilePath);
        const auto vtkMaterials = bakeColorRamp(colorStopsForPreset(rampPreset), maxCellCountSingleFile);

        //将粒子材质数组插入到额外材质数组的尾部
        const size_t materialOffset = data.materialAllFiles.roughs.size();
        data.materialAllFiles.roughs.insert(
                data.materialAllFiles.roughs.end(),
                vtkMaterials.begin(), vtkMaterials.end());

        //创建模块和管线，所有文件都相同
        SDL_Log("Initializing OptiX...");
        const OptixPipelineCompileOptions pipelineCompileOptions = {
                .usesMotionBlur = 0,
                .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
                .numPayloadValues = 12,      //3个颜色分量+1个当前追踪深度+8个降噪器辅助参数
                .numAttributeValues = 3,
                .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
                .pipelineLaunchParamsVariableName = "params",
                .usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE)
        };
        data.modules = createModules(data.context, pipelineCompileOptions, isDebugMode);
        data.programGroups = createProgramGroups(data.context, data.modules);
        data.pipeline = linkPipeline(data.context, data.programGroups, pipelineCompileOptions);

        //等待所有线程完成
        while (!workers.empty()) {
            workers.front().join();
            workers.pop_front();
        }

        //为每个文件创建着色器绑定表。由于创建SBT需要全局材质数组，则不能并行
        SDL_Log("Creating shader binding table...");

        //创建raygen和miss记录（设备指针）
        auto [raygenRecordPtr, missRecordPtr] = createRaygenMissSBTRecord(
                data.programGroups[0], data.programGroups[1],
                {0.7f, 0.8f, 0.9f});
        data.raygenMissPtr = {raygenRecordPtr, missRecordPtr};
        //创建额外球体和三角形记录（Record结构体）
        auto addGeoRecord = createAddSphereTriangleSBTRecord(
                data.programGroups[2], data.programGroups[3],
                data.programGroups[4], data.programGroups[5],
                data.addSpheres, data.addTriangles, data.materialAllFiles);

        for (size_t i = 0; i < fileCount; i++) {
            //创建VTK粒子的SBT记录
            std::vector<std::pair<size_t, float3 *>> particleSBTData;
            particleSBTData.reserve(data.vtkParticleAllFiles[i].size());

            for (const auto & particle : data.vtkParticleAllFiles[i]) {
                particleSBTData.emplace_back(particle.id + materialOffset, particle.dev_normals);
            }
            auto sbtRecords = createVTKParticleSBTRecord(
                    data.programGroups[4], data.programGroups[5],
                    particleSBTData, data.materialAllFiles);

            //将当前文件的VTK SBT记录和额外几何体的记录合并
            //每个文件的所有实例都需要对应一个SBT记录，额外几何体的SBT记录需要在每个文件的记录列表中拷贝一份
            sbtRecords.insert(sbtRecords.begin(), addGeoRecord.begin(), addGeoRecord.end());
            const size_t hitgroupRecordCount = sbtRecords.size();

            //拷贝SBT记录数组到设备
            CUdeviceptr dev_hitgroupSBTRecords = 0;
            cudaCheckError(cudaMalloc(
                    reinterpret_cast<void **>(&dev_hitgroupSBTRecords),
                    hitgroupRecordCount * sizeof(HitGroupSbtRecord)));
            cudaCheckError(cudaMemcpy(
                    reinterpret_cast<void *>(dev_hitgroupSBTRecords), sbtRecords.data(),
                    hitgroupRecordCount * sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));

            //创建SBT结构体
            const OptixShaderBindingTable sbt = {
                    .raygenRecord = raygenRecordPtr,
                    .missRecordBase = missRecordPtr,
                    .missRecordStrideInBytes = sizeof(MissSbtRecord),
                    .missRecordCount = 1,
                    .hitgroupRecordBase = dev_hitgroupSBTRecords,
                    .hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord),
                    .hitgroupRecordCount = static_cast<unsigned int>(hitgroupRecordCount)
            };

            //仅添加当前文件所有实例的SBT记录
            data.sbtAllFiles.emplace_back(sbt, dev_hitgroupSBTRecords);
        }

        SDL_Log("Renderer data loaded.");
        return data;
    }

    void RendererMesh::startRender(RendererMeshData & data, const RenderLoopData & loopData) {
        SDL_Log("Starting renderer...");
        const size_t addGeoCount = data.addSpheres.size() + data.addTriangles.size();

        //初始化降噪器
        DenoiserArgs denoiserArgs = initDenoiser(data.context, loopData.windowWidth, loopData.windowHeight);

        //初始化随机数生成器
        curandState * dev_stateArray = nullptr;
        RandomGenerator::initDeviceRandomGenerators(dev_stateArray, loopData.windowWidth, loopData.windowHeight);

        //设置窗口参数
        auto camera = SDL_GraphicsWindowConfigureCamera(
                loopData.cameraCenter,
                loopData.cameraTarget,
                loopData.upDirection, loopData.apiType
        );
        auto args = SDL_CreateGraphicsWindow(
                loopData.windowTitle, loopData.windowWidth, loopData.windowHeight,
                loopData.apiType, loopData.targetFPS,
                loopData.mouseSensitivity, loopData.pitchLimitDegree,
                loopData.cameraMoveSpeedStride, loopData.initialSpeedNTimesStride,
                loopData.isGraphicsAPIDebugMode);

        //设置全局参数
        GlobalParams params = {
                .handle = 0,
                .stateArray = dev_stateArray
        };
        CUdeviceptr dev_params = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_params), sizeof(GlobalParams)));

        RayGenParams raygenData = {
                .width = static_cast<unsigned int>(loopData.windowWidth),
                .height = static_cast<unsigned int>(loopData.windowHeight),
                .colorBuffer = reinterpret_cast<float4 *>(denoiserArgs.denoiserInputBuffers[0]),
                .albedoBuffer = reinterpret_cast<float4 *>(denoiserArgs.denoiserInputBuffers[1]),
                .normalBuffer = reinterpret_cast<float4 *>(denoiserArgs.denoiserInputBuffers[2]),
        };

        //当前使用的文件下标，即加速结构索引
        size_t currentFileIndex = 0;

        //循环每个文件
        SDL_Event event;
        SDL_GraphicsWindowKeyMouseInput input; //定义在循环之外，更换文件时键盘移动不中断
        unsigned long long frameCount;

        while (true) {
            //准备当前文件资源
            auto & particleThisFile = data.vtkParticleAllFiles[currentFileIndex];
            auto & asThisFile = data.asAllFiles[currentFileIndex];
            auto & sbtRecordThisFile = data.sbtAllFiles[currentFileIndex];

            frameCount = 0;
            const auto frameCountPerFile = static_cast<size_t>(
                    data.durations[currentFileIndex] * static_cast<float>(loopData.targetFPS * loopData.renderSpeedRatio));

            //启动渲染当前文件
            while (frameCount < frameCountPerFile) {
                SDL_GraphicsWindowFrameStart(args);
                SDL_GraphicsWindowUpdateCamera(event, input, args, camera);

                //更新粒子变换矩阵
                float transform[12];
                for (size_t i = addGeoCount; i < asThisFile.instanceCount; i++) {
                    //计算该粒子运动总位移向量
                    const auto & velocity = particleThisFile[i - addGeoCount].velocity;
                    const float3 totalShift = velocity * data.durations[currentFileIndex];

                    //获取当前帧运动的位移，加上偏移量得到当前帧的位置
                    const float3 shift = totalShift / static_cast<float>(frameCountPerFile);
                    MathHelper::constructTransformMatrix(
                            {loopData.particleOffset + shift * static_cast<float>(frameCount)},
                            {0.0f, 0.0f, 0.0f}, loopData.particleScale, transform);
                    memcpy(asThisFile.pin_instances[i].transform, transform, 12 * sizeof(float));
                }

                //更新额外几何体变换矩阵并拷贝至设备内存
                (*data.func)(asThisFile.pin_instances, asThisFile.instanceCount, frameCount);
                cudaCheckError(cudaMemcpy(
                        asThisFile.dev_instances, asThisFile.pin_instances,
                        asThisFile.instanceCount * sizeof(OptixInstance), cudaMemcpyHostToDevice));
                //更新IAS
                updateIAS(data.context, asThisFile.ias, asThisFile.dev_instances, asThisFile.instanceCount);

                //更新raygen
                auto [outputCudaArray, outputSurfaceObject] = SDL_GraphicsWindowPrepareFrame(args);
                raygenData.cameraCenter = camera.cameraCenter;
                raygenData.cameraU = camera.cameraU;
                raygenData.cameraV = camera.cameraV;
                raygenData.cameraW = camera.cameraW;
                cudaCheckError(cudaMemcpy(
                        reinterpret_cast<void *>(data.raygenMissPtr.first + OPTIX_SBT_RECORD_HEADER_SIZE),
                        &raygenData, sizeof(RayGenParams), cudaMemcpyHostToDevice));

                //更新全局参数
                params.handle = std::get<0>(asThisFile.ias);
                cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_params),&params, sizeof(GlobalParams),cudaMemcpyHostToDevice));

                //启动
                optixCheckError(optixLaunch(
                        data.pipeline, nullptr, dev_params, sizeof(GlobalParams),
                        &sbtRecordThisFile.first, loopData.windowWidth, loopData.windowHeight, 1));
                cudaCheckError(cudaDeviceSynchronize());

                //降噪，当按下tab时不降噪
                if (!input.keyTab) {
                    denoiseOutput(denoiserArgs, outputCudaArray);
                } else {
                    skipDenoise(denoiserArgs, raygenData.colorBuffer, outputCudaArray);
                }

                //显示
                SDL_GraphicsWindowPresentFrame(args);
                SDL_GraphicsWindowFrameFinish(args);

                //更新帧计数
                frameCount++;
            }

            //下一个文件
            if (input.keyQuit) break;
            currentFileIndex++;

            //循环动画
            if (currentFileIndex >= data.fileCount) {
                currentFileIndex = 0;
            }
        }

        //清理窗口资源和全局资源
        SDL_Log("Render finished.");
        SDL_DestroyGraphicsWindow(args);
        cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_params)));
        RandomGenerator::freeDeviceRandomGenerators(dev_stateArray);

        //释放降噪器资源
        freeDenoiserResources(denoiserArgs);
    }

    void RendererMesh::freeRendererData(RendererMeshData & data) {
        SDL_Log("Render finished, cleaning up resources...");

        //释放设备端额外几何体数据
        for (auto & addSphere: data.addSpheres) {
            cudaCheckError(cudaFree(addSphere.dev_centers));
            cudaCheckError(cudaFree(addSphere.dev_radii));
            addSphere = {};
        }
        for (auto & addTriangle: data.addTriangles) {
            cudaCheckError(cudaFree(addTriangle.dev_vertices));
            cudaCheckError(cudaFree(addTriangle.dev_normals));
            addTriangle = {};
        }
        //释放VTK粒子数据
        for (auto & vtkParticleThisFile: data.vtkParticleAllFiles) {
            VTKMeshReader::freeVTKData(vtkParticleThisFile);
        }

        //释放加速结构和实例数组
        const size_t sharedGasCount = data.addGAS.size();
        for (auto & asThisFile: data.asAllFiles) {
            cleanupAccelerationStructure(asThisFile.ias);
            if (asThisFile.gas.size() > sharedGasCount) {
                std::vector<GAS> ownGas(
                        asThisFile.gas.begin() + static_cast<std::ptrdiff_t>(sharedGasCount),
                        asThisFile.gas.end());
                cleanupAccelerationStructure(ownGas);
            }

            cudaCheckError(cudaFreeHost(asThisFile.pin_instances));
            cudaCheckError(cudaFree(asThisFile.dev_instances));
        }
        cleanupAccelerationStructure(data.addGAS);

        //清理OptiX资源
        freeSBTRecords(data.sbtAllFiles);
        unlinkPipeline(data.pipeline);
        destroyProgramGroups(data.programGroups);
        destroyModules(data.modules);
        destroyContext(data.context);

        SDL_Log("Cleanup completed.");
        data = {};
    }

    void RendererMesh::writeCacheFilesAndExit(
            const std::string & seriesFilePath, const std::string & seriesFileName,
            const std::string & cacheFilePath, size_t maxCacheLoadThreadCount)
    {
        VTKMeshReader::writeVTKDataCache(seriesFilePath, seriesFileName, cacheFilePath, maxCacheLoadThreadCount);
        exit(0);
    }

    void RendererMesh::setAddGeoInsUpdateFunc(RendererMeshData & data, UpdateAddInstancesFunc func) {
        if (func == nullptr) {
            SDL_Log("Update instances function pointer is null, additional instance will never be updated!");
        } else {
            SDL_Log("Update instances function set.");
            data.func = func;
        }
    }
}