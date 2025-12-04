#include <Global/RendererTime.cuh>

namespace project {
    //转换额外输入为渲染器格式：将输入数据拷贝至设备内存并释放输入数组内存
    void mapAddGeoAndMatData(RendererTimeData & data, GeometryData & addGeoData, MaterialData & addMatData) {
        //球体
        std::vector<RendererSphere> addSpheres; addSpheres.reserve(addGeoData.spheres.size());
        for (size_t i = 0; i < addGeoData.spheres.size(); i++) {
            const auto & currentSpheres = addGeoData.spheres[i];
            const size_t currentSphereCount = currentSpheres.size();

            float3 * dev_centers = nullptr;
            float * dev_radii = nullptr;
            cudaCheckError(cudaMalloc(&dev_centers, currentSphereCount * sizeof(float3)));
            cudaCheckError(cudaMalloc(&dev_radii, currentSphereCount * sizeof(float)));

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

        data.addSpheres = std::move(addSpheres);
        data.addTriangles = std::move(addTriangles);

        data.materialAllFiles = {
                .roughs = std::move(addMatData.roughs),
                .metals = std::move(addMatData.metals)
        };
    }

    //为额外几何体构建GAS。此GAS和VTK粒子GAS合并，无需单独释放
    std::vector<GAS> buildAddDataGAS(RendererTimeData & data) {
        std::vector<GAS> addGAS;

        for (size_t i = 0; i < data.addSpheres.size(); i++) {
            addGAS.push_back(buildGASForSpheres(data.context, data.addSpheres[i]));
        }
        for (size_t i = 0; i < data.addTriangles.size(); i++) {
            addGAS.push_back(buildGASForTriangles(data.context, data.addTriangles[i]));
        }
        return addGAS;
    }

    //读取VTK文件，子线程函数
    void readVTKFile(
            size_t currentFileIndex, size_t totalFileCount, size_t addGeoCount,
            const std::string & currentVTKFilePathName, std::atomic<size_t> & processedFileCount,
            std::atomic<size_t> & maxPointCountSingleFile,
            const std::vector<GAS> & allGAS, RendererTimeData & data)
    {
        auto & asThisFile = data.asAllFiles[currentFileIndex];
        auto & vtkParticlesThisFile = data.vtkParticleAllFiles[currentFileIndex];

        //读取VTK文件
        auto [positions, ids, quats, velocities, shapeIDs] =
                VTKTimeReader::readVTKFile(currentVTKFilePathName, maxPointCountSingleFile);
        const auto particleCountThisFile = positions.size();
        vtkParticlesThisFile.reserve(particleCountThisFile);
        for (size_t i = 0; i < particleCountThisFile; i++) {
            vtkParticlesThisFile.emplace_back(positions[i], ids[i], quats[i], velocities[i], shapeIDs[i]);
        }

        //构建实例
        const auto instanceCount = addGeoCount + particleCountThisFile;
        asThisFile.gas.reserve(instanceCount);
        constexpr float defaultTransform[12] = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
        };
        cudaCheckError(cudaMallocHost(
                &asThisFile.pin_instances, instanceCount * sizeof(OptixInstance)));

        for (size_t i = 0; i < instanceCount; i++) {
            //额外几何体的GAS下标为i，粒子GAS下标由shapeID决定
            const auto & instanceGAS = allGAS[i < addGeoCount ? i : addGeoCount + vtkParticlesThisFile[i - addGeoCount].shapeID];
            asThisFile.gas.push_back(instanceGAS);

            OptixInstance instance = {
                    .instanceId = 0,
                    .sbtOffset = static_cast<unsigned int>(i),
                    .visibilityMask = 1,
                    .flags = 0,
                    .traversableHandle = instanceGAS.first
            };
            memcpy(instance.transform, defaultTransform, 12 * sizeof(float));
            memcpy(asThisFile.pin_instances + i, &instance, sizeof(OptixInstance));
        }

        //分配设备内存并拷贝实例
        cudaCheckError(cudaMalloc(
                &asThisFile.dev_instances, instanceCount * sizeof(OptixInstance)));
        cudaCheckError(cudaMemcpy(
                asThisFile.dev_instances, asThisFile.pin_instances,
                instanceCount * sizeof(OptixInstance), cudaMemcpyHostToDevice));
        asThisFile.instanceCount = instanceCount;

        //构建IAS
        cudaStream_t stream = nullptr;
        cudaCheckError(cudaStreamCreate(&stream));
        asThisFile.ias = buildIAS(data.context, asThisFile.dev_instances, instanceCount, stream);
        cudaCheckError(cudaStreamDestroy(stream));

        //打印进度
        processedFileCount++;
        SDL_Log("[%zd/%zd] (%.2f%%) Read VTK file %s.",
                processedFileCount.load(std::memory_order_relaxed), totalFileCount,
                static_cast<float>(processedFileCount) / static_cast<float>(totalFileCount) * 100.0f, currentVTKFilePathName.c_str());
    }

    RendererTimeData RendererTime::commitRendererData(
            GeometryData & addGeoData, MaterialData & addMatData, const std::string & particleMaterials,
            const std::string & seriesFilePath, const std::string & seriesFileName,
            const std::string & stlFilePath, const std::string & optixCacheFilePath,
            bool isDebugMode, size_t maxFileReadThreadCount)
    {
        SDL_Log("Loading renderer data...");
        RendererTimeData data = {
                .seriesFilePath = seriesFilePath,
                .seriesFileName = seriesFileName,
                .stlFilePath = stlFilePath
        };

        //初始化OptiX上下文
        data.context = createContext(optixCacheFilePath, isDebugMode);

        //转换输入并为额外几何体构建GAS
        SDL_Log("Loading additional geometry and material data...");
        mapAddGeoAndMatData(data, addGeoData, addMatData);
        auto gasAllFiles = buildAddDataGAS(data);
        data.addGeoCount = gasAllFiles.size();

        //读取STL文件目录下的所有STL文件，按文件名的字典顺序排序
        const auto particleDatas = VTKTimeReader::readSTLFiles(stlFilePath);

        //构建粒子GAS
        SDL_Log("Building particle GAS...");
        for (const auto & stlData: particleDatas) {
            gasAllFiles.push_back(buildGASForParticle(data.context, stlData, nullptr));
        }

        //读取series文件
        auto [vtkFiles, durations, fileCount] =
                VTKTimeReader::readSeriesFile(seriesFilePath, seriesFileName);
        data.durations = std::move(durations);
        data.fileCount = fileCount;

        data.vtkParticleAllFiles.resize(fileCount);
        data.asAllFiles.resize(fileCount);
        data.sbtAllFiles.reserve(fileCount);

        //多线程读取VTK文件
        const size_t threadCount = std::min<size_t>(std::thread::hardware_concurrency(), maxFileReadThreadCount);
        SDL_Log("Reading VTK file using %zd concurrent threads...", threadCount);

        std::atomic<size_t> processedFileCount(0);
        std::atomic<size_t> maxParticleCountSingleFile(0);
        std::deque<std::thread> workers;
        for (size_t i = 0; i < fileCount; i++) {
            workers.emplace_back(
                    readVTKFile, i, fileCount, data.addGeoCount, vtkFiles[i],
                    std::ref(processedFileCount), std::ref(maxParticleCountSingleFile),
                    std::cref(gasAllFiles), std::ref(data));
            if (workers.size() == threadCount) {
                workers.front().join();
                workers.pop_front();
            }
        }

        //创建模块和管线
        SDL_Log("Initializing OptiX...");
        const OptixPipelineCompileOptions pipelineCompileOptions = {
                .usesMotionBlur = 0,
                .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
                .numPayloadValues = 12,
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

        //生成材质，需要等待所有线程完成以获取单个文件最大粒子数
        SDL_Log("Generating materials...");
        const size_t materialOffset = data.materialAllFiles.roughs.size();
        const auto rampPreset = resolveColorRampPreset(particleMaterials);
        SDL_Log("Using color ramp preset: %s", presetName(rampPreset));
        const auto vtkMaterials = bakeColorRamp(
                colorStopsForPreset(rampPreset),
                maxParticleCountSingleFile.load(std::memory_order_relaxed));
        data.materialAllFiles.roughs.insert(
                data.materialAllFiles.roughs.end(),
                vtkMaterials.begin(), vtkMaterials.end());

        //创建着色器绑定表
        SDL_Log("Creating shader binding table...");
        auto [raygenRecordPtr, missRecordPtr] = createRaygenMissSBTRecord(
                data.programGroups[0], data.programGroups[1],
                {0.7f, 0.8f, 0.9f});
        data.raygenMissPtr = {raygenRecordPtr, missRecordPtr};
        auto addGeoRecord = createAddSphereTriangleSBTRecord(
                data.programGroups[2], data.programGroups[3],
                data.programGroups[4], data.programGroups[5],
                data.addSpheres, data.addTriangles, data.materialAllFiles);

        for (size_t i = 0; i < fileCount; i++) {
            std::vector<std::pair<size_t, float3 *>> particleSBTData;
            particleSBTData.reserve(data.vtkParticleAllFiles[i].size());

            for (const auto & particle : data.vtkParticleAllFiles[i]) {
                particleSBTData.emplace_back(particle.id + materialOffset, particleDatas[particle.shapeID].dev_normals);
            }
            auto sbtRecords = createVTKParticleSBTRecord(
                    data.programGroups[4], data.programGroups[5],
                    particleSBTData, data.materialAllFiles);

            sbtRecords.insert(sbtRecords.begin(), addGeoRecord.begin(), addGeoRecord.end());
            const size_t hitgroupRecordCount = sbtRecords.size();

            CUdeviceptr dev_hitgroupSBTRecords = 0;
            cudaCheckError(cudaMalloc(
                    reinterpret_cast<void **>(&dev_hitgroupSBTRecords),
                    hitgroupRecordCount * sizeof(HitGroupSbtRecord)));
            cudaCheckError(cudaMemcpy(
                    reinterpret_cast<void *>(dev_hitgroupSBTRecords), sbtRecords.data(),
                    hitgroupRecordCount * sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));

            const OptixShaderBindingTable sbt = {
                    .raygenRecord = raygenRecordPtr,
                    .missRecordBase = missRecordPtr,
                    .missRecordStrideInBytes = sizeof(MissSbtRecord),
                    .missRecordCount = 1,
                    .hitgroupRecordBase = dev_hitgroupSBTRecords,
                    .hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord),
                    .hitgroupRecordCount = static_cast<unsigned int>(hitgroupRecordCount)
            };

            data.sbtAllFiles.emplace_back(sbt, dev_hitgroupSBTRecords);
        }

        SDL_Log("Renderer data loaded.");
        return data;
    }

    //计算四元数的插值：使用 Slerp (Spherical Linear Interpolation，球面线性插值)
    //w, x, y, z
    float4 slerp(float4 q1, float4 q2, float t) {
        float dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;

        //如果点积为负，反转其中一个四元数，以确保走“最短路径”
        if (dot < 0.0f) {
            q2.w = -q2.w;
            q2.x = -q2.x;
            q2.y = -q2.y;
            q2.z = -q2.z;
            dot = -dot;
        }

        //如果两个四元数非常接近，使用线性插值 (Nlerp) 避免除以零
        if (dot > 0.9995f) {
            float4 result = {
                    q1.w + t * (q2.w - q1.w),
                    q1.x + t * (q2.x - q1.x),
                    q1.y + t * (q2.y - q1.y),
                    q1.z + t * (q2.z - q1.z)
            };
            //归一化
            float mag = std::sqrt(result.w * result.w + result.x * result.x + result.y * result.y + result.z * result.z);
            if (mag > 0.0) {
                result.w /= mag; result.x /= mag; result.y /= mag; result.z /= mag;
            }
            return result;
        }

        //使用标准 Slerp 公式
        float theta_0 = std::acos(dot);        // theta_0 = angle between input vectors
        float theta = theta_0 * t;                // theta = angle between v0 and result
        float sin_theta = std::sin(theta);     // compute this value only once
        float sin_theta_0 = std::sin(theta_0); // compute this value only once

        float s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;  //sin(theta_0 - theta) / sin(theta_0)
        float s1 = sin_theta / sin_theta_0;

        return {
                (s0 * q1.w) + (s1 * q2.w),
                (s0 * q1.x) + (s1 * q2.x),
                (s0 * q1.y) + (s1 * q2.y),
                (s0 * q1.z) + (s1 * q2.z)
        };
    }

    //四元数转为欧拉角（角度制）
    float3 quatToEuler(const float4 & q) {
        float3 angles;

        //Roll (x-axis)
        const float sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
        const float cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
        angles.x = std::atan2(sinr_cosp, cosr_cosp);

        //Pitch (y-axis)
        const float sinp = 2 * (q.w * q.y - q.z * q.x);
        if (std::abs(sinp) >= 1) {
            angles.y = std::copysign(PI / 2, sinp); // use 90 degrees if out of range
        } else {
            angles.y = std::asin(sinp);
        }

        //Yaw (z-axis)
        const float siny_cosp = 2 * (q.w * q.z + q.x * q.y);
        const float cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
        angles.z = std::atan2(siny_cosp, cosy_cosp);

        //转为角度
        angles.x = MathHelper::radianToDegree(angles.x);
        angles.y = MathHelper::radianToDegree(angles.y);
        angles.z = MathHelper::radianToDegree(angles.z);

        return angles;
    }

    void RendererTime::startRender(RendererTimeData & data, const RenderLoopData & loopData) {
        SDL_Log("Starting renderer...");

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

        //当前使用的文件下标
        size_t currentFileIndex = 0;

        //循环每个文件
        SDL_Event event;
        SDL_GraphicsWindowKeyMouseInput input;
        unsigned long long frameCount;

        SDL_Log("Starting...");
        while (true) {
            //准备当前文件资源
            auto & particleThisFile = data.vtkParticleAllFiles[currentFileIndex];
            auto & asThisFile = data.asAllFiles[currentFileIndex];
            auto & sbtRecordThisFile = data.sbtAllFiles[currentFileIndex];

            frameCount = 0;
            const auto frameCountThisFile = static_cast<size_t>(
                    data.durations[currentFileIndex] * static_cast<float>(loopData.targetFPS * loopData.renderSpeedRatio));

            //启动渲染当前文件
            while (frameCount < frameCountThisFile) {
                SDL_GraphicsWindowFrameStart(args);
                SDL_GraphicsWindowUpdateCamera(event, input, args, camera);

                //更新粒子变换矩阵
                float transform[12];
                for (size_t i = data.addGeoCount; i < asThisFile.instanceCount; i++) {
                    //当前文件下标
                    const size_t localIndex = i - data.addGeoCount;
                    const auto & particleCur  = particleThisFile[localIndex];

                    //下一文件下标
                    size_t nextFileIndex = currentFileIndex;
                    if (currentFileIndex + 1 < data.fileCount) {
                        nextFileIndex = currentFileIndex + 1;
                    }
                    const auto & particleNext = data.vtkParticleAllFiles[nextFileIndex][localIndex];

                    //本文件内的时间插值因子
                    float factor = 1.0f;
                    if (frameCountThisFile > 1) {
                        factor = static_cast<float>(frameCount) / static_cast<float>(frameCountThisFile - 1);
                    }

                    //计算该粒子运动总位移向量
                    const auto & velocity = particleCur.velocity;
                    const float3 totalShift = velocity * data.durations[currentFileIndex];
                    const float3 shiftThisFrame = totalShift / static_cast<float>(frameCountThisFile);
                    const float3 shift = particleCur.position + shiftThisFrame * static_cast<float>(frameCount);

                    //旋转插值：quatCur -> quatNext 的 slerp
                    const float4 quatCur  = particleCur.quat;
                    const float4 quatNext = particleNext.quat;
                    const float4 quat = slerp(quatCur, quatNext, factor);
                    const float3 rotate = quatToEuler(quat);

                    //构造变换矩阵
                    MathHelper::constructTransformMatrix(
                            loopData.particleOffset + shift,
                            rotate, loopData.particleScale, transform);
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

                //降噪
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

    void RendererTime::freeRendererData(RendererTimeData & data) {
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

        //释放加速结构和实例数组
        for (auto & asThisFile: data.asAllFiles) {
            cleanupAccelerationStructure(asThisFile.ias);
            cudaCheckError(cudaFreeHost(asThisFile.pin_instances));
            cudaCheckError(cudaFree(asThisFile.dev_instances));
        }
        cleanupAccelerationStructure(data.gasAllFiles);

        //清理OptiX资源
        freeSBTRecords(data.sbtAllFiles);
        unlinkPipeline(data.pipeline);
        destroyProgramGroups(data.programGroups);
        destroyModules(data.modules);
        destroyContext(data.context);

        SDL_Log("Cleanup completed.");
        data = {};
    }

    void RendererTime::setAddGeoInsUpdateFunc(RendererTimeData & data, UpdateAddInstancesFunc func) {
        if (func == nullptr) {
            SDL_Log("Update instances function pointer is null, additional instance will never be updated!");
        } else {
            SDL_Log("Update instances function set.");
            data.func = func;
        }
    }

}