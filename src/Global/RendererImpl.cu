#include <Global/RendererImpl.cuh>

namespace project {
    template <typename T>
    struct SbtRecord {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };
    typedef SbtRecord<RayGenParams>   RayGenSbtRecord;
    typedef SbtRecord<MissParams>     MissSbtRecord;
    typedef SbtRecord<HitGroupParams> HitGroupSbtRecord;

    OptixDeviceContext createContext(bool isDebug) {
        SDL_Log("Creating context...");
        cudaCheckError(cudaFree(nullptr));
        optixCheckError(optixInit());
        const OptixDeviceContextOptions options = {
                .logCallbackFunction = &optixLogCallBackFunction,
                .logCallbackData = nullptr,
                .logCallbackLevel = 4,
                .validationMode = isDebug ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF
        };
        OptixDeviceContext optixDeviceContext = nullptr;
        CUcontext cudaContext = nullptr;
        optixCheckError(optixDeviceContextCreate(cudaContext, &options, &optixDeviceContext));
        optixCheckError(optixDeviceContextSetCacheLocation(optixDeviceContext, "../cache"));
        SDL_Log("Context created.");
        return optixDeviceContext;
    }
    void destroyContext(OptixDeviceContext & context) {
        SDL_Log("Destroying context...");
        optixCheckError(optixDeviceContextDestroy(context));
        context = nullptr;
    }

    GAS buildGASForSpheres(OptixDeviceContext & context, const std::vector<Sphere> & spheres) {
        SDL_Log("Building GAS for spheres...");

        //GAS为静态，无需更新，则设置为可压缩且高质量
        const OptixAccelBuildOptions buildOptions = {
                .buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
                .operation = OPTIX_BUILD_OPERATION_BUILD
        };

        //将球体数据分为球心数组和半径数组，并拷贝至设备内存
        const size_t sphereCount = spheres.size();
        std::vector<float3> centers(sphereCount);
        std::vector<float> radii(sphereCount);
        for (size_t i = 0; i < sphereCount; i++) {
            centers[i] = spheres[i].center;
            radii[i] = spheres[i].radius;
        }
        CUdeviceptr dev_sphereCenters = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_sphereCenters), sphereCount * sizeof(float3)));
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_sphereCenters), centers.data(), sphereCount * sizeof(float3), cudaMemcpyHostToDevice));
        CUdeviceptr dev_sphereRadii = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_sphereRadii), sphereCount * sizeof(float)));
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_sphereRadii), radii.data(), sphereCount * sizeof(float), cudaMemcpyHostToDevice));

        //设置构建输入
        const unsigned int buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        const OptixBuildInput buildInput = {
                .type = OPTIX_BUILD_INPUT_TYPE_SPHERES,
                .sphereArray = {
                        .vertexBuffers = &dev_sphereCenters,
                        .vertexStrideInBytes = sizeof(float3),
                        .numVertices = static_cast<unsigned int>(sphereCount),
                        .radiusBuffers = &dev_sphereRadii,
                        .radiusStrideInBytes = sizeof(float),
                        .flags = buildInputFlags,
                        .numSbtRecords = 1,
                }
        };

        //计算加速结构内存并分配空间
        OptixAccelBufferSizes bufferSizes = {};
        optixCheckError(optixAccelComputeMemoryUsage(context, &buildOptions, &buildInput, 1, &bufferSizes));
        CUdeviceptr dev_tempBuffer = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_tempBuffer), bufferSizes.tempSizeInBytes));
        CUdeviceptr dev_output = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_output), bufferSizes.outputSizeInBytes));

        //计算压缩后大小
        CUdeviceptr dev_compressedSize = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_compressedSize), sizeof(unsigned long long)));
        const OptixAccelEmitDesc emitProperty = {
                .result = dev_compressedSize,
                .type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
        };

        //构建GAS
        OptixTraversableHandle handle = 0;
        optixCheckError(optixAccelBuild(
                context, nullptr, &buildOptions,
                &buildInput, 1,
                dev_tempBuffer, bufferSizes.tempSizeInBytes,
                dev_output, bufferSizes.outputSizeInBytes,
                &handle, &emitProperty, 1));
        cudaCheckError(cudaDeviceSynchronize());

        //释放临时空间
        cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_tempBuffer)));
        cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_sphereRadii)));
        cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_sphereCenters)));

        //压缩GAS
        unsigned long long compressedSize = 0;
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(&compressedSize), reinterpret_cast<void *>(dev_compressedSize), sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        if (compressedSize < bufferSizes.outputSizeInBytes) {
            SDL_Log("Compressing GAS: %zd --> %zd.", bufferSizes.outputSizeInBytes, compressedSize);

            //重新分配空间
            CUdeviceptr dev_compressedGasOutput = 0;
            cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_compressedGasOutput), compressedSize));
            optixCheckError(optixAccelCompact(
                    context, nullptr, handle,
                    dev_compressedGasOutput, compressedSize,
                    &handle));
            cudaCheckError(cudaDeviceSynchronize());

            //释放原有空间
            cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_output)));
            dev_output = dev_compressedGasOutput;
        } else {
            SDL_Log("Does not compress GAS!");
        }

        SDL_Log("GAS for spheres built.");
        return {handle, dev_output};
    }
    GAS buildGASForTriangles(OptixDeviceContext & context, const std::vector<Triangle> & triangles) {
        SDL_Log("Building GAS for triangles...");

        const OptixAccelBuildOptions buildOptions = {
                .buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
                .operation = OPTIX_BUILD_OPERATION_BUILD
        };

        //三角形的顶点需要作为GAS构建输入，法线和顶点一同传入Hit着色器中用于生成反射光线
        const size_t triangleCount = triangles.size();

        //非索引模式，提取顶点数据到顶点数组
        std::vector<float3> triangleVertices(triangleCount * 3);
        for (size_t i = 0; i < triangleCount; i++) {
            triangleVertices[i * 3 + 0] = triangles[i].vertices[0];
            triangleVertices[i * 3 + 1] = triangles[i].vertices[1];
            triangleVertices[i * 3 + 2] = triangles[i].vertices[2];
        }
        CUdeviceptr dev_triangleVertices = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_triangleVertices), triangleCount * 3 * sizeof(float3)));
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_triangleVertices), triangleVertices.data(), triangleCount * 3 * sizeof(float3), cudaMemcpyHostToDevice));

        const unsigned int buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        const OptixBuildInput buildInput = {
                .type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
                .triangleArray = {
                        .vertexBuffers = &dev_triangleVertices,
                        .numVertices = static_cast<unsigned int>(triangleVertices.size()),
                        .vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
                        .vertexStrideInBytes = sizeof(float3),
                        .flags = buildInputFlags,
                        .numSbtRecords = 1
                }
        };

        //计算加速结构内存并分配空间
        OptixAccelBufferSizes bufferSizes = {};
        optixCheckError(optixAccelComputeMemoryUsage(context, &buildOptions, &buildInput, 1, &bufferSizes));
        CUdeviceptr dev_tempBuffer = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_tempBuffer), bufferSizes.tempSizeInBytes));
        CUdeviceptr dev_output = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_output), bufferSizes.outputSizeInBytes));

        //计算压缩后大小
        CUdeviceptr dev_compressedSize = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_compressedSize), sizeof(unsigned long long)));
        const OptixAccelEmitDesc emitProperty = {
                .result = dev_compressedSize,
                .type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
        };

        //构建GAS
        OptixTraversableHandle handle = 0;
        optixCheckError(optixAccelBuild(
                context, nullptr, &buildOptions,
                &buildInput, 1,
                dev_tempBuffer, bufferSizes.tempSizeInBytes,
                dev_output, bufferSizes.outputSizeInBytes,
                &handle, &emitProperty, 1));
        cudaCheckError(cudaDeviceSynchronize());

        //释放临时空间
        cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_tempBuffer)));
        cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_triangleVertices)));

        //压缩GAS
        unsigned long long compressedSize = 0;
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(&compressedSize), reinterpret_cast<void *>(dev_compressedSize), sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        if (compressedSize < bufferSizes.outputSizeInBytes) {
            SDL_Log("Compressing GAS: %zd --> %zd.", bufferSizes.outputSizeInBytes, compressedSize);

            //重新分配空间
            CUdeviceptr dev_compressedGasOutput = 0;
            cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_compressedGasOutput), compressedSize));
            optixCheckError(optixAccelCompact(
                    context, nullptr, handle,
                    dev_compressedGasOutput, compressedSize,
                    &handle));
            cudaCheckError(cudaDeviceSynchronize());

            //释放原有空间
            cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_output)));
            dev_output = dev_compressedGasOutput;
        } else {
            SDL_Log("Does not compress GAS!");
        }

        SDL_Log("GAS for triangles built.");
        return {handle, dev_output};
    }
    void cleanupAccelerationStructure(GAS & data) {
        std::vector vector = {data};
        cleanupAccelerationStructure(vector);
        data = {};
    }
    void cleanupAccelerationStructure(std::vector<GAS> & data) {
        for (auto & item: data) {
            cudaCheckError(cudaFree(reinterpret_cast<void *>(item.second)));
        }
        data = {};
    }
    void cleanupAccelerationStructure(IAS & data) {
        std::vector vector = {data};
        cleanupAccelerationStructure(vector);
        data = {};
    }
    void cleanupAccelerationStructure(std::vector<IAS> & data) {
        for (auto & item : data) {
            const auto & [handle, ptr, _] = item;
            cudaCheckError(cudaFree(reinterpret_cast<void *>(ptr)));
        }
        data = {};
    }

    std::vector<OptixInstance> createInstances(const std::vector<GAS> & data) {
        SDL_Log("Creating instances...");

        const size_t instanceCount = data.size();
        std::vector<OptixInstance> instances(instanceCount);

        for (size_t i = 0; i < instanceCount; i++) {
            auto & instance = instances[i];
            instance.instanceId = 0;
            instance.sbtOffset = i;
            instance.visibilityMask = 1;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.traversableHandle = data[i].first;
        }

        //此函数只部分初始化实例，在渲染前需要通过实例更新函数更新实例变换矩阵
        SDL_Log("Instances created.");
        return instances;
    }

    IAS buildIAS(OptixDeviceContext & context, const std::vector<OptixInstance> & instances) {
        const size_t instanceCount = instances.size();

        //同GAS，实例数组作为构建输入，需要被拷贝至设备内存
        CUdeviceptr dev_instances = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_instances), instanceCount * sizeof(OptixInstance)));
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_instances), instances.data(), instanceCount * sizeof(OptixInstance), cudaMemcpyHostToDevice));

        //构建输入，允许更新且高性能
        const OptixAccelBuildOptions buildOptions = {
                .buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_BUILD,
                .operation = OPTIX_BUILD_OPERATION_BUILD
        };
        const OptixBuildInput buildInput = {
                .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
                .instanceArray = {
                        .instances = dev_instances,
                        .numInstances = static_cast<unsigned int>(instanceCount)
                }
        };

        //计算IAS空间
        OptixAccelBufferSizes bufferSizes;
        optixCheckError(optixAccelComputeMemoryUsage(context, &buildOptions, &buildInput, 1, &bufferSizes));
        CUdeviceptr dev_tempBuffer = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_tempBuffer), bufferSizes.tempSizeInBytes));
        CUdeviceptr dev_output = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_output), bufferSizes.outputSizeInBytes));

        //构建IAS，不压缩，支持压缩的加速结构不支持更新
        OptixTraversableHandle handle = 0;
        optixCheckError(optixAccelBuild(context, nullptr, &buildOptions, &buildInput, 1, dev_tempBuffer, bufferSizes.tempSizeInBytes,
                                        dev_output, bufferSizes.outputSizeInBytes, &handle, nullptr, 0));
        cudaCheckError(cudaDeviceSynchronize());

        //释放临时空间
        cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_tempBuffer)));
        cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_instances)));

        return {handle, dev_output, bufferSizes};
    }
    void updateIAS(OptixDeviceContext & context, IAS & ias, const std::vector<OptixInstance> & instances) {
        const size_t instanceCount = instances.size();
        CUdeviceptr dev_instances = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_instances), instanceCount * sizeof(OptixInstance)));
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_instances), instances.data(), instanceCount * sizeof(OptixInstance), cudaMemcpyHostToDevice));

        const OptixAccelBuildOptions buildOptions = {
                .buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_BUILD, //flags需要保持和构建时一致
                .operation = OPTIX_BUILD_OPERATION_UPDATE   //更新
        };
        const OptixBuildInput buildInput = {
                .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
                .instanceArray = {
                        .instances = dev_instances,
                        .numInstances = static_cast<unsigned int>(instanceCount)
                }
        };
        auto [handle, ptr, bufferSizes] = ias;

        //分配更新所需临时空间，更新操作在原有空间上进行，无需释放原有空间
        CUdeviceptr dev_tempBuffer = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_tempBuffer), bufferSizes.tempUpdateSizeInBytes));

        //更新
        optixCheckError(optixAccelBuild(
                context, nullptr, &buildOptions,
                &buildInput, 1,
                dev_tempBuffer, bufferSizes.tempUpdateSizeInBytes, //改为tempUpdateSize
                ptr, bufferSizes.outputSizeInBytes,
                &handle, nullptr, 0));
        cudaCheckError(cudaDeviceSynchronize());

        cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_tempBuffer)));
        cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_instances)));
    }

    std::array<OptixModule, 3> createModules(
            OptixDeviceContext & context, const OptixPipelineCompileOptions & pipelineCompileOptions, bool isDebugMode)
    {
        SDL_Log("Creating modules...");

        const OptixModuleCompileOptions moduleCompileOptions = {
                .optLevel = isDebugMode ? OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 : OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
                .debugLevel = isDebugMode ? OPTIX_COMPILE_DEBUG_LEVEL_FULL : OPTIX_COMPILE_DEBUG_LEVEL_NONE
        };
        SDL_Log("Reading compiled PTX...");
        const std::string optixShader = FileHelper::readTextFile("../shader/shader.ptx");
        if (!optixShader.empty()) {
            SDL_Log("Read PTX string:\n%s...", optixShader.substr(0, 200).c_str());
        } else {
            SDL_Log("Failed to read PTX string!");
            exit(-1);
        }

        //创建ptx模块
        OptixModule module = nullptr;
        optixCheckError(optixModuleCreate(
                context, &moduleCompileOptions, &pipelineCompileOptions,
                optixShader.c_str(), optixShader.size(),
                optixPerCallLogBuffer, optixPerCallLogSize, &module));
        optixCheckPerCallLog();

        //获取内置球体求交模块
        const OptixBuiltinISOptions sphereISOptions = {
                .builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE,
                .usesMotionBlur = false
        };
        OptixModule sphereModule = nullptr;
        optixCheckError(optixBuiltinISModuleGet(
                context, &moduleCompileOptions,
                &pipelineCompileOptions, &sphereISOptions,
                &sphereModule));

        //获取内置三角形求交模块
        const OptixBuiltinISOptions triangleISOptions = {
                .builtinISModuleType = OPTIX_PRIMITIVE_TYPE_TRIANGLE,
                .usesMotionBlur = false
        };
        OptixModule triangleModule = nullptr;
        optixCheckError(optixBuiltinISModuleGet(
                context, &moduleCompileOptions,
                &pipelineCompileOptions, &triangleISOptions,
                &triangleModule));

        SDL_Log("Modules created.");
        return {module, sphereModule, triangleModule};
    }
    void destroyModules(std::array<OptixModule, 3> & modules) {
        SDL_Log("Destroying modules...");
        for (auto & item: modules) {
            optixCheckError(optixModuleDestroy(item));
        }
        modules = {};
    }

    std::array<OptixProgramGroup, 6> createProgramGroups(OptixDeviceContext & context, std::array<OptixModule, 3> & modules) {
        SDL_Log("Creating program groups...");
        const OptixProgramGroupOptions options = {};

        //raygen
        SDL_Log("Creating raygen program...");
        OptixProgramGroup rayGenProgramGroup = nullptr;
        const OptixProgramGroupDesc raygenProgramDesc = {
                .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                .raygen = {.module = modules[0], .entryFunctionName = "__raygen__raygenProgram"}
        };
        optixCheckError(optixProgramGroupCreate(
                context, &raygenProgramDesc,
                1, &options,
                optixPerCallLogBuffer, optixPerCallLogSize, &rayGenProgramGroup));
        optixCheckPerCallLog();

        //miss
        SDL_Log("Creating miss program...");
        OptixProgramGroup missProgramGroup = nullptr;
        const OptixProgramGroupDesc missProgramDesc = {
                .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
                .miss = {.module = modules[0], .entryFunctionName = "__miss__missProgram"}
        };
        optixCheckError(optixProgramGroupCreate(
                context, &missProgramDesc,
                1, &options,
                optixPerCallLogBuffer, optixPerCallLogSize, &missProgramGroup));
        optixCheckPerCallLog();

        //closesthit - sphere
        SDL_Log("Creating closesthit program for sphere...");
        OptixProgramGroup hitGroupProgramGroupSphereRough = nullptr;
        const OptixProgramGroupDesc hitProgramDescSphereRough = {
                .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                .hitgroup = {
                        .moduleCH = modules[0], .entryFunctionNameCH = "__closesthit__closesthitProgramSphereRough",
                        .moduleAH = nullptr, .entryFunctionNameAH = nullptr,
                        .moduleIS = modules[1], .entryFunctionNameIS = nullptr //使用内置球体IS模块
                }
        };
        optixCheckError(optixProgramGroupCreate(
                context, &hitProgramDescSphereRough,
                1, &options,
                optixPerCallLogBuffer, optixPerCallLogSize,
                &hitGroupProgramGroupSphereRough));
        optixCheckPerCallLog();
        OptixProgramGroup hitGroupProgramGroupSphereMetal = nullptr;
        const OptixProgramGroupDesc hitProgramDescSphereMetal = {
                .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                .hitgroup = {
                        .moduleCH = modules[0], .entryFunctionNameCH = "__closesthit__closesthitProgramSphereMetal",
                        .moduleAH = nullptr, .entryFunctionNameAH = nullptr,
                        .moduleIS = modules[1], .entryFunctionNameIS = nullptr
                }
        };
        optixCheckError(optixProgramGroupCreate(
                context, &hitProgramDescSphereMetal,
                1, &options,
                optixPerCallLogBuffer, optixPerCallLogSize,
                &hitGroupProgramGroupSphereMetal));
        optixCheckPerCallLog();

        //closesthit - triangle
        SDL_Log("Creating closesthit program for triangle...");
        OptixProgramGroup hitGroupProgramGroupTriangleRough = nullptr;
        const OptixProgramGroupDesc hitProgramDescTriangleRough = {
                .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                .hitgroup = {
                        .moduleCH = modules[0], .entryFunctionNameCH = "__closesthit__closesthitProgramTriangleRough",
                        .moduleAH = nullptr, .entryFunctionNameAH = nullptr,
                        .moduleIS = modules[2], .entryFunctionNameIS = nullptr //使用内置三角形IS模块
                }
        };
        optixCheckError(optixProgramGroupCreate(
                context, &hitProgramDescTriangleRough,
                1, &options,
                optixPerCallLogBuffer, optixPerCallLogSize,
                &hitGroupProgramGroupTriangleRough));
        optixCheckPerCallLog();
        OptixProgramGroup hitGroupProgramGroupTriangleMetal = nullptr;
        const OptixProgramGroupDesc hitProgramDescTriangleMetal = {
                .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                .hitgroup = {
                        .moduleCH = modules[0], .entryFunctionNameCH = "__closesthit__closesthitProgramTriangleMetal",
                        .moduleAH = nullptr, .entryFunctionNameAH = nullptr,
                        .moduleIS = modules[2], .entryFunctionNameIS = nullptr
                }
        };
        optixCheckError(optixProgramGroupCreate(
                context, &hitProgramDescTriangleMetal,
                1, &options,
                optixPerCallLogBuffer, optixPerCallLogSize,
                &hitGroupProgramGroupTriangleMetal));
        optixCheckPerCallLog();

        SDL_Log("Modules created.");
        return {
            rayGenProgramGroup, missProgramGroup,
            hitGroupProgramGroupSphereRough, hitGroupProgramGroupSphereMetal,    //sphere
            hitGroupProgramGroupTriangleRough, hitGroupProgramGroupTriangleMetal //triangle
        };
    }
    void destroyProgramGroups(std::array<OptixProgramGroup, 6> & programGroups) {
        SDL_Log("Destroying program groups...");
        for (auto & item: programGroups) {
            optixCheckError(optixProgramGroupDestroy(item));
        }
        programGroups = {};
    }

    OptixPipeline linkPipeline(
            OptixDeviceContext & context, const std::array<OptixProgramGroup, 6> & linkedProgramGroups,
            const OptixPipelineCompileOptions & pipelineCompileOptions)
    {
        SDL_Log("Linking pipeline...");
        const OptixPipelineLinkOptions pipelineLinkOptions = {
                .maxTraceDepth = RAY_TRACE_DEPTH
        };

        //将所有传入的程序组都用于创建管线
        OptixPipeline optixPipeline = nullptr;
        optixCheckError(optixPipelineCreate(
                context, &pipelineCompileOptions,
                &pipelineLinkOptions, linkedProgramGroups.data(), linkedProgramGroups.size(),
                optixPerCallLogBuffer, optixPerCallLogSize, &optixPipeline));
        optixCheckPerCallLog();

        //计算最终硬件执行时所需的具体堆栈大小
        OptixStackSizes stackSizes = {};
        for (auto linkedProgramGroup : linkedProgramGroups) {
            optixCheckError(optixUtilAccumulateStackSizes(linkedProgramGroup, &stackSizes, optixPipeline));
        }

        SDL_Log("Pipeline linked.");
        return optixPipeline;
    }
    void unlinkPipeline(OptixPipeline & optixPipeline) {
        SDL_Log("Unlinking pipeline...");
        optixCheckError(optixPipelineDestroy(optixPipeline));
        optixPipeline = nullptr;
    }

    std::pair<OptixShaderBindingTable, std::vector<CUdeviceptr>> createShaderBindingTable(
            const std::array<OptixProgramGroup, 6> & programGroups, const GeometryData & geoData, const MaterialData & matData)
    {
        SDL_Log("Creating shader binding table...");
        std::vector<CUdeviceptr> ptrs;

        //raygen
        SDL_Log("Creating raygen record...");
        RayGenSbtRecord rayGenSbtRecord = {};
        optixCheckError(optixSbtRecordPackHeader(programGroups[0], &rayGenSbtRecord));
        CUdeviceptr dev_raygenRecord = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_raygenRecord), sizeof(RayGenSbtRecord)));
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_raygenRecord), &rayGenSbtRecord, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));
        ptrs.push_back(dev_raygenRecord);

        //miss
        SDL_Log("Creating miss record...");
        MissSbtRecord missSbtRecord = {};
        optixCheckError(optixSbtRecordPackHeader(programGroups[1], &missSbtRecord));
        missSbtRecord.data = {0.7f, 0.8f, 0.9f};
        CUdeviceptr dev_missRecord = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_missRecord), sizeof(MissSbtRecord)));
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_missRecord), &missSbtRecord, sizeof(MissSbtRecord), cudaMemcpyHostToDevice));
        ptrs.push_back(dev_missRecord);

        //hit
        const auto & spheres = geoData.spheres;
        const auto & triangles = geoData.triangles;
        const auto & particles = geoData.particles;
        const auto & roughs = matData.roughs;
        const auto & metals = matData.metals;
        std::vector<HitGroupSbtRecord> hitGroupSbtRecords;

        //为所有球体的中心和半径分配设备内存
        const size_t sphereCount = spheres.size();
        std::vector<float3> sphereCenters(sphereCount);
        std::vector<float> sphereRadii(sphereCount);
        for (size_t i = 0; i < sphereCount; i++) {
            sphereCenters[i] = spheres[i].center;
            sphereRadii[i] = spheres[i].radius;
        }

        CUdeviceptr dev_sphereCenters = 0;
        CUdeviceptr dev_sphereRadii = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_sphereCenters), sphereCount * sizeof(float3)));
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_sphereRadii), sphereCount * sizeof(float)));
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_sphereCenters), sphereCenters.data(), sphereCount * sizeof(float3), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_sphereRadii), sphereRadii.data(), sphereCount * sizeof(float), cudaMemcpyHostToDevice));
        ptrs.push_back(dev_sphereCenters);
        ptrs.push_back(dev_sphereRadii);

        for (const auto & sphere : spheres) {
            HitGroupSbtRecord record = {.data = {
                    GeometryType::SPHERE, sphere.materialType
            }};

            //设置球体中心和半径指针（所有记录共享同一指针，因为使用primitiveIndex访问）
            record.data.sphere.centers = reinterpret_cast<float3 *>(dev_sphereCenters);
            record.data.sphere.radii = reinterpret_cast<float *>(dev_sphereRadii);

            switch (sphere.materialType) {
                case MaterialType::ROUGH: //sphere - rough
                    optixCheckError(optixSbtRecordPackHeader(
                            programGroups[2], &record));
                    record.data.rough.albedo = roughs[sphere.materialIndex].albedo;
                    break;
                case MaterialType::METAL: //sphere - metal
                    optixCheckError(optixSbtRecordPackHeader(
                            programGroups[3], &record));
                    record.data.metal.albedo = metals[sphere.materialIndex].albedo;
                    record.data.metal.fuzz = metals[sphere.materialIndex].fuzz;
                    break;
                default:;
            }
            hitGroupSbtRecords.push_back(record);
        }

        //hit - triangle
        SDL_Log("Creating closesthit - triangle record...");

        // 为所有三角形的顶点法线分配设备内存
        size_t triangleCount = triangles.size();
        size_t vertexNormalCount = triangleCount * 3;
        std::vector<float3> vertexNormals(vertexNormalCount);
        for (size_t i = 0; i < triangleCount; i++) {
            vertexNormals[i * 3 + 0] = triangles[i].normals[0];
            vertexNormals[i * 3 + 1] = triangles[i].normals[1];
            vertexNormals[i * 3 + 2] = triangles[i].normals[2];
        }

        CUdeviceptr dev_vertexNormals = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_vertexNormals), vertexNormalCount * sizeof(float3)));
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_vertexNormals), vertexNormals.data(), vertexNormalCount * sizeof(float3), cudaMemcpyHostToDevice));
        ptrs.push_back(dev_vertexNormals);

        for (const auto & triangle : triangles) {
            HitGroupSbtRecord record = {.data = {
                    GeometryType::TRIANGLE, triangle.materialType
            }};

            //设置顶点法线指针（所有记录共享同一指针）
            record.data.triangles.vertexNormals = reinterpret_cast<float3 *>(dev_vertexNormals);

            switch (triangle.materialType) {
                case MaterialType::ROUGH: //triangle - rough
                    optixCheckError(optixSbtRecordPackHeader(
                            programGroups[4], &record));
                    record.data.rough.albedo = roughs[triangle.materialIndex].albedo;
                    break;
                case MaterialType::METAL: //triangle - metal
                    optixCheckError(optixSbtRecordPackHeader(
                            programGroups[5], &record));
                    record.data.metal.albedo = metals[triangle.materialIndex].albedo;
                    record.data.metal.fuzz = metals[triangle.materialIndex].fuzz;
                    break;
                default:;
            }
            hitGroupSbtRecords.push_back(record);
        }

        //hit - particle
        //不同于三角形，此处为粒子对象中一组三角形创建一个记录，即每个粒子一条记录
        SDL_Log("Creating closesthit - particle record...");
        for (const auto & particle : particles) {
            //为当前粒子的顶点法线分配设备内存
            triangleCount = particle.triangles.size();
            vertexNormalCount = triangleCount * 3;
            vertexNormals.clear();
            vertexNormals.resize(vertexNormalCount);

            //由于item.triangles中已经包含了交换后的数据，直接使用即可
            for (size_t j = 0; j < triangleCount; j++) {
                vertexNormals[j * 3 + 0] = particle.triangles[j].normals[0];
                vertexNormals[j * 3 + 1] = particle.triangles[j].normals[1];
                vertexNormals[j * 3 + 2] = particle.triangles[j].normals[2];
            }
            dev_vertexNormals = 0;
            cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_vertexNormals), vertexNormalCount * sizeof(float3)));
            cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_vertexNormals), vertexNormals.data(), vertexNormalCount * sizeof(float3), cudaMemcpyHostToDevice));
            ptrs.push_back(dev_vertexNormals); //保存指针以便后续释放。由于添加到数组中，无需单独释放此内存

            HitGroupSbtRecord record = {.data = {
                    //GeometryType用于着色，保持三角形不变
                    GeometryType::TRIANGLE, particle.materialType
            }};
            // 设置顶点法线指针
            record.data.triangles.vertexNormals = reinterpret_cast<float3 *>(dev_vertexNormals);

            switch (particle.materialType) {
                case MaterialType::ROUGH: //triangle - rough
                    optixCheckError(optixSbtRecordPackHeader(
                            programGroups[4], &record));
                    record.data.rough.albedo = roughs[particle.materialIndex].albedo;
                    break;
                case MaterialType::METAL: //triangle - metal
                    optixCheckError(optixSbtRecordPackHeader(
                            programGroups[5], &record));
                    record.data.metal.albedo = metals[particle.materialIndex].albedo;
                    record.data.metal.fuzz = metals[particle.materialIndex].fuzz;
                    break;
                default:;
            }
            hitGroupSbtRecords.push_back(record);
        }

        //将hit记录一次性拷贝到显存
        const size_t hitSbtRecordCount = hitGroupSbtRecords.size();
        CUdeviceptr dev_hitSbtRecords = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_hitSbtRecords), hitSbtRecordCount * sizeof(HitGroupSbtRecord)));
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_hitSbtRecords), hitGroupSbtRecords.data(), hitSbtRecordCount * sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));
        ptrs.push_back(dev_hitSbtRecords);

        SDL_Log("Shader binding table created.");
        const OptixShaderBindingTable sbt = {
                .raygenRecord = dev_raygenRecord,
                .missRecordBase = dev_missRecord,
                .missRecordStrideInBytes = sizeof(MissSbtRecord),
                .missRecordCount = 1,
                .hitgroupRecordBase = dev_hitSbtRecords,
                .hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord),
                .hitgroupRecordCount = static_cast<unsigned int>(hitSbtRecordCount)
        };
        return {sbt, ptrs};
    }
    void freeShaderBindingTable(std::pair<OptixShaderBindingTable, std::vector<CUdeviceptr>> & sbt) {
        SDL_Log("Freeing device memory for Shader binding table...");
        for (auto & item: sbt.second) {
            cudaCheckError(cudaFree(reinterpret_cast<void*>(item)));
        }
        sbt = {};
    }
}