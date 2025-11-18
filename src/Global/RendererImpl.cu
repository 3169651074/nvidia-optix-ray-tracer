#include <Global/RendererImpl.cuh>

namespace project {
    OptixDeviceContext createContext(const std::string & cacheFilePath, bool isDebug) {
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
        optixCheckError(optixDeviceContextSetCacheLocation(optixDeviceContext, cacheFilePath.c_str()));
        SDL_Log("Context created.");
        return optixDeviceContext;
    }
    void destroyContext(OptixDeviceContext & context) {
        SDL_Log("Destroying context...");
        optixCheckError(optixDeviceContextDestroy(context));
        context = nullptr;
    }

    //构建加速结构辅助函数
    GAS buildASImpl(
            OptixDeviceContext & context, const OptixAccelBuildOptions & buildOptions,
            const OptixBuildInput & buildInput, bool isCompress, cudaStream_t stream)
    {
        //计算加速结构内存并分配空间
        OptixAccelBufferSizes bufferSizes = {};
        optixCheckError(optixAccelComputeMemoryUsage(
                context, &buildOptions, &buildInput, 1, &bufferSizes));
        CUdeviceptr dev_tempBuffer = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_tempBuffer), bufferSizes.tempSizeInBytes));
        CUdeviceptr dev_output = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_output), bufferSizes.outputSizeInBytes));

        //计算压缩后大小
        CUdeviceptr dev_compressedSize = 0;
        OptixAccelEmitDesc emitProperty = {};
        if (isCompress) {
            cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_compressedSize), sizeof(unsigned long long)));
            emitProperty = {
                    .result = dev_compressedSize,
                    .type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
            };
        }

        //构建GAS
        OptixTraversableHandle handle = 0;
        optixCheckError(optixAccelBuild(
                context, stream, &buildOptions,
                &buildInput, 1,
                dev_tempBuffer, bufferSizes.tempSizeInBytes,
                dev_output, bufferSizes.outputSizeInBytes,
                &handle,
                isCompress ? &emitProperty : nullptr, isCompress ? 1 : 0));
        cudaCheckError(cudaStreamSynchronize(stream));

        //释放临时空间
        cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_tempBuffer)));

        //压缩GAS
        if (isCompress) {
            unsigned long long compressedSize = 0;
            cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(&compressedSize), reinterpret_cast<void *>(dev_compressedSize), sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            if (compressedSize < bufferSizes.outputSizeInBytes) {
                //重新分配空间
                CUdeviceptr dev_compressedGasOutput = 0;
                cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_compressedGasOutput), compressedSize));
                optixCheckError(optixAccelCompact(
                        context, stream, handle,
                        dev_compressedGasOutput, compressedSize,
                        &handle));
                cudaCheckError(cudaStreamSynchronize(stream));
                //释放原有空间
                cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_output)));
                dev_output = dev_compressedGasOutput;
            }
        }

        return {handle, dev_output};
    }

    GAS buildGASForSpheres(
            OptixDeviceContext & context, const RendererSphere & spheres, cudaStream_t stream)
    {
        //GAS为静态，无需更新，则设置为可压缩且高质量
        const OptixAccelBuildOptions buildOptions = {
                .buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
                .operation = OPTIX_BUILD_OPERATION_BUILD
        };

        //设置构建输入
        const unsigned int buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        const OptixBuildInput buildInput = {
                .type = OPTIX_BUILD_INPUT_TYPE_SPHERES,
                .sphereArray = {
                        .vertexBuffers = reinterpret_cast<const CUdeviceptr *>(&spheres.dev_centers),
                        .vertexStrideInBytes = sizeof(float3),
                        .numVertices = static_cast<unsigned int>(spheres.count),
                        .radiusBuffers = reinterpret_cast<const CUdeviceptr *>(&spheres.dev_radii),
                        .radiusStrideInBytes = sizeof(float),
                        .flags = buildInputFlags,
                        .numSbtRecords = 1,
                }
        };

        return buildASImpl(context, buildOptions, buildInput, true, stream);
    }
    GAS buildGASForTriangles(
            OptixDeviceContext & context, const RendererTriangle & triangles, cudaStream_t stream)
    {
        //构建选项和输入同球体GAS
        const OptixAccelBuildOptions buildOptions = {
                .buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
                .operation = OPTIX_BUILD_OPERATION_BUILD
        };

        const unsigned int buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        const OptixBuildInput buildInput = {
                .type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
                .triangleArray = {
                        .vertexBuffers = reinterpret_cast<const CUdeviceptr *>(&triangles.dev_vertices),
                        .numVertices = static_cast<unsigned int>(triangles.count),
                        .vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
                        .vertexStrideInBytes = sizeof(float3),
                        .flags = buildInputFlags,
                        .numSbtRecords = 1
                }
        };

        return buildASImpl(context, buildOptions, buildInput, true, stream);
    }
    GAS buildGASForParticle(
            OptixDeviceContext & context, const RendererParticle & particle, cudaStream_t stream)
    {
        const OptixAccelBuildOptions buildOptions = {
                .buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
                .operation = OPTIX_BUILD_OPERATION_BUILD
        };
        const unsigned int buildInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
        const OptixBuildInput buildInput = {
                .type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
                .triangleArray = {
                        .vertexBuffers = reinterpret_cast<const CUdeviceptr *>(&particle.dev_vertices),
                        .numVertices = static_cast<unsigned int>(particle.count * 3),
                        .vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3,
                        .vertexStrideInBytes = sizeof(float3),
                        .flags = buildInputFlags,
                        .numSbtRecords = 1
                }
        };

        return buildASImpl(context, buildOptions, buildInput, true, stream);
    }

    IAS buildIAS(
            OptixDeviceContext & context, const OptixInstance * dev_instances,
            size_t instanceCount, cudaStream_t stream)
    {
        //构建选项：允许更新且快速构建
        const OptixAccelBuildOptions buildOptions = {
                .buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_BUILD,
                .operation = OPTIX_BUILD_OPERATION_BUILD
        };
        const OptixBuildInput buildInput = {
                .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
                .instanceArray = {
                        .instances = reinterpret_cast<CUdeviceptr>(dev_instances),
                        .numInstances = static_cast<unsigned int>(instanceCount)
                }
        };

        //计算IAS空间
        OptixAccelBufferSizes bufferSizes;
        optixCheckError(optixAccelComputeMemoryUsage(
                context, &buildOptions, &buildInput, 1, &bufferSizes));
        CUdeviceptr dev_tempBuffer = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_tempBuffer), bufferSizes.tempSizeInBytes));
        CUdeviceptr dev_output = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_output), bufferSizes.outputSizeInBytes));

        //构建IAS，不压缩：支持压缩的加速结构不支持更新
        OptixTraversableHandle handle = 0;
        optixCheckError(optixAccelBuild(context, stream, &buildOptions, &buildInput, 1, dev_tempBuffer, bufferSizes.tempSizeInBytes,
                                        dev_output, bufferSizes.outputSizeInBytes, &handle, nullptr, 0));
        cudaCheckError(cudaStreamSynchronize(stream));

        //释放临时空间
        cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_tempBuffer)));
        return {handle, dev_output, bufferSizes};
    }
    void updateIAS(
            OptixDeviceContext & context, IAS & ias, const OptixInstance * dev_instances,
            size_t instanceCount, cudaStream_t stream)
    {
        const OptixAccelBuildOptions buildOptions = {
                .buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_BUILD, //flags需要保持和构建时一致
                .operation = OPTIX_BUILD_OPERATION_UPDATE   //更新
        };
        const OptixBuildInput buildInput = {
                .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
                .instanceArray = {
                        .instances = reinterpret_cast<CUdeviceptr>(dev_instances),
                        .numInstances = static_cast<unsigned int>(instanceCount)
                }
        };
        auto [handle, ptr, bufferSizes] = ias;

        //分配更新所需临时空间，更新操作在原有空间上进行，无需释放原有空间
        CUdeviceptr dev_tempBuffer = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_tempBuffer), bufferSizes.tempUpdateSizeInBytes));

        //更新
        optixCheckError(optixAccelBuild(
                context, stream, &buildOptions,
                &buildInput, 1,
                dev_tempBuffer, bufferSizes.tempUpdateSizeInBytes, //改为tempUpdateSize
                ptr, bufferSizes.outputSizeInBytes,
                &handle, nullptr, 0));
        cudaCheckError(cudaStreamSynchronize(stream));

        //释放临时空间
        cudaCheckError(cudaFree(reinterpret_cast<void *>(dev_tempBuffer)));
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

    std::array<OptixProgramGroup, 6> createProgramGroups(
            OptixDeviceContext & context, std::array<OptixModule, 3> & modules)
    {
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

    std::pair<CUdeviceptr, CUdeviceptr> createRaygenMissSBTRecord(
            OptixProgramGroup & raygenProgramGroup, OptixProgramGroup & missProgramGroup,
            const float3 & backgroundColor)
    {
        //raygen
        RayGenSbtRecord rayGenSbtRecord = {};
        optixCheckError(optixSbtRecordPackHeader(raygenProgramGroup, &rayGenSbtRecord));
        CUdeviceptr dev_raygenRecord = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_raygenRecord), sizeof(RayGenSbtRecord)));
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_raygenRecord), &rayGenSbtRecord, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

        //miss
        MissSbtRecord missSbtRecord = {};
        optixCheckError(optixSbtRecordPackHeader(missProgramGroup, &missSbtRecord));
        missSbtRecord.data.backgroundColor = backgroundColor;
        CUdeviceptr dev_missRecord = 0;
        cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&dev_missRecord), sizeof(MissSbtRecord)));
        cudaCheckError(cudaMemcpy(reinterpret_cast<void *>(dev_missRecord), &missSbtRecord, sizeof(MissSbtRecord), cudaMemcpyHostToDevice));

        return {dev_raygenRecord, dev_missRecord};
    }

    std::vector<HitGroupSbtRecord> createAddSphereTriangleSBTRecord(
            OptixProgramGroup & sphereRoughProgramGroup, OptixProgramGroup & sphereMetalProgramGroup,
            OptixProgramGroup & triangleRoughProgramGroup, OptixProgramGroup & triangleMetalProgramGroup,
            const std::vector<RendererSphere> & spheres, const std::vector<RendererTriangle> & triangles,
            const RendererMaterial & globalMaterials)
    {
        std::vector<HitGroupSbtRecord> records;
        records.reserve(spheres.size() + triangles.size());

        //sphere
        for (const auto & sphere : spheres) {
            HitGroupSbtRecord record = {};

            //设置球体中心和半径指针
            record.data.sphere.centers = sphere.dev_centers;
            record.data.sphere.radii = sphere.dev_radii;

            switch (sphere.materialType) {
                case MaterialType::ROUGH: //sphere - rough
                    optixCheckError(optixSbtRecordPackHeader(
                            sphereRoughProgramGroup, &record));
                    record.data.rough.albedo = globalMaterials.roughs[sphere.materialIndex];
                    break;
                case MaterialType::METAL: //sphere - metal
                    optixCheckError(optixSbtRecordPackHeader(
                            sphereMetalProgramGroup, &record));
                    record.data.metal.albedo = globalMaterials.metals[sphere.materialIndex].first;
                    record.data.metal.fuzz = globalMaterials.metals[sphere.materialIndex].second;
                    break;
                default:;
            }
            records.push_back(record);
        }

        //triangle
        for (const auto & triangle : triangles) {
            HitGroupSbtRecord record = {};

            //设置顶点法线指针
            record.data.triangles.vertexNormals = triangle.dev_normals;

            switch (triangle.materialType) {
                case MaterialType::ROUGH: //triangle - rough
                    optixCheckError(optixSbtRecordPackHeader(
                            triangleRoughProgramGroup, &record));
                    record.data.rough.albedo = globalMaterials.roughs[triangle.materialIndex];
                    break;
                case MaterialType::METAL: //triangle - metal
                    optixCheckError(optixSbtRecordPackHeader(
                            triangleMetalProgramGroup, &record));
                    record.data.metal.albedo = globalMaterials.metals[triangle.materialIndex].first;
                    record.data.metal.fuzz = globalMaterials.metals[triangle.materialIndex].second;
                    break;
                default:;
            }
            records.push_back(record);
        }
        return records;
    }

    std::vector<HitGroupSbtRecord> createVTKParticleSBTRecord(
            OptixProgramGroup & triangleRoughProgramGroup, OptixProgramGroup & triangleMetalProgramGroup,
            const std::vector<RendererParticle> & particles,
            const RendererMaterial & globalMaterials, size_t materialOffset)
    {
        std::vector<HitGroupSbtRecord> records;
        records.reserve(particles.size());

        for (const auto & particle : particles) {
            HitGroupSbtRecord record = {};

            //设置顶点法线指针
            record.data.triangles.vertexNormals = particle.dev_normals;

            //粗糙材质
            optixCheckError(optixSbtRecordPackHeader(
                    triangleRoughProgramGroup, &record));
            record.data.rough.albedo = globalMaterials.roughs[particle.id + materialOffset];
            records.push_back(record);
        }
        return records;
    }

    void freeSBTRecords(std::vector<SBT> & sbtAllfiles) {
        for (auto & sbtThisFile: sbtAllfiles) {
            cudaCheckError(cudaFree(reinterpret_cast<void *>(sbtThisFile.second)));
        }
        sbtAllfiles = {};
    }
}