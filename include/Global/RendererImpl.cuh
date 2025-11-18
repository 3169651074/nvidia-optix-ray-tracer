#ifndef RENDEREROPTIX_RENDERERIMPL_CUH
#define RENDEREROPTIX_RENDERERIMPL_CUH

#include <Global/HostFunctions.cuh>
#include <Global/Shader.cuh>

namespace project {
    //缓存读写最大并发线程数
    constexpr size_t MAX_CACHE_LOAD_THREAD_COUNT = 8;

    //着色器绑定表数据类型
    template <typename T>
    struct SbtRecord {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };
    typedef SbtRecord<RayGenParams>   RayGenSbtRecord;
    typedef SbtRecord<MissParams>     MissSbtRecord;
    typedef SbtRecord<HitGroupParams> HitGroupSbtRecord;
    typedef std::pair<OptixShaderBindingTable, CUdeviceptr> SBT;

    //加速结构数据类型
    typedef std::pair<OptixTraversableHandle, CUdeviceptr> GAS;
    typedef std::tuple<OptixTraversableHandle, CUdeviceptr, OptixAccelBufferSizes> IAS;

    //以下定义的类型由输入类型转换，作为渲染参数传递
    //一个RenderSphere表示一个或一组球体，将会被视为一个实例：构建到同一个GAS，对应一个SBT记录
    typedef struct RendererSphere {
        MaterialType materialType;
        size_t materialIndex;

        float3 * dev_centers;
        float * dev_radii;
        size_t count;
    } RendererSphere;

    //每三个顶点/法线表示一个三角形
    typedef struct RendererTriangle {
        MaterialType materialType;
        size_t materialIndex;

        //dev_vertices/dev_normals的长度为3倍count
        float3 * dev_vertices;
        float3 * dev_normals;
        size_t count;
    } RendererTriangle;

    //一个VTK粒子，包含多个三角形
    typedef struct RendererParticle {
        size_t id;
        float3 velocity;

        //一个粒子内所有三角形连续存储（SOA），则不使用RenderTriangle数组（AOS）
        //同三角形，dev_vertices/dev_normals的长度为3倍count
        float3 * dev_vertices;
        float3 * dev_normals;
        size_t count;
    } RendererParticle;

    //所有材质。VTK粒子材质偏移量为额外Rough材质数量
    typedef struct RendererMaterial {
        std::vector<float3> roughs;
        std::vector<std::pair<float3, float>> metals;
    } RendererMaterial;

    //一个文件的所有加速结构变量
    typedef struct RendererAS {
        //每个文件的GAS为addGAS和所有粒子GAS的合并
        std::vector<GAS> gas;
        //每个文件IAS包含额外几何体和该文件所有VTK粒子
        IAS ias;

        //实例数组。每个文件一组，将额外几何体和VTK几何体组合到一个数组中
        OptixInstance * pin_instances;
        OptixInstance * dev_instances;
        size_t instanceCount;
    } RendererAS;

    //初始化optix上下文
    OptixDeviceContext createContext(const std::string & cacheFilePath, bool isDebugMode);
    void destroyContext(OptixDeviceContext & context);

    //构建GAS。每次为一个实例中的一个或多个几何体构建一个GAS
    GAS buildGASForSpheres(
            OptixDeviceContext & context, const RendererSphere & spheres, cudaStream_t stream = nullptr);
    GAS buildGASForTriangles(
            OptixDeviceContext & context, const RendererTriangle & triangles, cudaStream_t stream = nullptr);
    GAS buildGASForParticle(
            OptixDeviceContext & context, const RendererParticle & particle, cudaStream_t stream = nullptr);

    //构建IAS
    IAS buildIAS(
            OptixDeviceContext & context, const OptixInstance * dev_instances,
            size_t instanceCount, cudaStream_t stream = nullptr);
    //使用新的实例数组更新IAS。若需要重建，则释放原有IAS后重新调用buildIAS
    void updateIAS(
            OptixDeviceContext & context, IAS & ias, const OptixInstance * dev_instances,
            size_t instanceCount, cudaStream_t stream = nullptr);

    //释放加速结构
    void cleanupAccelerationStructure(GAS & data);
    void cleanupAccelerationStructure(std::vector<GAS> & data);
    void cleanupAccelerationStructure(IAS & data);
    void cleanupAccelerationStructure(std::vector<IAS> & data);

    //创建模块，返回通过ptx创建的模块(0) 并获取内置球体求交模块(1) 和三角形求交模块(2)
    std::array<OptixModule, 3> createModules(
            OptixDeviceContext & context,
            const OptixPipelineCompileOptions & pipelineCompileOptions,
            bool isDebugMode);
    void destroyModules(std::array<OptixModule, 3> & modules);

    //创建程序组，包含raygen(0), miss(1), chSphereRough(2), chSphereMetal(3), ...
    std::array<OptixProgramGroup, 6> createProgramGroups(
            OptixDeviceContext & context, std::array<OptixModule, 3> & modules);
    void destroyProgramGroups(std::array<OptixProgramGroup, 6> & programGroups);

    //连接管线
    OptixPipeline linkPipeline(
            OptixDeviceContext & context,
            const std::array<OptixProgramGroup, 6> & linkedProgramGroups,
            const OptixPipelineCompileOptions & pipelineCompileOptions);
    void unlinkPipeline(OptixPipeline & optixPipeline);

    //创建着色器绑定表
    //创建raygen和miss记录，返回记录的设备指针
    std::pair<CUdeviceptr, CUdeviceptr> createRaygenMissSBTRecord(
            OptixProgramGroup & raygenProgramGroup, OptixProgramGroup & missProgramGroup,
            const float3 & backgroundColor);

    //创建额外球体和三角形记录，每个RendererSphere/RendererTriangle一个记录
    //返回碰撞记录数组，在调用端合并后统一拷贝至设备内存
    std::vector<HitGroupSbtRecord> createAddSphereTriangleSBTRecord(
            OptixProgramGroup & sphereRoughProgramGroup, OptixProgramGroup & sphereMetalProgramGroup,
            OptixProgramGroup & triangleRoughProgramGroup, OptixProgramGroup & triangleMetalProgramGroup,
            const std::vector<RendererSphere> & spheres, const std::vector<RendererTriangle> & triangles,
            const RendererMaterial & globalMaterials);

    //创建一个文件所有VTK粒子的记录，每个RendererParticle一个记录
    std::vector<HitGroupSbtRecord> createVTKParticleSBTRecord(
            OptixProgramGroup & triangleRoughProgramGroup, OptixProgramGroup & triangleMetalProgramGroup,
            const std::vector<RendererParticle> & particles,
            const RendererMaterial & globalMaterials, size_t materialOffset);

    //释放一组SBT记录的设备内存
    void freeSBTRecords(std::vector<SBT> & sbtAllfiles);
}

#endif //RENDEREROPTIX_RENDERERIMPL_CUH
