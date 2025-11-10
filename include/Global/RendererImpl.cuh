#ifndef RENDEREROPTIX_RENDERERIMPL_CUH
#define RENDEREROPTIX_RENDERERIMPL_CUH

#include <Global/HostFunctions.cuh>
#include <Global/Shader.cuh>

namespace project {
    //几何体类型，每个几何体构建一条SBT记录
    typedef struct Sphere {
        MaterialType materialType;
        size_t materialIndex;

        float3 center;
        float radius;
    } Sphere;

    //由一个三角形组成的几何体（单图元几何体）
    typedef struct Triangle {
        MaterialType materialType;
        size_t materialIndex;

        std::array<float3, 3> vertices;
        std::array<float3, 3> normals;
    } Triangle;

    //由一组三角形组成的粒子（多图元几何体）
    typedef struct Particle {
        MaterialType materialType;
        size_t materialIndex;

        std::vector<Triangle> triangles;
    } Particle;

    //材质类型
    typedef struct Rough {
        float3 albedo;
    } Rough;

    typedef struct Metal {
        float3 albedo;
        float fuzz;
    } Metal;

    //几何体输入信息
    typedef struct GeometryData {
        std::vector<Sphere> spheres;
        std::vector<Triangle> triangles;
        std::vector<Particle> particles;
    } GeometryData;

    //材质输入信息
    typedef struct MaterialData {
        std::vector<Rough> roughs;
        std::vector<Metal> metals;
    } MaterialData;

    typedef std::pair<OptixTraversableHandle, CUdeviceptr> GAS;
    typedef std::tuple<OptixTraversableHandle, CUdeviceptr, OptixAccelBufferSizes> IAS;

    //初始化optix上下文
    OptixDeviceContext createContext(bool isDebugMode = true);
    void destroyContext(OptixDeviceContext & context);

    //构建GAS
    GAS buildGASForSpheres(OptixDeviceContext & context, const std::vector<Sphere> & spheres);
    GAS buildGASForTriangles(OptixDeviceContext & context, const std::vector<Triangle> & triangles);

    //释放加速结构内存
    void cleanupAccelerationStructure(GAS & data);
    void cleanupAccelerationStructure(std::vector<GAS> & data);
    void cleanupAccelerationStructure(IAS & data);
    void cleanupAccelerationStructure(std::vector<IAS> & data);

    //构建实例列表，依次将传入的GAS句柄赋值给实例
    //实例作为IAS的构建输入，和GAS构建输入一样需要在构建前临时拷贝至设备内存
    std::vector<OptixInstance> createInstances(const std::vector<GAS> & data);

    //构建IAS
    IAS buildIAS(OptixDeviceContext & context, const std::vector<OptixInstance> & instances);
    //使用实例数组更新IAS。若需要重建，则释放原有IAS后重新调用buildIAS
    void updateIAS(OptixDeviceContext & context, IAS & ias, const std::vector<OptixInstance> & instances);

    //创建模块，返回通过ptx创建的模块并获取内置球体求交模块和三角形求交模块
    std::array<OptixModule, 3> createModules(
            OptixDeviceContext & context, const OptixPipelineCompileOptions & pipelineCompileOptions, bool isDebugMode = true);
    void destroyModules(std::array<OptixModule, 3> & modules);

    //创建程序组，包含raygen(0), miss(1), chSphereRough(2), chSphereMetal(3), ...
    std::array<OptixProgramGroup, 6> createProgramGroups(OptixDeviceContext & context, std::array<OptixModule, 3> & modules);
    void destroyProgramGroups(std::array<OptixProgramGroup, 6> & programGroups);

    //连接管线
    OptixPipeline linkPipeline(
            OptixDeviceContext & context, const std::array<OptixProgramGroup, 6> & linkedProgramGroups,
            const OptixPipelineCompileOptions & pipelineCompileOptions);
    void unlinkPipeline(OptixPipeline & optixPipeline);

    //创建着色器绑定表
    std::pair<OptixShaderBindingTable, std::vector<CUdeviceptr>> createShaderBindingTable(
            const std::array<OptixProgramGroup, 6> & programGroups, const GeometryData & geoData, const MaterialData & matData);
    void freeShaderBindingTable(std::pair<OptixShaderBindingTable, std::vector<CUdeviceptr>> & sbt);
}

#endif //RENDEREROPTIX_RENDERERIMPL_CUH
