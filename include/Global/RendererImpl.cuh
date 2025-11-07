#ifndef RENDEREROPTIX_RENDERERIMPL_CUH
#define RENDEREROPTIX_RENDERERIMPL_CUH

#include <Global/HostFunctions.cuh>
#include <Global/Shader.cuh>

namespace project {
    //几何体类型
    typedef struct Sphere {
        MaterialType materialType;
        size_t materialIndex;

        float3 center;
        float radius;
    } Sphere;

    typedef struct Triangle {
        MaterialType materialType;
        size_t materialIndex;

        float3 vertices[3];
        float3 normals[3];
    } Triangle;

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
    } GeometryData;

    //材质输入信息
    typedef struct MaterialData {
        std::vector<Rough> roughs;
        std::vector<Metal> metals;
    } MaterialData;

    //初始化optix上下文
    OptixDeviceContext createContext(bool isDebugMode = true);
    void destroyContext(OptixDeviceContext & context);

    //构建GAS
    std::pair<OptixTraversableHandle, CUdeviceptr> buildGASForSpheres(OptixDeviceContext & context, const std::vector<Sphere> & spheres);
    std::pair<OptixTraversableHandle, CUdeviceptr> buildGASForTriangles(OptixDeviceContext & context, const std::vector<Triangle> & triangles);

    //释放加速结构内存
    void cleanupAccelerationStructure(std::pair<OptixTraversableHandle, CUdeviceptr> & data);
    void cleanupAccelerationStructure(std::vector<std::pair<OptixTraversableHandle, CUdeviceptr>> & data);

    //构建实例列表，依次将传入的GAS句柄赋值给实例
    //此函数返回实例指针，指针指向设备内存中的实例对象数组开头
    OptixInstance * createInstances(const std::vector<std::pair<OptixTraversableHandle, size_t>> & data);
    void freeInstances(OptixInstance * & dev_instances);

    //构建IAS
    std::tuple<OptixTraversableHandle, CUdeviceptr, OptixAccelBufferSizes> buildIAS(
            OptixDeviceContext & context, const OptixInstance * dev_instances, size_t instanceCount);
    //使用实例数组更新IAS。若需要重建，则释放原有IAS后重新调用buildIAS
    void updateIAS(
            OptixDeviceContext & context, std::tuple<OptixTraversableHandle, CUdeviceptr, OptixAccelBufferSizes> & ias,
            const OptixInstance * dev_instances, size_t instanceCount);

    //创建模块，返回通过ptx创建的模块并获取内置球体求交模块和三角形求交模块
    std::array<OptixModule, 3> createModules(
            OptixDeviceContext & context, const OptixPipelineCompileOptions & pipelineCompileOptions, bool isDebugMode);
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
