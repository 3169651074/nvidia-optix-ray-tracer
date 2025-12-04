#ifndef RENDEREROPTIX_RENDERERMESH_CUH
#define RENDEREROPTIX_RENDERERMESH_CUH

#include <Util/ColorRamp.cuh>
#include <Util/VTKMeshReader.cuh>

/*
 * 全局共用：额外几何体及其GAS；材质数据 = 额外材质数据 + VTK材质数据，全局使用一个材质数组
 * 由于页面锁定内存大小有限，则几何数据和材质数据不使用页面锁定内存，仅为实例数组分配页面锁定内存
 *
 * 几何体数据：输入时转换为SOA结构，并拷贝至显存，渲染结束后释放（加速结构构建和SBT记录均需要使用）
 * 材质数据：读取完额外几何体信息和VTK信息后生成材质数组，立即拷贝至显存，渲染结束后释放
 *
 * 实例数据：获取到额外几何体数据/VTK粒子数据后为每个文件分配固定大小的页面锁定内存和设备内存，渲染结束后释放
 */
namespace project {
    //渲染器数据，在外部函数之间传递
    //此结构体中的几何体均为指针，指向设备内存中的几何体数据
    typedef struct RendererMeshData {
        //额外几何体
        std::vector<RendererSphere> addSpheres;
        std::vector<RendererTriangle> addTriangles;
        //额外几何体GAS
        std::vector<GAS> addGAS;

        //VTK粒子
        std::vector<std::vector<RendererMeshParticle>> vtkParticleAllFiles;
        std::vector<float> durations;
        size_t fileCount;

        //VTK文件路径
        std::string seriesFilePath, seriesFileName;
        std::string cacheFilePath;

        //所有材质，材质数据直接存储
        RendererMaterial materialAllFiles;

        //加速结构
        std::vector<RendererAS> asAllFiles;

        //OptiX变量
        OptixDeviceContext context;
        std::array<OptixModule, 3> modules;
        std::array<OptixProgramGroup, 6> programGroups;
        OptixPipeline pipeline;
        OptixDenoiser denoiser;

        //raygen和miss记录
        std::pair<CUdeviceptr, CUdeviceptr> raygenMissPtr;
        //所有文件的着色器绑定表
        std::vector<SBT> sbtAllFiles;

        //额外几何体实例更新函数
        UpdateAddInstancesFunc func;
    } RendererMeshData;

    //提交额外几何体和材质数据。此函数完成所有准备工作
    class RendererMesh {
    public:
        //初始化渲染器
        static RendererMeshData commitRendererData(
                GeometryData & addGeoData, MaterialData & addMatData, const std::string particleMaterials,
                const std::string & seriesFilePath, const std::string & seriesFileName,
                const std::string & cacheFilePath, bool isDebugMode, size_t maxCacheLoadThreadCount);

        //生成VTK粒子缓存文件并退出
        static void writeCacheFilesAndExit(
                const std::string & seriesFilePath, const std::string & seriesFileName,
                const std::string & cacheFilePath, size_t maxCacheLoadThreadCount);

        //设置额外实例更新函数，额外实例按照输入顺序排在整个实例列表pin_instances的头部，函数将在每帧被调用
        static void setAddGeoInsUpdateFunc(RendererMeshData & data, UpdateAddInstancesFunc func);

        //启动循环
        static void startRender(RendererMeshData & data, const RenderLoopData & loopData);

        //释放渲染器数据
        static void freeRendererData(RendererMeshData & data);
    };
}

#endif //RENDEREROPTIX_RENDERERMESH_CUH
