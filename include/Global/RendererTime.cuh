#ifndef RENDEREROPTIX_RENDERERTIME_CUH
#define RENDEREROPTIX_RENDERERTIME_CUH

#include <Util/ColorRamp.cuh>
#include <GraphicsAPI/SDL_GraphicsWindow.cuh>
#include <Util/VTKTimeReader.cuh>

/*
 * 非网格输入数据：所有粒子几何只构建一次，每帧根据VTK文件中每个粒子的速度和旋转四元数更新IAS
 */
namespace project {
    typedef struct RendererTimeData {
        //额外几何体
        std::vector<RendererSphere> addSpheres;
        std::vector<RendererTriangle> addTriangles;
        size_t addGeoCount;

        //VTK粒子
        std::vector<std::vector<RendererTimeParticleReference>> vtkParticleAllFiles;
        std::vector<float> durations;
        size_t fileCount;

        //VTK文件路径
        std::string seriesFilePath, seriesFileName;
        std::string stlFilePath;

        //所有材质
        RendererMaterial materialAllFiles;

        /*
         * Time数据只有一份GAS，但是需要为每个文件的初始位置构建一个IAS
         * 则仍然需要将每个文件的AS分别存储
         * 每个文件的RendererAS中GAS数组均相同（拷贝GAS指针），IAS各自独立
         * IAS相对于GAS，占用空间极小，不会造成显著显存占用
         */
        std::vector<RendererAS> asAllFiles;
        std::vector<GAS> gasAllFiles;

        //OptiX变量
        OptixDeviceContext context;
        std::array<OptixModule, 3> modules;
        std::array<OptixProgramGroup, 6> programGroups;
        OptixPipeline pipeline;
        OptixDenoiser denoiser;

        //raygen和miss记录
        std::pair<CUdeviceptr, CUdeviceptr> raygenMissPtr;
        //着色器绑定表
        std::vector<SBT> sbtAllFiles;

        //额外几何体实例更新函数
        UpdateAddInstancesFunc func;
    } RendererTimeData;

    class RendererTime {
    public:
        //初始化渲染器
        static RendererTimeData commitRendererData(
                GeometryData & addGeoData, MaterialData & addMatData, const std::string & particleMaterials,
                const std::string & seriesFilePath, const std::string & seriesFileName,
                const std::string & stlFilePath, const std::string & optixCacheFilePath,
                bool isDebugMode, size_t maxFileReadThreadCount);

        //设置额外实例更新函数
        static void setAddGeoInsUpdateFunc(RendererTimeData & data, UpdateAddInstancesFunc func);

        //启动循环
        static void startRender(RendererTimeData & data, const RenderLoopData & loopData);

        //释放渲染器数据
        static void freeRendererData(RendererTimeData & data);
    };
}

#endif //RENDEREROPTIX_RENDERERTIME_CUH
