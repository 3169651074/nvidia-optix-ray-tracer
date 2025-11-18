#ifndef RENDEREROPTIX_RENDERER_CUH
#define RENDEREROPTIX_RENDERER_CUH

#include <Util/ColorRamp.cuh>
#include <Util/VTKReader.cuh>
#include <GraphicsAPI/SDL_GraphicsWindow.cuh>
#include <thread>
#include <atomic>

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
    //实例更新函数类型
    typedef void (*UpdateAddInstancesFunc)(OptixInstance * pin_instancesThisFile, unsigned long long frameCount);

    //以下定义的类型仅用于接收输入，不用于渲染
    //几何体
    typedef struct Sphere {
        float3 center;
        float radius;
    } Sphere;
    typedef struct Triangle {
        std::array<float3, 3> vertices;
        std::array<float3, 3> normals;
    } Triangle;

    //材质
    typedef struct MaterialIndex {
        MaterialType materialType;
        size_t materialIndex;
    } MaterialIndex;

    //输入信息。二维输入数组中，每个一维数组的所有几何体将会被视为一个实例
    typedef struct GeometryData {
        std::vector<std::vector<Sphere>> spheres;
        std::vector<MaterialIndex> sphereMaterialIndices;   //一个实例使用一个材质

        std::vector<std::vector<Triangle>> triangles;
        std::vector<MaterialIndex> triangleMaterialIndices;
    } GeometryData;
    typedef struct MaterialData {
        std::vector<float3> roughs;
        std::vector<std::pair<float3, float>> metals;
    } MaterialData;

    //渲染器数据，在外部函数之间传递
    //此结构体中的几何体均为指针，指向设备内存中的几何体数据
    typedef struct RendererData {
        //额外几何体
        std::vector<RendererSphere> addSpheres;
        std::vector<RendererTriangle> addTriangles;
        //额外几何体GAS
        std::vector<GAS> addGAS;

        //VTK粒子
        std::vector<std::vector<RendererParticle>> vtkParticleAllFiles;
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

        //raygen和miss记录
        std::pair<CUdeviceptr, CUdeviceptr> raygenMissPtr;
        //所有文件的着色器绑定表
        std::vector<SBT> sbtAllFiles;

        //额外几何体实例更新函数
        UpdateAddInstancesFunc func;
    } RendererData;

    //提交额外几何体和材质数据。此函数完成所有准备工作
    RendererData commitRendererData(
            GeometryData & addGeoData, MaterialData & addMatData,
            const std::string & seriesFilePath, const std::string & seriesFileName,
            const std::string & cacheFilePath);

    //生成VTK粒子缓存文件并退出
    void writeCacheFilesAndExit(
            const std::string & seriesFilePath, const std::string & seriesFileName,
            const std::string & cacheFilePath);

    //设置额外实例更新函数，额外实例按照输入顺序排在整个实例列表pin_instances的头部，函数将在每帧被调用
    void setAddGeoInsUpdateFunc(RendererData & data, UpdateAddInstancesFunc func);

    //启动循环
    typedef struct RenderLoopData {
        SDL_GraphicsWindowAPIType apiType;
        int windowWidth, windowHeight;
        const char * windowTitle;

        size_t targetFPS;
        float3 cameraCenter, cameraTarget, upDirection;

        size_t renderSpeedRatio;
        float3 particleOffset, particleScale;
        float mouseSensitivity, pitchLimitDegree;
        float cameraMoveSpeedStride;
        size_t initialSpeedNTimesStride;

        RenderLoopData(
                SDL_GraphicsWindowAPIType apiType, int windowWidth, int windowHeight, const char * windowTitle,
                size_t targetFPS, const float3 & cameraCenter, const float3 & cameraTarget, const float3 & upDirection,
                size_t renderSpeedRatio, const float3 & particleOffset = {}, const float3 & particleScale = {1.0f, 1.0f, 1.0f},
                float mouseSensitivity = 0.002f, float pitchLimitDegree = 85.0f, float cameraMoveSpeedStride = 0.002f, size_t initialSpeedNTimesStride = 10);
    } RenderLoopData;
    void startRender(RendererData & data, const RenderLoopData & loopData);

    //释放数据
    void freeRendererData(RendererData & data);
}

#endif //RENDEREROPTIX_RENDERER_CUH
