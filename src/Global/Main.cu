#include <Global/Renderer.cuh>
using namespace project;

namespace {
    constexpr const char * seriesFilePath = "../files/";
    constexpr const char * seriesFileName = "particle_mesh-middle.vtk.series";
    constexpr const char * cacheFilePath = "../cache/";

    void updateInstancesTransforms(OptixInstance * pin_instances, unsigned long long frameCount) {
        float sphereTransform[12];
        MathHelper::constructTransformMatrix(
                {0.0f, 0.0f, -1000.5f},
                {0.0f, 0.0f, 0.0f},
                {1.0f, 1.0f, 1.0f}, sphereTransform);
        memcpy(pin_instances[0].transform, sphereTransform, 12 * sizeof(float));
    }
}

#undef main
int main(int argc, char * argv[]) {
    //额外几何体
    const std::vector<Sphere> spheres = {
            {float3{0.0f, 0.0f, 0.0f}, 1000.0f},
    };
    const std::vector<MaterialIndex> sphereMaterialIndices = {
            {MaterialType::ROUGH, 3}
    };
    GeometryData geoData = {
            .spheres = {spheres},
            .sphereMaterialIndices = sphereMaterialIndices
    };

    //额外材质
    const std::vector<float3> roughs = {
            {.65, .05, .05},
            {.73, .73, .73},
            {.12, .45, .15},
            {.70, .60, .50},
    };
    const std::vector<std::pair<float3, float>> metals = {
            {{0.8f, 0.85f, 0.88f}, 0.0f},
    };
    MaterialData matData = {
            .roughs = roughs,
            .metals = metals
    };

//#define GENERATE_CACHE_AND_EXIT
#ifdef GENERATE_CACHE_AND_EXIT
    writeCacheFilesAndExit(seriesFilePath, seriesFileName, cacheFilePath);
#endif

    auto data = commitRendererData(geoData, matData, seriesFilePath, seriesFileName, cacheFilePath);
    setAddGeoInsUpdateFunc(data, updateInstancesTransforms);
    const RenderLoopData loopData(
            SDL_GraphicsWindowAPIType::OPENGL, 1200, 800,
            "Test", 60,
            {5.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 1.0f},
            4);
    startRender(data, loopData);
    freeRendererData(data);

    return 0;
}
