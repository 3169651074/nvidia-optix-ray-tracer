#include <Util/ProgramArgumentParser.cuh>
using namespace project;

static std::vector<std::array<float, 12>> instanceTransforms;
static void updateInstancesTransforms(
        OptixInstance * pin_instances, size_t instanceCount, unsigned long long frameCount)
{
    memcpy(pin_instances[0].transform, instanceTransforms[0].data(), 12 * sizeof(float));
}

#undef main
int main(int argc, char * argv[]) {
    //解析JSON参数
    auto [geoData, matData, loopData, transforms,
          seriesFilePath, seriesFileName, cacheFilePath, stlFilePath,
          particleMaterials, isMesh, isDebugMode, isWriteCache,
          cacheProcessThreadCount] = ProgramArgumentParser::parseProgramArguments();
    const size_t maxCacheLoadThreadCount = std::max<size_t>(1, cacheProcessThreadCount);
    instanceTransforms = std::move(transforms);

    if (isMesh) {
        //Mesh输入
        if (isWriteCache) {
            RendererMesh::writeCacheFilesAndExit(seriesFilePath, seriesFileName, cacheFilePath, maxCacheLoadThreadCount);
        }

        auto data = RendererMesh::commitRendererData(
                geoData, matData, particleMaterials,
                seriesFilePath, seriesFileName, cacheFilePath,
                isDebugMode, maxCacheLoadThreadCount);
        RendererMesh::setAddGeoInsUpdateFunc(data, &updateInstancesTransforms);
        RendererMesh::startRender(data, loopData);
        RendererMesh::freeRendererData(data);
    } else {
        //Time输入
        auto data = RendererTime::commitRendererData(
                geoData, matData, particleMaterials,
                seriesFilePath, seriesFileName,
                stlFilePath, cacheFilePath,
                isDebugMode, maxCacheLoadThreadCount);
        RendererTime::setAddGeoInsUpdateFunc(data, &updateInstancesTransforms);
        RendererTime::startRender(data, loopData);
        RendererTime::freeRendererData(data);
    }

    return 0;
}
