#ifndef RENDEREROPTIX_VTKTIMEREADER_CUH
#define RENDEREROPTIX_VTKTIMEREADER_CUH

#include <Global/RendererImpl.cuh>

#define VTK_READER_IO_BUFFER_SIZE_KB (1024 * 1)
#define VTK_READER_ERROR_EXIT_CODE (-1)

namespace project {
    class VTKTimeReader {
    public:
        //读取series文件
        static std::tuple<std::vector<std::string>, std::vector<float>, size_t> readSeriesFile(
                const std::string & seriesFilePath, const std::string & seriesFileName);

        //读取目录下所有STL文件，并将数据拷贝至设备内存，按文件名的字典顺序排序
        static std::vector<RendererTimeParticleData> readSTLFiles(const std::string & stlFilePath);

        //子线程调用函数：读取VTK文件
        static std::tuple<
                std::vector<float3>, std::vector<size_t>, std::vector<float4>,
                std::vector<float3>, std::vector<size_t>
        > readVTKFile(const std::string & filePath, std::atomic<size_t> & maxPointCountSingleFile);
    };
}

#endif //RENDEREROPTIX_VTKTIMEREADER_CUH
