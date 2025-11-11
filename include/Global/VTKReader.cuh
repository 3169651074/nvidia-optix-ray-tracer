#ifndef RENDEREROPTIX_VTKREADER_CUH
#define RENDEREROPTIX_VTKREADER_CUH

#include <Global/RendererImpl.cuh>

#define VTK_READER_IO_BUFFER_SIZE_KB (1024 * 1)
#define VTK_READER_ERROR_EXIT_CODE (-1)

namespace project {
    class VTKReader {
    public:
        //读取series文件，返回文件信息：将VTK文件相对路径组合和seriesFilePath组合，将每个文件出现的时间转换为持续时间
        static std::tuple<std::vector<std::string>, std::vector<float>, size_t> readSeriesFile(
                const std::string & seriesFilePath, const std::string & seriesFileName);

        //读取series文件及其引用的所有VTK文件数据，将数据写入缓存文件后退出程序，每个VTK文件对应一个缓存文件
        static void writeVTKDataCache(
                const std::string & seriesFilePath, const std::string & seriesFileName,
                const std::string & cacheFilePath, size_t materialOffset);

        //读取单个缓存文件信息，由子线程执行
        static std::vector<Particle> readVTKDataCache(const std::string & cacheFilePathName);

        //读取所有VTK文件中单个文件最多的Cell数量
        static size_t maxCellCountSingleVTKFile(const std::string & cacheFilePath);
    };
}

#endif //RENDEREROPTIX_VTKREADER_CUH
