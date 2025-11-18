#ifndef RENDEREROPTIX_VTKREADER_CUH
#define RENDEREROPTIX_VTKREADER_CUH

#include <Global/RendererImpl.cuh>

#define VTK_READER_IO_BUFFER_SIZE_KB (1024 * 1)
#define VTK_READER_ERROR_EXIT_CODE (-1)

namespace project {
    /*
     * VTK缓存文件管理
     * 每个VTK文件对应一个缓存文件，缓存文件包含每个粒子的ID（用于索引材质）和整个文件的顶点数组、法线数组
     * 顶点数组用于构建GAS，法线数组用于传入着色器中进行着色
     *
     * 缓存文件结构：
     * particleXX.cache：
     * [粒子总数 size_t]
     * 粒子#0:
     *   [ID size_t] [速度 float3] [顶点数 size_t] [顶点数组 float3*N] [法线数组 float3*N]
     * 粒子#1:
     *   [ID size_t] [速度 float3] [顶点数 size_t] [顶点数组 float3*N] [法线数组 float3*N]
     *
     * metadata.cache：[所有VTK文件中的最大Cell数 size_t]
     */
    class VTKReader {
    public:
        //读取series文件，返回文件信息：将VTK文件相对路径组合和seriesFilePath组合，将每个文件出现的时间转换为持续时间
        static std::tuple<std::vector<std::string>, std::vector<float>, size_t> readSeriesFile(
                const std::string & seriesFilePath, const std::string & seriesFileName);

        //读取series文件及其引用的所有VTK文件数据，将数据写入缓存文件后退出程序，每个VTK文件对应一个缓存文件
        static void writeVTKDataCache(
                const std::string & seriesFilePath, const std::string & seriesFileName,
                const std::string & cacheFilePath);

        //读取单个缓存文件信息，由子线程执行
        //此函数将粒子顶点和法线数据拷贝至设备内存，需要在渲染结束后释放
        static std::vector<RendererParticle> readVTKDataCache(const std::string & cacheFilePathName);
        static void freeVTKData(std::vector<RendererParticle> & particles);

        //读取所有VTK文件中单个文件最多的Cell数量
        static size_t readMaxCellCountAllVTKFile(const std::string & cacheFilePath);
    };
}

#endif //RENDEREROPTIX_VTKREADER_CUH
