#include <Util/VTKMeshReader.cuh>
using namespace project;

//C++
#include <filesystem>
namespace filesystem = std::filesystem; //using用于类型别名，不能用于命名空间别名

//JSON
#include <JSON/json.hpp>
using json = nlohmann::json;

#define checkStreamOpen(io, path)\
    do {                         \
        if (!io.is_open()) {     \
            SDL_Log("Failed to open file: %s!", (path).c_str());\
            exit(VTK_READER_ERROR_EXIT_CODE);\
        }                        \
    } while (false)

/*
 * 在Ubuntu上，VTK头文件使用nvcc编译时可能会报错，原因为nvcc不支持VTK所使用的C++高级特性
 * 因此需要将VTK源代码使用GCC进行编译
 * 由于此文件（VTKReader.cu）由于头文件链式包含，已经包含了OptiX头文件，而OptiX头文件必须使用nvcc进行编译
 * 则需要将使用VTK函数的部分单独放到一个.cpp文件中
 */
namespace vtk_reader {
    //读取单个VTK文件，返回[id数组，速度数组，顶点二维数组，法线二维数组]
    //顶点、法线二维数组中每个一维数组存储一个粒子的所有顶点和法线
    std::tuple<
            std::vector<size_t>, std::vector<float3>,
            std::vector<std::vector<float3>>, std::vector<std::vector<float3>>
    > readVTKMeshFile(const std::string & filePath, std::atomic<size_t> & maxCellCountSingleFile);
}

//所有写入文件的大小均使用固定长度的uint64_t
namespace {
    //子线程函数：读取VTK文件，将其转换为渲染器粒子数组，并写入缓存文件
    void writeVTKFileCache(
            size_t totalFileCount, const std::string & vtkFilePathName,
            const std::string & cacheFilePathName, std::atomic<size_t> & processedFileCount,
            std::atomic<size_t> & maxCellCountSingleFile)
    {
        //读取VTK粒子数组
        const auto [ids, velocities,
                    verticesThisFile, normalsThisFile]
                    = vtk_reader::readVTKMeshFile(vtkFilePathName, maxCellCountSingleFile);

        //打开数据文件
        std::ofstream out(cacheFilePathName, std::ios::out | std::ios::binary);
        checkStreamOpen(out, cacheFilePathName);
        auto * outBuffer = new char [1024 * VTK_READER_IO_BUFFER_SIZE_KB];
        out.rdbuf()->pubsetbuf(outBuffer, sizeof(outBuffer));

        //写入粒子数量
        const uint64_t particleCount = ids.size();
        out.write(reinterpret_cast<const char *>(&particleCount), sizeof(uint64_t));

        for (size_t i = 0; i < particleCount; i++) {
            //写入粒子基础信息
            const uint64_t & id = ids[i];
            out.write(reinterpret_cast<const char *>(&id), sizeof(uint64_t));
            const float3 & vel = velocities[i];
            out.write(reinterpret_cast<const char *>(&vel), sizeof(float3));

            //写入此粒子的顶点/法线数量
            const uint64_t vertexCountThisCell = verticesThisFile[i].size();
            out.write(reinterpret_cast<const char *>(&vertexCountThisCell), sizeof(uint64_t));

            //写入顶点数组和法线数组
            out.write(reinterpret_cast<const char *>(verticesThisFile[i].data()), static_cast<std::streamsize>(vertexCountThisCell * sizeof(float3)));
            out.write(reinterpret_cast<const char *>(normalsThisFile[i].data()), static_cast<std::streamsize>(vertexCountThisCell * sizeof(float3)));
        }

        out.close();
        delete[] outBuffer;

        //统计已处理文件个数：原子操作计数器
        processedFileCount++;
        SDL_Log("[%zd/%zd] (%.2f%%) Created cache for VTK file %s.",
                processedFileCount.load(std::memory_order_relaxed), totalFileCount,
                static_cast<float>(processedFileCount) / static_cast<float>(totalFileCount) * 100.0f, cacheFilePathName.c_str());
    }
}

namespace project {
    std::tuple<std::vector<std::string>, std::vector<float>, size_t> VTKMeshReader::readSeriesFile(
            const std::string & seriesFilePath, const std::string & seriesFileName)
    {
        const auto filePath = seriesFilePath + seriesFileName;
        SDL_Log("Reading series file: %s...", filePath.c_str());

        //打开JSON文件
        std::ifstream file(filePath);
        checkStreamOpen(file, filePath);

        //读取数据到JSON对象
        json data;
        try {
            //使用 parse() 函数将文件流解析成一个 json 对象
            data = json::parse(file);
        } catch (json::parse_error & e) {
            SDL_Log("JSON parsing error: %s", e.what());
            exit(VTK_READER_ERROR_EXIT_CODE);
        }

        //从JSON对象中提取数据
        const std::string version = data["file-series-version"];
        SDL_Log("Series file version: %s", version.c_str());
        if (!(data.contains("files") && data["files"].is_array())) {
            SDL_Log("Failed to parse files array in series file!");
            exit(VTK_READER_ERROR_EXIT_CODE);
        }

        std::vector<std::string> files;
        std::vector<float> times;
        for (const auto & item : data["files"]) {
            const std::string name = item["name"]; files.push_back(seriesFilePath + name); //合并series文件路径
            const float time = item["time"]; times.push_back(time);
        }

        //打印前5条记录
        const size_t entryCount = files.size();
        const size_t printEntryCount = std::min<size_t>(entryCount, 5);
        SDL_Log("First %zd entries:", printEntryCount);
        for (size_t i = 0; i < printEntryCount; i++) {
            SDL_Log("Time: %f --> VTK file: %s", times[i], files[i].c_str());
        }

        //转换时间信息
        //time的输入为每个文件出现的时间，转换为每个文件持续的时间：后一个文件的出现时间减去当前文件出现时间
        std::vector<float> fileDurations(entryCount);
        if (entryCount == 1) {
            fileDurations[0] = 1000.0f;
        } else {
            for (size_t i = 0; i < entryCount - 1; i++) {
                fileDurations[i] = times[i + 1] - times[i];
            }
            //最后一个文件的出现时间使用倒数第二个文件的时间
            fileDurations[entryCount - 1] = fileDurations[entryCount - 2];
        }
        file.close();

        SDL_Log("Series file parse completed, found %zd entries.", entryCount);
        return {files, fileDurations, entryCount};
    }

    void VTKMeshReader::writeVTKDataCache(
            const std::string & seriesFilePath, const std::string & seriesFileName,
            const std::string & cacheFilePath, size_t maxCacheLoadThreadCount)
    {
        SDL_Log("Generating VTK data cache...");

        //读取series文件
        const auto [vtkFiles, durations, fileCount] = readSeriesFile(seriesFilePath, seriesFileName);

        //确保缓存文件目录存在
        filesystem::path path(cacheFilePath);
        filesystem::create_directories(path);

        //若已存在缓存文件，则删除旧缓存
        bool isPrintedWarning = false;
        for (const auto & entry : filesystem::directory_iterator(path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".cache") {
                if (!isPrintedWarning) {
                    SDL_Log("Existing cache files found in cache directory: %s. They will be removed.", cacheFilePath.c_str());
                    isPrintedWarning = true;
                }
                filesystem::remove(entry.path());
            }
        }

        //启动多个线程处理数据，每个文件一个线程
        SDL_Log("Starting processing VTK files...");
        std::atomic<size_t> processedFileCount(0), maxCellCountSingleFile(0);

        //启动线程数不超过设定的最大值
        const size_t threadCount = std::min<size_t>(std::thread::hardware_concurrency(), maxCacheLoadThreadCount);
        SDL_Log("Writing VTK cache using %zd concurrent threads...", threadCount);

        std::deque<std::thread> workers;
        for (size_t i = 0; i < fileCount; i++) {
            workers.emplace_back(
                    writeVTKFileCache, fileCount,
                    vtkFiles[i], cacheFilePath + "particle" + std::to_string(i) + ".cache",
                    std::ref(processedFileCount), std::ref(maxCellCountSingleFile));
            if (workers.size() == threadCount) {
                workers.front().join();
                workers.pop_front();
            }
        }
        while (!workers.empty()) {
            workers.front().join();
            workers.pop_front();
        }

        //将所有文件最大Cell数量耗时较长写入独立文件用于快速构造材质数组，同时使得材质构造较为灵活
        const auto maxCellCount = static_cast<uint64_t>(maxCellCountSingleFile.load(std::memory_order_relaxed));
        SDL_Log("Max cell count in a single VTK file: %zd.", maxCellCount);
        std::ofstream metaData(cacheFilePath + "metadata.cache", std::ios::out);
        if (!metaData.is_open()) {
            SDL_Log("Failed to open file: %s!", (cacheFilePath + "metadata.cache").c_str());
            exit(VTK_READER_ERROR_EXIT_CODE);
        }
        metaData << maxCellCount;
        metaData.close();

        SDL_Log("VTK data cache files are written to cache directory: %s.", cacheFilePath.c_str());
    }

    std::vector<RendererMeshParticle> VTKMeshReader::readVTKDataCache(const std::string & cacheFilePathName) {
        //打开缓存文件
        std::ifstream in(cacheFilePathName, std::ios::in | std::ios::binary);
        checkStreamOpen(in, cacheFilePathName);
        auto * inBuffer = new char [1024 * VTK_READER_IO_BUFFER_SIZE_KB];
        in.rdbuf()->pubsetbuf(inBuffer, sizeof(inBuffer));

        //读取粒子数量
        uint64_t particleCount;
        in.read(reinterpret_cast<char *>(&particleCount), sizeof(uint64_t));

        std::vector<RendererMeshParticle> particles; particles.reserve(particleCount);
        std::vector<float3> verticesThisCell, normalsThisCell;

        //逐个读取每个粒子信息
        for (size_t i = 0; i < particleCount; i++) {
            //ID
            uint64_t id;
            in.read(reinterpret_cast<char *>(&id), sizeof(uint64_t));

            //速度
            float3 velocity;
            in.read(reinterpret_cast<char *>(&velocity), sizeof(float3));

            //顶点/法线数量
            uint64_t vertexCount;
            in.read(reinterpret_cast<char *>(&vertexCount), sizeof(uint64_t));

            //顶点和法线数组（需要resize以接收来自流的数据，无需清理原有内容）
            verticesThisCell.resize(vertexCount); normalsThisCell.resize(vertexCount);
            in.read(reinterpret_cast<char *>(verticesThisCell.data()), static_cast<std::streamsize>(vertexCount * sizeof(float3)));
            in.read(reinterpret_cast<char *>(normalsThisCell.data()), static_cast<std::streamsize>(vertexCount * sizeof(float3)));

            //将顶点和法线数组拷贝至设备内存
            float3 * dev_vertices;
            float3 * dev_normals;
            cudaCheckError(cudaMalloc(&dev_vertices, vertexCount * sizeof(float3)));
            cudaCheckError(cudaMemcpy(dev_vertices, verticesThisCell.data(), vertexCount * sizeof(float3), cudaMemcpyHostToDevice));
            cudaCheckError(cudaMalloc(&dev_normals, vertexCount * sizeof(float3)));
            cudaCheckError(cudaMemcpy(dev_normals, normalsThisCell.data(), vertexCount * sizeof(float3), cudaMemcpyHostToDevice));

            particles.push_back({
                .id = static_cast<size_t>(id),
                .velocity = velocity,
                .dev_vertices = dev_vertices,
                .dev_normals = dev_normals,
                .triangleCount = vertexCount / 3
            });
        }

        in.close();
        delete[] inBuffer;
        return particles;
    }

    void VTKMeshReader::freeVTKData(std::vector<RendererMeshParticle> & particles) {
        for (const auto & particle: particles) {
            //cudaCheckError(cudaFree(particle.dev_vertices));
            cudaCheckError(cudaFree(particle.dev_normals));
        }
        particles = {};
    }

    size_t VTKMeshReader::readMaxCellCountAllVTKFile(const std::string & cacheFilePath) {
        std::ifstream metaData(cacheFilePath + "metadata.cache", std::ios::in);
        checkStreamOpen(metaData, cacheFilePath + "metadata.cache");
        uint64_t maxCount;
        metaData >> maxCount;
        metaData.close();

        SDL_Log("Max cell count in a single VTK file: %zd.", maxCount);
        return maxCount;
    }
}