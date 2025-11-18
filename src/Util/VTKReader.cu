#include <Util/VTKReader.cuh>
using namespace project;

//C++
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>
#include <filesystem>
namespace filesystem = std::filesystem; //using用于类型别名，不能用于命名空间别名

//JSON
#include <JSON/json.hpp>
using json = nlohmann::json;

//VTK
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkFieldData.h>
#include <vtkCell.h>
#include <vtkCellTypes.h>
#include <vtkPolyDataNormals.h>
#include <vtkPointData.h>

#define checkStreamOpen(io, path)\
    do {                     \
        if (!io.is_open()) { \
            SDL_Log("Failed to open file: %s!", (path).c_str());\
            exit(VTK_READER_ERROR_EXIT_CODE);\
        }                    \
    } while (false)

namespace {
    //检查VTK文件头
    void checkVTKFileHeader(const std::string & filePath) {
        std::ifstream vtkFile(filePath);
        checkStreamOpen(vtkFile, filePath);
        std::string line;
        getline(vtkFile, line);
        if (line.find("# vtk DataFile Version") == std::string::npos) {
            SDL_Log("Illegal vtk file header in file %s: %s!", filePath.c_str(), line.c_str());
            exit(VTK_READER_ERROR_EXIT_CODE);
        }
        vtkFile.close();
    }

    //读取单个VTK文件，返回[id数组，速度数组，顶点二维数组，法线二维数组]
    //顶点、法线二维数组中每个一维数组存储一个粒子的所有顶点和法线
    std::tuple<
            std::vector<size_t>, std::vector<float3>,
            std::vector<std::vector<float3>>, std::vector<std::vector<float3>>
    > readVTKFile(const std::string & filePath, std::atomic<size_t> & maxCellCountSingleFile) {
        std::vector<size_t> ids;
        std::vector<float3> velocities;
        std::vector<std::vector<float3>> verticesThisFile, normalsThisFile;

        //获取vtkPolyData指针
        vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
        reader->SetFileName(filePath.c_str());
        reader->Update();
        vtkPolyData * polyData = reader->GetOutput();
        if (polyData == nullptr || polyData->GetNumberOfPoints() == 0) {
            SDL_Log("Failed to get poly data pointer or there is no points in file!");
            exit(VTK_READER_ERROR_EXIT_CODE);
        }
        const vtkIdType cellCount = polyData->GetNumberOfCells();
        maxCellCountSingleFile = std::max<size_t>(maxCellCountSingleFile, cellCount);

        //读取几何数据
        vtkCellData * cellData = polyData->GetCellData();
        vtkDataArray * idArray = cellData ? cellData->GetArray("id") : nullptr;
        vtkDataArray * velArray = cellData ? cellData->GetArray("vel") : nullptr;
        if (cellData == nullptr || idArray == nullptr || velArray == nullptr) {
            SDL_Log("Failed to read cell data!");
            exit(VTK_READER_ERROR_EXIT_CODE);
        }

        //计算全局顶点法向量
        vtkNew<vtkPolyDataNormals> normalsFilter;
        normalsFilter->SetInputData(polyData);
        normalsFilter->SetComputePointNormals(true); //计算顶点法向量
        normalsFilter->SetComputeCellNormals(false); //不计算面法向量
        normalsFilter->SetSplitting(false);          //不要因为法线差异而分裂顶点
        normalsFilter->SetConsistency(true);         //尝试使所有法线方向一致
        normalsFilter->SetAutoOrientNormals(true);   //将法向量定向到外侧
        normalsFilter->Update();

        vtkDataArray * vtkDataNormals = normalsFilter->GetOutput()->GetPointData()->GetNormals();
        vtkPoints * meshPoints = polyData->GetPoints();

        //逐个Cell读取
        for (vtkIdType i = 0; i < cellCount; i++) {
            //获取Cell作为独立几何对象
            vtkCell * cell = polyData->GetCell(i);

            //检查几何类型是否为vtkTriangleStrip
            if (strcmp(vtkCellTypes::GetClassNameFromTypeId(cell->GetCellType()), "vtkTriangleStrip") != 0) {
                SDL_Log("Found illegal cell type: %s!", vtkCellTypes::GetClassNameFromTypeId(cell->GetCellType()));
                exit(VTK_READER_ERROR_EXIT_CODE);
            }

            //ID
            ids.push_back(static_cast<size_t>(idArray->GetTuple1(i)));

            //速度
            const double * vel = velArray->GetTuple3(i);
            velocities.push_back({
                static_cast<float>(vel[0]),
                static_cast<float>(vel[1]),
                static_cast<float>(vel[2]),
            });

            //读取该Cell的所有顶点坐标和法向量
            vtkIdList * pointIds = cell->GetPointIds();
            const vtkIdType cellPointCount = cell->GetNumberOfPoints();

            //一个 TriangleStrip 会生成cellPointCount - 2个三角形
            const size_t triangleCount = cellPointCount - 2;
            std::vector<float3> verticesThisCell, normalsThisCell;
            verticesThisCell.reserve(triangleCount * 3); normalsThisCell.reserve(triangleCount * 3);

            for (vtkIdType triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++) {
                /*
                 * 根据三角形索引调整顶点顺序
                 * 偶数三角形：正常顺序 (v_i, v_{i+1}, v_{i+2})
                 * 奇数三角形：翻转顺序 (v_{i}, v_{i+2}, v_{i+1})
                 */
                std::array<vtkIdType, 3> trianglePointIndexes = {
                        pointIds->GetId(triangleIndex + 0),
                        pointIds->GetId(triangleIndex + 1),
                        pointIds->GetId(triangleIndex + 2)
                };
                if ((triangleIndex & 1) != 0) {
                    std::swap(trianglePointIndexes[1], trianglePointIndexes[2]);
                }

                //添加三个顶点和法线
                for (vtkIdType pointIdx: trianglePointIndexes) {
                    double coords[3];
                    meshPoints->GetPoint(pointIdx, coords);
                    verticesThisCell.push_back({
                        static_cast<float>(coords[0]),
                        static_cast<float>(coords[1]),
                        static_cast<float>(coords[2])
                    });
                    double normal[3];
                    vtkDataNormals->GetTuple(pointIdx, normal);
                    normalsThisCell.push_back({
                        static_cast<float>(normal[0]),
                        static_cast<float>(normal[1]),
                        static_cast<float>(normal[2])
                    });
                }
            }

            //将顶点数组和法线数组移动到文件全局数组
            verticesThisFile.push_back(std::move(verticesThisCell));
            normalsThisFile.push_back(std::move(normalsThisCell));
        }
        return {ids, velocities, verticesThisFile, normalsThisFile};
    }

    //子线程函数：读取VTK文件，将其转换为渲染器粒子数组，并写入缓存文件
    void writeVTKFileCache(
            size_t totalFileCount, const std::string & vtkFilePathName,
            const std::string & cacheFilePathName, std::atomic<size_t> & processedFileCount,
            std::atomic<size_t> & maxCellCountSingleFile)
    {
        //读取VTK粒子数组
        const auto [ids, velocities, verticesThisFile, normalsThisFile] = readVTKFile(vtkFilePathName, maxCellCountSingleFile);

        //打开数据文件
        std::ofstream out(cacheFilePathName, std::ios::out | std::ios::binary);
        checkStreamOpen(out, cacheFilePathName);
        auto * outBuffer = new char [1024 * VTK_READER_IO_BUFFER_SIZE_KB];
        out.rdbuf()->pubsetbuf(outBuffer, sizeof(outBuffer));

        //写入粒子数量
        const size_t particleCount = ids.size();
        out.write(reinterpret_cast<const char *>(&particleCount), sizeof(size_t));

        for (size_t i = 0; i < particleCount; i++) {
            //写入粒子基础信息
            const size_t & id = ids[i];
            out.write(reinterpret_cast<const char *>(&id), sizeof(size_t));
            const float3 & vel = velocities[i];
            out.write(reinterpret_cast<const char *>(&vel), sizeof(float3));

            //写入此粒子的顶点/法线数量
            const size_t vertexCountThisCell = verticesThisFile[i].size();
            out.write(reinterpret_cast<const char *>(&vertexCountThisCell), sizeof(size_t));

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
    std::tuple<std::vector<std::string>, std::vector<float>, size_t> VTKReader::readSeriesFile(
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

    void VTKReader::writeVTKDataCache(
            const std::string & seriesFilePath, const std::string & seriesFileName,
            const std::string & cacheFilePath)
    {
        SDL_Log("Generating VTK data cache...");

        //读取series文件
        const auto [vtkFiles, durations, fileCount] = readSeriesFile(seriesFilePath, seriesFileName);

        //确保缓存文件目录存在
        filesystem::path path(cacheFilePath);
        filesystem::create_directories(path);

        //启动多个线程处理数据，每个文件一个线程
        SDL_Log("Starting processing VTK files...");
        std::atomic<size_t> processedFileCount(0), maxCellCountSingleFile(0);
        std::vector<std::thread> threads;

        for (size_t i = 0; i < fileCount; i++) {
            threads.emplace_back(
                    writeVTKFileCache, fileCount,
                    vtkFiles[i], cacheFilePath + "particle" + std::to_string(i) + ".cache",
                    std::ref(processedFileCount), std::ref(maxCellCountSingleFile));
        }

        //等待所有线程结束
        for (auto & t : threads) {
            t.join();
        }

        //将所有文件最大Cell数量耗时较长写入独立文件用于快速构造材质数组，同时使得材质构造较为灵活
        const auto maxCellCount = maxCellCountSingleFile.load(std::memory_order_relaxed);
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

    std::vector<RendererParticle> VTKReader::readVTKDataCache(const std::string & cacheFilePathName) {
        //打开缓存文件
        std::ifstream in(cacheFilePathName, std::ios::in | std::ios::binary);
        checkStreamOpen(in, cacheFilePathName);
        auto * inBuffer = new char [1024 * VTK_READER_IO_BUFFER_SIZE_KB];
        in.rdbuf()->pubsetbuf(inBuffer, sizeof(inBuffer));

        //读取粒子数量
        size_t particleCount;
        in.read(reinterpret_cast<char *>(&particleCount), sizeof(size_t));

        std::vector<RendererParticle> particles; particles.reserve(particleCount);
        std::vector<float3> verticesThisCell, normalsThisCell;

        //逐个读取每个粒子信息
        for (size_t i = 0; i < particleCount; i++) {
            //ID
            size_t id;
            in.read(reinterpret_cast<char *>(&id), sizeof(size_t));

            //速度
            float3 velocity;
            in.read(reinterpret_cast<char *>(&velocity), sizeof(float3));

            //顶点/法线数量
            size_t vertexCount;
            in.read(reinterpret_cast<char *>(&vertexCount), sizeof(size_t));

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
                .id = id,
                .velocity = velocity,
                .dev_vertices = dev_vertices,
                .dev_normals = dev_normals,
                .count = vertexCount / 3
            });
        }

        in.close();
        delete[] inBuffer;
        return particles;
    }

    void VTKReader::freeVTKData(std::vector<RendererParticle> & particles) {
        for (const auto & particle: particles) {
            //cudaCheckError(cudaFree(particle.dev_vertices));
            cudaCheckError(cudaFree(particle.dev_normals));
        }
        particles = {};
    }

    size_t VTKReader::readMaxCellCountAllVTKFile(const std::string & cacheFilePath) {
        std::ifstream metaData(cacheFilePath + "metadata.cache", std::ios::in);
        checkStreamOpen(metaData, cacheFilePath + "metadata.cache");
        size_t maxCount;
        metaData >> maxCount;
        metaData.close();

        SDL_Log("Max cell count in a single VTK file: %zd.", maxCount);
        return maxCount;
    }
}