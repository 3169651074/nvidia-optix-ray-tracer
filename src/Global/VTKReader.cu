#include <Global/VTKReader.cuh>
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

namespace {
    //获取当前可执行文件的绝对路径
    std::string currentExecutableDirectory() {
        char buffer[300];
        std::string execDir;
#ifdef _WIN32
        //Windows
        if (GetModuleFileNameA(nullptr, buffer, sizeof(buffer))) {
            execDir = std::string(buffer);
            execDir = execDir.substr(0, execDir.find_last_of("\\/")); //去掉文件名，保留目录
        } else {
            SDL_Log("Error: Unable to get executable directory!");
            exit(VTK_READER_ERROR_EXIT_CODE);
        }
#else
        //Linux
        ssize_t count = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
        if (count != -1) {
            buffer[count] = '\0'; //Null-terminate the string
            execDir = std::string(dirname(buffer)); //获取目录部分
        } else {
            SDL_Log("Error: Unable to get executable directory!");
            exit(VTK_READER_ERROR_EXIT_CODE);
        }
#endif
        return execDir;
    }

    //确保目录存在，可以传入绝对路径或当前工作目录（std::filesystem::current_path()）
    void ensureDirectoryExists(const std::string & path) {
        try {
            //使用std::filesystem::create_directories递归创建目录
            if (std::filesystem::create_directories(path)) {
                SDL_Log("Create directory: %s.", path.c_str());
            }
        } catch (const std::filesystem::filesystem_error & e) {
            SDL_Log("Exception in when creating path %s: %s!", path.c_str(), e.what());
        }
    }

    //检查VTK文件头
    void checkVTKFileHeader(const std::string & filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            SDL_Log("Failed to open vtk file: %s!", filePath.c_str());
            exit(VTK_READER_ERROR_EXIT_CODE);
        }
        std::string line;
        getline(file, line);
        if (line.find("# vtk DataFile Version") == std::string::npos) {
            SDL_Log("Illegal vtk file header in file %s: %s!", filePath.c_str(), line.c_str());
            exit(VTK_READER_ERROR_EXIT_CODE);
        }
        file.close();
    }

    //使用VTK库读取VTK文件信息
    typedef struct VTKParticle {
        size_t id;
        float3 velocity;
        float2 boundingBoxRanges[3];
        float3 centroid;
        //组成此粒子的所有三角形顶点，保留triangle strip的格式
        std::vector<float3> vertices;
        //每个顶点对应的法向量，和顶点一一对应
        std::vector<float3> verticesNormals;
    } VTKParticle;

    //读取单个VTK文件
    std::vector<VTKParticle> readVTKFile(const std::string & filePath) {
        //读取VTK文件并获取vtkPolyData指针
        vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
        reader->SetFileName(filePath.c_str());
        reader->Update();
        vtkPolyData * polyData = reader->GetOutput();
        if (polyData == nullptr || polyData->GetNumberOfPoints() == 0) {
            SDL_Log("Failed to get poly data pointer or there is no points in file!");
            exit(VTK_READER_ERROR_EXIT_CODE);
        }
        //全局点数量和粒子数量
        const vtkIdType numCells = polyData->GetNumberOfCells();

        std::vector<VTKParticle> ret;
        ret.reserve(numCells);

        //读取几何数据
        vtkCellData * cellData = polyData->GetCellData();
        vtkDataArray * idArray = cellData ? cellData->GetArray("id") : nullptr;
        vtkDataArray * velArray = cellData ? cellData->GetArray("vel") : nullptr;
        if (cellData == nullptr || idArray == nullptr || velArray == nullptr) {
            SDL_Log("Failed to read cell data, in file!");
            exit(VTK_READER_ERROR_EXIT_CODE);
        }

        //计算全局顶点法向量
        vtkNew<vtkPolyDataNormals> normalsFilter;
        normalsFilter->SetInputData(polyData);
        normalsFilter->SetComputePointNormals(true); // 计算顶点法向量
        normalsFilter->SetComputeCellNormals(false); // 不计算面法向量
        normalsFilter->SetSplitting(false);          // 不要因为法线差异而分裂顶点
        normalsFilter->SetConsistency(true);         // 尝试使所有法线方向一致
        normalsFilter->SetAutoOrientNormals(true);   // 将法向量定向到外侧
        normalsFilter->Update();

        vtkPolyData * resultWithNormals = normalsFilter->GetOutput();
        vtkDataArray * normals = resultWithNormals->GetPointData()->GetNormals();
        vtkPoints * meshPoints = polyData->GetPoints();

        //逐个Cell读取
        for (vtkIdType i = 0; i < numCells; i++) {
            VTKParticle particle{};

            //获取Cell作为独立几何对象
            vtkCell * cell = polyData->GetCell(i);

            //检查几何类型是否为vtkTriangleStrip
            if (strcmp(vtkCellTypes::GetClassNameFromTypeId(cell->GetCellType()), "vtkTriangleStrip") != 0) {
                SDL_Log("Found illegal cell type: %s, aborting!", vtkCellTypes::GetClassNameFromTypeId(cell->GetCellType()));
                exit(VTK_READER_ERROR_EXIT_CODE);
            }

            //ID
            particle.id = static_cast<size_t>(idArray->GetTuple1(i));

            //速度
            const double * vel = velArray->GetTuple3(i);
            particle.velocity = float3{static_cast<float>(vel[0]), static_cast<float>(vel[1]), static_cast<float>(vel[2])};

            //包围盒
            double bounds[6];
            cell->GetBounds(bounds);
            for (size_t j = 0; j < 3; j++) {
                particle.boundingBoxRanges[j] = float2{static_cast<float>(bounds[j * 2]), static_cast<float>(bounds[j * 2 + 1])};
            }

            //质心
            const vtkIdType numCellPoints = cell->GetNumberOfPoints();
            if (numCellPoints > 0) {
                double centroid[3] = {0.0, 0.0, 0.0};
                vtkPoints * points = cell->GetPoints();

                //累加所有顶点坐标
                for (vtkIdType j = 0; j < numCellPoints; j++) {
                    double point[3];
                    points->GetPoint(j, point);
                    centroid[0] += point[0];
                    centroid[1] += point[1];
                    centroid[2] += point[2];
                }

                //取平均值
                centroid[0] /= static_cast<double>(numCellPoints);
                centroid[1] /= static_cast<double>(numCellPoints);
                centroid[2] /= static_cast<double>(numCellPoints);

                particle.centroid = float3{
                        static_cast<float>(centroid[0]),
                        static_cast<float>(centroid[1]),
                        static_cast<float>(centroid[2])};
            } else {
                SDL_Log("There is no points in cell %zd, in file %s!", i, filePath.c_str());
                exit(-1);
            }

            //读取该Cell的所有顶点坐标和法向量
            particle.vertices.reserve(numCellPoints);
            particle.verticesNormals.reserve(numCellPoints);

            vtkIdList * pointIds = cell->GetPointIds();
            for (vtkIdType j = 0; j < numCellPoints; j++) {
                const vtkIdType pointId = pointIds->GetId(j);

                //获取顶点坐标
                double coords[3];
                meshPoints->GetPoint(pointId, coords);
                particle.vertices.push_back(float3{
                        static_cast<float>(coords[0]),
                        static_cast<float>(coords[1]),
                        static_cast<float>(coords[2]),
                });

                //获取顶点法向量
                double normal[3];
                normals->GetTuple(pointId, normal);
                particle.verticesNormals.push_back(float3{
                        static_cast<float>(normal[0]),
                        static_cast<float>(normal[1]),
                        static_cast<float>(normal[2]),
                });
            }
            ret.push_back(particle);
        }
        return ret;
    }

    //读取一组VTK文件，获取单个文件Cell的最大数量
    size_t maxCellCountSingleFile(const std::vector<std::string> & vtkFiles) {
        SDL_Log("Finding out maximum cell count in a single VTK file...");

        size_t maxCount = 0;
        for (const auto & filePath: vtkFiles) {
            //检查文件头
            checkVTKFileHeader(filePath);

            //获取vtkPolyData指针
            vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
            reader->SetFileName(filePath.c_str());
            reader->Update();
            vtkPolyData * polyData = reader->GetOutput();
            if (polyData == nullptr) {
                SDL_Log("Failed to get poly data pointer in file %s!", filePath.c_str());
                exit(VTK_READER_ERROR_EXIT_CODE);
            }

            //获取全局点数量
            const vtkIdType numCells = polyData->GetNumberOfCells();
            maxCount = std::max<size_t>(maxCount, numCells);
        }

        SDL_Log("Max cell count in a single VTK file: %zd.", maxCount);
        return maxCount;
    }

    //子线程函数：读取VTK文件，将其转换为渲染器粒子数组，并写入缓存文件
    void writeVTKFileCache(size_t totalFileCount, size_t materialOffset, const std::string & vtkFilePathName,
                           const std::string & cacheFilePathName, std::atomic<size_t> & processedFileCount)
    {
        //读取VTK粒子数组
        const auto vtkParticles = readVTKFile(vtkFilePathName);
        const auto particleCount = vtkParticles.size();

        //打开文件
        std::ofstream out(cacheFilePathName, std::ios::out | std::ios::binary);
        if (!out.is_open()) {
            SDL_Log("Failed to open file: %s!", cacheFilePathName.c_str());
            exit(VTK_READER_ERROR_EXIT_CODE);
        }
        auto * buffer = new char [1024 * VTK_READER_IO_BUFFER_SIZE_KB];
        out.rdbuf()->pubsetbuf(buffer, sizeof(buffer));

        //写入粒子数量（元数据，用于读取相应数量的粒子）
        out.write(reinterpret_cast<const char *>(&particleCount), sizeof(size_t));

        //逐个转换每个粒子，转换完成后立即写入到文件
        for (size_t i = 0; i < particleCount; i++) {
            const auto & particle = vtkParticles[i];

            //写入粒子基础信息
            out.write(reinterpret_cast<const char *>(&particle.id), sizeof(size_t));                     //ID
            out.write(reinterpret_cast<const char *>(&particle.velocity), sizeof(float3));               //速度
            out.write(reinterpret_cast<const char *>(&particle.boundingBoxRanges), 3 * sizeof(float2));  //包围盒
            out.write(reinterpret_cast<const char *>(&particle.centroid), sizeof(float3));               //重心
            const size_t materialIndex = materialOffset + i;
            out.write(reinterpret_cast<const char *>(&materialIndex), sizeof(size_t));                   //材质索引

            //写入粒子几何数据：顶点数量，顶点数组和法线数组
            const size_t triangleCount = particle.vertices.size() - 2; //vtkTriangleStrip由N个点组成N - 2个三角形
            const size_t vertexCount = triangleCount * 3;
            out.write(reinterpret_cast<const char *>(&vertexCount), sizeof(size_t));

            //为了减少IO开销，将所有顶点和法线数据合并后一次性写入
            std::vector<float3> particleVertices; particleVertices.reserve(vertexCount);
            std::vector<float3> particleNormals;  particleNormals.reserve(vertexCount);

            for (size_t j = 0; j < triangleCount; j++) {
                particleVertices.push_back(particle.vertices[j]);
                particleNormals.push_back(particle.verticesNormals[j]);

                size_t idx1 = j + 1, idx2 = j + 2;
                //偶数三角形，顶点顺序保持不变。奇数三角形，第2，3个顶点需要取反以保持面法线方向一致
                if ((j & 1) != 0) {
                    std::swap(idx1, idx2);
                }
                particleVertices.push_back(particle.vertices[idx1]);
                particleVertices.push_back(particle.vertices[idx2]);
                particleNormals.push_back(particle.verticesNormals[idx1]);
                particleNormals.push_back(particle.verticesNormals[idx2]);
            }

            //写入三角形顶点和法线数据
            out.write(reinterpret_cast<const char *>(particleVertices.data()), vertexCount * sizeof(float3));
            out.write(reinterpret_cast<const char *>(particleNormals.data()), vertexCount * sizeof(float3));
        }

        out.close();
        delete[] buffer;

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
        if (!file.is_open()) {
            SDL_Log("Could not open the series file!");
            exit(VTK_READER_ERROR_EXIT_CODE);
        }

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
            const std::string & cacheFilePath, size_t materialOffset)
    {
        SDL_Log("Generating VTK data cache...");

        //读取series文件
        const auto [vtkFiles, durations, fileCount] = readSeriesFile(seriesFilePath, seriesFileName);

        //确保缓存文件目录存在
        filesystem::path path(cacheFilePath);
filesystem::create_directories(path);

        //由于获取所有文件最大Cell数量耗时较长，则将其写入独立文件，用于快速构造材质数组，同时使得材质构造较为灵活
        const auto maxCellCount = maxCellCountSingleFile(vtkFiles);
        SDL_Log("Max cell count in a single VTK file: %zd.", maxCellCount);
        std::ofstream metaData(cacheFilePath + "metadata.cache", std::ios::out);
        if (!metaData.is_open()) {
            SDL_Log("Failed to open file: %s!", (cacheFilePath + "metadata.cache").c_str());
            exit(VTK_READER_ERROR_EXIT_CODE);
        }
        metaData << maxCellCount;
        metaData.close();

        //启动多个线程处理数据，每个文件一个线程
        SDL_Log("Starting processing VTK files...");
        std::atomic<size_t> processedFileCount(0);
        std::vector<std::thread> threads;
        for (size_t i = 0; i < fileCount; i++) {
            threads.emplace_back(
                    writeVTKFileCache, fileCount, materialOffset,
                    vtkFiles[i], cacheFilePath + "particle" + std::to_string(i) + ".cache",
                    std::ref(processedFileCount));
        }

        //等待所有线程结束
        for (auto & t : threads) {
            t.join();
        }

        SDL_Log("VTK data cache files are written to cache directory: %s.", cacheFilePath.c_str());
        exit(0);
    }

    std::vector<Particle> VTKReader::readVTKDataCache(const std::string & cacheFilePathName) {
        //打开缓存文件
        std::ifstream in(cacheFilePathName, std::ios::in | std::ios::binary);
        if (!in.is_open()) {
            SDL_Log("Failed to open cache file: %s!", cacheFilePathName.c_str());
            exit(VTK_READER_ERROR_EXIT_CODE);
        }
        auto * buffer = new char [1024 * VTK_READER_IO_BUFFER_SIZE_KB];
        in.rdbuf()->pubsetbuf(buffer, sizeof(buffer));

        //读取粒子数量
        size_t particleCount;
        in.read(reinterpret_cast<char *>(&particleCount), sizeof(size_t));

        std::vector<Particle> particleForThisFile(particleCount);

        //逐个读取每个粒子
        for (size_t i = 0; i < particleCount; i++) {
            //ID（未使用）速度 包围盒（未使用）重心（未使用）材质索引 顶点数量 顶点数组 法线数组
            size_t id;
            in.read(reinterpret_cast<char *>(&id), sizeof(size_t));
            float3 velocity;
            in.read(reinterpret_cast<char *>(&velocity), sizeof(float3));
            float2 boundingBox[3];
            in.read(reinterpret_cast<char *>(&boundingBox), 3 * sizeof(float2));
            float3 centroid;
            in.read(reinterpret_cast<char *>(&centroid), sizeof(float3));
            size_t materialIndex;
            in.read(reinterpret_cast<char *>(&materialIndex), sizeof(size_t));

            size_t vertexCount;
            in.read(reinterpret_cast<char *>(&vertexCount), sizeof(size_t));

            std::vector<float3> vertexes(vertexCount), normals(vertexCount);
            in.read(reinterpret_cast<char *>(vertexes.data()), vertexCount * sizeof(float3));
            in.read(reinterpret_cast<char *>(normals.data()), vertexCount * sizeof(float3));

            //还原顶点数组和法线数组为三角形数组
            const size_t triangleCount = vertexCount / 3;
            std::vector<Triangle> triangles(triangleCount);
            for (size_t j = 0; j < triangleCount; j++) {
                triangles[j].vertices = std::array<float3, 3>{vertexes[j * 3 + 0], vertexes[j * 3 + 1], vertexes[j * 3 + 2]};
                triangles[j].normals = std::array<float3, 3>{normals[j * 3 + 0], normals[j * 3 + 1], normals[j * 3 + 2]};
            }

            //添加到文件缓存数据
            const Particle particle = {
                    .materialType = MaterialType::ROUGH,
                    .materialIndex = materialIndex,
                    .triangles = triangles,
                    .velocity = velocity
            };
            particleForThisFile[i] = particle;
        }
        in.close();
        delete[] buffer;

        return particleForThisFile;
    }

    size_t VTKReader::maxCellCountSingleVTKFile(const std::string & cacheFilePath) {
        std::ifstream metaData(cacheFilePath + "metadata.cache", std::ios::in);
        if (!metaData.is_open()) {
            SDL_Log("Failed to open file: %s!", (cacheFilePath + "metadata.cache").c_str());
            exit(VTK_READER_ERROR_EXIT_CODE);
        }
        size_t maxCount;
        metaData >> maxCount;
        metaData.close();

        SDL_Log("Max cell count in a single VTK file: %zd.", maxCount);
        return maxCount;
    }
}