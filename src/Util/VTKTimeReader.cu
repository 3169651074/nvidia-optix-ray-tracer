#include <Util/VTKTimeReader.cuh>

//C++
#include <filesystem>
namespace filesystem = std::filesystem;

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

namespace vtk_reader {
    std::pair<
            std::vector<float3>, std::vector<float3>
    > readSTLFile(const std::string & filePath);

    std::tuple<
            std::vector<float3>, std::vector<size_t>, std::vector<float4>,
            std::vector<float3>, std::vector<size_t>
    > readVTKTimeFile(const std::string & filePath, std::atomic<size_t> & maxPointCountSingleFile);
}

namespace project {
    std::tuple<std::vector<std::string>, std::vector<float>, size_t>
    VTKTimeReader::readSeriesFile(const std::string &seriesFilePath, const std::string &seriesFileName) {
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
            const std::string name = item["name"]; files.push_back(seriesFilePath + name);
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

    std::vector<RendererTimeParticleData> VTKTimeReader::readSTLFiles(const std::string & stlFilePath) {
        //查找目录下所有.stl文件
        SDL_Log("Scanning STL files...");

        std::vector<filesystem::path> stlFilePaths;
        for (const auto & entry : filesystem::directory_iterator(stlFilePath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".stl") {
                stlFilePaths.push_back(entry.path());
            }
        }

        //按字典顺序排序
        std::sort(stlFilePaths.begin(), stlFilePaths.end());

        //逐个读取文件数据，并拷贝至设备内存
        SDL_Log("Found %zd STL files, collecting STL data...", stlFilePaths.size());
        std::vector<RendererTimeParticleData> stlFileDatas;
        stlFileDatas.reserve(stlFilePaths.size());

        for (const auto & path: stlFilePaths) {
            const auto [vertices, normals] = vtk_reader::readSTLFile(path.string());
            RendererTimeParticleData particleData;

            cudaCheckError(cudaMalloc(&particleData.dev_vertices, vertices.size() * sizeof(float3)));
            cudaCheckError(cudaMemcpy(particleData.dev_vertices, vertices.data(), vertices.size() * sizeof(float3), cudaMemcpyHostToDevice));

            cudaCheckError(cudaMalloc(&particleData.dev_normals, normals.size() * sizeof(float3)));
            cudaCheckError(cudaMemcpy(particleData.dev_normals, normals.data(), normals.size() * sizeof(float3), cudaMemcpyHostToDevice));

            particleData.triangleCount = normals.size();
            stlFileDatas.push_back(particleData);
        }

        return stlFileDatas;
    }

    std::tuple<
            std::vector<float3>, std::vector<size_t>, std::vector<float4>,
            std::vector<float3>, std::vector<size_t>
    > VTKTimeReader::readVTKFile(const std::string &filePath, std::atomic<size_t> & maxPointCountSingleFile) {
        return vtk_reader::readVTKTimeFile(filePath, maxPointCountSingleFile);
    }
}