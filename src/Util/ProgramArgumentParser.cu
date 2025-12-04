#include <Util/ProgramArgumentParser.cuh>

namespace project {
    std::vector<std::array<float, 12>> ProgramArgumentParser::parseSphereData(
            GeometryData & geoData, const json & sphereData)
    {
        //额外几何体当前使用静态的变换矩阵，在此处预计算
        std::vector<std::array<float, 12>> transforms(sphereData.size());

        for (size_t i = 0; i < sphereData.size(); i++) {
            const auto & sphere = sphereData[i];

            //逐个对象成员解析
            const auto center = sphere["center"].get<std::vector<float>>();
            const auto radius = sphere["radius"].get<float>();
            const auto matType = sphere["mat-type"].get<std::string>() == "ROUGH"
                                 ? MaterialType::ROUGH : MaterialType::METAL;
            const auto matIndex = sphere["mat-index"].get<size_t>();
            const auto shift = sphere["shift"].get<std::vector<float>>();
            const auto rotate = sphere["rotate"].get<std::vector<float>>();
            const auto scale = sphere["scale"].get<std::vector<float>>();

            geoData.spheres.push_back({
                                              {{center[0], center[1], center[2]}, radius}
                                      });
            geoData.sphereMaterialIndices.push_back({
                                                            matType, matIndex
                                                    });
            //计算变换矩阵
            float transform[12];
            MathHelper::constructTransformMatrix(
                    {shift[0], shift[1], shift[2]},
                    {rotate[0], rotate[1], rotate[2]},
                    {scale[0], scale[1], scale[2]}, transform);
            //拷贝到数组
            memcpy(transforms[i].data(), transform, 12 * sizeof(float));
        }
        return transforms;
    }

    std::tuple<
            GeometryData, MaterialData, RenderLoopData,
            std::vector<std::array<float, 12>>,
            std::string, std::string, std::string, std::string,
            std::string, bool, bool, bool,
            size_t
    > ProgramArgumentParser::parseProgramArguments() {
        SDL_Log("Parsing program arguments from JSON file: %s", CONFIG_FILE_PATH);

        //打开 JSON 文件
        std::ifstream file(CONFIG_FILE_PATH);
        if (!file) {
            SDL_Log("Failed to open config: %s!", CONFIG_FILE_PATH);
            exit(COMMAND_PARSER_ERROR_EXIT_CODE);
        }

        //解析JSON数据
        json data;
        try {
            data = json::parse(file);

            //额外材质信息
            SDL_Log("Parsing additional material data...");
            MaterialData matData = {};
            for (const auto & rough: data["roughs"]) {
                const auto albedo = rough["albedo"].get<std::vector<float>>();
                matData.roughs.push_back({albedo[0], albedo[1], albedo[2]});
            }
            for (const auto & metal : data["metals"]) {
                const auto albedo = metal["albedo"].get<std::vector<float>>();
                const auto fuzz = metal["fuzz"].get<float>();
                matData.metals.push_back({
                                                 {albedo[0], albedo[1], albedo[2]}, fuzz
                                         });
            }

            //额外几何体信息
            SDL_Log("Parsing additional geometry data...");
            GeometryData geoData = {};
            const auto sphereTransforms = parseSphereData(geoData, data["spheres"]);

            //渲染窗口信息
            SDL_Log("Parsing render loop data...");

            const auto loopData = data["loop-data"];
            const std::string apiTypeStr = loopData["api"].get<std::string>();
            SDL_GraphicsWindowAPIType apiType;
            //平台检查：Linux不支持D3D11/D3D12
#ifndef _WIN32
            if (apiTypeStr == "D3D11" || apiTypeStr == "D3D12") {
                SDL_Log("Error: Direct3D (D3D11/D3D12) is only supported on Windows!");
                SDL_Log(R"(Please use "OGL" or "VK" instead.)");
                exit(COMMAND_PARSER_ERROR_EXIT_CODE);
            }
#endif
            if (apiTypeStr == "OGL") {
                apiType = SDL_GraphicsWindowAPIType::OPENGL;
            } else if (apiTypeStr == "VK") {
                apiType = SDL_GraphicsWindowAPIType::VULKAN;
#ifdef _WIN32
            } else if (apiTypeStr == "D3D11") {
                apiType = SDL_GraphicsWindowAPIType::DIRECT3D11;
            } else if (apiTypeStr == "D3D12") {
                apiType = SDL_GraphicsWindowAPIType::DIRECT3D12;
#endif
            } else {
                SDL_Log(R"(Invalid api type, must be "OGL", "VK", "D3D11" or "D3D12"!)");
                exit(COMMAND_PARSER_ERROR_EXIT_CODE);
            }
            const auto isMesh = data.at("mesh").get<bool>();
            const auto seriesFilePath = data.at("series-path").get<std::string>();
            const auto seriesFileName = data.at("series-name").get<std::string>();
            const auto cacheFilePath = data.at("cache-path").get<std::string>();
            const auto stlFilePath = data.at("stl-path").get<std::string>();
            const auto particleMaterials = data.at("particle-material-preset").get<std::string>();
            const auto windowWidth = loopData["window-width"].get<int>();
            const auto windowHeight = loopData["window-height"].get<int>();
            const auto targetFPS = loopData["fps"].get<size_t>();
            const auto cameraCenterVec = loopData["camera-center"].get<std::vector<float>>();
            const auto cameraTargetVec = loopData["camera-target"].get<std::vector<float>>();
            const auto upDirectionVec = loopData["up-direction"].get<std::vector<float>>();
            const auto renderSpeedRatio = loopData["render-speed-ratio"].get<size_t>();
            const auto particleShiftVec = loopData["particle-shift"].get<std::vector<float>>();
            const auto particleScaleVec = loopData["particle-scale"].get<std::vector<float>>();
            const auto mouseSensitivity = loopData["mouse-sensitivity"].get<float>();
            const auto pitchLimitDegree = loopData["camera-pitch-limit-degree"].get<float>();
            const auto cameraSpeedStride = loopData["camera-speed-stride"].get<float>();
            const auto initialSpeedRatio = loopData["camera-initial-speed-ratio"].get<size_t>();
            const auto isDebugMode = data.at("debug-mode").get<bool>();
            const auto isCache = data.at("cache").get<bool>();
            const auto threadCount = data.at("cache-process-thread-count").get<size_t>();

            const RenderLoopData retLoopData(
                    apiType,
                    windowWidth, windowHeight, "RendererOptiX",
                    targetFPS,
                    {cameraCenterVec[0], cameraCenterVec[1], cameraCenterVec[2]},
                    {cameraTargetVec[0], cameraTargetVec[1], cameraTargetVec[2]},
                    {upDirectionVec[0], upDirectionVec[1], upDirectionVec[2]},
                    renderSpeedRatio,
                    {particleShiftVec[0], particleShiftVec[1], particleShiftVec[2]},
                    {particleScaleVec[0], particleScaleVec[1], particleScaleVec[2]},
                    mouseSensitivity,
                    pitchLimitDegree,
                    cameraSpeedStride,
                    initialSpeedRatio,
                    isDebugMode
            );
            file.close();

            return {
                    geoData, matData, retLoopData, sphereTransforms,
                    seriesFilePath, seriesFileName, cacheFilePath, stlFilePath,
                    particleMaterials, isMesh, isDebugMode, isCache, threadCount
            };
        } catch (json::parse_error & e) {
            SDL_Log("JSON parsing error: %s", e.what());
            exit(COMMAND_PARSER_ERROR_EXIT_CODE);
        }
    }
}