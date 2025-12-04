#ifndef RENDEREROPTIX_COLORRAMP_CUH
#define RENDEREROPTIX_COLORRAMP_CUH

#include <Global/HostFunctions.cuh>
#include <unordered_map>

//颜色预设，使用颜色映射条带模板生成材质数组
namespace project {
    typedef enum class ColorRampPreset {
        Viridis,
        Plasma,
        Spectral,
        Terrain,
        Heatmap,
        Grayscale
    } ColorRampPreset;

    typedef struct ColorStop {
        float position;
        float3 color;
    } ColorStop;

    static inline float3 lerpColor(const float3 &a, const float3 &b, float t) {
        return float3{
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t,
                a.z + (b.z - a.z) * t
        };
    }

    static inline std::vector<ColorStop> colorStopsForPreset(ColorRampPreset preset) {
        switch (preset) {
            case ColorRampPreset::Plasma:
                return {
                        {0.00f, {0.050f, 0.030f, 0.527f}},
                        {0.25f, {0.537f, 0.062f, 0.549f}},
                        {0.50f, {0.871f, 0.191f, 0.494f}},
                        {0.75f, {0.992f, 0.580f, 0.288f}},
                        {1.00f, {0.940f, 0.975f, 0.131f}}
                };
            case ColorRampPreset::Spectral:
                return {
                        {0.00f, {0.619f, 0.003f, 0.258f}},
                        {0.20f, {0.835f, 0.243f, 0.310f}},
                        {0.40f, {0.957f, 0.427f, 0.263f}},
                        {0.60f, {0.993f, 0.681f, 0.380f}},
                        {0.80f, {0.741f, 0.858f, 0.407f}},
                        {1.00f, {0.400f, 0.761f, 0.647f}}
                };
            case ColorRampPreset::Terrain:
                return {
                        {0.00f, {0.149f, 0.149f, 0.149f}},
                        {0.25f, {0.114f, 0.451f, 0.208f}},
                        {0.50f, {0.639f, 0.784f, 0.325f}},
                        {0.75f, {0.988f, 0.972f, 0.745f}},
                        {1.00f, {0.996f, 0.922f, 0.545f}}
                };
            case ColorRampPreset::Heatmap:
                return {
                        {0.00f, {0.050f, 0.050f, 0.300f}},
                        {0.25f, {0.000f, 0.000f, 1.000f}},
                        {0.50f, {0.000f, 1.000f, 1.000f}},
                        {0.75f, {1.000f, 1.000f, 0.000f}},
                        {1.00f, {1.000f, 0.000f, 0.000f}}
                };
            case ColorRampPreset::Grayscale:
                return {
                        {0.00f, {0.050f, 0.050f, 0.050f}},
                        {1.00f, {0.950f, 0.950f, 0.950f}}
                };
            case ColorRampPreset::Viridis:
            default:
                return {
                        {0.00f, {0.267f, 0.004f, 0.329f}},
                        {0.25f, {0.283f, 0.141f, 0.458f}},
                        {0.50f, {0.254f, 0.265f, 0.530f}},
                        {0.75f, {0.196f, 0.509f, 0.364f}},
                        {1.00f, {0.993f, 0.906f, 0.144f}}
                };
        }
    }

    static inline std::vector<float3> bakeColorRamp(const std::vector<ColorStop> &stops, size_t count) {
        std::vector<float3> colors;
        if (count == 0 || stops.empty()) return colors;

        colors.resize(count);
        if (count == 1) {
            colors[0] = stops.back().color;
            return colors;
        }

        for (size_t i = 0; i < count; ++i) {
            const float u = static_cast<float>(i) / static_cast<float>(count - 1);
            const ColorStop *lower = &stops.front();
            const ColorStop *upper = &stops.back();

            for (size_t s = 1; s < stops.size(); ++s) {
                if (u <= stops[s].position) {
                    upper = &stops[s];
                    lower = &stops[s - 1];
                    break;
                }
                lower = &stops[s];
            }

            const float span = upper->position - lower->position;
            const float t = span > 0.0f ? (u - lower->position) / span : 0.0f;
            colors[i] = lerpColor(lower->color, upper->color, std::clamp(t, 0.0f, 1.0f));
        }
        return colors;
    }

    static inline std::string toLower(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    static inline const char * presetName(ColorRampPreset preset) {
        switch (preset) {
            case ColorRampPreset::Plasma:    return "plasma";
            case ColorRampPreset::Spectral:  return "spectral";
            case ColorRampPreset::Terrain:   return "terrain";
            case ColorRampPreset::Heatmap:   return "heatmap";
            case ColorRampPreset::Grayscale: return "grayscale";
            case ColorRampPreset::Viridis:
            default:                         return "viridis";
        }
    }

    static inline ColorRampPreset resolveColorRampPreset(const std::string & particleMaterials) {
        const std::unordered_map<std::string, ColorRampPreset> mapping = {
                {"viridis",   ColorRampPreset::Viridis},
                {"plasma",    ColorRampPreset::Plasma},
                {"spectral",  ColorRampPreset::Spectral},
                {"terrain",   ColorRampPreset::Terrain},
                {"heatmap",   ColorRampPreset::Heatmap},
                {"grayscale", ColorRampPreset::Grayscale}
        };

        //忽略大小写：将输入转换为小写
        std::string s = particleMaterials;
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        try {
            return mapping.at(s);
        } catch (const std::out_of_range & e) {
            SDL_Log("Error when processing color preset: no such value!");
            return ColorRampPreset::Viridis;
        }
    }
}

#endif //RENDEREROPTIX_COLORRAMP_CUH
