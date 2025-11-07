#ifndef RENDEREROPTIX_SHADER_CUH
#define RENDEREROPTIX_SHADER_CUH

#include <Global/DeviceFunctions.cuh>

//此文件定义的结构体中的参数仅一帧内有效
namespace project {
    //几何体类型
    typedef enum class GeometryType {
        SPHERE, TRIANGLE
    } GeometryType;

    //材质类型
    typedef enum class MaterialType {
        ROUGH, METAL
    } MaterialType;

    //全局参数
    constexpr unsigned int RAY_TRACE_DEPTH = 10;

    typedef struct GlobalParams {
        OptixTraversableHandle handle;
        curandState * stateArray;
    } GlobalParams;

    //raygen参数
    typedef struct RayGenParams {
        unsigned int width, height;
        cudaSurfaceObject_t surfaceObject;

        float3 cameraCenter;
        float3 cameraU, cameraV, cameraW;
    } RayGenParams;

    //miss参数
    typedef struct MissParams {
        float3 backgroundColor;
    } MissParams;

    //hit参数
    typedef struct HitGroupParams {
        GeometryType geometryType;
        MaterialType materialType;

        //全局几何数据，通过hit参数结构体传递，为统一访问方式，不使用内置几何信息获取函数
        //“全局几何数据”为包含命中几何体的GAS内的所有几何体
        union {
            //球体球心数组和半径数组
            struct {
                float3 * centers;
                float * radii;
            } sphere;           //作为union的成员变量名，不是类型名
            //三角形顶点法线数组
            struct {
                float3 * vertexNormals;
            } triangles;
        };

        //材质基础颜色，此设计只允许GAS内所有几何体相同基础颜色，仅当需要GAS内几何体有不同颜色时需要变为数组
        union {
            //粗糙材质
            struct {
                float3 albedo;
            } rough;
            //金属材质
            struct {
                float3 albedo;
                float fuzz;
            } metal;
        };
    } HitGroupParams;
}

#endif //RENDEREROPTIX_SHADER_CUH
