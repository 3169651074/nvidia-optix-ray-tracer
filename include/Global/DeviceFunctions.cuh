#ifndef RENDEREROPTIX_DEVICEFUNCTIONS_CUH
#define RENDEREROPTIX_DEVICEFUNCTIONS_CUH

/*
 * 设备端头文件，设备端源文件和主机端源文件通用
 * 可用于着色器文件
 */

#include <optix_device.h>
#include <curand_kernel.h>

#define OPTIX_GLOBAL_PARAM extern "C" __constant__
#define OPTIX_PROGRAM extern "C" __global__

namespace project {
    // ====== 数值常量 ======

    constexpr float FLOAT_ZERO_VALUE = 1e-6f;
    constexpr float FLOAT_INFINITY_VALUE = 1e16f;
    constexpr float PI = 3.1415926;

    // ====== 数学工具函数 ======

    class MathHelper {
    public:
        //角度弧度转换
        __host__ __device__ __forceinline__ static float degreeToRadian(float degree) {
            return degree * PI / 180.0f;
        }
        __host__ __device__ __forceinline__ static float radianToDegree(float radian) {
            return radian * 180.0f / PI;
        }

        //浮点数比较
        __host__ __device__ __forceinline__ static bool floatValueNearZero(float val) {
            return abs(val) < FLOAT_ZERO_VALUE;
        }
        __host__ __device__ __forceinline__ static bool floatValueEquals(float v1, float v2) {
            return abs(v1 - v2) < FLOAT_ZERO_VALUE;
        }

        //构造变换矩阵
        typedef struct Matrix {
            float data[5][5];
            size_t row, col;

            //矩阵乘法
            __host__ __device__ __forceinline__ Matrix operator*(const Matrix & right) const {
                Matrix ret{{}, row, right.col};

                for (size_t i = 1; i <= ret.row; i++) {
                    for (size_t j = 1; j <= ret.col; j++) {
                        float sum = 0.0f;
                        for (size_t n = 1; n <= col; n++) {
                            sum += data[i][n] * right.data[n][j];
                        }
                        ret.data[i][j] = sum;
                    }
                }
                return ret;
            }

            //构造变换矩阵
            __host__ __device__ __forceinline__ static Matrix constructShiftMatrix(const float3 & shift) {
                return {
                        {
                                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f, 0.0f, shift.x,
                                0.0f, 0.0f, 1.0f, 0.0f, shift.y,
                                0.0f, 0.0f, 0.0f, 1.0f, shift.z,
                                0.0f, 0.0f, 0.0f, 0.0f, 1.0f
                        }, 4, 4
                };
            }

            __host__ __device__ __forceinline__ static Matrix constructScaleMatrix(const float3 & scale) {
                return {
                        {
                                0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, scale.x, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, scale.y, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, scale.z, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 1.0f
                        }, 4, 4
                };
            }

            __host__ __device__ __forceinline__ static Matrix constructRotateMatrix(float degree, int axis) {
                const float theta = MathHelper::degreeToRadian(degree);
                switch (axis) {
                    case 0:
                        return {
                                {
                                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                        0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, cos(theta), -sin(theta), 0.0f,
                                        0.0f, 0.0f, sin(theta), cos(theta), 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f, 1.0f
                                }, 4, 4
                        };
                    case 1:
                        return {
                                {
                                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                        0.0f, cos(theta), 0.0f, sin(theta), 0.0f,
                                        0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                        0.0f, -sin(theta), 0.0f, cos(theta), 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f, 1.0f
                                }, 4, 4
                        };
                    case 2:
                        return {
                                {
                                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                        0.0f, cos(theta), -sin(theta), 0.0f, 0.0f,
                                        0.0f, sin(theta), cos(theta), 0.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 0.0f, 1.0f
                                }, 4, 4
                        };
                    default: return {};
                }
            }

            __host__ __device__ __forceinline__ static Matrix constructRotateMatrix(const float3 & rotate) {
                const auto mx = constructRotateMatrix(rotate.x, 0);
                const auto my = constructRotateMatrix(rotate.y, 1);
                const auto mz = constructRotateMatrix(rotate.z, 2);
                return mx * my * mz;
            }
        } Matrix;

        __host__ __device__ static void constructTransformMatrix(
                const float3 & shift, const float3 & rotate, const float3 & scale, float * result)
        {
            //分别构造SRT矩阵
            const auto s = Matrix::constructShiftMatrix(shift);
            const auto r = Matrix::constructRotateMatrix(rotate);
            const auto sc = Matrix::constructScaleMatrix(scale);

            //求总变换矩阵
            const auto transform = s * r * sc;

            //取前3行
            for (size_t i = 0; i < 3; i++) {
                memcpy(result + i * 4, &transform.data[i + 1][1], 4 * sizeof(float));
            }
        }
    };

    // ========== 颜色转换函数 ==========

    __host__ __device__ __forceinline__ static uchar4 colorToUchar4(const float3 & c) {
        //裁剪到[0, 1]范围
        const float cx = max(0.0f, min(c.x, 1.0f));
        const float cy = max(0.0f, min(c.y, 1.0f));
        const float cz = max(0.0f, min(c.z, 1.0f));

        //转换为SRGB
        constexpr float invGamma = 1.0f / 2.4f;
        const float px = powf(cx, invGamma);
        const float py = powf(cy, invGamma);
        const float pz = powf(cz, invGamma);
        const float sx = cx < 0.0031308f ? 12.92f * cx : 1.055f * px - 0.055f;
        const float sy = cy < 0.0031308f ? 12.92f * cy : 1.055f * py - 0.055f;
        const float sz = cz < 0.0031308f ? 12.92f * cz : 1.055f * pz - 0.055f;

        //裁剪SRGB到[0, 1]范围
        const float srgb_x = max(0.0f, min(sx, 1.0f));
        const float srgb_y = max(0.0f, min(sy, 1.0f));
        const float srgb_z = max(0.0f, min(sz, 1.0f));

        //转换到单字节整数
        constexpr unsigned int N = 255u;
        constexpr unsigned int Np1 = 256u;

        return {
                (unsigned char)min((unsigned int)(srgb_x * (float)Np1), N),
                (unsigned char)min((unsigned int)(srgb_y * (float)Np1), N),
                (unsigned char)min((unsigned int)(srgb_z * (float)Np1), N),
                255u
        };
    }
    __host__ __device__ __forceinline__ static uchar4 colorToUchar4(const float4 & c) {
        return colorToUchar4(float3{c.x, c.y, c.z});
    }

    // ========== 随机数生成函数 ==========

    __device__ __forceinline__ static float randomDouble(curandState * state) {
        return curand_uniform(state);
    }

    __device__ __forceinline__ static float randomDouble(curandState * state, float min, float max) {
        return min + (max - min) * randomDouble(state);
    }

    template<typename T>
    __device__ __forceinline__ static T randomInteger(curandState * state, T min, T max) {
        const float val = min + ((max + 1) - min) * randomDouble(state);
        return static_cast<T>(val);
    }

    // ========== float2 运算 ==========

    //数乘除
    __host__ __device__ __forceinline__ static void operator*=(float2 & obj, float num) {
        obj.x *= num;
        obj.y *= num;
    }
    __host__ __device__ __forceinline__ static float2 operator*(const float2 & obj, float num) {
        float2 ret = obj; ret *= num; return ret;
    }
    __host__ __device__ __forceinline__ static float2 operator*(float num, const float2 & obj) {
        return obj * num;
    }
    __host__ __device__ __forceinline__ static void operator/=(float2 & obj, float num) {
        obj.x /= num;
        obj.y /= num;
    }
    __host__ __device__ __forceinline__ static float2 operator/(const float2 & obj, float num) {
        float2 ret = obj; ret /= num; return ret;
    }
    __host__ __device__ __forceinline__ static float2 operator/(float num, const float2 & obj) {
        return obj / num;
    }

    //数加减
    __host__ __device__ __forceinline__ static void operator+=(float2 & obj, float num) {
        obj.x += num;
        obj.y += num;
    }
    __host__ __device__ __forceinline__ static float2 operator+(const float2 & obj, float num) {
        float2 ret = obj; ret += num; return ret;
    }
    __host__ __device__ __forceinline__ static void operator-=(float2 & obj, float num) {
        obj.x -= num;
        obj.y -= num;
    }
    __host__ __device__ __forceinline__ static float2 operator-(const float2 & obj, float num) {
        float2 ret = obj; ret -= num; return ret;
    }

    //向量加减
    __host__ __device__ __forceinline__ static void operator+=(float2 & obj1, const float2 & obj2) {
        obj1.x += obj2.x;
        obj1.y += obj2.y;
    }
    __host__ __device__ __forceinline__ static float2 operator+(const float2 & obj1, const float2 & obj2) {
        float2 ret = obj1; ret += obj2; return ret;
    }
    __host__ __device__ __forceinline__ static void operator-=(float2 & obj1, const float2 & obj2) {
        obj1.x -= obj2.x;
        obj1.y -= obj2.y;
    }
    __host__ __device__ __forceinline__ static float2 operator-(const float2 & obj1, const float2 & obj2) {
        float2 ret = obj1; ret -= obj2; return ret;
    }

    //求模长
    __host__ __device__ __forceinline__ static float lengthSquared(const float2 & obj) {
        return obj.x * obj.x + obj.y * obj.y;
    }

    //单位化
    __host__ __device__ __forceinline__ static float2 normalize(const float2 & obj) {
        const float len2 = lengthSquared(obj);
        if (len2 <= FLOAT_ZERO_VALUE * FLOAT_ZERO_VALUE) {
            return make_float2(1.0f, 0.0f); //安全缺省
        }
        const float invLen = rsqrtf(len2);
        return obj * invLen;
    }

    // ========== float3 运算 ==========

    //取反
    __host__ __device__ __forceinline__ static float3 operator-(const float3 & obj) {
        return {
            -obj.x, -obj.y, -obj.z
        };
    }

    //数乘除
    __host__ __device__ __forceinline__ static void operator*=(float3 & obj, float num) {
        obj.x *= num;
        obj.y *= num;
        obj.z *= num;
    }
    __host__ __device__ __forceinline__ static float3 operator*(const float3 & obj, float num) {
        float3 ret = obj; ret *= num; return ret;
    }
    __host__ __device__ __forceinline__ static float3 operator*(float num, const float3 & obj) {
        return obj * num;
    }

    //向量分量乘除
    __host__ __device__ __forceinline__ static void operator*=(float3 & obj1, const float3 & obj2) {
        obj1.x *= obj2.x;
        obj1.y *= obj2.y;
        obj1.z *= obj2.z;
    }
    __host__ __device__ __forceinline__ static float3 operator*(const float3 & obj1, const float3 & obj2) {
        float3 ret = obj1; ret *= obj2; return ret;
    }
    __host__ __device__ __forceinline__ static void operator/=(float3 & obj1, const float3 & obj2) {
        obj1.x /= obj2.x;
        obj1.y /= obj2.y;
        obj1.z /= obj2.z;
    }
    __host__ __device__ __forceinline__ static float3 operator/(const float3 & obj1, const float3 & obj2) {
        float3 ret = obj1; ret /= obj2; return ret;
    }
    __host__ __device__ __forceinline__ static void operator/=(float3 & obj, float num) {
        obj.x /= num;
        obj.y /= num;
        obj.z /= num;
    }
    __host__ __device__ __forceinline__ static float3 operator/(const float3 & obj, float num) {
        float3 ret = obj; ret /= num; return ret;
    }
    __host__ __device__ __forceinline__ static float3 operator/(float num, const float3 & obj) {
        return obj / num;
    }

    //数加减
    __host__ __device__ __forceinline__ static void operator+=(float3 & obj, float num) {
        obj.x += num;
        obj.y += num;
        obj.z += num;
    }
    __host__ __device__ __forceinline__ static float3 operator+(const float3 & obj, float num) {
        float3 ret = obj; ret += num; return ret;
    }
    __host__ __device__ __forceinline__ static void operator-=(float3 & obj, float num) {
        obj.x -= num;
        obj.y -= num;
        obj.z -= num;
    }
    __host__ __device__ __forceinline__ static float3 operator-(const float3 & obj, float num) {
        float3 ret = obj; ret -= num; return ret;
    }

    //向量加减
    __host__ __device__ __forceinline__ static void operator+=(float3 & obj1, const float3 & obj2) {
        obj1.x += obj2.x;
        obj1.y += obj2.y;
        obj1.z += obj2.z;
    }
    __host__ __device__ __forceinline__ static float3 operator+(const float3 & obj1, const float3 & obj2) {
        float3 ret = obj1; ret += obj2; return ret;
    }
    __host__ __device__ __forceinline__ static void operator-=(float3 & obj1, const float3 & obj2) {
        obj1.x -= obj2.x;
        obj1.y -= obj2.y;
        obj1.z -= obj2.z;
    }
    __host__ __device__ __forceinline__ static float3 operator-(const float3 & obj1, const float3 & obj2) {
        float3 ret = obj1; ret -= obj2; return ret;
    }

    //求模长
    __host__ __device__ __forceinline__ static float lengthSquared(const float3 & obj) {
        return obj.x * obj.x + obj.y * obj.y + obj.z * obj.z;
    }
    __host__ __device__ __forceinline__ static float length(const float3 & obj) {
        return sqrt(lengthSquared(obj));
    }

    //单位化
    __host__ __device__ __forceinline__ static float3 normalize(const float3 & obj) {
        const float len2 = lengthSquared(obj);
        if (len2 <= FLOAT_ZERO_VALUE * FLOAT_ZERO_VALUE) {
            return make_float3(0.0f, 0.0f, 1.0f); //安全缺省
        }
        const float invLen = rsqrtf(len2);
        return obj * invLen;
    }

    //点乘叉乘
    __host__ __device__ __forceinline__ static float dot(const float3 & obj1, const float3 & obj2) {
        return obj1.x * obj2.x + obj1.y * obj2.y + obj1.z * obj2.z;
    }
    __host__ __device__ __forceinline__ static float3 cross(const float3 & obj1, const float3 & obj2) {
        return {
                obj1.y * obj2.z - obj1.z * obj2.y,
                obj1.z * obj2.x - obj1.x * obj2.z,
                obj1.x * obj2.y - obj1.y * obj2.x
        };
    }

    //向量绕轴旋转
    __host__ __device__ static float3 rotate(const float3 & origin, const float3 & axis, float angle) {
        const float3 k = normalize(axis);
        const float cosTheta = cos(angle);
        const float sinTheta = sin(angle);
        const float3 & v = origin;

        //part1: v * cos(theta)
        const float3 part1 = v * cosTheta;
        //part2: (k x v) * sin(theta)
        const float3 part2 = cross(k, v) * sinTheta;
        //part3: k * (k . v) * (1 - cos(theta))
        const float3 part3 = k * dot(k, v) * (1 - cosTheta);

        return part1 + part2 + part3;
    }

    // ========== float3随机数生成函数 ==========

    //生成每个分量都在指定范围内的随机向量
    __device__ __forceinline__ static float3 randomVector(curandState * state, float componentMin = 0.0f, float componentMax = 1.0f) {
        return {
                randomDouble(state, componentMin, componentMax),
                randomDouble(state, componentMin, componentMax),
                randomDouble(state, componentMin, componentMax)
        };
    }

    //生成平面（x，y，0）上模长不大于指定长度的向量
    __device__ __forceinline__ static float3 randomPlaneVector(curandState * state, float maxLength = 1.0f) {
        float x, y;
        do {
            x = randomDouble(state, -1.0f, 1.0f);
            y = randomDouble(state, -1.0f, 1.0f);
        } while (x * x + y * y > maxLength * maxLength);
        return {x, y, 0.0f};
    }

    //生成模长为length的空间向量
    __device__ static float3 randomSpaceVector(curandState * state, float length = 1.0f) {
        //先生成单位向量，再缩放到指定模长
        float3 ret{};
        float lengthSquare;
        do {
            ret.x = randomDouble(state, -1.0f, 1.0f);
            ret.y = randomDouble(state, -1.0f, 1.0f);
            ret.z = randomDouble(state, -1.0f, 1.0f);
            lengthSquare = lengthSquared(ret);
        } while (lengthSquare < FLOAT_ZERO_VALUE * FLOAT_ZERO_VALUE);
        ret = normalize(ret);
        return ret * length;
    }

    //生成遵守按指定轴余弦分布的随机向量，非单位向量
    __device__ static float3 randomCosineVector(curandState * state, int axis, bool toPositive = true) {
        float coord[3];
        const auto r1 = randomDouble(state);
        const auto r2 = randomDouble(state);
        coord[0] = cos(2.0f * PI * r1) * 2.0f * sqrt(r2);
        coord[1] = sin(2.0f * PI * r1) * 2.0f * sqrt(r2);
        coord[2] = sqrt(1.0f - r2);
        switch (axis) {
            case 0: {
                const float tmp = coord[0]; coord[0] = coord[2]; coord[2] = tmp; break;
            }
            case 1: {
                const float tmp = coord[1]; coord[1] = coord[2]; coord[2] = tmp; break;
            }
            case 2:
            default:;
        }
        if (!toPositive) {
            coord[axis] = -coord[axis];
        }
        return {coord[0], coord[1], coord[2]};
    }
}

#endif //RENDEREROPTIX_DEVICEFUNCTIONS_CUH
