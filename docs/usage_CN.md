# 使用指南

## 1. 准备 VTK 数据

渲染器需要以下输入数据：

- 一个.vtk.series序列文件
- 一组.vtk文件，或一组.cache缓存文件
- config.json配置文件

### VTK文件格式

VTK文件格式需要匹配src/Util/VTKReaderImpl.cpp中的文件读取方式。若已有VTK文件格式和以下格式不同，则应修改VTKReaderImpl.cpp以匹配当前文件格式

- 若使用Mesh模式，每个VTK文件需要包括：

1. 一个包含一组Cell的vtkPolyData，全部Cell类型为 vtkTriangleStrip
2. vtkDataArray1：id  --整数数组
3. vtkDataArray2：vel --每个元素包含三个浮点数的速度数组
4. 每个Cell需要包含一组顶点坐标，并能被vtkPolyDataNormals用于计算顶点法线

- 若使用普通模式，每个VTK文件需要包括：
  
1. 每个粒子的中心位置，ID，旋转四元数，速度，形状ID
2. 一组存储几何数据的STL文件，所有STL文件都将被读取，并按文件名的字典顺序和粒子的形状ID属性进行匹配，此时每个VTK文件无需包含粒子几何信息

## 2. 配置项目

编辑 `files/config.json` 文件，设置以下关键参数：

```json
{
  "mesh": true/false,
  "series-path": "",
  "series-name": "",
  "cache-path": "",
  "cache": true/false //仅在mesh输入时需要设置
  "stl-path": ""      //仅在非mesh输入时需要设置
}
```

## 3. 运行程序

- Mesh输入：

1. 将config.json中的cache属性设置为true，生成缓存文件
2. 将cache改为false，再次运行程序，开始循环渲染粒子动画
3. 每一帧，所有粒子的位置根据当前加载的VTK文件和粒子的速度确定，渲染器会使用速度和帧数量计算帧位移

- 非Mesh输入：

1. 正确放置VTK文件、STL文件和Series文件
2. 直接运行可执行程序即可

## 4. 交互控制

- **ESC**：退出程序

### 键盘控制相机中心位置

- **W**：向前移动
- **S**：向后移动
- **A**：向左移动
- **D**：向右移动
- **Space**：向上移动
- **Left Shift**：向下移动

### 鼠标控制相机朝向

- **鼠标移动**：旋转相机视角
- **鼠标滚轮**：调节相机移动速度。向上滚动提升速度，向下滚动降低速度

### 额外功能

- 渲染器集成了OptiX降噪器，默认开启降噪，可以按住Tab键以临时禁用降噪，查看光线追踪的原始渲染结果

## 5. 注意事项

- Mesh输入下，什么时候需要重新生成缓存

1. series文件引用的任何一个VTK文件被修改
2. 操作系统使用不同的二进制内存布局，如小端序 -- 大端序
3. 缓存文件损坏

- 重新生成缓存时，只需要将cache参数改为true并运行程序即可，原有缓存文件会被自动删除
- 虽然从缓存文件中读取无需使用到VTK库，但是可执行文件已经链接了VTK动态库，缺少VTK动态库文件将导致程序无法运行

## 6. 高级用法

### 自定义实例更新

在 `Main.cu` 中，可以自定义 `updateInstancesTransforms` 函数来实现动态几何体变换：

```cpp
static void updateInstancesTransforms(
    OptixInstance * pin_instances, size_t instanceCount, unsigned long long frameCount)
{
/*
  pin_instances：当前帧的实例数组，在页面锁定内存中
  instanceCount：实例总数，包括自定义添加的额外实例和当前VTK文件的所有粒子，每个粒子一个实例
  frameCount：当前帧计数，可以使用此参数构造动画
*/

    //示例：球体整体绕中心位置转动
    static const Point3 initialCenter = {0.0, 2.0, 0.0};
    const float radius = 2.0f;
    const float speed = 0.02f;
    const float angle = static_cast<float>(frameCount) * speed;
    const float3 newCenter = {
            initialCenter[0] + radius * std::cos(angle) * 1.5f,
            initialCenter[1] + radius * std::sin(angle) * std::cos(angle),
            initialCenter[2] + radius * std::sin(angle) * 1.5f
    };
    const float3 newCenter2 = {-newCenter.x, newCenter.y,-newCenter.z};
    const float3 newCenter3 = {-newCenter.x, newCenter.y + 5.0f, newCenter.z};

    //计算变换矩阵并更新 pin_instances[i].transform
    float transform[12];
    MathHelper::constructTransformMatrix(
        newCenter,          //位移
        {0.0f, 0.0f, 0.0f}, //旋转
        {1.0f, 1.0f, 1.0f}, //缩放
        transform
    );

    //若此动画应用于第一个自定义实例
    memcpy(pin_instances[0].transform, transform, 12 * sizeof(float));

/*
注意：
  此函数中应只修改额外几何体的实例变换矩阵，所有额外几何体实例按照
 【球体 -> 三角形】的顺序从实例数组收个元素开始向后存放，instanceCount为当前文件包含所有粒子实例的实例总数，应当记录自定义实例个数，且不应该在此函数内修改VTK粒子实例的变换矩阵，自定义实例排列在所有VTK粒子实例之前，即实例数组的头部
*/
}
```

- 修改完成后，需要重新编译项目，但无需重新生成缓存文件（Mesh输入）
