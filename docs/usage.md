# Usage Guide

## 1. Prepare VTK Data

The renderer requires the following input data:

- A .vtk.series sequence file
- A set of .vtk files, or a set of .cache cache files
- config.json configuration file

### VTK File Format

The VTK file format needs to match the file reading method in src/Util/VTKReaderImpl.cpp. If existing VTK files have a different format, VTKReaderImpl.cpp should be modified to match the current file format.

- If using Mesh mode, each VTK file needs to include:

1. A vtkPolyData containing a set of Cells, all Cell types are vtkTriangleStrip
2. vtkDataArray1: id -- integer array
3. vtkDataArray2: vel -- velocity array with each element containing three floats
4. Each Cell needs to contain a set of vertex coordinates and can be used by vtkPolyDataNormals to calculate vertex normals

- If using normal mode, each VTK file needs to include:
  
1. Center position, ID, rotation quaternion, velocity, and shape ID for each particle
2. A set of STL files storing geometry data. All STL files will be read and matched with particle shape ID properties in dictionary order of filenames. In this case, each VTK file does not need to contain particle geometry information

## 2. Configure Project

Edit the `files/config.json` file and set the following key parameters:

```json
{
  "mesh": true/false,
  "series-path": "",
  "series-name": "",
  "cache-path": "",
  "cache": true/false //Only needs to be set for mesh input
  "stl-path": ""      //Only needs to be set for non-mesh input
}
```

## 3. Run Program

- Mesh input:

1. Set the cache property in config.json to true to generate cache files
2. Change cache to false and run the program again to start loop rendering particle animation
3. For each frame, all particle positions are determined based on the currently loaded VTK file and particle velocity. The renderer will use velocity and frame count to calculate frame displacement

- Non-Mesh input:

1. Place VTK files, STL files, and Series files correctly
2. Run the executable directly

## 4. Interactive Controls

- **ESC**: Exit program

### Keyboard Control Camera Center Position

- **W**: Move forward
- **S**: Move backward
- **A**: Move left
- **D**: Move right
- **Space**: Move up
- **Left Shift**: Move down

### Mouse Control Camera Direction

- **Mouse Movement**: Rotate camera view
- **Mouse Wheel**: Adjust camera movement speed. Scroll up to increase speed, scroll down to decrease speed

### Additional Features

- The renderer integrates OptiX denoiser, which is enabled by default. You can hold Tab key to temporarily disable denoising and view the raw ray tracing rendering results

## 5. Notes

- Under Mesh input, when do you need to regenerate cache:

1. Any VTK file referenced by the series file has been modified
2. Operating system uses different binary memory layout, such as little-endian -- big-endian
3. Cache file is corrupted

- When regenerating cache, simply change the cache parameter to true and run the program. Original cache files will be automatically deleted
- Although reading from cache files does not require the VTK library, the executable has already linked VTK dynamic libraries. Missing VTK dynamic library files will prevent the program from running

## 6. Advanced Usage

### Custom Instance Updates

In `Main.cu`, you can customize the `updateInstancesTransforms` function to implement dynamic geometry transformations:

```cpp
static void updateInstancesTransforms(
    OptixInstance * pin_instances, size_t instanceCount, unsigned long long frameCount)
{
/*
  pin_instances: Instance array for the current frame, in page-locked memory
  instanceCount: Total number of instances, including custom added extra instances and all particles from the current VTK file, one instance per particle
  frameCount: Current frame count, can be used to construct animations
*/

    //Example: Sphere rotating around center position
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

    //Calculate transformation matrix and update pin_instances[i].transform
    float transform[12];
    MathHelper::constructTransformMatrix(
        newCenter,          //Translation
        {0.0f, 0.0f, 0.0f}, //Rotation
        {1.0f, 1.0f, 1.0f}, //Scale
        transform
    );

    //If this animation applies to the first custom instance
    memcpy(pin_instances[0].transform, transform, 12 * sizeof(float));

/*
Note:
  This function should only modify the instance transformation matrices of extra geometry. All extra geometry instances are stored starting from the first element of the instance array in the order of [Sphere -> Triangle], followed by particle instances. instanceCount is the total number of instances including all particle instances from the current file. You should record the number of custom instances and should not modify the transformation matrices of VTK particle instances within this function. Custom instances are arranged before all VTK particle instances, i.e., at the head of the instance array.
*/
}
```

- After modification, you need to recompile the project, but do not need to regenerate cache files (for Mesh input)