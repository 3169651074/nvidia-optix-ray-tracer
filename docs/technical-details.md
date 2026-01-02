# Technical Details

## Architecture Overview

RendererOptiX adopts a layered architecture design, from bottom to top including: OptiX rendering layer, graphics API abstraction layer, renderer interface layer, and application layer.

## OptiX Rendering Pipeline

### Context Initialization

The OptiX device context is the core of rendering, responsible for managing all OptiX resources:

```cpp
OptixDeviceContext createContext(const std::string & cacheFilePath, bool isDebugMode)
```

- Create OptiX context using CUDA device
- Configure cache path to accelerate shader compilation
- Optional validation mode

## Acceleration Structures

### Geometry Acceleration Structure (GAS)

GAS is used to accelerate queries for individual geometry types:

- **Sphere GAS**: Uses built-in sphere intersector
- **Triangle GAS**: Uses built-in triangle intersector
- **Particle GAS**: Build one triangle GAS per VTK particle

Build process:

1. Prepare geometry data (SOA format)
2. Set up build input
3. Calculate required memory
4. Allocate device memory
5. Build acceleration structure

### Instance Acceleration Structure (IAS)

IAS combines multiple GAS into a scene:

- One IAS per VTK file
- IAS contains additional geometry instances and all particle instances
- Supports dynamic update of instance transformation matrices

## Shader Modules

### Module Creation

The project uses three OptiX modules:

1. **Custom Module**: Loaded from PTX file, contains user-defined shaders
2. **Built-in Sphere Module**: OptiX built-in sphere intersector
3. **Built-in Triangle Module**: OptiX built-in triangle intersector

### Program Groups

Create six program groups:

- `raygen`: Ray generation program
- `miss`: Miss handler
- `chSphereRough`: Sphere rough material closest hit
- `chSphereMetal`: Sphere metal material closest hit
- `chTriangleRough`: Triangle rough material closest hit
- `chTriangleMetal`: Triangle metal material closest hit

## Shader Binding Table (SBT)

SBT associates program groups with scene geometry:

```text
SBT Structure:
├── RayGen Record (1)
├── Miss Record (1)
└── HitGroup Records (N)
    ├── Additional sphere records
    ├── Additional triangle records
    └── VTK particle records (one per particle)
```

Each record contains:

- Program group handle
- Parameter data (material index, geometry data pointers, etc.)

## Ray Tracing Flow

### RayGen Program

1. Calculate ray direction for pixel coordinates
2. Initialize ray payload (color, depth, etc.)
3. Call `optixTrace` to launch ray
4. Process returned results
5. Write to color buffer

### ClosestHit Program

1. Read geometry data
2. Calculate intersection normal
3. Select shading model based on material type
4. Recursively launch reflection/refraction rays
5. Return shading result

### Miss Program

Return background color

## Image Denoising

Use OptiX AI Denoiser (a convolutional neural network-based denoiser) to reduce ray tracing noise:

1. **Input Buffers**:
   - Color buffer (primary input)
   - Normal buffer (guide)
   - Albedo buffer (guide)

2. **Denoising Process**:
   - Allocate denoiser state and temporary buffers
   - Call `optixDenoiserInvoke`
   - Copy results to output buffer

3. **Output**: Denoised color buffer

## Memory Management

### Data Structures

#### SOA (Structure of Arrays) Layout

Geometry data is stored in SOA format to improve GPU access efficiency:

```cpp
// Sphere data
float3 * dev_centers;  // All sphere centers stored contiguously
float * dev_radii;     // All radii stored contiguously

// Triangle data
float3 * dev_vertices;  // All vertices stored contiguously (3*count)
float3 * dev_normals;   // All normals stored contiguously (3*count)
```

### Pinned Memory

Instance arrays use pinned memory:

- Accelerates CPU-GPU data transfer
- Supports asynchronous transfer
- Used for frequently updated data

### Memory Allocation Strategy

1. **Geometry Data**: Use regular device memory (`cudaMalloc`)
2. **Instance Arrays**: Use pinned memory (`cudaMallocHost`)
3. **Acceleration Structures**: Managed by OptiX
4. **Buffers**: Use CUDA arrays (interop with graphics API)

## Graphics API Integration

### CUDA-Graphics API Interop

Use CUDA graphics API interop to pass OptiX rendering results to graphics API, and use SDL to display rendering results:

#### OpenGL

- Register OpenGL texture
- Map resource
- Unmap after use

#### Vulkan

Use Vulkan external memory extension:

- Create Vulkan image supporting external memory
- Get memory handle
- Import memory in CUDA

#### Direct3D

Use Direct3D shared resources:

- Create shared texture
- Use `cudaGraphicsD3D11RegisterResource` or `cudaGraphicsD3D12RegisterResource`
- Map and unmap resources

### Window Management

Use SDL2 for window and input management:

- Create window and graphics context
- Handle keyboard and mouse input
- Manage event loop

## VTK Data Processing

### Series File Format

VTK series files are in JSON format:

```json
{
  "file-series-version": "1.0",
  "files": [
    {"name": "file0.vtk", "time": 0.0},
    {"name": "file1.vtk", "time": 0.1},
    ...
  ]
}
```

## Cache System

### Cache File Format

In Mesh mode, each VTK file corresponds to one cache file:

```text
particleXX.cache:
  [Total particles: size_t]
  Particle#0:
    [ID: size_t]
    [Velocity: float3]
    [Vertex count: size_t]
    [Vertex array: float3 * N]
    [Normal array: float3 * N]
  Particle#1:
    ...
```

### Cache Advantages

- Avoid repeated VTK file parsing
- Binary format, faster read speed
- Support multi-threaded parallel loading

### Data Conversion

VTK data is converted to renderer internal format:

1. **Read VTK File**: Parse using VTK library
2. **Extract Particle Data**: Get Cell data for each particle
3. **Convert to Triangles**: Convert Cell to triangle mesh
4. **Calculate Normals**: Compute normals for each triangle
5. **Store to Device Memory**: Copy to GPU

## Material System

### Material Types

#### Rough Material

Uses Lambertian diffuse reflection model:

- Albedo color determines diffuse color
- Fully diffuse, no specular reflection

#### Metal Material

Uses Cook-Torrance microfacet model:

- Albedo color determines metal color
- Fuzz parameter controls surface roughness
- Supports specular reflection

### Material Indexing

Material array organization:

```text
[Additional Rough Materials] [Additional Metal Materials] [VTK Rough Materials] [VTK Metal Materials]
```

VTK particle material indices need to add offset for additional materials.

## Performance Optimization

### Acceleration Structure Optimization

1. **GAS**:
   - `OPTIX_BUILD_FLAG_ALLOW_COMPACTION`: Compact acceleration structure to save VRAM
   - `OPTIX_BUILD_FLAG_PREFER_FAST_TRACE`: Optimize tracing performance

2. **IAS**:
   - `OPTIX_BUILD_FLAG_ALLOW_UPDATE`: Allow incremental updates
   - Build IAS only once per VTK file, update IAS between frames instead of rebuilding

### Memory Optimization

1. **Data Layout**: Use SOA to improve cache efficiency
2. **Memory Reuse**: Avoid frequent allocation/deallocation
3. **Pinned Memory**: Use pinned memory for host-side instance arrays to improve copy efficiency, can be extended to double buffering with CUDA streams

### Rendering Optimization

1. **Denoiser**: Eliminate ray tracing noise through denoising. Current implementation samples 1 ray per pixel
2. **Recursion Depth Limit**: Limit ray recursion depth (currently 5)
3. **Adaptive Sampling**: Can implement adaptive sampling as needed

## Debug Support

### OptiX Validation Mode

Enabling validation mode can detect:

- Invalid acceleration structures
- Shader parameter errors
- Memory access violations

### Graphics API Debugging

- OpenGL: Use `glDebugMessageCallback`
- Vulkan: Enable validation layers
- Direct3D: Use debug device

When debug mode is enabled, debug information is printed to console

### Logging System

Use SDL logging system to output renderer information, including:

- Configuration parsing
- Data loading progress
- Error messages

## Limitations

1. **Single GPU**: Multi-GPU rendering not supported
2. **Simplified Material Model**: Uses simplified physically-based material models
3. **No Volume Rendering**: Volume data rendering not supported

## Future Improvement Directions

1. **More Complex Materials**: Implement more physically-based material models and light sources
2. **Importance Sampling**: Importance sampling for light sources to improve image quality
3. **Volume Rendering**: Support ray tracing for volume data
4. **Multi-GPU Support**: Distributed rendering, tile-based rendering
5. **Object Selection**: Support particle selection in rendering to view real-time information
6. **Real-time Editing**: 3D editor supporting runtime scene modification