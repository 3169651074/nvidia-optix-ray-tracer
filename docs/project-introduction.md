# Project Introduction

## Overview

RendererOptiX is a high-performance real-time renderer based on NVIDIA OptiX 9.0, specifically designed for rendering VTK (Visualization Toolkit) format particle data. This project combines modern ray tracing technology with multiple graphics APIs to provide a flexible and efficient rendering solution.

## Project Goals

- **High-Performance Rendering**: Achieve real-time rendering using GPU ray tracing technology
- **Ease of Use**: Unified rendering parameter configuration through JSON files
- **Cross-Platform Support**: Support for Windows and Linux platforms
- **Multi-API Compatibility**: Support for displaying rendered images using OpenGL, Vulkan, Direct3D11, and Direct3D12
- **Extensibility**: Modular design for easy extension of new features

## Core Features

### 1. OptiX Ray Tracing

The project uses NVIDIA OptiX 9.0 as the core rendering engine, implementing:

- Real-time ray tracing
- Multiple geometry types support (spheres, triangles, VTK particles)
- Material system (rough materials and metallic materials)
- Denoising processing with integrated OptiX AI Denoiser

### 2. VTK Data Support

- Support for reading VTK series files (`.vtk.series`)
- Automatic parsing of particle geometry data
- Support for particle caching to accelerate loading
- Multi-threaded cache file reading and writing

### 3. Graphics APIs

The project implements interoperability between modern graphics APIs and CUDA, supporting:

- **OpenGL**: Cross-platform standard graphics API
- **Vulkan**: Modern low-level graphics API
- **Direct3D11**: Windows platform (Windows only)
- **Direct3D12**: Windows platform modern API (Windows only)

### 4. Interactive Controls

- Mouse control for camera rotation
- Keyboard control for camera movement (WASD movement + Space/Left Shift for up/down)
- Mouse wheel to adjust movement speed (scroll up to accelerate movement)
- Left-click to release mouse for free movement, click again to return to window
- Configurable camera parameters

## Architecture Design

### Overall Architecture

```text
┌─────────────────────────────────────────┐
│     Application Entry (Main.cu)        │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Renderer Interface Layer               │
│  (RendererTime/RendererMesh)           │
│  - Data submission and initialization   │
│  - Render loop control                  │
│  - Resource management                  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     Renderer Implementation Layer       │
│     (RendererImpl.cuh)                 │
│  - OptiX context management            │
│  - Acceleration structure building      │
│    (GAS/IAS)                           │
│  - Shader Binding Table (SBT)          │
│  - Denoiser management                 │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     Graphics API Layer                  │
│     (SDL_GraphicsWindow)               │
│  - Window management                   │
│  - Graphics resource management        │
│  - CUDA-SDL-Graphics API interop       │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│     Utility Layer (Util)               │
│  - VTK file reading                    │
│  - JSON configuration parsing          │
│  - Color mapping and particle material │
│    generation                          │
└─────────────────────────────────────────┘
```

### Core Modules

#### 1. Renderer Core (Renderer)

The `Renderer` class provides the main interface for the renderer:

- `commitRendererData()`: Submit geometry and material data, initialize renderer
- `startRender()`: Start render loop
- `freeRendererData()`: Free rendering resources

#### 2. OptiX Implementation (RendererImpl)

Implements OptiX-related low-level operations:

- **Context Management**: Create and destroy OptiX device context
- **Acceleration Structures**: Build Geometry Acceleration Structure (GAS) and Instance Acceleration Structure (IAS)
- **Shader Modules**: Load and compile PTX shaders
- **Program Groups**: Create ray generation, miss, and closest hit program groups
- **Denoiser**: Initialize and manage OptiX denoiser

#### 3. Graphics API Wrapper

`SDL_GraphicsWindow` provides a unified graphics API interface:

- Window creation and management
- Graphics resource (textures, buffers) management
- CUDA and graphics API interoperability
- Input event handling

#### 4. VTK Reader (VTKReader)

Responsible for reading and processing VTK data:

- Parse VTK series files
- Read particle geometry data
- Cache management (read/write cache files)
- Multi-threaded data loading

## Data Flow

### Rendering Pipeline

1. **Initialization Phase**
   - Parse JSON configuration file
   - Read VTK data or load cache
   - Build geometry and material data
   - Initialize OptiX context and pipeline

2. **Per-Frame Rendering**
   - Update camera parameters
   - Update instance transformation matrices
   - Execute OptiX ray tracing
   - Apply denoising
   - Copy results to graphics API resources
   - Present to screen

3. **Cleanup Phase**
   - Free device memory
   - Destroy acceleration structures
   - Close OptiX context
   - Destroy graphics API resources

### Memory Management

- **Host Memory**: Store configuration data and metadata
- **Device Memory**: Store geometry data, material data, acceleration structures
- **Pinned Memory**: Used for instance arrays, accelerating CPU-GPU transfers
- **CUDA Arrays**: Used for sharing rendering results with graphics APIs

## Technology Stack

- **CUDA**: GPU computation and memory management
- **OptiX 9.0**: Ray tracing engine
- **VTK 9.5**: VTK file reading
- **SDL2**: Window and input management
- **Vulkan/DirectX/OpenGL**: Graphics rendering backends
- **nlohmann/json**: JSON configuration parsing
- **CMake**: Build system

## Application Scenarios

- **Scientific Visualization**: Render large-scale particle data
- **Data Exploration**: Interactively view VTK datasets
- **Rendering Research**: Learn and experiment with ray tracing technology
- **Performance Testing**: GPU ray tracing performance evaluation

## Future Plans

- Support for more geometry types
- Implement more complex material models
- Add volume rendering support
- Optimize large-scale scene rendering performance
- Support multi-GPU rendering
- Add post-processing effects