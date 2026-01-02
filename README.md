[简体中文](./README_CN.md)

# RendererOptiX

A VTK renderer based on NVIDIA OptiX that supports real-time rendering of VTK particle data in specific formats, with support for OpenGL, Vulkan, and Direct3D (Windows) rendering.

## Project Introduction

RendererOptiX is a high-performance real-time renderer that uses NVIDIA OptiX 9.0 for ray tracing rendering. This project can read and render particle data in specific VTK formats and generate cache files. It supports rough and metal material types, and provides interactive keyboard and mouse controls.

**This project serves as a relatively basic implementation using OptiX and can be used as an introductory example for the NVIDIA OptiX library, providing reference for GPU ray tracer and OptiX API learners.**

**A set of VTK particle examples is provided in the files folder in the repository, which can be used to verify the renderer's functionality**

## Features

- **High-performance ray tracing**: Based on NVIDIA OptiX 9.0
- **VTK data support**: Supports reading and rendering VTK particle sequence files in specific formats
- **Multiple materials**: Supports rough and metal materials
- **Multiple graphics APIs**: Supports OpenGL, Vulkan, Direct3D11, Direct3D12
- **Interactive controls**: Supports mouse and keyboard camera controls
- **Cache system**: Supports VTK data caching to accelerate loading
- **Configurable rendering**: Flexible rendering parameter configuration via JSON configuration files

## Documentation Index

- [Project Introduction](docs/project-introduction.md) - Project overview, architecture design, and core concepts
- [Build Guide](docs/build-guide.md) - Build environment requirements, build steps, and dependency configuration
- [Configuration Reference](docs/configuration.md) - Configuration file format and parameter descriptions
- [Usage Guide](docs/usage.md) - Configuration instructions and usage examples
- [Technical Details](docs/technical-details.md) - Technical architecture, implementation principles, and performance optimization

## Notes

- Since this renderer relies on libraries with large file sizes, and the VTK library it depends on does not provide precompiled versions officially, the project repository does not include dependency library files. Before running the renderer, all dependency libraries must be satisfied and the project must be compiled from source.

- When regeneration requirements are met, shader PTX files need to be recompiled or caches need to be regenerated, otherwise the renderer will not work properly. Please refer to the corresponding sections in the documentation for specific requirements.