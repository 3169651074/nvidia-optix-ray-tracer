# Configuration Reference

This document details all parameters of the `config.json` configuration file.

## Configuration File Location

Default configuration file path: `files/config.json`

You can modify the `CONFIG_FILE_PATH` constant in `include/Util/ProgramArgumentParser.cuh` to change the path.

## File Path Configuration

```json
// Whether this is mesh input: true when each VTK file contains complete particle geometry
"mesh": false,
// .vtk.series file path
"series-path": "../files/",
// .vtk.series file name
"series-name": "particle.vtk.series",
// Path for generated cache files (OptiX cache and data cache in mesh mode)
"cache-path": "../cache/",
// Path to STL files for non-mesh input
"stl-path": "../files/shape/separated/",
```

**Note**:

- Paths can be relative (to the executable) or absolute
- On Linux, if using MangoHud for OSD display, the executable path changes and absolute paths should be used
- Use `/` as path separator (supported on both Windows and Linux)

## Cache Configuration

```json
// Whether to generate cache files and exit the program
// true: Program reads VTK files on startup, generates cache files, then exits
// false: Reads already generated cache files and starts rendering
"cache": false
// Number of CPU threads used for reading/writing cache files, adjust according to your CPU model
"cache-process-thread-count": 8
```

## Debug Configuration

```json
// Whether to enable debug mode for OptiX and graphics API
// Enabling this reduces performance, only set to true during debugging and development
"debug-mode": false
```

## Material Configuration

### Particle Material Color Presets

**Type**: string  
**Description**: Used to generate color ramps for particle materials  
**Values**: Choose any one from the following values, case-insensitive
```json
"viridis" "plasma" "spectral"
"terrain" "heatmap" "grayscale"
```
For other color mappings, refer to the implementation in `include/Util/ColorRamp.cuh` and add your own color mapping enumerations

### roughs

**Type**: `array of objects`  
**Description**: Rough material array  
**Structure**:

```json
"roughs": [
  {
    "albedo": [r, g, b]
  }
]
```

#### albedo

**Type**: `array of float` (3 elements)  
**Description**: Albedo color (RGB)  
**Range**: `[0.0, 1.0]`  
**Example**:

```json
"roughs": [
  {"albedo": [0.65, 0.05, 0.05]},  // Red
  {"albedo": [0.73, 0.73, 0.73]},  // Gray
  {"albedo": [0.12, 0.45, 0.15]},  // Green
  {"albedo": [0.70, 0.60, 0.50]}   // Beige
]
```

### metals

**Type**: `array of objects`  
**Description**: Metal material array  
**Structure**:

```json
"metals": [
  {
    "albedo": [r, g, b],
    "fuzz": float
  }
]
```

#### albedo

**Type**: `array of float` (3 elements)  
**Description**: Metal color (RGB)  
**Range**: `[0.0, 1.0]`  
**Example**:

```json
{"albedo": [0.8, 0.85, 0.88]}
```

#### fuzz

**Type**: `float`  
**Description**: Surface roughness  
**Range**: `[0.0, 1.0]`  
**Description**:

- `0.0`: Perfect specular reflection
- Higher values mean rougher surface

**Example**:

```json
{"albedo": [0.8, 0.85, 0.88], "fuzz": 0.0}  // Specular
{"albedo": [0.7, 0.6, 0.5], "fuzz": 0.3}    // Rough metal
```

## Geometry Configuration

### spheres

**Type**: `array of objects`  
**Description**: Sphere array  
**Structure**:

```json
"spheres": [
  {
    "center": [x, y, z],
    "radius": float,
    "mat-type": "ROUGH" | "METAL",
    "mat-index": integer,
    "shift": [x, y, z],
    "rotate": [x, y, z],
    "scale": [x, y, z]
  }
]
```

#### center

**Type**: `array of float` (3 elements)  
**Description**: Sphere center position  
**Example**:

```json
"center": [0.0, 0.0, 0.0]
```

#### radius

**Type**: `float`  
**Description**: Sphere radius  
**Example**:

```json
"radius": 1000.0
```

#### mat-type

**Type**: `string`  
**Description**: Material type  
**Values**: `"ROUGH"` or `"METAL"`  
**Example**:

```json
"mat-type": "ROUGH"
```

#### mat-index

**Type**: `integer`  
**Description**: Material index (corresponding to index in `roughs` or `metals` array)  
**Note**: Index starts from 0  
**Example**:

```json
"mat-index": 3
```

#### shift

**Type**: `array of float` (3 elements)  
**Description**: Translation vector  
**Example**:

```json
"shift": [0.0, 0.0, -1000.5]
```

#### rotate

**Type**: `array of float` (3 elements)  
**Description**: Rotation angles (degrees)  
**Note**: Current implementation may not fully support this, recommend using `[0.0, 0.0, 0.0]`  
**Example**:

```json
"rotate": [0.0, 0.0, 0.0]
```

#### scale

**Type**: `array of float` (3 elements)  
**Description**: Scale factors  
**Example**:

```json
"scale": [1.0, 1.0, 1.0]
```

### triangles

**Type**: `array of objects`  
**Description**: Triangle array (currently may not be fully implemented)  
**Structure**:

```json
"triangles": [
  {
    "vertices": [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]],
    "normals": [[nx1, ny1, nz1], [nx2, ny2, nz2], [nx3, ny3, nz3]],
    "mat-type": "ROUGH" | "METAL",
    "mat-index": integer
  }
]
```

## Render Loop Configuration (loop-data)

### api

**Type**: `string`  
**Description**: Graphics API type  
**Values**:

- `"OGL"`: OpenGL
- `"VK"`: Vulkan
- `"D3D11"`: Direct3D 11 (Windows only)
- `"D3D12"`: Direct3D 12 (Windows only)

**Example**:

```json
"api": "VK"
```

### window-width

**Type**: `integer`  
**Description**: Window width (pixels)  
**Example**:

```json
"window-width": 1200
```

### window-height

**Type**: `integer`  
**Description**: Window height (pixels)  
**Example**:

```json
"window-height": 800
```

### fps

**Type**: `integer`  
**Description**: Target frame rate  
**Example**:

```json
"fps": 60
```

### camera-center

**Type**: `array of float` (3 elements)  
**Description**: Initial camera position  
**Example**:

```json
"camera-center": [5.0, 0.0, 0.0]
```

### camera-target

**Type**: `array of float` (3 elements)  
**Description**: Initial camera target point (look-at position)  
**Example**:

```json
"camera-target": [0.0, 0.0, 0.0]
```

### up-direction

**Type**: `array of float` (3 elements)  
**Description**: Camera up vector (no need to normalize)  
**Example**:

```json
"up-direction": [0.0, 0.0, 1.0]
```

### camera-pitch-limit-degree

**Type**: `float`  
**Description**: Camera pitch angle limit (degrees)  
**Range**: Should be less than 90.0  
**Default**: `85.0`  
**Example**:

```json
"camera-pitch-limit-degree": 85.0
```

### camera-speed-stride

**Type**: `float`  
**Description**: Camera speed change amount adjusted by mouse wheel  
**Default**: `0.002`  
**Example**:

```json
"camera-speed-stride": 0.002
```

### camera-initial-speed-ratio

**Type**: `integer`  
**Description**: Multiplier for initial camera speed relative to `camera-speed-stride`  
**Default**: `10`  
**Example**:

```json
"camera-initial-speed-ratio": 10
```

### mouse-sensitivity

**Type**: `float`  
**Description**: Mouse sensitivity  
**Default**: `0.002`  
**Example**:

```json
"mouse-sensitivity": 0.002
```

### render-speed-ratio

**Type**: `integer`  
**Description**: Particle motion speed multiplier  
**Description**:

- Higher values mean slower particle motion
- `1` means original speed
- Used to control animation playback speed

**Example**:

```json
"render-speed-ratio": 4
```

### particle-shift

**Type**: `array of float` (3 elements)  
**Description**: Global translation for all particles  
**Example**:

```json
"particle-shift": [0.0, 0.0, 0.0]
```

### particle-scale

**Type**: `array of float` (3 elements)  
**Description**: Global scale for all particles  
**Example**:

```json
"particle-scale": [1.0, 1.0, 1.0]
```

## Complete Configuration Example

```json
{
  "series-path": "../files/",
  "series-name": "particle_mesh-short.vtk.series",
  "cache-path": "../cache/",
  "cache": false,
  "debug-mode": false,
  "cache-process-thread-count": 8,
  "roughs": [
    {"albedo": [0.65, 0.05, 0.05]},
    {"albedo": [0.73, 0.73, 0.73]},
    {"albedo": [0.12, 0.45, 0.15]},
    {"albedo": [0.70, 0.60, 0.50]}
  ],
  "metals": [
    {"albedo": [0.8, 0.85, 0.88], "fuzz": 0.0}
  ],
  "spheres": [
    {
      "center": [0.0, 0.0, 0.0],
      "radius": 1000.0,
      "mat-type": "ROUGH",
      "mat-index": 3,
      "shift": [0.0, 0.0, -1000.5],
      "rotate": [0.0, 0.0, 0.0],
      "scale": [1.0, 1.0, 1.0]
    }
  ],
  "triangles": [],
  "loop-data": {
    "api": "VK",
    "window-width": 1200,
    "window-height": 800,
    "fps": 60,
    "camera-center": [5.0, 0.0, 0.0],
    "camera-target": [0.0, 0.0, 0.0],
    "up-direction": [0.0, 0.0, 1.0],
    "camera-pitch-limit-degree": 85.0,
    "camera-speed-stride": 0.002,
    "camera-initial-speed-ratio": 10,
    "mouse-sensitivity": 0.002,
    "render-speed-ratio": 4,
    "particle-shift": [0.0, 0.0, 0.0],
    "particle-scale": [1.0, 1.0, 1.0]
  }
}
```

## Configuration Validation

The program validates the configuration file on startup:

- JSON format check
- Field check
- Type check
- Platform compatibility check (e.g., D3D11/D3D12 only available on Windows)
- Due to CUDA calls in the project, D3D rendering cannot be used on Linux through translation tools like Proton

If there are configuration errors, the program will output error messages and exit. Check the console for logs and error information.

## Next Steps

- [Usage Guide](usage.md) - Usage examples
- [Technical Details](technical-details.md) - Implementation principles