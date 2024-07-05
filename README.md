# Galitef

- Vulkan-based glTF renderer with full PBR capabilities.
- On-the-fly cubemap generation and usage
- Compliant with glTF 2.0 PBR standards

## Libraries Used

- glfw
- stb_image
- fastgltf (glTF parsing)

## Usage

Launch through command line with first two arguments specifying the viewport width nad viewport height, respectively. Followed by relative model and relative HDRI paths.

## In-viewport Usage Commands

- WASD for flycam movement along with Q and E for vertical control
- Mouse for camera movement
- Esc to exit

## Example

```
galitef.exe 1280 720 ./models/WaterBottle.glb ./hdri/metro_noord_4k.hdr
```

## To-do

- Make renderer fully compliant with glTF standards
- Add pre-made cubemap usage functionality
- Create solution for dealing with models where the normal map is accompanied by no tangents