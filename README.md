LightweightVK [![Build Status](https://github.com/corporateshark/lightweightvk/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/corporateshark/lightweightvk/actions)
========================

LightweightVK is a deeply refactored **bindless-only** fork of [IGL](https://github.com/facebook/igl) which is designed to run on top of **Vulkan 1.3** with optional **mesh shaders** and **ray tracing** support.

The main goals of LightweightVK:

1. **Lean.** Minimalistic API without bloat (no `std::vector`, `std::unordered_map` etc in the API).
2. **Bindless.** Utilize Vulkan 1.3+ dynamic rendering, descriptor indexing, and buffer device address features for modern bindless-only API design. Ray tracing features are fully integrated with the bindless-only design.
3. **Agile.** A playground for experiments to enable quick exploration of ideas and adoption of Vulkan API changes.
4. **Multilingual.** In addition to raw SPIR-V, there's built-in support for `Glslang` (GLSL) and `Slang` shaders, which is invaluable for rapid prototyping.

Designed for rapid prototyping of modern Vulkan-based renderers.

There are **no plans to keep this fork in sync with the IGL upstream**, since the **API was completely redesigned** in a *bindless* manner.

Discord: https://discord.com/invite/bEyHyKCrvq

## Supported rendering backends

 * Vulkan 1.3 (Windows, Linux, MacOS, Android)
   * mandatory **VK_KHR_dynamic_rendering_local_read**
   * mandatory **VK_KHR_maintenance5**
   * mandatory **VK_KHR_push_descriptor**
   * mandatory **VK_EXT_surface_maintenance1**
   * optional **VK_KHR_maintenance6**
   * optional **VK_KHR_acceleration_structure**
   * optional **VK_KHR_ray_tracing_pipeline**
   * optional **VK_KHR_ray_query**
   * optional **VK_KHR_fragment_shading_rate**
   * optional **VK_EXT_fragment_density_map**
   * optional **VK_EXT_fragment_density_map2**
   * optional **VK_EXT_dynamic_rendering_unused_attachments**
   * optional **VK_EXT_host_image_copy**
   * optional **VK_EXT_layer_settings**
   * optional **VK_EXT_mesh_shader**
   * optional **VK_ARM_tensors**
   * optional **VK_ARM_data_graph**

## Supported platforms

 * Linux
 * Windows
 * MacOS (via KosmicKrisp)
 * Android

## API Support

|                               | Windows                    | Linux                      | MacOS                      | Android                    |
| ----------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| Vulkan 1.3                    | :heavy_check_mark:         | :heavy_check_mark:         | :heavy_check_mark:         | :heavy_check_mark:         |
| VK_KHR_acceleration_structure | :heavy_check_mark:         | :heavy_check_mark:         |                            | :heavy_check_mark:         |
| VK_KHR_ray_tracing_pipeline   | :heavy_check_mark:         | :heavy_check_mark:         |                            | :heavy_check_mark:         |
| VK_KHR_ray_query              | :heavy_check_mark:         | :heavy_check_mark:         |                            | :heavy_check_mark:         |
| VK_EXT_mesh_shader            | :heavy_check_mark:         | :heavy_check_mark:         |                            | :heavy_check_mark:         |
| OpenXR 1.1                    | :heavy_check_mark:         |                            |                            |                            |

On MacOS, `KosmicKrisp` and `VulkanSDK 1.4.363+` are required.

## Build

Before building, run the deployment scripts:

```
python3 deploy_deps.py
python3 deploy_content.py
```

These scripts download external third-party dependencies. Please check [LICENSE.md](./LICENSE.md) for the full list.

### Windows

```
cd build
cmake .. -G "Visual Studio 18 2026"
```

### Linux

```
sudo apt-get install clang xorg-dev libxinerama-dev libxcursor-dev libgles2-mesa-dev libegl1-mesa-dev libglfw3-dev libglew-dev libstdc++-12-dev extra-cmake-modules libxkbcommon-x11-dev wayland-protocols
cd build
cmake .. -G "Unix Makefiles"
```

:heavy_exclamation_mark: Use `cmake .. -G "Unix Makefiles" -DLVK_WITH_WAYLAND=ON` to build for Wayland, X11 is used by default.

### MacOS

:heavy_exclamation_mark: Be sure that `VulkanSDK 1.4.363+` for MacOS is installed https://vulkan.lunarg.com/sdk/home#mac

```
cd build
cmake .. -G "Xcode"
```

### Android

:heavy_exclamation_mark: Be sure that [Android Studio](https://developer.android.com/studio) is set up.

:heavy_exclamation_mark: Be sure that the `ANDROID_NDK` environment variable points to your Android NDK.

:heavy_exclamation_mark: Be sure that the `JAVA_HOME` environment variable is set to the path of the Java Runtime.

:heavy_exclamation_mark: Be sure that the `adb` platform tool is in the `PATH` environment variable.

```
cd build
cmake .. -DLVK_WITH_SAMPLES_ANDROID=ON
cd android/001_HelloTriangle            # or any other sample
./gradlew assembleDebug                 # or assembleRelease
```
You can also open the project in Android Studio and build it from there.

Before running demo apps on your device, connect the device to a desktop machine and run the deployment script:

```
python3 deploy_content_android.py
```

> NOTE: To run demos on an Android device, it should support Vulkan 1.3. Please check https://vulkan.gpuinfo.org/listdevices.php?platform=android

> NOTE: At the moment, demo apps do not support touch input on Android.

## Using the Slang compiler

By default, the [Slang](https://github.com/shader-slang/slang) compiler is disabled,
and only GLSL shaders are supported through the [glslang](https://github.com/KhronosGroup/glslang) compiler.
To enable [Slang](https://github.com/shader-slang/slang), configure the project with the following CMake option:

```
cmake .. -DLVK_WITH_SLANG=ON
```

## Arm tensors and data graphs (neural graphics)

LightweightVK exposes [VK_ARM_tensors](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_ARM_tensors.html)
and [VK_ARM_data_graph](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_ARM_data_graph.html) in the same bindless
manner as textures: tensors are accessed from shaders through the `kTensors<Type>_<Rank>[]` arrays (descriptor set 0, binding 5)
using `tensorReadARM()`/`tensorWriteARM()`, and neural networks are run as data graph pipelines created from SPIR-V graph modules,
for example loaded from [VGF](https://github.com/arm/ai-ml-sdk-vgf-library) files.

```c
lvk::Holder<lvk::TensorHandle> tensor = ctx_->createTensor({.format = lvk::Format_R_I8, .rank = 4, .dimensions = {1, 544, 960, 12}, .usage = lvk::TensorUsageBits_Shader | lvk::TensorUsageBits_DataGraph});
...
buffer.cmdDispatch({w, h, 1}, {.tensors = {tensor}}); // GLSL: tensorWriteARM(kTensorsI8_4[pc.tensor], coords, value)
buffer.cmdDispatchDataGraph(graphPipeline, {tensor}, {output}); // the tensors the pipeline was created against
```

The sample `DEMO_003_NeuralSuperSampling` runs [Arm Neural Super Sampling](https://huggingface.co/Arm/neural-super-sampling)
on the Bistro scene, 960x540 -> 1920x1080 by default and 1920x1080 -> 3840x2160 with `--4k`. GPUs without native support
can use the [Arm ML Emulation Layer for Vulkan](https://github.com/arm/ai-ml-emulation-layer-for-vulkan); LightweightVK
builds a [fork](https://github.com/rokuz/ai-ml-emulation-layer-for-vulkan) of it carrying runtime performance work, which
is downloaded and built from source (no global installation) with the following CMake option:

```
cmake .. -DLVK_WITH_ML_EMULATION_LAYER=ON
```

`DEMO_003_NeuralSuperSampling` enables the layer whenever the GPU lacks `VK_ARM_tensors`; run it with `--no-ml-emulation` to
disable it.
`lvk/HelpersVgf.h` loads VGF models (`lvk::VgfModel`) and creates their data graph pipelines; it is built when `LVK_WITH_TENSORS` is
enabled (the default).

## Screenshots

Check out [https://github.com/corporateshark/lightweightvk/samples](https://github.com/corporateshark/lightweightvk/tree/master/samples).

A comprehensive set of examples can be found in this repository [3D Graphics Rendering Cookbook: 2nd edition](https://github.com/PacktPublishing/3D-Graphics-Rendering-Cookbook-Second-Edition) and in the book
[Vulkan 3D Graphics Rendering Cookbook - 2nd Edition](https://www.amazon.com/Vulkan-Graphics-Rendering-Cookbook-High-Performance/dp/1803248114).

[![Vulkan 3D Graphics Rendering Cookbook](.github/screenshot01.jpg)](https://github.com/PacktPublishing/3D-Graphics-Rendering-Cookbook-Second-Edition/tree/main/Chapter11/06_FinalDemo/src)
![image](.github/samples/007_RayTracingAO.jpg)
[![Solar System Demo](.github/screenshot02.jpg)](https://github.com/corporateshark/lightweightvk/blob/master/samples/DEMO_001_SolarSystem.cpp)

## Interop with raw Vulkan API calls

The header file `lvk/vulkan/VulkanUtils.h` offers a collection of functions that allow you to access the underlying Vulkan API objects from LightweightVK handles. This makes it easy to mix LVK and Vulkan code, as shown in the following example:

```c
lvk::Holder<lvk::BufferHandle> vertexBuffer = ctx_->createBuffer({...});
...
lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();
VkCommandBuffer cmdBuf = getVkCommandBuffer(buffer);
VkBuffer buf = getVkBuffer(vertexBuffer);
vkCmdUpdateBuffer(cmdBuf, buf, 0, sizeof(params), &params);
ctx_->submit(buffer, ctx_->getCurrentSwapchainTexture());
```

If you'd like to add more helper functions, feel free to submit a pull request.

## License

LightweightVK is released under the MIT license, see [LICENSE.md](./LICENSE.md) for the full text as well as third-party library
acknowledgements.
