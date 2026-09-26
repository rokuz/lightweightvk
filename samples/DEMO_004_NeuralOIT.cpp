/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 Order-independent transparency on a fully transparent Bistro:

   A-buffer             the reference: the K nearest fragments of a pixel, kept sorted by a lock-free insertion and blended front
                        to back; the fragments pushed out of the list are tail-blended behind them
   Weighted blended     "Weighted Blended Order-Independent Transparency" (McGuire and Bavoil, JCGT 2013): one geometry pass
                        accumulates every fragment weighted by a function of its depth and its alpha, and the pixel is the
                        weighted average revealed against the background. `--weight-distance` is the distance at which the
                        depth term of the weight reaches 0.03; it has to be tuned per scene, and 250 is the value that
                        minimizes the error on Bistro at this scale
   DFAOIT               "Deep and Fast Approximate Order Independent Transparency" (Tsopouridis et al., CGF 2024): per-pixel
                        features from one geometry pass, the colour of the pixel predicted from them by a small neural network
                        dispatched as a VK_ARM_data_graph pipeline over tensors. The k nearest fragments of a pixel are kept
                        exactly and composited analytically, and the network only predicts what is behind them; k is the
                        paper's DFAOIT_k, and the sample reads it out of the network as `inputs - 10` rather than being told.
                        A pixel with no more than k fragments therefore never reads the network, and is left out of its
                        input tensor: the pack pass compacts the rest into `kTensorHeight` rows

 The networks come from `third-party/content/src/dfaoit/`: `tools/train_dfaoit.py` trains one and writes it as an unspecialized VGF
 named after its topology, `tools/make_dfaoit_model.py` shapes it for a tensor, and the sample loads that one with
 `lvk::VgfModel`. The network is quantized to 8 bits, the integer graph an NPU runs natively: int8 tensors in and out, the
 features quantized to `round(x * 255) - 128` on the way in and the tail read back as `(v + 128) / 255`. Two are deployed:
 `16x16` runs in half the time of `32x16` and matches it above alpha 0.5, where the tail is attenuated enough that the extra
 capacity has nothing left to resolve.

 `--technique abuffer|wboit|dfaoit`, `--network 16x16|32x16`, `--alpha A`, `--layers N`, `--weight-distance D`, `--compare` (mean
 squared error against the A-buffer), `--diff` (show the difference), `--timings` (GPU time of every stage). On Android the
 same options are read from `args.txt` next to the OBB, and the numbers go to logcat.
*/

#include <algorithm>
#include <cstring>
#include <string>

#include "Bistro.h"

#include <lvk/HelpersVgf.h>

#define MODEL_PATH "src/bistro/Exterior/exterior.obj"
#define CACHE_FILE_NAME "cache.data"

constexpr uint32_t kRenderWidth = 1920;
constexpr uint32_t kRenderHeight = 1080;

#if defined(ANDROID)
constexpr uint32_t kTensorWidth = 1170;
constexpr uint32_t kTensorHeight = 408;
#else
constexpr uint32_t kTensorWidth = kRenderWidth;
constexpr uint32_t kTensorHeight = 810;
#endif // ANDROID

constexpr uint32_t kMaxOitLayers = 64;

constexpr size_t getKeysSize(uint32_t numPixels, uint32_t numSlots) {
  return sizeof(uint64_t) * numPixels * numSlots;
}

const char* kCodeMeshVS = R"(
layout (location=0) in vec3 pos;
layout (location=1) in vec2 uv;
layout (location=2) in uint normal;
layout (location=3) in uint mtlIndex;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 model;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  vec2 materials;
  vec2 aBuffer;
  vec2 nearest;
  uint sampler0;
  uint layers;
  uint viewWidth;
  uint viewHeight;
  float alpha;
} pc;

layout (location=0) out vec3 normalWS;
layout (location=1) out vec2 uvOut;
layout (location=2) flat out uint materialId;
layout (location=3) out float viewDistance;

vec2 unpackSnorm2x8(uint d) {
  return vec2(uvec2(d, d >> 8) & 255u) / 127.5 - 1.0;
}

vec3 unpackOctahedral16(uint data) {
  vec2 v = unpackSnorm2x8(data);
  vec3 n = vec3(v, 1.0 - abs(v.x) - abs(v.y));
  float t = max(-n.z, 0.0);
  n.x += (n.x > 0.0) ? -t : t;
  n.y += (n.y > 0.0) ? -t : t;
  return normalize(n);
}

void main() {
  vec4 posVS = pc.perFrame.view * pc.perFrame.model * vec4(pos, 1.0);
  gl_Position = pc.perFrame.proj * posVS;
  normalWS = unpackOctahedral16(normal);
  uvOut = uv;
  materialId = mtlIndex;
  viewDistance = length(posVS.xyz);
}
)";

const char* kCodeMeshFS = R"(
layout (location=0) in vec3 normalWS;
layout (location=1) in vec2 uv;
layout (location=2) flat in uint materialId;

struct Material {
  vec4 ambient;
  vec4 diffuse;
  uint texAmbient;
  uint texDiffuse;
  uint texAlpha;
  uint padding;
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 model;
};

layout(std430, buffer_reference) readonly buffer Materials {
  Material mtl[];
};

layout(std430, buffer_reference) coherent buffer Keys {
  uint64_t keys[];
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  Materials materials;
  Keys aBuffer;
  Keys nearest;
  uint sampler0;
  uint layers;
  uint viewWidth;
  uint viewHeight;
  float alpha;
  float weightDistance;
} pc;

vec4 shade() {
  Material mtl = pc.materials.mtl[materialId];
  vec4 Ka = mtl.ambient * textureBindless2D(mtl.texAmbient, pc.sampler0, uv);
  vec4 Kd = mtl.diffuse * textureBindless2D(mtl.texDiffuse, pc.sampler0, uv);
  float mask = mtl.texAlpha > 0 ? textureBindless2D(mtl.texAlpha, pc.sampler0, uv).r : 1.0;

  vec3 n = normalize(normalWS);
  float NdotL1 = clamp(dot(n, normalize(vec3(-1, 1, +1))), 0.0, 1.0);
  float NdotL2 = clamp(dot(n, normalize(vec3(-1, 1, -1))), 0.0, 1.0);
  float NdotL = 0.5 * (NdotL1 + NdotL2);

  return vec4(clamp(Ka.rgb + Kd.rgb * (0.4 + 0.6 * NdotL), vec3(0.0), vec3(1.0)), pc.alpha * mask * Kd.a);
}

const uint64_t kEmptyKey = 0xFFFFFFFFFFFFFFFFul;

uint64_t insertKey(Keys list, uint numSlots, vec4 color) {
  if (gl_HelperInvocation)
    return kEmptyKey;

  uint first = (uint(gl_FragCoord.y) * pc.viewWidth + uint(gl_FragCoord.x)) * numSlots;
  uint64_t key = (uint64_t(floatBitsToUint(gl_FragCoord.z)) << 32) | uint64_t(packUnorm4x8(color));

  if (numSlots <= 8u && key >= list.keys[first + numSlots - 1u]) {
    return key;
  }

  for (uint slot = 0u; slot != numSlots && key != kEmptyKey; slot++) {
    uint64_t previous = atomicMin(list.keys[first + slot], key);
    key = previous > key ? previous : key;
  }

  return key;
}
)";

const char* kCodeABufferFS = R"(
layout (location=0) out vec4 out_Tail;

void main() {
  vec4 color = shade();

  if (color.a < 1.0 / 255.0)
    discard;

  uint64_t evicted = insertKey(pc.aBuffer, pc.layers & 0x7FFFFFFFu, color);

  out_Tail = vec4(0.0);

  if (evicted != kEmptyKey && (pc.layers & 0x80000000u) == 0u) {
    vec4 tail = unpackUnorm4x8(uint(evicted));
    out_Tail = vec4(tail.rgb * tail.a, tail.a);
  }
}
)";

const char* kCodeWeightedFS = R"(
layout (location=3) in float viewDistance;

layout (location=0) out vec4 out_Accumulated;
layout (location=1) out vec4 out_Revealage;
layout (location=2) out float out_Count;

void main() {
  vec4 color = shade();

  if (color.a < 1.0 / 255.0)
    discard;

  vec3 premultiplied = color.rgb * color.a;

  float distanceWeight = clamp(0.03 / (1e-5 + pow(viewDistance / pc.weightDistance, 4.0)), 1e-2, 3e3);
  float alphaWeight = min(1.0, max(max(premultiplied.r, premultiplied.g), max(premultiplied.b, color.a)) * 40.0 + 0.01);

  out_Accumulated = vec4(premultiplied, color.a) * (alphaWeight * alphaWeight * distanceWeight);
  out_Revealage = vec4(color.a);
  out_Count = 1.0;
}
)";

const char* kCodeWeightedResolveCS = R"(
layout (local_size_x = 16, local_size_y = 16) in;

layout (set = 0, binding = 0) uniform texture2D kTextures2D[];
layout (set = 0, binding = 2, rgba16f) uniform writeonly image2D kImagesRGBA16F[];

layout(std430, buffer_reference) readonly buffer Nearest {
  uint words[];
};

layout(push_constant) uniform constants {
  vec4 background;
  Nearest nearest;
  uvec2 slots;
  uvec2 counter;
  uint texSum;
  uint texAccumulated;
  uint texCount;
  uint imageOut;
  uint tensorInput;
  uint tensorOutput;
  uint viewWidth;
  uint viewHeight;
  uint tensorWidth;
  uint tensorHeight;
} pc;

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

  if (coord.x >= int(pc.viewWidth) || coord.y >= int(pc.viewHeight))
    return;

  vec4 accumulated = texelFetch(kTextures2D[pc.texSum], coord, 0);
  float revealage = texelFetch(kTextures2D[pc.texAccumulated], coord, 0).r;

  vec3 color = accumulated.rgb / max(accumulated.a, 1e-5);

  imageStore(kImagesRGBA16F[pc.imageOut], coord, vec4(mix(color, pc.background.rgb, revealage), 1.0));
}
)";

const char* kCodeFeaturesFS = R"(
layout (location=0) out vec4 out_Sum;
layout (location=1) out vec4 out_Accumulated;
layout (location=2) out float out_Count;

void main() {
  vec4 color = shade();

  if (color.a < 1.0 / 255.0)
    discard;

  out_Sum = color;
  out_Accumulated = vec4(color.rgb * color.a, color.a);
  out_Count = 1.0;

  insertKey(pc.nearest, kDfaoitNumNearest, color);
}
)";

const char* kCodeFullscreenVS = R"(
layout (location=0) out vec2 uv;

void main() {
  uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * vec2(2, -2) + vec2(-1, 1), 0.0, 1.0);
}
)";

const char* kCodeCompositeFS = R"(
layout (location=0) in vec2 uv;

layout (location=0) out vec4 out_FragColor;

layout(std430, buffer_reference) readonly buffer Keys {
  uint64_t keys[];
};

layout(push_constant) uniform constants {
  Keys aBuffer;
  uint layers;
  uint viewWidth;
  uint viewHeight;
  uint tailTexture;
  uint sampler0;
} pc;

const uint64_t kEmptyKey = 0xFFFFFFFFFFFFFFFFul;

void blendOver(inout vec4 dst, vec4 src) {
  dst.rgb += (1.0 - dst.a) * src.rgb;
  dst.a += (1.0 - dst.a) * src.a;
}

void main() {
  vec4 tail = textureBindless2D(pc.tailTexture, pc.sampler0, uv);
  vec4 color = vec4(0.0);

  if (!gl_HelperInvocation) {
    uint first = (uint(gl_FragCoord.y) * pc.viewWidth + uint(gl_FragCoord.x)) * pc.layers;

    for (uint i = 0u; i != pc.layers; i++) {
      uint64_t key = pc.aBuffer.keys[first + i];
      if (key == kEmptyKey)
        break;
      vec4 fragment = unpackUnorm4x8(uint(key));
      blendOver(color, vec4(fragment.rgb * fragment.a, fragment.a));
    }
  }

  blendOver(color, tail);

  out_FragColor = color;
}
)";

const char* kCodeDfaoitCommon = R"(
layout (local_size_x = 16, local_size_y = 16) in;

layout (set = 0, binding = 0) uniform texture2D kTextures2D[];

layout(std430, buffer_reference) readonly buffer Nearest {
  uint words[];
};

layout(std430, buffer_reference) buffer Slots {
  uint value[];
};

layout(std430, buffer_reference) buffer Counter {
  uint value[];
};

layout(push_constant) uniform constants {
  vec4 background;
  Nearest nearest;
  Slots slots;
  Counter counter;
  uint texSum;
  uint texAccumulated;
  uint texCount;
  uint imageOut;
  uint tensorInput;
  uint tensorOutput;
  uint viewWidth;
  uint viewHeight;
  uint tensorWidth;
  uint tensorHeight;
} pc;

const uint kNoSlot = 0xFFFFFFFFu;

struct Pixel {
  uint count;
  vec3 nearest;
  float attenuation;
  float transmittance;
  float inputs[kDfaoitNumInputs];
};

Pixel loadPixel(ivec2 coord) {
  vec4 sum = texelFetch(kTextures2D[pc.texSum], coord, 0);
  vec4 accumulated = texelFetch(kTextures2D[pc.texAccumulated], coord, 0);
  uint index = uint(coord.y) * pc.viewWidth + uint(coord.x);

  Pixel p;
  p.count = uint(texelFetch(kTextures2D[pc.texCount], coord, 0).r + 0.5);
  p.transmittance = accumulated.a;

  vec4 nearSum = vec4(0.0);
  vec3 exact = vec3(0.0);
  float alphas[kDfaoitNumNearest];

  p.attenuation = 1.0;

  for (uint i = 0u; i != kDfaoitNumNearest; i++) {
    vec4 f = p.count > i ? unpackUnorm4x8(pc.nearest.words[2u * kDfaoitNumNearest * index + 2u * i]) : vec4(0.0);
    exact += p.attenuation * f.a * f.rgb;
    p.attenuation *= 1.0 - f.a;
    nearSum += f;
    alphas[i] = f.a;
  }

  p.nearest = exact;

  vec4 average = p.count > kDfaoitNumNearest ? max(sum - nearSum, vec4(0.0)) / float(p.count - kDfaoitNumNearest) : vec4(0.0);

  vec3 squashed = accumulated.rgb / (1.0 + accumulated.rgb);

  p.inputs[0] = average.a;
  p.inputs[1] = average.r;
  p.inputs[2] = average.g;
  p.inputs[3] = average.b;
  p.inputs[4] = squashed.r;
  p.inputs[5] = squashed.g;
  p.inputs[6] = squashed.b;
  p.inputs[7] = p.nearest.r;
  p.inputs[8] = p.nearest.g;
  p.inputs[9] = p.nearest.b;

  for (uint i = 0u; i != kDfaoitNumNearest; i++) {
    p.inputs[10u + i] = alphas[i];
  }

  return p;
}
)";

const char* kCodeDfaoitPackCS = R"(
shared uint sGroupCount;
shared uint sGroupBase;

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  bool inside = coord.x < int(pc.viewWidth) && coord.y < int(pc.viewHeight);

  Pixel p;
  bool wanted = false;
  if (inside) {
    p = loadPixel(coord);
    wanted = p.count > kDfaoitNumNearest;
  }

  if (gl_LocalInvocationIndex == 0u)
    sGroupCount = 0u;
  barrier();
  uint local = wanted ? atomicAdd(sGroupCount, 1u) : 0u;
  barrier();
  if (gl_LocalInvocationIndex == 0u)
    sGroupBase = atomicAdd(pc.counter.value[0], sGroupCount);
  barrier();

  if (!inside)
    return;

  uint slot = kNoSlot;

  if (wanted) {
    const uint candidate = sGroupBase + local;
    if (candidate < pc.tensorWidth * pc.tensorHeight) {
      slot = candidate;
      int8_t quantized[kDfaoitNumInputs];
      for (uint i = 0u; i != kDfaoitNumInputs; i++) {
        quantized[i] = int8_t(clamp(int(round(p.inputs[i] * 255.0)) - 128, -128, 127));
      }
      tensorWriteARM(kTensorsI8_4[pc.tensorInput], uint[](0, slot / pc.tensorWidth, slot % pc.tensorWidth, 0), quantized);
    }
  }

  pc.slots.value[uint(coord.y) * pc.viewWidth + uint(coord.x)] = slot;
}
)";

const char* kCodeDfaoitResolveCS = R"(
layout (set = 0, binding = 2, rgba16f) uniform writeonly image2D kImagesRGBA16F[];

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

  if (coord.x >= int(pc.viewWidth) || coord.y >= int(pc.viewHeight))
    return;

  Pixel p = loadPixel(coord);

  vec3 color = p.nearest;

  if (p.count > kDfaoitNumNearest) {
    const uint slot = pc.slots.value[uint(coord.y) * pc.viewWidth + uint(coord.x)];
    if (slot != kNoSlot) {
      int8_t outputs[kDfaoitNumOutputs];
      tensorReadARM(kTensorsI8_4[pc.tensorOutput], uint[](0, slot / pc.tensorWidth, slot % pc.tensorWidth, 0), outputs);
      color += p.attenuation * (vec3(outputs[0], outputs[1], outputs[2]) + 128.0) / 255.0;
    } else {
      const float coverage = 1.0 - pow(1.0 - p.inputs[0], float(p.count - kDfaoitNumNearest));
      color += p.attenuation * coverage * vec3(p.inputs[1], p.inputs[2], p.inputs[3]);
    }
  }

  imageStore(kImagesRGBA16F[pc.imageOut], coord, vec4(color + p.transmittance * pc.background.rgb, 1.0));
}
)";

const char* kCodeErrorCS = R"(
layout (local_size_x = 1, local_size_y = 64) in;

layout (set = 0, binding = 0) uniform texture2D kTextures2D[];

layout(std430, buffer_reference) writeonly buffer RowStats {
  vec4 row[];
};

layout(push_constant) uniform constants {
  RowStats rows;
  uint texResult;
  uint texReference;
  uint texCount;
  uint layers;
  uint viewWidth;
  uint viewHeight;
} pc;

void main() {
  uint y = gl_GlobalInvocationID.y;

  if (y >= pc.viewHeight)
    return;

  vec4 stats = vec4(0.0);

  for (uint x = 0u; x != pc.viewWidth; x++) {
    vec3 d = texelFetch(kTextures2D[pc.texResult], ivec2(x, y), 0).rgb - texelFetch(kTextures2D[pc.texReference], ivec2(x, y), 0).rgb;
    float count = texelFetch(kTextures2D[pc.texCount], ivec2(x, y), 0).r;
    stats.x += dot(d, d);
    stats.y += count;
    stats.z = max(stats.z, count);
    stats.w += count > float(pc.layers) ? 1.0 : 0.0;
  }

  pc.rows.row[y] = stats;
}
)";

const char* kCodePresentVS = R"(
layout (location=0) out vec2 uv;

layout(push_constant) uniform constants {
  mat4 clipRotation;
  uint texResult;
  uint texReference;
  uint sampler0;
  uint showDifference;
  float gain;
} pc;

void main() {
  uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = pc.clipRotation * vec4(uv * vec2(2, -2) + vec2(-1, 1), 0.0, 1.0);
}
)";

const char* kCodePresentFS = R"(
layout (location=0) in vec2 uv;

layout (location=0) out vec4 out_FragColor;

layout(push_constant) uniform constants {
  mat4 clipRotation;
  uint texResult;
  uint texReference;
  uint sampler0;
  uint showDifference;
  float gain;
} pc;

void main() {
  vec3 color = textureBindless2D(pc.texResult, pc.sampler0, uv).rgb;

  if (pc.showDifference != 0u) {
    color = pc.gain * abs(color - textureBindless2D(pc.texReference, pc.sampler0, uv).rgb);
  }

  out_FragColor = vec4(color, 1.0);
}
)";

enum Technique {
  Technique_ABuffer = 0,
  Technique_WeightedBlended,
  Technique_DFAOIT,
};

enum GPUTimestamp {
  GPUTimestamp_Begin = 0,
  GPUTimestamp_ABuffer,
  GPUTimestamp_Features,
  GPUTimestamp_Inference,
  GPUTimestamp_Resolve,
  GPUTimestamp_Present,
  GPUTimestamp_NUM,
};

struct PerFrame {
  mat4 proj;
  mat4 view;
  mat4 model;
};

struct MeshPushConstants {
  uint64_t perFrame;
  uint64_t materials;
  uint64_t aBuffer;
  uint64_t nearest;
  uint32_t sampler0;
  uint32_t layers;
  uint32_t viewWidth;
  uint32_t viewHeight;
  float alpha;
  float weightDistance;
};

struct CompositePushConstants {
  uint64_t aBuffer;
  uint32_t layers;
  uint32_t viewWidth;
  uint32_t viewHeight;
  uint32_t tailTexture;
  uint32_t sampler0;
};

struct DfaoitPushConstants {
  vec4 background;
  uint64_t nearest;
  uint64_t slots;
  uint64_t counter;
  uint32_t texSum;
  uint32_t texAccumulated;
  uint32_t texCount;
  uint32_t imageOut;
  uint32_t tensorInput;
  uint32_t tensorOutput;
  uint32_t viewWidth;
  uint32_t viewHeight;
  uint32_t tensorWidth;
  uint32_t tensorHeight;
};

struct ErrorPushConstants {
  uint64_t rows;
  uint32_t texResult;
  uint32_t texReference;
  uint32_t texCount;
  uint32_t layers;
  uint32_t viewWidth;
  uint32_t viewHeight;
};

struct RowStats {
  float squaredError = 0.0f;
  float numFragments = 0.0f;
  float maxFragments = 0.0f;
  float numPixelsAboveLayers = 0.0f;
};

struct PresentPushConstants {
  mat4 clipRotation;
  uint32_t texResult;
  uint32_t texReference;
  uint32_t sampler0;
  uint32_t showDifference;
  float gain;
};

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      .width = (int)kRenderWidth,
      .height = (int)kRenderHeight,
      .resizable = false,
      .initialCameraPos = vec3(-100, 40, -47),
      .initialCameraTarget = vec3(0, 35, 0),
      .contextConfig =
          {
              .enableValidationGpuAV = false,
              .enableMLEmulationLayer = true,
          },
  };
  VULKAN_APP_DECLARE(app, cfg);

  int technique = Technique_ABuffer;
  int numLayers = 16;
  float weightDistance = 250.0f;
  float startAlpha = 0.3f;
  bool compare = false;
  bool showDifference = false;
  bool measureTime = false;
  const char* networkName = "16x16";

  for (size_t i = 0; i < app.args_.size(); i++) {
    const char* arg = app.args_[i].c_str();
    const char* next = i + 1 < app.args_.size() ? app.args_[i + 1].c_str() : nullptr;
    if (!strcmp(arg, "--technique") && next) {
      technique = !strcmp(next, "dfaoit") ? Technique_DFAOIT : !strcmp(next, "wboit") ? Technique_WeightedBlended : Technique_ABuffer;
      i++;
    } else if (!strcmp(arg, "--weight-distance") && next) {
      weightDistance = (float)atof(next);
      i++;
    } else if (!strcmp(arg, "--alpha") && next) {
      startAlpha = (float)atof(next);
      i++;
    } else if (!strcmp(arg, "--layers") && next) {
      numLayers = atoi(next);
      i++;
    } else if (!strcmp(arg, "--compare")) {
      compare = true;
    } else if (!strcmp(arg, "--diff")) {
      compare = true;
      showDifference = true;
    } else if (!strcmp(arg, "--timings")) {
      measureTime = true;
    } else if (!strcmp(arg, "--network") && next) {
      networkName = next;
      i++;
    }
  }

  lvk::IContext* ctx = app.ctx_.get();

  {
    const std::string cacheFileName = app.folderContentRoot_ + CACHE_FILE_NAME;

    if (!loadFromCache(app, cacheFileName.c_str())) {
      if (!LVK_VERIFY(loadAndCache(app, cacheFileName.c_str(), MODEL_PATH))) {
        LVK_ASSERT_MSG(false, "Cannot load 3D model. Run `deploy_content.py` before running this app.");
        VULKAN_APP_EXIT();
      }
    }
  }

  loadMaterialTextures(app, "src/bistro/Exterior/");

  lvk::Holder<lvk::BufferHandle> sbMaterials = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_Device,
      .size = sizeof(GPUMaterial) * materials_.size(),
      .data = materials_.data(),
      .debugName = "Buffer: materials",
  });
  lvk::Holder<lvk::BufferHandle> vb0 = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Vertex,
      .storage = lvk::StorageType_Device,
      .size = sizeof(VertexData) * vertexData_.size(),
      .data = vertexData_.data(),
      .debugName = "Buffer: vertex",
  });
  lvk::Holder<lvk::BufferHandle> ib0 = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Index,
      .storage = lvk::StorageType_Device,
      .size = sizeof(uint32_t) * indexData_.size(),
      .data = indexData_.data(),
      .debugName = "Buffer: index",
  });
  lvk::Holder<lvk::BufferHandle> bufPerFrame = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(PerFrame),
      .debugName = "Buffer: per frame",
  });

  const uint32_t numIndices = (uint32_t)indexData_.size();

  vertexData_ = {};
  indexData_ = {};

  char modelFileName[256] = {};
  snprintf(modelFileName,
           sizeof(modelFileName) - 1,
           "%ssrc/dfaoit/dfaoit-%s-%ux%u.vgf",
           app.folderContentRoot_.c_str(),
           networkName,
           kTensorWidth,
           kTensorHeight);

  lvk::VgfModel vgf;
  std::vector<uint32_t> layerSizes;

  if (!vgf.loadFromFile(modelFileName)) {
    LLOGW("Cannot load `%s`. Run `tools/train_dfaoit.py`, then `tools/make_dfaoit_model.py --render %ux%u`.\n",
          modelFileName,
          kTensorWidth,
          kTensorHeight);
  } else {
    const ldr::Span<const lvk::DataGraphConstant> constants = vgf.getConstants();
    for (size_t i = 0; i + 1 < constants.size(); i += 2) {
      const lvk::DataGraphConstant& weights = constants[i];
      const lvk::DataGraphConstant& biases = constants[i + 1];
      if (weights.rank != 4 || biases.rank != 1 || weights.format != lvk::Format_R_I8) {
        layerSizes.clear();
        break;
      }
      if (layerSizes.empty()) {
        layerSizes.push_back((uint32_t)weights.dimensions[3]);
      }
      layerSizes.push_back((uint32_t)weights.dimensions[0]);
    }
    if (layerSizes.size() != 4) {
      LLOGW("`%s` is not the three-layer network this sample expects\n", modelFileName);
      layerSizes.clear();
    }
  }

  const bool hasNetwork = layerSizes.size() == 4;
  const bool hasDataGraph = hasNetwork && ctx->supportsTensors() && ctx->supportsDataGraph();

  if (hasNetwork && !hasDataGraph) {
    LLOGW("VK_ARM_tensors and VK_ARM_data_graph are not available (configure with LVK_WITH_ML_EMULATION_LAYER=ON to emulate them)\n");
  }

  const uint32_t numNearest = hasNetwork ? std::max(2u, layerSizes[0] - 10u) : 2u;
  const std::string nearest = "const uint kDfaoitNumNearest = " + std::to_string(numNearest) + ";\n";

  const std::string codeABufferFS = std::string(kCodeMeshFS) + kCodeABufferFS;
  const std::string codeWeightedFS = std::string(kCodeMeshFS) + kCodeWeightedFS;
  const std::string codeFeaturesFS = nearest + kCodeMeshFS + kCodeFeaturesFS;
  const std::string network = hasNetwork ? nearest + "const uint kDfaoitNumInputs = " + std::to_string(layerSizes[0]) + ";\n" +
                                               "const uint kDfaoitNumHidden1 = " + std::to_string(layerSizes[1]) + ";\n" +
                                               "const uint kDfaoitNumHidden2 = " + std::to_string(layerSizes[2]) + ";\n" +
                                               "const uint kDfaoitNumOutputs = " + std::to_string(layerSizes[3]) + ";\n"
                                         : std::string();
  const std::string codeResolveCS = network + kCodeDfaoitCommon + kCodeDfaoitResolveCS;
  const std::string codePackCS = network + kCodeDfaoitCommon + kCodeDfaoitPackCS;

  lvk::Holder<lvk::ShaderModuleHandle> smMeshVert = ctx->createShaderModule({kCodeMeshVS, lvk::Stage_Vert, "Shader Module: mesh (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smABufferFrag =
      ctx->createShaderModule({codeABufferFS.c_str(), lvk::Stage_Frag, "Shader Module: A-buffer (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smWeightedFrag =
      ctx->createShaderModule({codeWeightedFS.c_str(), lvk::Stage_Frag, "Shader Module: weighted blended (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smWeightedResolveComp =
      ctx->createShaderModule({kCodeWeightedResolveCS, lvk::Stage_Comp, "Shader Module: weighted blended resolve (comp)"});
  lvk::Holder<lvk::ShaderModuleHandle> smFeaturesFrag =
      ctx->createShaderModule({codeFeaturesFS.c_str(), lvk::Stage_Frag, "Shader Module: DFAOIT features (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smFullscreenVert =
      ctx->createShaderModule({kCodeFullscreenVS, lvk::Stage_Vert, "Shader Module: fullscreen (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smCompositeFrag =
      ctx->createShaderModule({kCodeCompositeFS, lvk::Stage_Frag, "Shader Module: A-buffer composite (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smPresentFrag =
      ctx->createShaderModule({kCodePresentFS, lvk::Stage_Frag, "Shader Module: present (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smPresentVert =
      ctx->createShaderModule({kCodePresentVS, lvk::Stage_Vert, "Shader Module: present (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smErrorComp =
      ctx->createShaderModule({kCodeErrorCS, lvk::Stage_Comp, "Shader Module: error (comp)"});
  lvk::Holder<lvk::ShaderModuleHandle> smResolveComp;
  lvk::Holder<lvk::ShaderModuleHandle> smPackComp;

  if (hasDataGraph) {
    smResolveComp = ctx->createShaderModule({codeResolveCS.c_str(), lvk::Stage_Comp, "Shader Module: DFAOIT resolve (comp)"});
    smPackComp = ctx->createShaderModule({codePackCS.c_str(), lvk::Stage_Comp, "Shader Module: DFAOIT pack (comp)"});
  }

  lvk::Holder<lvk::SamplerHandle> sampler = ctx->createSampler({
      .mipMap = lvk::SamplerMip_Linear,
      .wrapU = lvk::SamplerWrap_Repeat,
      .wrapV = lvk::SamplerWrap_Repeat,
      .debugName = "Sampler: linear",
  });
  lvk::Holder<lvk::SamplerHandle> samplerClamp = ctx->createSampler({
      .wrapU = lvk::SamplerWrap_Clamp,
      .wrapV = lvk::SamplerWrap_Clamp,
      .debugName = "Sampler: clamp",
  });

  const lvk::Format formatColor = lvk::Format_RGBA_F16;
  const lvk::Format formatCount = lvk::Format_R_F16;

  const lvk::VertexInput vertexInput = {
      .attributes = {{.location = 0, .format = lvk::VertexFormat_Float3, .offset = offsetof(VertexData, position)},
                     {.location = 1, .format = lvk::VertexFormat_HalfFloat2, .offset = offsetof(VertexData, uv)},
                     {.location = 2, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, normal)},
                     {.location = 3, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, mtlIndex)}},
      .inputBindings = {{.stride = sizeof(VertexData)}},
  };
  const lvk::ColorAttachment blendPremultiplied = {
      .format = formatColor,
      .blendEnabled = true,
      .srcRGBBlendFactor = lvk::BlendFactor_One,
      .srcAlphaBlendFactor = lvk::BlendFactor_One,
      .dstRGBBlendFactor = lvk::BlendFactor_OneMinusSrcAlpha,
      .dstAlphaBlendFactor = lvk::BlendFactor_OneMinusSrcAlpha,
  };
  const lvk::ColorAttachment blendAdditive = {
      .format = formatColor,
      .blendEnabled = true,
      .srcRGBBlendFactor = lvk::BlendFactor_One,
      .srcAlphaBlendFactor = lvk::BlendFactor_One,
      .dstRGBBlendFactor = lvk::BlendFactor_One,
      .dstAlphaBlendFactor = lvk::BlendFactor_One,
  };
  const lvk::ColorAttachment blendAdditiveTransmittance = {
      .format = formatColor,
      .blendEnabled = true,
      .srcRGBBlendFactor = lvk::BlendFactor_One,
      .srcAlphaBlendFactor = lvk::BlendFactor_Zero,
      .dstRGBBlendFactor = lvk::BlendFactor_One,
      .dstAlphaBlendFactor = lvk::BlendFactor_OneMinusSrcAlpha,
  };
  const lvk::ColorAttachment blendRevealage = {
      .format = formatColor,
      .blendEnabled = true,
      .srcRGBBlendFactor = lvk::BlendFactor_Zero,
      .srcAlphaBlendFactor = lvk::BlendFactor_Zero,
      .dstRGBBlendFactor = lvk::BlendFactor_OneMinusSrcColor,
      .dstAlphaBlendFactor = lvk::BlendFactor_OneMinusSrcColor,
  };
  const lvk::ColorAttachment blendCount = {
      .format = formatCount,
      .blendEnabled = true,
      .srcRGBBlendFactor = lvk::BlendFactor_One,
      .srcAlphaBlendFactor = lvk::BlendFactor_One,
      .dstRGBBlendFactor = lvk::BlendFactor_One,
      .dstAlphaBlendFactor = lvk::BlendFactor_One,
  };

  lvk::Holder<lvk::RenderPipelineHandle> pipelineABuffer = ctx->createRenderPipeline({
      .vertexInput = vertexInput,
      .smVert = smMeshVert,
      .smFrag = smABufferFrag,
      .color = {blendPremultiplied},
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: A-buffer",
  });
  lvk::Holder<lvk::RenderPipelineHandle> pipelineWeighted = ctx->createRenderPipeline({
      .vertexInput = vertexInput,
      .smVert = smMeshVert,
      .smFrag = smWeightedFrag,
      .color = {blendAdditive, blendRevealage, blendCount},
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: weighted blended",
  });
  lvk::Holder<lvk::RenderPipelineHandle> pipelineFeatures = ctx->createRenderPipeline({
      .vertexInput = vertexInput,
      .smVert = smMeshVert,
      .smFrag = smFeaturesFrag,
      .color = {blendAdditive, blendAdditiveTransmittance, blendCount},
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: DFAOIT features",
  });
  lvk::Holder<lvk::RenderPipelineHandle> pipelineComposite = ctx->createRenderPipeline({
      .smVert = smFullscreenVert,
      .smFrag = smCompositeFrag,
      .color = {blendPremultiplied},
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: A-buffer composite",
  });
  lvk::Holder<lvk::RenderPipelineHandle> pipelinePresent = ctx->createRenderPipeline({
      .smVert = smPresentVert,
      .smFrag = smPresentFrag,
      .color = {{.format = ctx->getSwapchainFormat()}},
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: present",
  });
  lvk::Holder<lvk::ComputePipelineHandle> pipelineError = ctx->createComputePipeline({.smComp = smErrorComp});
  lvk::Holder<lvk::ComputePipelineHandle> pipelineWeightedResolve = ctx->createComputePipeline({.smComp = smWeightedResolveComp});
  lvk::Holder<lvk::ComputePipelineHandle> pipelineResolve;
  lvk::Holder<lvk::ComputePipelineHandle> pipelinePack;

  if (hasDataGraph) {
    pipelineResolve = ctx->createComputePipeline({.smComp = smResolveComp});
    pipelinePack = ctx->createComputePipeline({.smComp = smPackComp});
  }

  lvk::Holder<lvk::QueryPoolHandle> queryPool = ctx->createQueryPool(GPUTimestamp_NUM, "Query pool: timestamps");

  lvk::Holder<lvk::BufferHandle> aBuffer;
  lvk::Holder<lvk::TextureHandle> tailTexture;
  lvk::Holder<lvk::TextureHandle> texExact;
  lvk::Holder<lvk::TextureHandle> texApproximate;
  lvk::Holder<lvk::TextureHandle> texSum;
  lvk::Holder<lvk::TextureHandle> texAccumulated;
  lvk::Holder<lvk::TextureHandle> texCount;
  lvk::Holder<lvk::BufferHandle> nearestBuffer;
  lvk::Holder<lvk::BufferHandle> slotsBuffer;
  lvk::Holder<lvk::BufferHandle> counterBuffer;
  lvk::Holder<lvk::BufferHandle> rowSums;
  lvk::Holder<lvk::TensorHandle> tensorInput;
  lvk::Holder<lvk::TensorHandle> tensorOutput;
  lvk::Holder<lvk::DataGraphPipelineHandle> pipelineGraph;

  uint32_t builtWidth = 0;
  uint32_t builtHeight = 0;
  uint32_t builtLayers = 0;
  bool graphReady = false;

  if (hasDataGraph) {
    const uint8_t usage = lvk::TensorUsageBits_Shader | lvk::TensorUsageBits_DataGraph;
    const int64_t dimsInput[] = {1, kTensorHeight, kTensorWidth, layerSizes.front()};
    const int64_t dimsOutput[] = {1, kTensorHeight, kTensorWidth, layerSizes.back()};
    lvk::Result result;

    if (vgf.getNumInputs() != 1 || vgf.getNumOutputs() != 1 || !vgf.getInput(0).matches(dimsInput) ||
        !vgf.getOutput(0).matches(dimsOutput)) {
      LLOGW("`%s` is not shaped for %ux%u\n", modelFileName, kTensorWidth, kTensorHeight);
    } else {
      tensorInput = ctx->createTensor(vgf.getInput(0).toTensorDesc(usage), "DFAOIT: input tensor", &result);
      if (result.isOk()) {
        tensorOutput = ctx->createTensor(vgf.getOutput(0).toTensorDesc(usage), "DFAOIT: output tensor", &result);
      }
      if (result.isOk()) {
        pipelineGraph = vgf.createDataGraphPipeline(*ctx, "DFAOIT: data graph", &result);
      }
      if (result.isOk()) {
        graphReady = true;
      } else {
        LLOGW("Cannot create the data graph: %s\n", result.message);
      }
    }
  }

  float alpha = startAlpha;
  bool tailBlend = true;
  float differenceGain = 8.0f;
  double meanSquaredError = 0.0;
  double meanFragments = 0.0;
  double maxFragments = 0.0;
  double percentAboveLayers = 0.0;
  double stageTime[GPUTimestamp_NUM] = {};
  double frameTime = 0.0;
  uint64_t frameIndex = 0;

  const vec4 background = vec4(0.38f, 0.51f, 0.71f, 1.0f);

  app.run([&](ldr::Span<const RenderView> views, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    const lvk::Dimensions dim = ctx->getDimensions(views[0].colorTexture);
    const uint32_t viewSize = dim.width * dim.height;
    const uint64_t fittingLayers = ctx->getMaxStorageBufferRange() / (sizeof(uint64_t) * viewSize);
    const int maxLayers = (int)std::max<uint64_t>(1, std::min<uint64_t>(kMaxOitLayers, fittingLayers));

    numLayers = std::clamp(numLayers, 1, maxLayers);

    if (builtWidth != dim.width || builtHeight != dim.height) {
      builtWidth = dim.width;
      builtHeight = dim.height;
      builtLayers = 0;

      auto createTarget = [&](lvk::Format format, uint8_t usage, const char* debugName) -> lvk::Holder<lvk::TextureHandle> {
        return ctx->createTexture({.format = format, .dimensions = dim, .usage = usage, .debugName = debugName});
      };
      const uint8_t usageAttachment = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled;

      tailTexture = createTarget(formatColor, usageAttachment, "Texture: A-buffer tail");
      texExact = createTarget(formatColor, usageAttachment, "Texture: A-buffer result");
      texApproximate = createTarget(formatColor, lvk::TextureUsageBits_Storage | lvk::TextureUsageBits_Sampled, "Texture: DFAOIT result");
      texSum = createTarget(formatColor, usageAttachment, "Texture: DFAOIT sum");
      texAccumulated = createTarget(formatColor, usageAttachment, "Texture: DFAOIT accumulated");
      texCount = createTarget(formatCount, usageAttachment, "Texture: DFAOIT count");
      nearestBuffer = ctx->createBuffer({
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_Device,
          .size = getKeysSize(viewSize, numNearest),
          .debugName = "Buffer: DFAOIT nearest fragments",
      });
      slotsBuffer = ctx->createBuffer({
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_Device,
          .size = sizeof(uint32_t) * viewSize,
          .debugName = "Buffer: DFAOIT tensor row of every pixel",
      });
      counterBuffer = ctx->createBuffer({
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_Device,
          .size = sizeof(uint32_t),
          .debugName = "Buffer: DFAOIT compaction counter",
      });
      rowSums = ctx->createBuffer({
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_HostVisible,
          .size = sizeof(RowStats) * dim.height,
          .debugName = "Buffer: error row statistics",
      });
    }

    if (builtLayers != (uint32_t)numLayers) {
      builtLayers = (uint32_t)numLayers;
      aBuffer = ctx->createBuffer({
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_Device,
          .size = getKeysSize(viewSize, builtLayers),
          .debugName = "Buffer: A-buffer",
      });
    }

    if (technique == Technique_DFAOIT && !graphReady) {
      technique = Technique_ABuffer;
    }

    const bool needApproximate = technique != Technique_ABuffer;
    const bool needExact = technique == Technique_ABuffer || compare;

    const PerFrame perFrame = {
        .proj = glm::perspective(float(45.0f * (M_PI / 180.0f)), views[0].aspectRatio, 0.5f, 500.0f),
        .view = app.camera_.getViewMatrix(),
        .model = glm::scale(mat4(1.0f), vec3(0.05f)),
    };

    const lvk::Viewport viewport = {0.0f, 0.0f, (float)dim.width, (float)dim.height, 0.0f, 1.0f};
    const lvk::ScissorRect scissor = {0, 0, dim.width, dim.height};
    const lvk::Dimensions groups16 = {(dim.width + 15) / 16, (dim.height + 15) / 16, 1};

    const MeshPushConstants pcMesh = {
        .perFrame = ctx->gpuAddress(bufPerFrame),
        .materials = ctx->gpuAddress(sbMaterials),
        .aBuffer = ctx->gpuAddress(aBuffer),
        .nearest = ctx->gpuAddress(nearestBuffer),
        .sampler0 = sampler.index(),
        .layers = builtLayers | (tailBlend ? 0u : 0x80000000u),
        .viewWidth = dim.width,
        .viewHeight = dim.height,
        .alpha = alpha,
        .weightDistance = weightDistance,
    };

    lvk::ICommandBuffer& buf = ctx->acquireCommandBuffer();

    buf.cmdUpdateBuffer(bufPerFrame, perFrame);
    processLoadedMaterialTextures(buf, sbMaterials);

    const bool measureTimeThisFrame = measureTime;

    auto timestamp = [&](GPUTimestamp t) {
      if (measureTimeThisFrame) {
        buf.cmdWriteTimestamp(queryPool, t);
      }
    };

    if (measureTimeThisFrame) {
      buf.cmdResetQueryPool(queryPool, 0, GPUTimestamp_NUM);
    }

    timestamp(GPUTimestamp_Begin);

    auto drawScene = [&](lvk::RenderPipelineHandle pipeline) {
      buf.cmdBindRenderPipeline(pipeline);
      buf.cmdBindViewport(viewport);
      buf.cmdBindScissorRect(scissor);
      buf.cmdBindVertexBuffer(0, vb0);
      buf.cmdBindIndexBuffer(ib0, lvk::IndexFormat_UI32);
      buf.cmdPushConstants(pcMesh);
      buf.cmdDrawIndexed(numIndices);
    };

    if (needExact) {
      buf.cmdPushDebugGroupLabel("A-buffer", 0xff0000ff);
      buf.cmdFillBuffer(aBuffer, 0, getKeysSize(viewSize, builtLayers), 0xFFFFFFFF);
      buf.cmdBeginRendering(
          lvk::RenderPass{.color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}}}},
          lvk::Framebuffer{.color = {{.texture = tailTexture}}},
          {.buffers = {aBuffer}});
      drawScene(pipelineABuffer);
      buf.cmdEndRendering();
      buf.cmdBeginRendering(lvk::RenderPass{.color = {{.loadOp = lvk::LoadOp_Clear,
                                                       .storeOp = lvk::StoreOp_Store,
                                                       .clearColor = {background.r, background.g, background.b, background.a}}}},
                            lvk::Framebuffer{.color = {{.texture = texExact}}},
                            {.sampledImages = {tailTexture}, .buffers = {aBuffer}});
      buf.cmdBindRenderPipeline(pipelineComposite);
      buf.cmdBindViewport(viewport);
      buf.cmdBindScissorRect(scissor);
      buf.cmdPushConstants(CompositePushConstants{
          .aBuffer = ctx->gpuAddress(aBuffer),
          .layers = builtLayers,
          .viewWidth = dim.width,
          .viewHeight = dim.height,
          .tailTexture = tailTexture.index(),
          .sampler0 = samplerClamp.index(),
      });
      buf.cmdDraw(3);
      buf.cmdEndRendering();
      buf.cmdPopDebugGroupLabel();
    }

    timestamp(GPUTimestamp_ABuffer);

    const DfaoitPushConstants pcResolve = {
        .background = background,
        .nearest = ctx->gpuAddress(nearestBuffer),
        .slots = ctx->gpuAddress(slotsBuffer),
        .counter = ctx->gpuAddress(counterBuffer),
        .texSum = texSum.index(),
        .texAccumulated = texAccumulated.index(),
        .texCount = texCount.index(),
        .imageOut = texApproximate.index(),
        .tensorInput = tensorInput.index(),
        .tensorOutput = tensorOutput.index(),
        .viewWidth = dim.width,
        .viewHeight = dim.height,
        .tensorWidth = kTensorWidth,
        .tensorHeight = kTensorHeight,
    };

    if (technique == Technique_WeightedBlended) {
      buf.cmdPushDebugGroupLabel("Weighted blended: accumulate", 0xffff00ff);
      buf.cmdBeginRendering(
          lvk::RenderPass{.color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0, 0, 0, 0}},
                                    {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {1, 1, 1, 1}},
                                    {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0, 0, 0, 0}}}},
          lvk::Framebuffer{.color = {{.texture = texSum}, {.texture = texAccumulated}, {.texture = texCount}}});
      drawScene(pipelineWeighted);
      buf.cmdEndRendering();
      buf.cmdPopDebugGroupLabel();

      timestamp(GPUTimestamp_Features);
      timestamp(GPUTimestamp_Inference);

      buf.cmdPushDebugGroupLabel("Weighted blended: resolve", 0xffff00ff);
      buf.cmdBindComputePipeline(pipelineWeightedResolve);
      buf.cmdPushConstants(pcResolve);
      buf.cmdDispatch(groups16, {.sampledImages = {texSum, texAccumulated}, .storageImages = {texApproximate}});
      buf.cmdPopDebugGroupLabel();

      timestamp(GPUTimestamp_Resolve);
    } else if (needApproximate) {
      buf.cmdPushDebugGroupLabel("DFAOIT: features", 0xff00ff00);
      buf.cmdFillBuffer(nearestBuffer, 0, getKeysSize(viewSize, numNearest), 0xFFFFFFFF);
      buf.cmdBeginRendering(
          lvk::RenderPass{.color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0, 0, 0, 0}},
                                    {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0, 0, 0, 1}},
                                    {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0, 0, 0, 0}}}},
          lvk::Framebuffer{.color = {{.texture = texSum}, {.texture = texAccumulated}, {.texture = texCount}}},
          {.buffers = {nearestBuffer}});
      drawScene(pipelineFeatures);
      buf.cmdEndRendering();
      buf.cmdPopDebugGroupLabel();

      timestamp(GPUTimestamp_Features);

      buf.cmdPushDebugGroupLabel("DFAOIT: data graph", 0xff00ff00);
      buf.cmdFillBuffer(counterBuffer, 0, sizeof(uint32_t), 0);
      buf.cmdBindComputePipeline(pipelinePack);
      buf.cmdPushConstants(pcResolve);
      buf.cmdDispatch(groups16,
                      {
                          .sampledImages = {texSum, texAccumulated, texCount},
                          .buffers = {nearestBuffer, slotsBuffer, counterBuffer},
                          .tensors = {tensorInput},
                      });
      const lvk::TensorHandle graphInputs[] = {tensorInput};
      const lvk::TensorHandle graphOutputs[] = {tensorOutput};
      buf.cmdDispatchDataGraph(pipelineGraph, graphInputs, graphOutputs);
      buf.cmdPopDebugGroupLabel();

      timestamp(GPUTimestamp_Inference);

      buf.cmdPushDebugGroupLabel("DFAOIT: resolve", 0xff00ff00);
      buf.cmdBindComputePipeline(pipelineResolve);
      buf.cmdPushConstants(pcResolve);
      buf.cmdDispatch(groups16,
                      {
                          .sampledImages = {texSum, texAccumulated, texCount},
                          .storageImages = {texApproximate},
                          .buffers = {nearestBuffer, slotsBuffer},
                          .tensors = {tensorOutput},
                      });
      buf.cmdPopDebugGroupLabel();

      timestamp(GPUTimestamp_Resolve);
    } else {
      timestamp(GPUTimestamp_Features);
      timestamp(GPUTimestamp_Inference);
      timestamp(GPUTimestamp_Resolve);
    }

    const bool measureError = compare && needApproximate;

    if (measureError) {
      buf.cmdBindComputePipeline(pipelineError);
      buf.cmdPushConstants(ErrorPushConstants{
          .rows = ctx->gpuAddress(rowSums),
          .texResult = texApproximate.index(),
          .texReference = texExact.index(),
          .texCount = texCount.index(),
          .layers = builtLayers,
          .viewWidth = dim.width,
          .viewHeight = dim.height,
      });
      buf.cmdDispatch({1, (dim.height + 63) / 64, 1}, {.sampledImages = {texApproximate, texExact, texCount}, .buffers = {rowSums}});
    }

    const lvk::TextureHandle texResult = needApproximate ? texApproximate : texExact;
    const lvk::Framebuffer framebuffer = {.color = {{.texture = views[0].colorTexture}}};

    buf.cmdBeginRendering(lvk::RenderPass{.color = {{.loadOp = lvk::LoadOp_DontCare, .storeOp = lvk::StoreOp_Store}}},
                          framebuffer,
                          {.sampledImages = {texResult, measureError ? lvk::TextureHandle(texExact) : texResult}});
    buf.cmdBindRenderPipeline(pipelinePresent);
    buf.cmdBindViewport(viewport);
    buf.cmdBindScissorRect(scissor);
    buf.cmdPushConstants(PresentPushConstants{
        .clipRotation = views[0].clipRotation,
        .texResult = texResult.index(),
        .texReference = texExact.index(),
        .sampler0 = samplerClamp.index(),
        .showDifference = measureError && showDifference ? 1u : 0u,
        .gain = differenceGain,
    });
    buf.cmdDraw(3);

    app.imgui_->beginFrame(framebuffer);
    ImGui::Begin("Order-independent transparency", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Combo("Technique", &technique, "A-buffer\0Weighted blended\0DFAOIT\0");
    if (!graphReady) {
      ImGui::TextDisabled("the data graph is not available");
    }
    ImGui::SliderFloat("Alpha", &alpha, 0.0f, 1.0f);
    if (technique == Technique_WeightedBlended) {
      ImGui::SeparatorText("Weighted blended");
      ImGui::SliderFloat("Weight distance", &weightDistance, 1.0f, 500.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
    }
    ImGui::SeparatorText("A-buffer");
    ImGui::SliderInt("Layers", &numLayers, 1, maxLayers);
    ImGui::Checkbox("Tail blending", &tailBlend);
    ImGui::Text("A-buffer: %.1f MB", (8.0 * viewSize * builtLayers) / (1024.0 * 1024.0));
    ImGui::SeparatorText("Quality");
    ImGui::Checkbox("Compare with the A-buffer", &compare);
    if (measureError) {
      ImGui::Checkbox("Show the difference", &showDifference);
      ImGui::SliderFloat("Gain", &differenceGain, 1.0f, 64.0f);
      ImGui::Text("MSE: %.6f", meanSquaredError);
      ImGui::Text("Fragments per pixel: %.1f mean, %.0f max", meanFragments, maxFragments);
      ImGui::Text("Above %u layers: %.1f%% of pixels", builtLayers, percentAboveLayers);
    }
    ImGui::SeparatorText("GPU time");
    ImGui::Checkbox("Measure", &measureTime);
    if (measureTimeThisFrame) {
      ImGui::Text("A-buffer:  %6.2f ms", stageTime[GPUTimestamp_ABuffer]);
      ImGui::Text("%s %6.2f ms", technique == Technique_WeightedBlended ? "Geometry: " : "Features: ", stageTime[GPUTimestamp_Features]);
      ImGui::Text("Inference: %6.2f ms", stageTime[GPUTimestamp_Inference]);
      ImGui::Text("Resolve:   %6.2f ms", stageTime[GPUTimestamp_Resolve]);
      ImGui::Text("Present:   %6.2f ms", stageTime[GPUTimestamp_Present]);
      ImGui::Text("Frame:     %6.2f ms", frameTime);
    }
    if (const uint32_t remaining = numRemainingMaterialTextures()) {
      ImGui::Text("Loading textures: %u left", remaining);
    }
    ImGui::End();
    app.drawFPS();
    app.imgui_->endFrame(buf);

    buf.cmdEndRendering();

    timestamp(GPUTimestamp_Present);

    const lvk::SubmitHandle submitHandle = ctx->submit(buf, views[0].colorTexture);

    if (measureError) {
      ctx->wait(submitHandle);
      ctx->invalidateMappedMemory(rowSums, 0, sizeof(RowStats) * dim.height);
      const RowStats* rows = (const RowStats*)ctx->getMappedPtr(rowSums);
      double squaredError = 0.0;
      double numFragments = 0.0;
      double numAbove = 0.0;
      maxFragments = 0.0;
      for (uint32_t y = 0; y != dim.height; y++) {
        squaredError += rows[y].squaredError;
        numFragments += rows[y].numFragments;
        numAbove += rows[y].numPixelsAboveLayers;
        maxFragments = std::max(maxFragments, (double)rows[y].maxFragments);
      }
      meanSquaredError = squaredError / (3.0 * viewSize);
      meanFragments = numFragments / viewSize;
      percentAboveLayers = 100.0 * numAbove / viewSize;
      if (frameIndex % 100 == 0) {
        LLOGL("MSE %.8f | fragments per pixel: mean %.1f, max %.0f | above %u layers: %.1f%%\n",
              meanSquaredError,
              meanFragments,
              maxFragments,
              builtLayers,
              percentAboveLayers);
      }
    }

    if (measureTimeThisFrame) {
      uint64_t timestamps[GPUTimestamp_NUM] = {};
      ctx->getQueryPoolResults(queryPool, 0, GPUTimestamp_NUM, sizeof(timestamps), timestamps, sizeof(timestamps[0]));
      const double toMs = ctx->getTimestampPeriodToMs();
      for (uint32_t i = GPUTimestamp_Begin + 1; i != GPUTimestamp_NUM; i++) {
        const double ms = (double)(timestamps[i] - timestamps[i - 1]) * toMs;
        stageTime[i] = stageTime[i] > 0.0 ? glm::mix(stageTime[i], ms, 0.05) : ms;
      }
      const double frameMs = (double)(timestamps[GPUTimestamp_Present] - timestamps[GPUTimestamp_Begin]) * toMs;
      frameTime = frameTime > 0.0 ? glm::mix(frameTime, frameMs, 0.05) : frameMs;
      if (frameIndex % 100 == 0) {
        LLOGL("GPU time, ms: frame %.2f | A-buffer %.2f, %s %.2f, inference %.2f, resolve %.2f, present %.2f | %.1f FPS\n",
              frameTime,
              stageTime[GPUTimestamp_ABuffer],
              technique == Technique_WeightedBlended ? "geometry" : "features",
              stageTime[GPUTimestamp_Features],
              stageTime[GPUTimestamp_Inference],
              stageTime[GPUTimestamp_Resolve],
              stageTime[GPUTimestamp_Present],
              app.fpsCounter_.getFPS());
      }
    }

    frameIndex++;
  });

  ctx->wait({});
  cancelLoadingMaterialTextures();

  VULKAN_APP_EXIT();
}
