/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm Neural Super Sampling (NSS) on top of VK_ARM_tensors and VK_ARM_data_graph, fully bindless.
 *
 * The Bistro scene is rendered at 960x540 with subpixel jitter into color, depth and motion vector targets and upscaled
 * to 1920x1080 by the NSS v1 pipeline (`--4k` runs the 1920x1080 -> 3840x2160 model instead, which
 * `tools/make_nss_model.py` produces from the published one):
 *
 *   depth scatter (compute) -> preprocess (compute, writes the int8 input tensor) -> data graph (VK_ARM_data_graph,
 *   the neural network from a VGF file) -> postprocess (compute, kernel prediction filter + temporal accumulation)
 *
 * The NSS algorithm comes from the Arm Neural Graphics SDK for Game Engines, deployed by `deploy_deps.py`; the compute entry
 * points and the shader permutation are compiled into this file. On GPUs
 * without native VK_ARM_tensors support the Arm ML Emulation Layer for Vulkan is used when LightweightVK was configured
 * with `LVK_WITH_ML_EMULATION_LAYER=ON` (run with `--no-ml-emulation` to disable it). `--no-nss` starts with NSS disabled,
 * `--camera-rotate` rotates the camera continuously (temporal stability test; combine with `--screenshot-frame`),
 * `--ao-samples N` sets the number of ambient occlusion rays per pixel (0 disables it).
 * `--full-res` starts with the scene rendered at the output resolution and NSS off, which is what NSS is compared
 * against; the "F" key and the render mode combo box switch between the two at run time.
 *
 * 1) Run the script "deploy_deps.py" from the LightweightVK root folder.
 * 2) Run the script "deploy_content.py" from the LightweightVK root folder.
 * 3) Run this app.
 */

#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif // _USE_MATH_DEFINES
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdio.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>

#include <shared/UtilsCubemap.h>
#include <stb/stb_image.h>
#include <stb/stb_image_resize2.h>

#define FFX_CPU 1
#include <neural-graphics-sdk/sdk/include/FidelityFX/gpu/nss/ffx_nss_resources.h>
#undef FFX_CPU

#include <lvk/HelpersImGui.h>
#include <lvk/HelpersVgf.h>
#include <lvk/LVK.h>

#include <ktx-software/lib/src/gl_format.h>
#include <ktx.h>
#include <ldrutils/lutils/ScopeExit.h>

#include "Bistro.h"
#include "VulkanApp.h"

#define MODEL_PATH "src/bistro/Exterior/exterior.obj"

uint32_t kRenderWidth = 960;
uint32_t kRenderHeight = 540;
uint32_t kOutputWidth = 1920;
uint32_t kOutputHeight = 1080;
const char* kNssModelFile = "src/nss/2_nss-960x540-v1_0_1.vgf";
const char* kNssSdkFolder = "neural-graphics-sdk";
constexpr float kCameraNear = 0.5f;
constexpr float kCameraFar = 500.0f;
constexpr float kCameraFovY = float(45.0f * (M_PI / 180.0f));
constexpr float kSceneScale = 0.05f;

std::string folderThirdParty;
std::string folderContentRoot;
std::vector<std::string> folderShadersNSS;

const VulkanApp* app_ = nullptr;

std::vector<uint8_t> readContentFile(const std::string& path) {
  return app_ ? app_->loadFile(path.c_str()) : std::vector<uint8_t>();
}

// clang-format off

#define PER_FRAME_GLSL                                            \
  "layout(std430, buffer_reference) readonly buffer PerFrame {\n" \
  "  mat4 proj;\n"                                                \
  "  mat4 projNoJitter;\n"                                        \
  "  mat4 view;\n"                                                \
  "  mat4 viewProjPrev;\n"                                        \
  "  mat4 skyboxViewProjPrev;\n"                                  \
  "  mat4 light;\n"                                               \
  "  uint texSkyboxRadiance;\n"                                   \
  "  uint texSkyboxIrradiance;\n"                                 \
  "  uint texShadow;\n"                                           \
  "  uint sampler0;\n"                                            \
  "  uint samplerShadow0;\n"                                      \
  "};\n"

#define MOTION_VECTOR_GLSL                                 \
  "vec2 clipToUv(vec4 clip) {\n"                           \
  "  const vec2 ndc = clip.xy / clip.w;\n"                 \
  "  return vec2(0.5 + 0.5 * ndc.x, 0.5 - 0.5 * ndc.y);\n" \
  "}\n"                                                    \
  "vec2 motionVector(vec4 clipCur, vec4 clipPrev) {\n"     \
  "  return clipToUv(clipPrev) - clipToUv(clipCur);\n"     \
  "}\n"

const char* kAOPushConstants = R"(
  uint tlas;
  uint aoSamples;
  uint frameId;
  float aoRadius;
  float aoPower;
)";

const char* kAODisabled = R"(
float ambientOcclusion(vec3 worldPos, vec3 n, uint tlas, uint samples, uint frameId, float radius, float power) {
  return 1.0;
}
)";

const char* kAOEnabled = R"(
float traceAO(rayQueryEXT rq, uint tlas, vec3 origin, vec3 dir, float radius) {
  rayQueryInitializeEXT(rq, kTLAS[tlas], gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 0.0, dir, radius);
  while (rayQueryProceedEXT(rq)) {}
  return (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) ? 1.0 : 0.0;
}

uint lcg(inout uint prev) {
  prev = 1664525u * prev + 1013904223u;
  return prev & 0x00FFFFFFu;
}

float rnd(inout uint seed) {
  return float(lcg(seed)) / float(0x01000000);
}

uint tea(uint val0, uint val1) {
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0;
  for (uint n = 0; n < 16; n++) {
    s0 += 0x9e3779b9u;
    v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4u);
    v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761eu);
  }
  return v0;
}

void computeTBN(in vec3 n, out vec3 x, out vec3 y) {
  const float yz = -n.y * n.z;
  y = normalize((abs(n.z) > 0.9999) ? vec3(-n.x * n.y, 1.0 - n.y * n.y, yz) : vec3(-n.x * n.z, yz, 1.0 - n.z * n.z));
  x = cross(y, n);
}

vec3 sampleCosineHemisphere(inout uint seed, vec3 tangent, vec3 bitangent, vec3 n) {
  const float r1 = rnd(seed);
  const float r2 = rnd(seed);
  const float sq = sqrt(1.0 - r2);
  const float phi = 2.0 * 3.141592653589 * r1;
  const vec3 d = vec3(cos(phi) * sq, sin(phi) * sq, sqrt(r2));
  return d.x * tangent + d.y * bitangent + d.z * n;
}

float ambientOcclusion(vec3 worldPos, vec3 n, uint tlas, uint samples, uint frameId, float radius, float power) {
  if (samples == 0)
    return 1.0;
  const vec3 origin = worldPos + n * 0.001;
  vec3 tangent, bitangent;
  computeTBN(n, tangent, bitangent);
  uint seed = tea(uint(gl_FragCoord.y * 4003.0 + gl_FragCoord.x), frameId);
  float occlusion = 0.0;
  for (uint i = 0; i < samples; i++) {
    rayQueryEXT rq;
    occlusion += traceAO(rq, tlas, origin, sampleCosineHemisphere(seed, tangent, bitangent, n), radius);
  }
  return pow(clamp(1.0 - occlusion / float(samples), 0.0, 1.0), power);
}
)";

const char* kCodeVSHead = PER_FRAME_GLSL R"(
layout (location=0) in vec3 pos;
layout (location=1) in vec2 uv;
layout (location=2) in uint normal;
layout (location=3) in uint mtlIndex;

struct Material {
   vec4 ambient;
   vec4 diffuse;
   int texAmbient;
   int texDiffuse;
   int texAlpha;
   int padding;
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
  mat4 normal;
};

layout(std430, buffer_reference) readonly buffer Materials {
  Material mtl[];
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  PerObject perObject;
  Materials materials;
)";

const char* kCodeVSTail = R"(
} pc;

struct PerVertex {
  vec3 normal;
  vec2 uv;
  vec4 shadowCoords;
  vec3 worldPos;
};
layout (location=0) out PerVertex vtx;
layout (location=4) out vec4 clipCur;
layout (location=5) out vec4 clipPrev;
layout (location=6) flat out Material mtl;

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
  mat4 model = pc.perObject.model;
  mtl = pc.materials.mtl[mtlIndex];
  const vec4 worldPos = model * vec4(pos, 1.0);
  const vec4 viewPos = pc.perFrame.view * worldPos;
  gl_Position = pc.perFrame.proj * viewPos;
  clipCur = pc.perFrame.projNoJitter * viewPos;
  clipPrev = pc.perFrame.viewProjPrev * worldPos;

  vtx.normal = normalize(mat3(pc.perObject.normal) * unpackOctahedral16(normal));
  vtx.uv = uv;
  vtx.shadowCoords = pc.perFrame.light * worldPos;
  vtx.worldPos = worldPos.xyz;
}
)";

const char* kCodeFSHead = PER_FRAME_GLSL MOTION_VECTOR_GLSL R"(
struct Material {
  vec4 ambient;
  vec4 diffuse;
  int texAmbient;
  int texDiffuse;
  int texAlpha;
  int padding;
};

struct PerVertex {
  vec3 normal;
  vec2 uv;
  vec4 shadowCoords;
  vec3 worldPos;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  uvec2 perObject;
  uvec2 materials;
)";

const char* kCodeFSTail = R"(
} pc;

layout (location=0) in PerVertex vtx;
layout (location=4) in vec4 clipCur;
layout (location=5) in vec4 clipPrev;
layout (location=6) flat in Material mtl;

layout (location=0) out vec4 out_FragColor;
layout (location=1) out vec2 out_Motion;

float PCF3(vec3 uvw) {
  float size = 1.0 / textureBindlessSize2D(pc.perFrame.texShadow).x;
  float shadow = 0.0;
  for (int v=-1; v<=+1; v++)
    for (int u=-1; u<=+1; u++)
      shadow += textureBindless2DShadow(pc.perFrame.texShadow, pc.perFrame.samplerShadow0, uvw + size * vec3(u, v, 0));
  return shadow / 9;
}

float shadow(vec4 s) {
  s = s / s.w;
  if (s.z > -1.0 && s.z < 1.0) {
    float depthBias = -0.00005;
    float shadowSample = PCF3(vec3(s.x, 1.0 - s.y, s.z + depthBias));
    return mix(0.3, 1.0, shadowSample);
  }
  return 1.0;
}

void main() {
  vec4 alpha = textureBindless2D(mtl.texAlpha, pc.perFrame.sampler0, vtx.uv);
  if (mtl.texAlpha > 0 && alpha.r < 0.5)
    discard;
  vec4 Ka = mtl.ambient * textureBindless2D(mtl.texAmbient, pc.perFrame.sampler0, vtx.uv);
  vec4 Kd = mtl.diffuse * textureBindless2D(mtl.texDiffuse, pc.perFrame.sampler0, vtx.uv);
  if (Kd.a < 0.5)
    discard;
  vec3 n = normalize(vtx.normal);
  const vec4 f0 = vec4(0.04);
  vec4 diffuse = textureBindlessCube(pc.perFrame.texSkyboxIrradiance, pc.perFrame.sampler0, n) * Kd * (vec4(1.0) - f0);
  const float ao = ambientOcclusion(vtx.worldPos, n, pc.tlas, pc.aoSamples, pc.frameId, pc.aoRadius, pc.aoPower);
  out_FragColor = vec4((Ka + diffuse * shadow(vtx.shadowCoords) * ao).rgb, 1.0);
  out_Motion = motionVector(clipCur, clipPrev);
};
)";

const char* kShadowVS = PER_FRAME_GLSL R"(
layout (location=0) in vec3 pos;

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  PerObject perObject;
} pc;

void main() {
  gl_Position = pc.perFrame.proj * pc.perFrame.view * pc.perObject.model * vec4(pos, 1.0);
}
)";

const char* kShadowFS = R"(
void main() {
};
)";

const char* kSkyboxVS = PER_FRAME_GLSL R"(
layout (location=0) out vec3 textureCoords;
layout (location=1) out vec4 clipCur;
layout (location=2) out vec4 clipPrev;

const vec3 positions[8] = vec3[8](
	vec3(-1.0,-1.0, 1.0), vec3( 1.0,-1.0, 1.0), vec3( 1.0, 1.0, 1.0), vec3(-1.0, 1.0, 1.0),
	vec3(-1.0,-1.0,-1.0), vec3( 1.0,-1.0,-1.0), vec3( 1.0, 1.0,-1.0), vec3(-1.0, 1.0,-1.0)
);

const int indices[36] = int[36](
	0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7, 4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3
);

layout(push_constant) uniform constants
{
	PerFrame perFrame;
} pc;

void main() {
  mat4 view = pc.perFrame.view;
  view = mat4(view[0], view[1], view[2], vec4(0, 0, 0, 1));
  vec3 pos = positions[indices[gl_VertexIndex]];
  gl_Position = (pc.perFrame.proj * view * vec4(pos, 1.0)).xyww;
  clipCur = pc.perFrame.projNoJitter * view * vec4(pos, 1.0);
  clipPrev = pc.perFrame.skyboxViewProjPrev * vec4(pos, 1.0);

  textureCoords = pos;
}

)";
const char* kSkyboxFS = PER_FRAME_GLSL MOTION_VECTOR_GLSL R"(
layout (location=0) in vec3 textureCoords;
layout (location=1) in vec4 clipCur;
layout (location=2) in vec4 clipPrev;
layout (location=0) out vec4 out_FragColor;
layout (location=1) out vec2 out_Motion;

layout(push_constant) uniform constants {
  PerFrame perFrame;
} pc;

void main() {
  out_FragColor = vec4(textureBindlessCube(pc.perFrame.texSkyboxRadiance, pc.perFrame.sampler0, textureCoords).rgb, 1.0);
  out_Motion = motionVector(clipCur, clipPrev);
}
)";

const char* kCodeFullscreenVS = R"(
layout (location=0) out vec2 uv;

layout(push_constant) uniform constants {
	mat4 clipRotation;
	uint tex;
	uint smp;
} pc;

void main() {
  uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = pc.clipRotation * vec4(uv * vec2(2, -2) + vec2(-1, 1), 0.0, 1.0);
}
)";

const char* kCodeFullscreenFS = R"(
layout (location=0) in vec2 uv;
layout (location=0) out vec4 out_FragColor;

layout(push_constant) uniform constants {
	mat4 clipRotation;
	uint tex;
	uint smp;
} pc;

void main() {
  out_FragColor = vec4(textureBindless2D(pc.tex, pc.smp, uv).rgb, 1.0);
}
)";

#define NSS_CONFIG_GLSL R"(
#define FFX_GPU 1
#define FFX_GLSL 1
#define FFX_HALF 1

#define QUANTIZED 1
#define REVERSE_Z 0
#define NSS_SHADER_QUALITY_MODE 0
#define SCALE_PRESET_MODE NSS_SCALE_PRESET_X2
#define NSS_SUPPORT_TENSOR 1
#define NSS_BINDLESS 1
#define MANAGE_HISTORY 0
)"

const char* kNssDepthScatter = NSS_CONFIG_GLSL R"(

#define NSS_BIND_SRV_INPUT_DEPTH 1
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS 2
#define NSS_BIND_UAV_DEPTH_TM1 3
#define NSS_BIND_CB_NSS 4

#include "ffx_nss_depth_scatter.h"

layout(local_size_x = FFX_NSS_THREAD_GROUP_WIDTH, local_size_y = FFX_NSS_THREAD_GROUP_HEIGHT, local_size_z = FFX_NSS_THREAD_GROUP_DEPTH) in;

void main() {
  DepthScatter(int32_t2(gl_GlobalInvocationID.xy));
}
)";

const char* kNssPreprocess = NSS_CONFIG_GLSL R"(

#define NSS_BIND_SRV_INPUT_COLOR_JITTERED 0
#define NSS_BIND_SRV_INPUT_DEPTH 1
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS 2
#define NSS_BIND_SRV_HISTORY_UPSCALED_COLOR 3
#define NSS_BIND_SRV_FEEDBACK_TENSOR 4
#define NSS_BIND_SRV_INPUT_DEPTH_TM1 5
#define NSS_BIND_SRV_LUMA_DERIV_TM1 6
#define NSS_BIND_PREPROCESS_INPUT_TENSOR 8
#define NSS_BIND_UAV_LUMA_DERIV 9
#define NSS_BIND_UAV_NEAREST_DEPTH_COORD 10
#define NSS_BIND_CB_NSS 12

#define NSS_PREPROCESS 1

#include "ffx_nss_preprocess.h"

layout(local_size_x = FFX_NSS_THREAD_GROUP_WIDTH, local_size_y = FFX_NSS_THREAD_GROUP_HEIGHT, local_size_z = FFX_NSS_THREAD_GROUP_DEPTH) in;

void main() {
  Preprocess(int32_t2(gl_GlobalInvocationID.xy));
}
)";

const char* kNssPostprocess = NSS_CONFIG_GLSL R"(

#define NSS_BIND_SRV_INPUT_COLOR_JITTERED 0
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS 1
#define NSS_BIND_SRV_HISTORY_UPSCALED_COLOR 2
#define NSS_BIND_KPN_TENSOR 3
#define NSS_BIND_SRV_FEEDBACK_TENSOR 4
#define NSS_BIND_SRV_NEAREST_DEPTH_COORD 5
#define NSS_BIND_UAV_UPSCALED_OUTPUT 7
#define NSS_BIND_CB_NSS 9

#define NSS_POSTPROCESS 1

#include "ffx_nss_postprocess.h"

layout(local_size_x = FFX_NSS_THREAD_GROUP_WIDTH, local_size_y = FFX_NSS_THREAD_GROUP_HEIGHT, local_size_z = FFX_NSS_THREAD_GROUP_DEPTH) in;

void main() {
  Postprocess(int32_t2(gl_GlobalInvocationID.xy));
}
)";

const char* kNssFeedbackToImage = NSS_CONFIG_GLSL R"(

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 2, rgba8_snorm) uniform writeonly image2D kImagesRGBA8Snorm[];

layout(push_constant) uniform PushConstants {
  uint tensor;
  uint image;
  uint width;
  uint height;
} pc;

void main() {
  const uvec2 p = gl_GlobalInvocationID.xy;
  if (p.x >= pc.width || p.y >= pc.height) {
    return;
  }
  int8_t v[4];
  tensorReadARM(kTensorsI8_4[pc.tensor], uint[](0, p.y, p.x, 0), v);
  const vec4 snorm = max(vec4(v[0], v[1], v[2], v[3]) / 127.0, vec4(-1.0));
  imageStore(kImagesRGBA8Snorm[pc.image], ivec2(p), snorm);
}
)";

// clang-format on

lvk::IContext* ctx_ = nullptr;

lvk::Holder<lvk::TextureHandle> texColorLR_;
lvk::Holder<lvk::TextureHandle> texMotionLR_;
lvk::Holder<lvk::TextureHandle> texDepthLR_;
uint32_t sceneWidth_ = kRenderWidth;
uint32_t sceneHeight_ = kRenderHeight;
bool fullResScene_ = false;
bool fullResScenePending_ = false;
enum RenderMode {
  RenderMode_FullResolution = 0,
  RenderMode_HalfResolution,
  RenderMode_HalfResolutionNSS,
  RenderMode_NUM_MODES,
};
int renderMode_ = RenderMode_HalfResolutionNSS;
lvk::Framebuffer fbShadowMap_;
lvk::Framebuffer fbOffscreen_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshVert_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smShadowVert_;
lvk::Holder<lvk::ShaderModuleHandle> smShadowFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smFullscreenVert_;
lvk::Holder<lvk::ShaderModuleHandle> smFullscreenFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smSkyboxVert_;
lvk::Holder<lvk::ShaderModuleHandle> smSkyboxFrag_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Mesh_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Shadow_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Skybox_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Fullscreen_;
lvk::Holder<lvk::BufferHandle> vb0_, ib0_;
std::vector<lvk::Holder<lvk::AccelStructHandle>> BLAS_;
lvk::Holder<lvk::AccelStructHandle> TLAS_;
lvk::Holder<lvk::BufferHandle> sbInstances_;
bool rayQuery_ = false;
int aoSamples_ = 8;
float aoRadius_ = 20.0f;
float aoPower_ = 1.0f;
uint32_t frameId_ = 0;
lvk::Holder<lvk::BufferHandle> sbMaterials_;
lvk::Holder<lvk::BufferHandle> ubPerFrame_, ubPerFrameShadow_, ubPerObject_;
lvk::Holder<lvk::SamplerHandle> sampler_;
lvk::Holder<lvk::SamplerHandle> samplerShadow_;
lvk::Holder<lvk::SamplerHandle> samplerLinearClamp_;
lvk::Holder<lvk::TextureHandle> skyboxTextureReference_;
lvk::Holder<lvk::TextureHandle> skyboxTextureIrradiance_;
lvk::RenderPass renderPassOffscreen_;
lvk::RenderPass renderPassMain_;
lvk::RenderPass renderPassShadow_;
lvk::DepthState depthState_;
lvk::DepthState depthStateLEqual_;

bool isShadowMapDirty_ = true;

struct UniformsPerFrame {
  mat4 proj;
  mat4 projNoJitter;
  mat4 view;
  mat4 viewProjPrev;
  mat4 skyboxViewProjPrev;
  mat4 light;
  uint32_t texSkyboxRadiance = 0;
  uint32_t texSkyboxIrradiance = 0;
  uint32_t texShadow = 0;
  uint32_t sampler = 0;
  uint32_t samplerShadow = 0;
} perFrame_;

struct UniformsPerObject {
  mat4 model;
  mat4 normal;
};

bool initAccelerationStructures() {
  const glm::mat3x4 identity(1.0f);
  lvk::Holder<lvk::BufferHandle> transformBuffer = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(glm::mat3x4),
      .data = &identity,
      .debugName = "Buffer: BLAS transform",
  });

  const uint32_t totalPrimitiveCount = uint32_t(indexData_.size()) / 3;
  lvk::AccelStructDesc blasDesc{
      .type = lvk::AccelStructType_BLAS,
      .geometryType = lvk::AccelStructGeomType_Triangles,
      .vertexFormat = lvk::VertexFormat_Float3,
      .vertexBuffer = vb0_,
      .vertexStride = sizeof(VertexData),
      .numVertices = uint32_t(vertexData_.size()),
      .indexFormat = lvk::IndexFormat_UI32,
      .indexBuffer = ib0_,
      .transformBuffer = transformBuffer,
      .buildRange = {.primitiveCount = totalPrimitiveCount},
      .buildFlags = lvk::AccelStructBuildFlagBits_PreferFastTrace,
      .debugName = "BLAS",
  };

  const lvk::AccelStructSizes sizes = ctx_->getAccelStructSizes(blasDesc);
  const uint32_t maxStorageBufferSize = ctx_->getMaxStorageBufferRange();
  const uint32_t numBLAS = 1 + std::max(uint32_t(sizes.buildScratchSize / maxStorageBufferSize),
                                        uint32_t(sizes.accelerationStructureSize / maxStorageBufferSize));

  const glm::mat3x4 transform(glm::scale(mat4(1.0f), vec3(kSceneScale)));
  const uint32_t primitiveCount = totalPrimitiveCount / numBLAS;

  BLAS_.reserve(numBLAS);
  std::vector<lvk::AccelStructInstance> instances;
  instances.reserve(numBLAS);
  for (uint32_t first = 0; first < totalPrimitiveCount; first += primitiveCount) {
    blasDesc.buildRange.primitiveOffset = first * 3 * sizeof(uint32_t);
    blasDesc.buildRange.primitiveCount = std::min(primitiveCount, totalPrimitiveCount - first);
    BLAS_.emplace_back(ctx_->createAccelerationStructure(blasDesc));
    instances.emplace_back(lvk::AccelStructInstance{
        .transform = (const lvk::mat3x4&)transform,
        .instanceCustomIndex = 0,
        .mask = 0xff,
        .instanceShaderBindingTableRecordOffset = 0,
        .flags = lvk::AccelStructInstanceFlagBits_TriangleFacingCullDisable,
        .accelerationStructureReference = ctx_->gpuAddress(BLAS_.back()),
    });
  }

  sbInstances_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(lvk::AccelStructInstance) * instances.size(),
      .data = instances.data(),
      .debugName = "Buffer: TLAS instances",
  });
  TLAS_ = ctx_->createAccelerationStructure({
      .type = lvk::AccelStructType_TLAS,
      .geometryType = lvk::AccelStructGeomType_Instances,
      .instancesBuffer = sbInstances_,
      .buildRange = {.primitiveCount = uint32_t(instances.size())},
      .buildFlags = lvk::AccelStructBuildFlagBits_PreferFastTrace,
      .debugName = "TLAS",
  });

  LLOGL("Ray-traced ambient occlusion: %zu BLAS over %u triangles\n", BLAS_.size(), totalPrimitiveCount);

  return TLAS_.valid();
}

bool initModel(VulkanApp& app) {
  const std::string cacheFileName = folderContentRoot + "cache.data";

  if (!loadFromCache(app, cacheFileName.c_str())) {
    if (!LVK_VERIFY(loadAndCache(app, cacheFileName.c_str(), MODEL_PATH))) {
      LVK_ASSERT_MSG(false, "Cannot load 3D model");
      return false;
    }
  }

  loadMaterialTextures(app, "src/bistro/Exterior/");

  sbMaterials_ = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Storage,
                                     .storage = lvk::StorageType_Device,
                                     .size = sizeof(GPUMaterial) * materials_.size(),
                                     .data = materials_.data(),
                                     .debugName = "Buffer: materials"},
                                    nullptr);

  const uint8_t accelStructUsage = rayQuery_ ? lvk::BufferUsageBits_AccelStructBuildInputReadOnly : 0;
  vb0_ = ctx_->createBuffer({.usage = uint8_t(lvk::BufferUsageBits_Vertex | accelStructUsage),
                             .storage = lvk::StorageType_Device,
                             .size = sizeof(VertexData) * vertexData_.size(),
                             .data = vertexData_.data(),
                             .debugName = "Buffer: vertex"},
                            nullptr);
  ib0_ = ctx_->createBuffer({.usage = uint8_t(lvk::BufferUsageBits_Index | accelStructUsage),
                             .storage = lvk::StorageType_Device,
                             .size = sizeof(uint32_t) * indexData_.size(),
                             .data = indexData_.data(),
                             .debugName = "Buffer: index"},
                            nullptr);
  return rayQuery_ ? initAccelerationStructures() : true;
}

lvk::Format ktx2iglTextureFormat(ktx_uint32_t format) {
  switch (format) {
  case GL_RGBA32F:
    return lvk::Format_RGBA_F32;
  case GL_RG16F:
    return lvk::Format_RG_F16;
  default:;
  }
  LVK_ASSERT_MSG(false, "Code should NOT be reached");
  return lvk::Format_RGBA_UN8;
}

void loadCubemapTexture(const std::string& fileNameKTX, lvk::Holder<lvk::TextureHandle>& tex) {
  LVK_PROFILER_FUNCTION();

  ktxTexture1* texture = nullptr;
  const std::vector<uint8_t> blobKTX = readContentFile(fileNameKTX);
  (void)LVK_VERIFY(!blobKTX.empty() &&
                   ktxTexture1_CreateFromMemory(blobKTX.data(), blobKTX.size(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture) ==
                       KTX_SUCCESS);
  if (!texture) {
    return;
  }
  SCOPE_EXIT {
    ktxTexture_Destroy(ktxTexture(texture));
  };

  if (!LVK_VERIFY(texture->glInternalformat == GL_RGBA32F)) {
    LVK_ASSERT_MSG(false, "Texture format not supported");
    return;
  }

  const uint32_t width = texture->baseWidth;
  const uint32_t height = texture->baseHeight;

  if (tex.empty()) {
    tex = ctx_->createTexture({
        .type = lvk::TextureType_Cube,
        .format = ktx2iglTextureFormat(texture->glInternalformat),
        .dimensions = {width, height},
        .usage = lvk::TextureUsageBits_Sampled,
        .numMipLevels = lvk::calcNumMipLevels(width, height),
        .data = texture->pData,
        .dataNumMipLevels = kEnableTextureCompression ? lvk::calcNumMipLevels(width, height) : 1u,
        .generateMipmaps = !kEnableTextureCompression,
        .debugName = fileNameKTX.c_str(),
    });
  }
}

ktxTexture1* bitmapToCube(Bitmap& bmp) {
  LVK_ASSERT(bmp.comp_ == 3);
  LVK_ASSERT(bmp.type_ == eBitmapType_Cube);
  LVK_ASSERT(bmp.fmt_ == eBitmapFormat_Float);

  const int w = bmp.w_;
  const int h = bmp.h_;

  const uint32_t mipLevels = lvk::calcNumMipLevels(w, h);

  ktxTextureCreateInfo createInfo = {
      .glInternalformat = GL_RGBA32F,
      .vkFormat = VK_FORMAT_R32G32B32A32_SFLOAT,
      .baseWidth = static_cast<uint32_t>(w),
      .baseHeight = static_cast<uint32_t>(h),
      .baseDepth = 1u,
      .numDimensions = 2u,
      .numLevels = mipLevels,
      .numLayers = 1u,
      .numFaces = 6u,
      .generateMipmaps = KTX_FALSE,
  };

  ktxTexture1* texture = nullptr;
  (void)LVK_VERIFY(ktxTexture1_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture) == KTX_SUCCESS);

  const int numFacePixels = w * h;

  for (size_t face = 0; face != 6; face++) {
    const vec3* src = reinterpret_cast<vec3*>(bmp.data_.data()) + face * numFacePixels;
    size_t offset = 0;
    (void)LVK_VERIFY(ktxTexture_GetImageOffset(ktxTexture(texture), 0, 0, face, &offset) == KTX_SUCCESS);
    float* dst = (float*)(texture->pData + offset);
    for (int y = 0; y != h; y++) {
      for (int x = 0; x != w; x++) {
        const vec4 rgba = vec4(src[x + y * w], 1.0f);
        memcpy(dst, &rgba, sizeof(rgba));
        dst += 4;
      }
    }
  }

  return texture;
}

void generateMipmaps(const std::string& outFilename, ktxTexture1* cubemap) {
  LVK_PROFILER_FUNCTION();

  LLOGL("Generating mipmaps");

  LVK_ASSERT(cubemap);

  uint32_t prevWidth = cubemap->baseWidth;
  uint32_t prevHeight = cubemap->baseHeight;

  for (uint32_t face = 0; face != 6; face++) {
    LLOGL(".");
    for (uint32_t miplevel = 1; miplevel < cubemap->numLevels; miplevel++) {
      LLOGL(":");
      const uint32_t width = prevWidth > 1 ? prevWidth >> 1 : 1;
      const uint32_t height = prevHeight > 1 ? prevWidth >> 1 : 1;

      size_t prevOffset = 0;
      (void)LVK_VERIFY(ktxTexture_GetImageOffset(ktxTexture(cubemap), miplevel - 1, 0, face, &prevOffset) == KTX_SUCCESS);
      size_t offset = 0;
      (void)LVK_VERIFY(ktxTexture_GetImageOffset(ktxTexture(cubemap), miplevel, 0, face, &offset) == KTX_SUCCESS);

      stbir_resize_float_linear(reinterpret_cast<const float*>(cubemap->pData + prevOffset),
                                prevWidth,
                                prevHeight,
                                0,
                                reinterpret_cast<float*>(cubemap->pData + offset),
                                width,
                                height,
                                0,
                                STBIR_RGBA);

      prevWidth = width;
      prevHeight = height;
    }
    prevWidth = cubemap->baseWidth;
    prevHeight = cubemap->baseHeight;
  }

  LLOGL("\n");
  ktxTexture_WriteToNamedFile(ktxTexture(cubemap), outFilename.c_str());
}

void processCubemap(const std::string& inFilename, const std::string& outFilenameEnv, const std::string& outFilenameIrr) {
  LVK_PROFILER_FUNCTION();

  int sourceWidth, sourceHeight;
  float* pxs = stbi_loadf(inFilename.c_str(), &sourceWidth, &sourceHeight, nullptr, 3);
  SCOPE_EXIT {
    if (pxs) {
      stbi_image_free(pxs);
    }
  };

  if (!LVK_VERIFY(pxs)) {
    LVK_ASSERT_MSG(false, "Did you read the tutorial at the top of this file?");
    return;
  }

  {
    Bitmap bmp = convertEquirectangularMapToCubeMapFaces(Bitmap(sourceWidth, sourceHeight, 3, eBitmapFormat_Float, pxs));
    ktxTexture1* cube = bitmapToCube(bmp);
    generateMipmaps(outFilenameEnv, cube);
    ktxTexture_Destroy(ktxTexture(cube));
  }

  {
    constexpr int dstW = 256;
    constexpr int dstH = 128;

    std::vector<vec3> out(dstW * dstH);
    convolveDiffuse((vec3*)pxs, sourceWidth, sourceHeight, dstW, dstH, out.data(), 1024);

    Bitmap bmp = convertEquirectangularMapToCubeMapFaces(Bitmap(dstW, dstH, 3, eBitmapFormat_Float, out.data()));
    ktxTexture1* cube = bitmapToCube(bmp);
    generateMipmaps(outFilenameIrr, cube);
    ktxTexture_Destroy(ktxTexture(cube));
  }
}

void loadSkyboxTexture() {
  LVK_PROFILER_FUNCTION();

  const std::string skyboxFileName{"immenstadter_horn_2k"};
  const std::string skyboxSubdir{"src/skybox_hdr/"};

  const std::string fileNameRefKTX = folderContentRoot + skyboxFileName + "_ReferenceMap.ktx";
  const std::string fileNameIrrKTX = folderContentRoot + skyboxFileName + "_IrradianceMap.ktx";

#if !defined(ANDROID)
  if (!std::filesystem::exists(fileNameRefKTX) || !std::filesystem::exists(fileNameIrrKTX)) {
    const std::string inFilename = folderContentRoot + skyboxSubdir + skyboxFileName + ".hdr";
    LLOGL("Cubemap in KTX format not found. Extracting from HDR file `%s`...\n", inFilename.c_str());

    processCubemap(inFilename, fileNameRefKTX, fileNameIrrKTX);
  }
#endif // !ANDROID

  loadCubemapTexture(fileNameRefKTX, skyboxTextureReference_);
  loadCubemapTexture(fileNameIrrKTX, skyboxTextureIrradiance_);
}

void createShadowMap() {
  const uint32_t w = 4096;
  const uint32_t h = 4096;
  const lvk::TextureDesc desc = {
      .type = lvk::TextureType_2D,
      .format = lvk::Format_Z_UN16,
      .dimensions = {w, h},
      .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
      .numMipLevels = lvk::calcNumMipLevels(w, h),
      .debugName = "Shadow map",
  };
  fbShadowMap_ = {
      .depthStencil = {.texture = ctx_->createTexture(desc).release()},
  };
}

void createOffscreenFramebuffer() {
  texColorLR_ = ctx_->createTexture({
      .type = lvk::TextureType_2D,
      .format = lvk::Format_RGBA_F16,
      .dimensions = {sceneWidth_, sceneHeight_},
      .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
      .debugName = "LR color (jittered)",
  });
  texMotionLR_ = ctx_->createTexture({
      .type = lvk::TextureType_2D,
      .format = lvk::Format_RG_F16,
      .dimensions = {sceneWidth_, sceneHeight_},
      .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
      .debugName = "LR motion vectors",
  });
  texDepthLR_ = ctx_->createTexture({
      .type = lvk::TextureType_2D,
      .format = lvk::Format_Z_F32,
      .dimensions = {sceneWidth_, sceneHeight_},
      .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
      .debugName = "LR depth",
  });
  fbOffscreen_ = {
      .color = {{.texture = texColorLR_}, {.texture = texMotionLR_}},
      .depthStencil = {.texture = texDepthLR_},
  };
}

void createPipelines() {
  const lvk::VertexInput vdesc = {
      .attributes =
          {
              {.location = 0, .format = lvk::VertexFormat_Float3, .offset = offsetof(VertexData, position)},
              {.location = 1, .format = lvk::VertexFormat_HalfFloat2, .offset = offsetof(VertexData, uv)},
              {.location = 2, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, normal)},
              {.location = 3, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, mtlIndex)},
          },
      .inputBindings = {{.stride = sizeof(VertexData)}},
  };

  const lvk::VertexInput vdescs = {
      .attributes = {{.format = lvk::VertexFormat_Float3, .offset = offsetof(VertexData, position)}},
      .inputBindings = {{.stride = sizeof(VertexData)}},
  };

  const std::string codeVert = std::string(kCodeVSHead) + kAOPushConstants + kCodeVSTail;
  const std::string codeFrag = std::string(rayQuery_ ? kAOEnabled : kAODisabled) + kCodeFSHead + kAOPushConstants + kCodeFSTail;
  smMeshVert_ = ctx_->createShaderModule({codeVert.c_str(), lvk::Stage_Vert, "Shader Module: main (vert)"});
  smMeshFrag_ = ctx_->createShaderModule({codeFrag.c_str(), lvk::Stage_Frag, "Shader Module: main (frag)"});
  smShadowVert_ = ctx_->createShaderModule({kShadowVS, lvk::Stage_Vert, "Shader Module: shadow (vert)"});
  smShadowFrag_ = ctx_->createShaderModule({kShadowFS, lvk::Stage_Frag, "Shader Module: shadow (frag)"});
  smFullscreenVert_ = ctx_->createShaderModule({kCodeFullscreenVS, lvk::Stage_Vert, "Shader Module: fullscreen (vert)"});
  smFullscreenFrag_ = ctx_->createShaderModule({kCodeFullscreenFS, lvk::Stage_Frag, "Shader Module: fullscreen (frag)"});
  smSkyboxVert_ = ctx_->createShaderModule({kSkyboxVS, lvk::Stage_Vert, "Shader Module: skybox (vert)"});
  smSkyboxFrag_ = ctx_->createShaderModule({kSkyboxFS, lvk::Stage_Frag, "Shader Module: skybox (frag)"});

  renderPipelineState_Mesh_ = ctx_->createRenderPipeline(
      {
          .vertexInput = vdesc,
          .smVert = smMeshVert_,
          .smFrag = smMeshFrag_,
          .color = {{.format = ctx_->getFormat(texColorLR_)}, {.format = ctx_->getFormat(texMotionLR_)}},
          .depthFormat = ctx_->getFormat(texDepthLR_),
          .cullMode = lvk::CullMode_Back,
          .frontFace = lvk::WindingMode_CCW,
          .debugName = "Pipeline: mesh",
      },
      nullptr);

  renderPipelineState_Shadow_ = ctx_->createRenderPipeline(
      {
          .vertexInput = vdescs,
          .smVert = smShadowVert_,
          .smFrag = smShadowFrag_,
          .depthFormat = ctx_->getFormat(fbShadowMap_.depthStencil.texture),
          .cullMode = lvk::CullMode_None,
          .debugName = "Pipeline: shadow",
      },
      nullptr);

  renderPipelineState_Fullscreen_ = ctx_->createRenderPipeline(
      {
          .smVert = smFullscreenVert_,
          .smFrag = smFullscreenFrag_,
          .color = {{.format = ctx_->getSwapchainFormat()}},
          .cullMode = lvk::CullMode_None,
          .debugName = "Pipeline: fullscreen",
      },
      nullptr);

  renderPipelineState_Skybox_ = ctx_->createRenderPipeline(
      {
          .smVert = smSkyboxVert_,
          .smFrag = smSkyboxFrag_,
          .color = {{.format = ctx_->getFormat(texColorLR_)}, {.format = ctx_->getFormat(texMotionLR_)}},
          .depthFormat = ctx_->getFormat(texDepthLR_),
          .cullMode = lvk::CullMode_Front,
          .frontFace = lvk::WindingMode_CCW,
          .debugName = "Pipeline: skybox",
      },
      nullptr);
}

bool initScene(VulkanApp& app) {
  rayQuery_ = ctx_->supportsRayQuery();
  if (!rayQuery_) {
    LLOGW("VK_KHR_ray_query is not available: rendering without ambient occlusion\n");
  }

  ubPerFrame_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Uniform,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(UniformsPerFrame),
      .debugName = "Buffer: uniforms (per frame)",
  });
  ubPerFrameShadow_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Uniform,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(UniformsPerFrame),
      .debugName = "Buffer: uniforms (per frame shadow)",
  });
  ubPerObject_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Uniform,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(UniformsPerObject),
      .debugName = "Buffer: uniforms (per object)",
  });

  depthState_ = {.compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true};
  depthStateLEqual_ = {.compareOp = lvk::CompareOp_LessEqual, .isDepthWriteEnabled = true};

  sampler_ = ctx_->createSampler({
      .mipMap = lvk::SamplerMip_Linear,
      .wrapU = lvk::SamplerWrap_Repeat,
      .wrapV = lvk::SamplerWrap_Repeat,
      .debugName = "Sampler: linear",
  });
  samplerShadow_ = ctx_->createSampler({
      .wrapU = lvk::SamplerWrap_Clamp,
      .wrapV = lvk::SamplerWrap_Clamp,
      .depthCompareOp = lvk::CompareOp_LessEqual,
      .depthCompareEnabled = true,
      .debugName = "Sampler: shadow",
  });
  samplerLinearClamp_ = ctx_->createSampler({
      .wrapU = lvk::SamplerWrap_Clamp,
      .wrapV = lvk::SamplerWrap_Clamp,
      .debugName = "Sampler: linear clamp",
  });

  renderPassOffscreen_ = {
      .color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}},
                {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}}},
      .depth = {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearDepth = 1.0f},
  };
  renderPassMain_ = {
      .color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}}},
  };
  renderPassShadow_ = {
      .color = {},
      .depth = {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearDepth = 1.0f},
  };

  createShadowMap();
  createOffscreenFramebuffer();
  createPipelines();

  if (!initModel(app)) {
    return false;
  }

  loadSkyboxTexture();

  return true;
}

void destroyScene() {
  printf("Waiting for the loader thread to exit...\n");
  cancelLoadingMaterialTextures();

  TLAS_ = nullptr;
  BLAS_.clear();
  sbInstances_ = nullptr;
  vb0_ = nullptr;
  ib0_ = nullptr;
  sbMaterials_ = nullptr;
  ubPerFrame_ = nullptr;
  ubPerFrameShadow_ = nullptr;
  ubPerObject_ = nullptr;
  smMeshVert_ = nullptr;
  smMeshFrag_ = nullptr;
  smShadowVert_ = nullptr;
  smShadowFrag_ = nullptr;
  smFullscreenVert_ = nullptr;
  smFullscreenFrag_ = nullptr;
  smSkyboxVert_ = nullptr;
  smSkyboxFrag_ = nullptr;
  renderPipelineState_Mesh_ = nullptr;
  renderPipelineState_Shadow_ = nullptr;
  renderPipelineState_Skybox_ = nullptr;
  renderPipelineState_Fullscreen_ = nullptr;
  skyboxTextureReference_ = nullptr;
  skyboxTextureIrradiance_ = nullptr;
  sampler_ = nullptr;
  samplerShadow_ = nullptr;
  samplerLinearClamp_ = nullptr;
  ctx_->destroy(fbShadowMap_);
  texColorLR_ = nullptr;
  texMotionLR_ = nullptr;
  texDepthLR_ = nullptr;
}

struct NssConstants {
  float deviceToViewDepth[4];
  float jitterOffset[4];
  float jitterOffsetTm1[4];
  float scaleFactor[4];
  int32_t outputDims[2];
  int32_t inputDims[2];
  float invOutputDims[2];
  float invInputDims[2];
  int32_t depthTm1Size[2];
  float invDepthTm1Size[2];
  int32_t inputTensorSize[2];
  float inputTensorSizeRcp[2];
  int32_t kpnDimension[2];
  float motionVectorScale[2];
  float paddingScale[2];
  float depthClipRequiredSepScale;
  float depthClipPower;
  float kpnScale[2];
  int32_t debugViewMode;
  float notHistoryReset;
  float exposure[2];
  int32_t indexModulo[2];
  int32_t reducedInputModulo[2];
  int32_t lutOffset[2];
};

static_assert(sizeof(NssConstants) == 208);

struct NssPushConstants {
  uint64_t constants;
  uint32_t samplerPoint;
  uint32_t samplerLinear;
  uint32_t texColor;
  uint32_t texDepth;
  uint32_t texMotion;
  uint32_t texHistory;
  uint32_t texFeedback;
  uint32_t texLumaDerivTm1;
  uint32_t texDepthTm1;
  uint32_t texNearestDepthCoord;
  uint32_t imgDepthTm1;
  uint32_t imgLumaDeriv;
  uint32_t imgNearestDepthCoord;
  uint32_t imgOutput;
  uint32_t tensorInput;
  uint32_t tensorKpn;
};

std::string expandIncludes(const std::string& text, std::set<std::string>& included, bool& ok);

std::string loadShaderWithIncludes(const std::string& fileName, std::set<std::string>& included, bool& ok) {
  included.insert(std::filesystem::path(fileName).filename().string());

  std::vector<uint8_t> blob;
  for (const std::string& root : folderShadersNSS) {
    blob = readContentFile(root + fileName);
    if (!blob.empty()) {
      break;
    }
  }
  if (blob.empty()) {
    LLOGW("Cannot open shader `%s`\n", fileName.c_str());
    ok = false;
    return {};
  }
  return expandIncludes(std::string(blob.begin(), blob.end()), included, ok);
}

std::string expandIncludes(const std::string& text, std::set<std::string>& included, bool& ok) {
  std::string result;
  size_t lineStart = 0;
  while (lineStart <= text.size()) {
    size_t lineEnd = text.find('\n', lineStart);
    if (lineEnd == std::string::npos) {
      lineEnd = text.size();
    }
    std::string line = text.substr(lineStart, lineEnd - lineStart);
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    lineStart = lineEnd + 1;
    const size_t pos = line.find("#include \"");
    if (pos != std::string::npos && line.find_first_not_of(" \t") == pos) {
      const size_t start = pos + strlen("#include \"");
      const size_t end = line.find('"', start);
      const std::string includeName = line.substr(start, end - start);
      if (!included.contains(std::filesystem::path(includeName).filename().string())) {
        result += "// >>> " + includeName + "\n";
        result += loadShaderWithIncludes(includeName, included, ok);
        result += "// <<< " + includeName + "\n";
      }
      continue;
    }
    result += line;
    result += '\n';
  }
  return result;
}

struct NeuralSuperSampling {
  static constexpr uint32_t kAlignment = FFX_NSS_RESOURCE_ALIGNMENT;
  static constexpr uint32_t kThreadGroup = FFX_NSS_THREAD_GROUP_WIDTH;
  static_assert(FFX_NSS_THREAD_GROUP_WIDTH == FFX_NSS_THREAD_GROUP_HEIGHT, "The dispatches below assume square thread groups");
  inline static uint32_t kGraphWidth = 0;
  inline static uint32_t kGraphHeight = 0;
  inline static uint32_t kKpnWidth = 0;
  inline static uint32_t kKpnHeight = 0;
  static constexpr uint32_t kKpnChannels = 36;
  static constexpr uint32_t kInputChannels = 12;
  static constexpr uint32_t kFeedbackChannels = 4;
  inline static uint32_t kDepthTm1Width = 0;
  inline static uint32_t kDepthTm1Height = 0;
  static constexpr uint32_t kJitterPhaseCount = 8 * 2 * 2;

  static void configure() {
    kGraphWidth = (kRenderWidth + kAlignment - 1) / kAlignment * kAlignment;
    kGraphHeight = (kRenderHeight + kAlignment - 1) / kAlignment * kAlignment;
    kKpnWidth = kGraphWidth / 4;
    kKpnHeight = kGraphHeight / 4;
    kDepthTm1Width = kRenderWidth / 2;
    kDepthTm1Height = kRenderHeight / 2;
  }

  bool supported = false;
  bool enabled = true;
  bool resetHistory = true;
  float exposure = 1.0f;
  uint32_t frameIndex = 0;
  vec2 jitter = vec2(0.0f);
  vec2 jitterPrev = vec2(0.0f);

  lvk::Holder<lvk::TensorHandle> tensorInput;
  lvk::Holder<lvk::TensorHandle> tensorKpn;
  lvk::Holder<lvk::TensorHandle> tensorFeedback;
  lvk::Holder<lvk::TextureHandle> texFeedback;
  bool feedbackAliasing = false;
  lvk::Holder<lvk::TextureHandle> texLumaDeriv[2];
  lvk::Holder<lvk::TextureHandle> texNearestDepthCoord;
  lvk::Holder<lvk::TextureHandle> texDepthTm1;
  lvk::Holder<lvk::TextureHandle> texOutput[2];
  lvk::Holder<lvk::SamplerHandle> samplerPoint;
  lvk::Holder<lvk::SamplerHandle> samplerLinear;
  lvk::Holder<lvk::BufferHandle> bufConstants;
  lvk::Holder<lvk::ShaderModuleHandle> smDepthScatter;
  lvk::Holder<lvk::ShaderModuleHandle> smPreprocess;
  lvk::Holder<lvk::ShaderModuleHandle> smPostprocess;
  lvk::Holder<lvk::ShaderModuleHandle> smFeedbackToImage;
  lvk::VgfModel vgf;
  lvk::Holder<lvk::ComputePipelineHandle> pipelineDepthScatter;
  lvk::Holder<lvk::ComputePipelineHandle> pipelinePreprocess;
  lvk::Holder<lvk::ComputePipelineHandle> pipelinePostprocess;
  lvk::Holder<lvk::ComputePipelineHandle> pipelineFeedbackToImage;
  lvk::Holder<lvk::DataGraphPipelineHandle> pipelineGraph;
  lvk::TensorHandle graphInputs[1] = {};
  lvk::TensorHandle graphOutputs[2] = {};

  static float halton(int32_t index, int32_t base) {
    float f = 1.0f;
    float result = 0.0f;
    for (int32_t i = index; i > 0;) {
      f /= (float)base;
      result = result + f * (float)(i % base);
      i = (int32_t)floorf((float)i / (float)base);
    }
    return result;
  }

  static vec2 jitterOffset(uint32_t frame) {
    const int32_t index = (int32_t)(frame % kJitterPhaseCount) + 1;
    return vec2(halton(index, 2) - 0.5f, halton(index, 3) - 0.5f);
  }

  bool init(lvk::IContext* ctx) {
    supported = false;

    if (!ctx->supportsTensorsARM() || !ctx->supportsDataGraphARM()) {
      LLOGW("NSS requires VK_ARM_tensors and VK_ARM_data_graph (configure with LVK_WITH_ML_EMULATION_LAYER=ON to emulate them)\n");
      return false;
    }

    const std::vector<uint8_t> blobVGF = readContentFile(folderContentRoot + kNssModelFile);
    if (blobVGF.empty() || !vgf.load(blobVGF.data(), blobVGF.size())) {
      LLOGW("NSS: cannot load `%s`. Run `deploy_content.py`, and `tools/make_nss_model.py` for the other resolutions.\n", kNssModelFile);
      return false;
    }
    const int64_t dimsInput[] = {1, kGraphHeight, kGraphWidth, kInputChannels};
    const int64_t dimsKpn[] = {1, kKpnHeight, kKpnWidth, kKpnChannels};
    const int64_t dimsFeedback[] = {1, kGraphHeight, kGraphWidth, kFeedbackChannels};
    const int32_t indexInput = vgf.findInput(dimsInput);
    const int32_t indexKpn = vgf.findOutput(dimsKpn);
    const int32_t indexFeedback = vgf.findOutput(dimsFeedback);
    if (indexInput < 0 || indexKpn < 0 || indexFeedback < 0 || vgf.getNumInputs() != 1 || vgf.getNumOutputs() != 2 ||
        vgf.getInput(indexInput).format != lvk::Format_R_I8 || vgf.getOutput(indexKpn).format != lvk::Format_R_I8 ||
        vgf.getOutput(indexFeedback).format != lvk::Format_R_I8) {
      LLOGW("NSS: unexpected graph interface (this sample expects an int8 NSS v1 model that upscales two times)\n");
      return false;
    }

    lvk::Result result;
    tensorInput = ctx->createTensor(
        {
            .format = lvk::Format_R_I8,
            .rank = 4,
            .dimensions = {1, kGraphHeight, kGraphWidth, kInputChannels},
            .usage = lvk::TensorUsageBits_Shader | lvk::TensorUsageBits_DataGraph,
        },
        "NSS: input tensor",
        &result);
    if (!result.isOk()) {
      LLOGW("NSS: cannot create the input tensor: %s\n", result.message);
      return false;
    }
    tensorKpn = ctx->createTensor(
        {
            .format = lvk::Format_R_I8,
            .rank = 4,
            .dimensions = {1, kKpnHeight, kKpnWidth, kKpnChannels},
            .usage = lvk::TensorUsageBits_Shader | lvk::TensorUsageBits_DataGraph,
        },
        "NSS: KPN coefficients",
        &result);
    if (!result.isOk()) {
      LLOGW("NSS: cannot create the KPN tensor: %s\n", result.message);
      return false;
    }
    feedbackAliasing = (ctx->getTensorFormatSupport(lvk::Format_R_I8, lvk::TensorTiling_Linear) & lvk::TensorUsageBits_ImageAliasing) != 0;
    LLOGL("NSS: temporal feedback via %s\n", feedbackAliasing ? "tensor image aliasing" : "a tensor to image copy");
    tensorFeedback = ctx->createTensor(
        {
            .format = lvk::Format_R_I8,
            .rank = 4,
            .dimensions = {1, kGraphHeight, kGraphWidth, kFeedbackChannels},
            .tiling = feedbackAliasing ? lvk::TensorTiling_Linear : lvk::TensorTiling_Optimal,
            .usage = (uint8_t)(feedbackAliasing ? (lvk::TensorUsageBits_DataGraph | lvk::TensorUsageBits_ImageAliasing |
                                                   lvk::TensorUsageBits_TransferDst)
                                                : (lvk::TensorUsageBits_DataGraph | lvk::TensorUsageBits_Shader)),
        },
        "NSS: temporal feedback",
        &result);
    if (!result.isOk()) {
      LLOGW("NSS: cannot create the feedback tensor: %s\n", result.message);
      return false;
    }
    if (feedbackAliasing) {
      texFeedback = ctx->createTextureAliasingTensor(tensorFeedback,
                                                     {
                                                         .type = lvk::TextureType_2D,
                                                         .format = lvk::Format_RGBA_SN8,
                                                         .dimensions = {kGraphWidth, kGraphHeight},
                                                         .usage = lvk::TextureUsageBits_Sampled,
                                                     },
                                                     "NSS: temporal feedback (texture)",
                                                     &result);
      if (!result.isOk()) {
        LLOGW("NSS: cannot create the feedback texture: %s\n", result.message);
        return false;
      }
    } else {
      texFeedback = ctx->createTexture({
          .type = lvk::TextureType_2D,
          .format = lvk::Format_RGBA_SN8,
          .dimensions = {kGraphWidth, kGraphHeight},
          .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage,
          .debugName = "NSS: temporal feedback (texture)",
      });
    }

    for (uint32_t i = 0; i != 2; i++) {
      texLumaDeriv[i] = ctx->createTexture({
          .type = lvk::TextureType_2D,
          .format = lvk::Format_RGBA_SN8,
          .dimensions = {kGraphWidth, kGraphHeight},
          .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage,
          .debugName = i ? "NSS: luma derivative 1" : "NSS: luma derivative 0",
      });
      texOutput[i] = ctx->createTexture({
          .type = lvk::TextureType_2D,
          .format = lvk::Format_RGBA_F16,
          .dimensions = {kOutputWidth, kOutputHeight},
          .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage,
          .debugName = i ? "NSS: upscaled output 1" : "NSS: upscaled output 0",
      });
    }
    texNearestDepthCoord = ctx->createTexture({
        .type = lvk::TextureType_2D,
        .format = lvk::Format_R_UN8,
        .dimensions = {kGraphWidth, kGraphHeight},
        .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage,
        .debugName = "NSS: nearest depth coord",
    });
    texDepthTm1 = ctx->createTexture({
        .type = lvk::TextureType_2D,
        .format = lvk::Format_R_UI32,
        .dimensions = {kDepthTm1Width, kDepthTm1Height},
        .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage,
        .debugName = "NSS: reconstructed previous depth",
    });
    samplerPoint = ctx->createSampler({
        .minFilter = lvk::SamplerFilter_Nearest,
        .magFilter = lvk::SamplerFilter_Nearest,
        .wrapU = lvk::SamplerWrap_Clamp,
        .wrapV = lvk::SamplerWrap_Clamp,
        .debugName = "NSS: point clamp",
    });
    samplerLinear = ctx->createSampler({
        .wrapU = lvk::SamplerWrap_Clamp,
        .wrapV = lvk::SamplerWrap_Clamp,
        .debugName = "NSS: linear clamp",
    });
    bufConstants = ctx->createBuffer({
        .usage = lvk::BufferUsageBits_Storage,
        .storage = lvk::StorageType_Device,
        .size = sizeof(NssConstants),
        .debugName = "NSS: constants",
    });

    struct ComputePass {
      const char* code;
      const char* debugName;
      lvk::Holder<lvk::ShaderModuleHandle>* sm;
      lvk::Holder<lvk::ComputePipelineHandle>* pipeline;
    };
    const ComputePass passes[] = {
        {kNssDepthScatter, "NSS: depth scatter", &smDepthScatter, &pipelineDepthScatter},
        {kNssPreprocess, "NSS: preprocess", &smPreprocess, &pipelinePreprocess},
        {kNssPostprocess, "NSS: postprocess", &smPostprocess, &pipelinePostprocess},
        {kNssFeedbackToImage, "NSS: feedback to image", &smFeedbackToImage, &pipelineFeedbackToImage},
    };
    for (const ComputePass& pass : passes) {
      const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
      std::set<std::string> included;
      bool ok = true;
      const std::string source = expandIncludes(pass.code, included, ok);
      if (!ok || source.empty()) {
        return false;
      }
      lvk::ShaderModuleDesc desc(source.c_str(), lvk::Stage_Comp, pass.debugName);
      desc.optimizeSPIRV = false;
      *pass.sm = ctx->createShaderModule(desc);
      if (pass.sm->empty()) {
        LLOGW("NSS: cannot compile `%s`\n", pass.debugName);
        return false;
      }
      *pass.pipeline = ctx->createComputePipeline({.smComp = *pass.sm, .debugName = pass.debugName});
      LLOGL("NSS: compiled `%s` in %.1f ms\n",
            pass.debugName,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }

    graphInputs[indexInput] = tensorInput;
    graphOutputs[indexKpn] = tensorKpn;
    graphOutputs[indexFeedback] = tensorFeedback;
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    pipelineGraph = vgf.createDataGraphPipeline(*ctx, graphInputs, graphOutputs, "NSS: data graph", &result);
    if (!result.isOk()) {
      LLOGW("NSS: cannot create the data graph pipeline: %s\n", result.message);
      return false;
    }
    LLOGL("NSS: data graph pipeline created in %.1f ms\n",
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());

    supported = true;
    resetHistory = true;
    return true;
  }

  void destroy() {
    pipelineGraph = nullptr;
    pipelineDepthScatter = nullptr;
    pipelinePreprocess = nullptr;
    pipelinePostprocess = nullptr;
    pipelineFeedbackToImage = nullptr;
    smFeedbackToImage = nullptr;
    vgf = lvk::VgfModel();
    smDepthScatter = nullptr;
    smPreprocess = nullptr;
    smPostprocess = nullptr;
    bufConstants = nullptr;
    samplerPoint = nullptr;
    samplerLinear = nullptr;
    texDepthTm1 = nullptr;
    texNearestDepthCoord = nullptr;
    texOutput[0] = nullptr;
    texOutput[1] = nullptr;
    texLumaDeriv[0] = nullptr;
    texLumaDeriv[1] = nullptr;
    texFeedback = nullptr;
    tensorFeedback = nullptr;
    tensorKpn = nullptr;
    tensorInput = nullptr;
  }

  lvk::TextureHandle history() const {
    return texOutput[(frameIndex + 1) & 1];
  }
  lvk::TextureHandle output() const {
    return texOutput[frameIndex & 1];
  }

  NssConstants makeConstants(bool reset) const {
    NssConstants c = {};

    const float fMin = std::min(kCameraNear, kCameraFar);
    const float fMax = std::max(kCameraNear, kCameraFar);
    const float fQ = fMax / (fMin - fMax);
    c.deviceToViewDepth[0] = -1.0f * fQ;
    c.deviceToViewDepth[1] = fQ * fMin;
    const float aspect = (float)kRenderWidth / (float)kRenderHeight;
    const float cotHalfFovY = cosf(0.5f * kCameraFovY) / sinf(0.5f * kCameraFovY);
    c.deviceToViewDepth[2] = 1.0f / (cotHalfFovY / aspect);
    c.deviceToViewDepth[3] = 1.0f / cotHalfFovY;

    c.inputDims[0] = kRenderWidth;
    c.inputDims[1] = kRenderHeight;
    c.invInputDims[0] = 1.0f / kRenderWidth;
    c.invInputDims[1] = 1.0f / kRenderHeight;
    c.outputDims[0] = kOutputWidth;
    c.outputDims[1] = kOutputHeight;
    c.invOutputDims[0] = 1.0f / kOutputWidth;
    c.invOutputDims[1] = 1.0f / kOutputHeight;
    c.inputTensorSize[0] = kGraphWidth;
    c.inputTensorSize[1] = kGraphHeight;
    c.inputTensorSizeRcp[0] = 1.0f / kGraphWidth;
    c.inputTensorSizeRcp[1] = 1.0f / kGraphHeight;

    const vec2 jitterTm1 = reset ? jitter : jitterPrev;
    c.jitterOffsetTm1[0] = jitterTm1.x;
    c.jitterOffsetTm1[1] = jitterTm1.y;
    c.jitterOffsetTm1[2] = jitterTm1.x / kRenderWidth;
    c.jitterOffsetTm1[3] = jitterTm1.y / kRenderHeight;
    c.jitterOffset[0] = jitter.x;
    c.jitterOffset[1] = jitter.y;
    c.jitterOffset[2] = jitter.x / kRenderWidth;
    c.jitterOffset[3] = jitter.y / kRenderHeight;

    c.scaleFactor[0] = (float)kOutputWidth / kRenderWidth;
    c.scaleFactor[1] = (float)kOutputHeight / kRenderHeight;
    c.scaleFactor[2] = (float)kRenderWidth / kOutputWidth;
    c.scaleFactor[3] = (float)kRenderHeight / kOutputHeight;

    c.motionVectorScale[0] = (float)kRenderWidth;
    c.motionVectorScale[1] = (float)kRenderHeight;

    c.paddingScale[0] = (float)kRenderWidth / kGraphWidth;
    c.paddingScale[1] = (float)kRenderHeight / kGraphHeight;
    c.depthTm1Size[0] = kDepthTm1Width;
    c.depthTm1Size[1] = kDepthTm1Height;
    c.invDepthTm1Size[0] = 1.0f / kDepthTm1Width;
    c.invDepthTm1Size[1] = 1.0f / kDepthTm1Height;
    c.kpnDimension[0] = kKpnWidth;
    c.kpnDimension[1] = kKpnHeight;
    c.kpnScale[0] = (float)kKpnWidth / kGraphWidth;
    c.kpnScale[1] = (float)kKpnHeight / kGraphHeight;

    {
      const float halfViewport = sqrtf(float(kRenderWidth * kRenderWidth + kRenderHeight * kRenderHeight));
      const float cornerViewDirLen =
          sqrtf(c.deviceToViewDepth[2] * c.deviceToViewDepth[2] + c.deviceToViewDepth[3] * c.deviceToViewDepth[3] + 1.0f);
      c.depthClipRequiredSepScale = 1.37e-05f * cornerViewDirLen * halfViewport;
      constexpr float kReferenceViewportLength = 2202.9071700822983f;
      const float resolutionFactor = std::min(std::max(halfViewport / kReferenceViewportLength, 0.0f), 1.0f);
      c.depthClipPower = 1.0f + 2.0f * resolutionFactor;
    }

    c.debugViewMode = 0;
    c.notHistoryReset = reset ? 0.0f : 1.0f;
    c.exposure[0] = exposure;
    c.exposure[1] = 1.0f / exposure;

    const int32_t indexModulo[2] = {(int32_t)(kOutputWidth / kRenderWidth), (int32_t)(kOutputHeight / kRenderHeight)};
    c.indexModulo[0] = indexModulo[0];
    c.indexModulo[1] = indexModulo[1];
    c.reducedInputModulo[0] = 1;
    c.reducedInputModulo[1] = 1;
    {
      const int32_t baseX = (int32_t)floorf(0.5f * c.scaleFactor[0]);
      const int32_t baseY = (int32_t)floorf(0.5f * c.scaleFactor[1]);
      const int32_t jitteredX = (int32_t)floorf((jitter.x + 0.5f) * c.scaleFactor[0]);
      const int32_t jitteredY = (int32_t)floorf((jitter.y + 0.5f) * c.scaleFactor[1]);
      int32_t dx = (jitteredX - baseX) % indexModulo[0];
      int32_t dy = (jitteredY - baseY) % indexModulo[1];
      if (dx < 0) {
        dx += indexModulo[0];
      }
      if (dy < 0) {
        dy += indexModulo[1];
      }
      c.lutOffset[0] = dx;
      c.lutOffset[1] = dy;
    }
    return c;
  }

  void dispatch(lvk::IContext* ctx, lvk::ICommandBuffer& buffer) {
    const bool reset = resetHistory;
    resetHistory = false;

    const NssConstants constants = makeConstants(reset);
    buffer.cmdUpdateBuffer(bufConstants, 0, sizeof(constants), &constants);

    const lvk::TextureHandle lumaDerivTm1 = texLumaDeriv[(frameIndex + 1) & 1];
    const lvk::TextureHandle lumaDeriv = texLumaDeriv[frameIndex & 1];
    const lvk::TextureHandle texHistory = history();
    const lvk::TextureHandle texOut = output();

    if (reset) {
      buffer.cmdPushDebugGroupLabel("NSS: reset history", 0xff0000ff);
      const lvk::TextureHandle clearTargets[] = {texLumaDeriv[0], texLumaDeriv[1], texNearestDepthCoord, texOutput[0], texOutput[1]};
      for (const lvk::TextureHandle tex : clearTargets) {
        buffer.cmdClearColorImage(tex, {.float32 = {0.0f, 0.0f, 0.0f, 0.0f}});
      }
      if (feedbackAliasing) {
        const std::vector<uint8_t> zeros(lvk::getTensorDataSize(ctx->getTensorDesc(tensorFeedback)), 0);
        ctx->upload(tensorFeedback, zeros.data(), zeros.size());
      } else {
        buffer.cmdClearColorImage(texFeedback, {.float32 = {0.0f, 0.0f, 0.0f, 0.0f}});
      }
      buffer.cmdPopDebugGroupLabel();
    }

    NssPushConstants pc = {
        .constants = ctx->gpuAddress(bufConstants),
        .samplerPoint = samplerPoint.index(),
        .samplerLinear = samplerLinear.index(),
        .texColor = texColorLR_.index(),
        .texDepth = texDepthLR_.index(),
        .texMotion = texMotionLR_.index(),
        .texHistory = texHistory.index(),
        .texFeedback = texFeedback.index(),
        .texLumaDerivTm1 = lumaDerivTm1.index(),
        .texDepthTm1 = texDepthTm1.index(),
        .texNearestDepthCoord = texNearestDepthCoord.index(),
        .imgDepthTm1 = texDepthTm1.index(),
        .imgLumaDeriv = lumaDeriv.index(),
        .imgNearestDepthCoord = texNearestDepthCoord.index(),
        .imgOutput = texOut.index(),
        .tensorInput = tensorInput.index(),
        .tensorKpn = tensorKpn.index(),
    };

    auto groups = [](uint32_t size) -> uint32_t { return (size + kThreadGroup - 1) / kThreadGroup; };

    buffer.cmdPushDebugGroupLabel("NSS: depth scatter", 0xff0000ff);
    buffer.cmdClearColorImage(texDepthTm1, {.uint32 = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff}});
    buffer.cmdBindComputePipeline(pipelineDepthScatter);
    buffer.cmdPushConstants(pc);
    buffer.cmdDispatch({groups(kDepthTm1Width), groups(kDepthTm1Height), 1},
                       {
                           .sampledImages = {texDepthLR_, texMotionLR_},
                           .storageImages = {texDepthTm1},
                           .buffers = {bufConstants},
                       });
    buffer.cmdPopDebugGroupLabel();

    buffer.cmdPushDebugGroupLabel("NSS: preprocess", 0xff0000ff);
    buffer.cmdBindComputePipeline(pipelinePreprocess);
    buffer.cmdPushConstants(pc);
    buffer.cmdDispatch({groups(kGraphWidth), groups(kGraphHeight), 1},
                       {
                           .sampledImages = {texColorLR_, texDepthLR_, texMotionLR_, texHistory, texFeedback, lumaDerivTm1, texDepthTm1},
                           .storageImages = {lumaDeriv, texNearestDepthCoord},
                           .buffers = {bufConstants},
                           .tensors = {tensorInput},
                       });
    buffer.cmdPopDebugGroupLabel();

    buffer.cmdPushDebugGroupLabel("NSS: data graph", 0xff0000ff);
    buffer.cmdDispatchDataGraph(pipelineGraph, graphInputs, graphOutputs);
    buffer.cmdPopDebugGroupLabel();

    if (!feedbackAliasing) {
      buffer.cmdPushDebugGroupLabel("NSS: feedback tensor to image", 0xff0000ff);
      buffer.cmdBindComputePipeline(pipelineFeedbackToImage);
      const struct {
        uint32_t tensor;
        uint32_t image;
        uint32_t width;
        uint32_t height;
      } pcCopy = {
          .tensor = tensorFeedback.index(),
          .image = texFeedback.index(),
          .width = kGraphWidth,
          .height = kGraphHeight,
      };
      buffer.cmdPushConstants(pcCopy);
      buffer.cmdDispatch({groups(kGraphWidth), groups(kGraphHeight), 1},
                         {
                             .storageImages = {texFeedback},
                             .tensors = {tensorFeedback},
                         });
      buffer.cmdPopDebugGroupLabel();
    }

    buffer.cmdPushDebugGroupLabel("NSS: postprocess", 0xff0000ff);
    buffer.cmdBindComputePipeline(pipelinePostprocess);
    buffer.cmdPushConstants(pc);
    buffer.cmdDispatch({groups(kOutputWidth), groups(kOutputHeight), 1},
                       {
                           .sampledImages = {texColorLR_, texMotionLR_, texHistory, texFeedback, texNearestDepthCoord},
                           .storageImages = {texOut},
                           .buffers = {bufConstants},
                           .tensors = {tensorKpn, tensorFeedback},
                       });
    buffer.cmdPopDebugGroupLabel();
  }
};

NeuralSuperSampling nss_;

enum GPUTimestamp {
  GPUTimestamp_BeginScene = 0,
  GPUTimestamp_EndScene,
  GPUTimestamp_BeginNSS,
  GPUTimestamp_EndNSS,
  GPUTimestamp_NUM_TIMESTAMPS
};
lvk::Holder<lvk::QueryPoolHandle> queryPoolTimestamps_;
uint64_t pipelineTimestamps_[GPUTimestamp_NUM_TIMESTAMPS] = {};

void selectResolution(bool uhd) {
  kRenderWidth = uhd ? 1920 : 960;
  kRenderHeight = uhd ? 1080 : 540;
  kOutputWidth = kRenderWidth * 2;
  kOutputHeight = kRenderHeight * 2;
  kNssModelFile = uhd ? "src/nss/2_nss-1920x1080-v1_0_1.vgf" : "src/nss/2_nss-960x540-v1_0_1.vgf";
  sceneWidth_ = kRenderWidth;
  sceneHeight_ = kRenderHeight;
  NeuralSuperSampling::configure();
}

VULKAN_APP_MAIN {
#if !defined(ANDROID)
  bool uhd = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--4k")) {
      uhd = true;
    }
  }
  selectResolution(uhd);
#else
  selectResolution(false);
#endif

  const VulkanAppConfig cfg{
      .width = (int)kOutputWidth,
      .height = (int)kOutputHeight,
      .resizable = false,
      .initialCameraPos = vec3(-100, 40, -47),
      .initialCameraTarget = vec3(0, 35, 0),
      .contextConfig =
          {
              .enableValidationGpuAV = false,
              .swapchainUseSurfacePreTransform = true,
              .enableMLEmulationLayer = true,
          },
  };
  VULKAN_APP_DECLARE(app, cfg);

  bool cameraRotate = false;
#if !defined(ANDROID)
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--no-nss")) {
      renderMode_ = RenderMode_HalfResolution;
    } else if (!strcmp(argv[i], "--ao-samples") && i + 1 < argc) {
      aoSamples_ = std::clamp(atoi(argv[++i]), 0, 16);
    } else if (!strcmp(argv[i], "--camera-rotate")) {
      cameraRotate = true;
    } else if (!strcmp(argv[i], "--full-res")) {
      renderMode_ = RenderMode_FullResolution;
      fullResScene_ = true;
      fullResScenePending_ = true;
      sceneWidth_ = kOutputWidth;
      sceneHeight_ = kOutputHeight;
    }
  }
#endif // !ANDROID
  uint32_t frameCounter = 0;

  ctx_ = app.ctx_.get();
  app_ = &app;
  folderThirdParty = app.folderThirdParty_;
  folderContentRoot = app.folderContentRoot_;
  const std::filesystem::path sdk = std::filesystem::path(folderThirdParty) / kNssSdkFolder / "sdk" / "include" / "FidelityFX" / "gpu";
  folderShadersNSS = {
      (sdk / "").string(),
      (sdk / "nss" / "").string(),
  };

  if (kEnableTextureCompression) {
    printf("Compressing textures... It can take a while in debug builds...(needs to be done once)\n");
  }

  if (!initScene(app)) {
#if defined(ANDROID)
    return;
#else
    return EXIT_FAILURE;
#endif // ANDROID
  }

  nss_.init(ctx_);
  if (!nss_.supported && renderMode_ == RenderMode_HalfResolutionNSS) {
    renderMode_ = RenderMode_HalfResolution;
  }

  queryPoolTimestamps_ = ctx_->createQueryPool(GPUTimestamp_NUM_TIMESTAMPS, "queryPoolTimestamps_");

#if LVK_WITH_GLFW
  app.addKeyCallback([](GLFWwindow* window, int key, int scancode, int action, int mods) {
    const bool pressed = action != GLFW_RELEASE && !ImGui::GetIO().WantCaptureKeyboard;
    if (key == GLFW_KEY_N && pressed && nss_.supported) {
      renderMode_ = renderMode_ == RenderMode_HalfResolutionNSS ? RenderMode_HalfResolution : RenderMode_HalfResolutionNSS;
      nss_.resetHistory = true;
    }
    if (key == GLFW_KEY_R && pressed) {
      nss_.resetHistory = true;
    }
    if (key == GLFW_KEY_F && pressed) {
      renderMode_ = renderMode_ == RenderMode_FullResolution ? (nss_.supported ? RenderMode_HalfResolutionNSS : RenderMode_HalfResolution)
                                                             : RenderMode_FullResolution;
      nss_.resetHistory = true;
    }
  });
#endif // LVK_WITH_GLFW

  mat4 viewProjPrev = mat4(1.0f);
  mat4 skyboxViewProjPrev = mat4(1.0f);
  bool firstFrame = true;

  app.run([&](ldr::Span<const RenderView> views, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    const RenderView& view = views[0];

    if (cameraRotate) {
      const float angle = float(frameCounter) * 0.25f * float(M_PI / 180.0);
      const vec3 dir0 = glm::normalize(cfg.initialCameraTarget - cfg.initialCameraPos);
      const vec3 dir = glm::rotate(glm::mat4(1.0f), angle, vec3(0.0f, 1.0f, 0.0f)) * vec4(dir0, 0.0f);
      app.positioner_.lookAt(cfg.initialCameraPos, cfg.initialCameraPos + dir, vec3(0.0f, 1.0f, 0.0f));
    }

    fullResScenePending_ = renderMode_ == RenderMode_FullResolution;
    nss_.enabled = renderMode_ == RenderMode_HalfResolutionNSS;

    if (fullResScenePending_ != fullResScene_) {
      fullResScene_ = fullResScenePending_;
      sceneWidth_ = fullResScene_ ? kOutputWidth : kRenderWidth;
      sceneHeight_ = fullResScene_ ? kOutputHeight : kRenderHeight;
      createOffscreenFramebuffer();
      nss_.resetHistory = true;
    }

    lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

    processLoadedMaterialTextures(buffer, sbMaterials_);

    const bool useNSS = nss_.supported && nss_.enabled && !fullResScene_;
    nss_.jitterPrev = nss_.jitter;
    nss_.jitter = useNSS ? NeuralSuperSampling::jitterOffset(nss_.frameIndex) : vec2(0.0f);
    const float aspectRatio = (float)sceneWidth_ / (float)sceneHeight_;
    const mat4 projNoJitter = glm::perspectiveRH_ZO(kCameraFovY, aspectRatio, kCameraNear, kCameraFar);
    const mat4 proj =
        glm::translate(mat4(1.0f), vec3(-2.0f * nss_.jitter.x / sceneWidth_, 2.0f * nss_.jitter.y / sceneHeight_, 0.0f)) * projNoJitter;
    const mat4 viewMatrix = app.camera_.getViewMatrix();
    const mat4 skyboxView = mat4(viewMatrix[0], viewMatrix[1], viewMatrix[2], vec4(0, 0, 0, 1));
    if (firstFrame) {
      viewProjPrev = projNoJitter * viewMatrix;
      skyboxViewProjPrev = projNoJitter * skyboxView;
      firstFrame = false;
    }

    const mat4 shadowProj = glm::perspective(float(60.0f * (M_PI / 180.0f)), 1.0f, 10.0f, 4000.0f);
    const mat4 shadowView = mat4(vec4(0.772608519f, 0.532385886f, -0.345892131f, 0),
                                 vec4(0, 0.544812560f, 0.838557839f, 0),
                                 vec4(0.634882748f, -0.647876859f, 0.420926809f, 0),
                                 vec4(-58.9244843f, -30.4530792f, -508.410126f, 1.0f));
    const mat4 scaleBias = mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0);

    perFrame_ = UniformsPerFrame{
        .proj = proj,
        .projNoJitter = projNoJitter,
        .view = viewMatrix,
        .viewProjPrev = viewProjPrev,
        .skyboxViewProjPrev = skyboxViewProjPrev,
        .light = scaleBias * shadowProj * shadowView,
        .texSkyboxRadiance = skyboxTextureReference_.index(),
        .texSkyboxIrradiance = skyboxTextureIrradiance_.index(),
        .texShadow = fbShadowMap_.depthStencil.texture.index(),
        .sampler = sampler_.index(),
        .samplerShadow = samplerShadow_.index(),
    };

    frameId_++;

    const mat4 modelMatrix = glm::scale(mat4(1.0f), vec3(kSceneScale));

    const UniformsPerObject perObject = {
        .model = modelMatrix,
        .normal = glm::transpose(glm::inverse(modelMatrix)),
    };

    buffer.cmdUpdateBuffer(ubPerFrame_, 0, sizeof(perFrame_), &perFrame_);
    buffer.cmdUpdateBuffer(ubPerObject_, 0, sizeof(perObject), &perObject);

    if (isShadowMapDirty_) {
      const UniformsPerFrame perFrameShadow{
          .proj = shadowProj,
          .view = shadowView,
      };
      buffer.cmdUpdateBuffer(ubPerFrameShadow_, 0, sizeof(perFrameShadow), &perFrameShadow);
      buffer.cmdBeginRendering(renderPassShadow_, fbShadowMap_);
      {
        buffer.cmdBindRenderPipeline(renderPipelineState_Shadow_);
        buffer.cmdPushDebugGroupLabel("Render Shadows", 0xff0000ff);
        buffer.cmdBindDepthState(depthState_);
        buffer.cmdBindVertexBuffer(0, vb0_, 0);
        struct {
          uint64_t perFrame;
          uint64_t perObject;
        } bindings = {
            .perFrame = ctx_->gpuAddress(ubPerFrameShadow_),
            .perObject = ctx_->gpuAddress(ubPerObject_),
        };
        buffer.cmdPushConstants(bindings);
        buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI32);
        buffer.cmdDrawIndexed(static_cast<uint32_t>(indexData_.size()));
        buffer.cmdPopDebugGroupLabel();
      }
      buffer.cmdEndRendering();
      buffer.cmdTransitionToShaderReadOnly({fbShadowMap_.depthStencil.texture}, {});
      buffer.cmdGenerateMipmap(fbShadowMap_.depthStencil.texture);
      isShadowMapDirty_ = false;
    }

    buffer.cmdResetQueryPool(queryPoolTimestamps_, 0, GPUTimestamp_NUM_TIMESTAMPS);

    {
      buffer.cmdWriteTimestamp(queryPoolTimestamps_, GPUTimestamp_BeginScene);
      buffer.cmdBeginRendering(renderPassOffscreen_, fbOffscreen_);
      {
        buffer.cmdBindRenderPipeline(renderPipelineState_Mesh_);
        buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
        buffer.cmdBindDepthState(depthState_);
        buffer.cmdBindVertexBuffer(0, vb0_, 0);

        struct {
          uint64_t perFrame;
          uint64_t perObject;
          uint64_t materials;
          uint32_t tlas;
          uint32_t aoSamples;
          uint32_t frameId;
          float aoRadius;
          float aoPower;
        } bindings = {
            .perFrame = ctx_->gpuAddress(ubPerFrame_),
            .perObject = ctx_->gpuAddress(ubPerObject_),
            .materials = ctx_->gpuAddress(sbMaterials_),
            .tlas = TLAS_.empty() ? 0u : TLAS_.index(),
            .aoSamples = TLAS_.empty() ? 0u : uint32_t(aoSamples_),
            .frameId = uint32_t(frameId_),
            .aoRadius = aoRadius_,
            .aoPower = aoPower_,
        };
        buffer.cmdPushConstants(bindings);
        buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI32);
        buffer.cmdDrawIndexed(static_cast<uint32_t>(indexData_.size()));
        buffer.cmdPopDebugGroupLabel();

        buffer.cmdBindRenderPipeline(renderPipelineState_Skybox_);
        buffer.cmdPushDebugGroupLabel("Render Skybox", 0x00ff00ff);
        buffer.cmdBindDepthState(depthStateLEqual_);
        buffer.cmdDraw(3 * 6 * 2);
        buffer.cmdPopDebugGroupLabel();
      }
      buffer.cmdEndRendering();
      buffer.cmdWriteTimestamp(queryPoolTimestamps_, GPUTimestamp_EndScene);
    }

    buffer.cmdWriteTimestamp(queryPoolTimestamps_, GPUTimestamp_BeginNSS);
    if (useNSS) {
      nss_.dispatch(ctx_, buffer);
    }
    buffer.cmdWriteTimestamp(queryPoolTimestamps_, GPUTimestamp_EndNSS);

    {
      const lvk::TextureHandle tex = useNSS ? nss_.output() : lvk::TextureHandle(texColorLR_);
      const lvk::Framebuffer fbMain = {.color = {{.texture = view.colorTexture}}};

      buffer.cmdBeginRendering(renderPassMain_, fbMain, {.sampledImages = {tex}});
      {
        buffer.cmdBindRenderPipeline(renderPipelineState_Fullscreen_);
        buffer.cmdPushDebugGroupLabel("Swapchain Output", 0xff0000ff);
        buffer.cmdBindDepthState({});
        const lvk::Dimensions dim = ctx_->getDimensions(view.colorTexture);
        float c = 1.0f;
        float s = 0.0f;
        lvk::getSurfaceTransformRotation(ctx_->getSwapchainSurfaceTransform(), c, s);
        buffer.cmdBindViewport({0.0f, 0.0f, (float)dim.width, (float)dim.height, 0.0f, 1.0f});
        buffer.cmdBindScissorRect({0, 0, dim.width, dim.height});
        struct {
          mat4 clipRotation;
          uint32_t tex;
          uint32_t smp;
        } bindings = {
            .clipRotation = mat4(c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
            .tex = tex.index(),
            .smp = samplerLinearClamp_.index(),
        };
        buffer.cmdPushConstants(bindings);
        buffer.cmdDraw(3);
        buffer.cmdPopDebugGroupLabel();

        app.imgui_->beginFrame(fbMain);
        ImGui::Begin("Neural Super Sampling", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs);
        ImGui::Text("W/S/A/D - camera movement, Shift - fast movement");
        ImGui::Text("N - toggle NSS, R - reset history, F - full-resolution scene");
        ImGui::Separator();
        char modes[RenderMode_NUM_MODES][64] = {};
        snprintf(modes[RenderMode_FullResolution], sizeof(modes[0]) - 1, "Full-resolution (%ux%u)", kOutputWidth, kOutputHeight);
        snprintf(modes[RenderMode_HalfResolution], sizeof(modes[0]) - 1, "Half-resolution (%ux%u)", kRenderWidth, kRenderHeight);
        snprintf(modes[RenderMode_HalfResolutionNSS],
                 sizeof(modes[0]) - 1,
                 "Half-resolution + NSS (%ux%u -> %ux%u)",
                 kRenderWidth,
                 kRenderHeight,
                 kOutputWidth,
                 kOutputHeight);
        const char* items[RenderMode_NUM_MODES] = {modes[0], modes[1], modes[2]};
        const int numItems = nss_.supported ? RenderMode_NUM_MODES : RenderMode_NUM_MODES - 1;
        if (ImGui::Combo("Rendering", &renderMode_, items, numItems)) {
          nss_.resetHistory = true;
        }
        ImGui::Separator();
        if (rayQuery_) {
          ImGui::SliderInt("Ray-traced AO samples per pixel", &aoSamples_, 0, 16);
          ImGui::SliderFloat("AO radius", &aoRadius_, 1.0f, 200.0f);
          ImGui::SliderFloat("AO power", &aoPower_, 0.5f, 4.0f);
        } else {
          ImGui::Text("VK_KHR_ray_query is not available: no ambient occlusion");
        }
        ImGui::Separator();
        if (nss_.supported) {
          if (ImGui::Button("Reset history")) {
            nss_.resetHistory = true;
          }
          ImGui::SliderFloat("Exposure", &nss_.exposure, 0.1f, 8.0f);
          const double toMS = ctx_->getTimestampPeriodToMs();
          ImGui::Text("GPU scene: %.2f ms, NSS: %.2f ms",
                      double(pipelineTimestamps_[GPUTimestamp_EndScene] - pipelineTimestamps_[GPUTimestamp_BeginScene]) * toMS,
                      double(pipelineTimestamps_[GPUTimestamp_EndNSS] - pipelineTimestamps_[GPUTimestamp_BeginNSS]) * toMS);
        } else {
          ImGui::Text("VK_ARM_tensors / VK_ARM_data_graph are not available: showing the %ux%u input", kRenderWidth, kRenderHeight);
        }
        if (const uint32_t num = numRemainingMaterialTextures()) {
          ImGui::ProgressBar(1.0f - float(num) / cachedMaterials_.size(), ImVec2(-1, 0), "Loading materials...");
        }
        ImGui::End();
        app.drawFPS();
        app.imgui_->endFrame(buffer);
      }
      buffer.cmdEndRendering();
    }

    ctx_->submit(buffer, view.colorTexture);

    frameCounter++;

    ctx_->getQueryPoolResults(queryPoolTimestamps_,
                              0,
                              GPUTimestamp_NUM_TIMESTAMPS,
                              sizeof(pipelineTimestamps_),
                              pipelineTimestamps_,
                              sizeof(pipelineTimestamps_[0]));

    if ((frameCounter % 120) == 0) {
      const double toMS = ctx_->getTimestampPeriodToMs();
      LLOGL("GPU scene: %.3f ms, NSS: %.3f ms\n",
            double(pipelineTimestamps_[GPUTimestamp_EndScene] - pipelineTimestamps_[GPUTimestamp_BeginScene]) * toMS,
            double(pipelineTimestamps_[GPUTimestamp_EndNSS] - pipelineTimestamps_[GPUTimestamp_BeginNSS]) * toMS);
    }

    viewProjPrev = projNoJitter * viewMatrix;
    skyboxViewProjPrev = projNoJitter * skyboxView;
    if (useNSS) {
      nss_.frameIndex++;
    }
  });

  nss_.destroy();
  destroyScene();
  queryPoolTimestamps_ = nullptr;
  ctx_ = nullptr;

  VULKAN_APP_EXIT();
}
