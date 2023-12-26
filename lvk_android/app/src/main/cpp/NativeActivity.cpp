#include <android_native_app_glue.h>
#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <shared/UtilsFPS.h>

#include <lvk/LVK.h>
#include <lvk/vulkan/VulkanClasses.h>
#include <lvk/HelpersImGui.h>

#include <implot/implot.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <cstdio>
#include <thread>

#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

#include <ktx.h>
#include <ktx-software/lib/vkformat_enum.h>
#include <ktx-software/lib/gl_format.h>
#include <ldrutils/lmath/Colors.h>
#include <ldrutils/lutils/ScopeExit.h>

#include <shared/Camera.h>
#include <shared/UtilsCubemap.h>
#include <shared/UtilsFPS.h>

#include <taskflow/taskflow.hpp>

#include <chrono>

#define STBIR_NO_SIMD
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <stb/stb_image_resize2.h>

#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

#include <time.h>
clock_t start = clock();

int width_ = 800;
int height_ = 600;
FramesPerSecondCounter fps_;

std::unique_ptr<lvk::IContext> ctx_;

std::unique_ptr<lvk::IContext> createVulkanContext(ANativeWindow * window,
                                                   uint32_t width,
                                                   uint32_t height,
                                                   lvk::ContextConfig const & cfg) {
  std::unique_ptr<lvk::VulkanContext> ctx =
    std::make_unique<lvk::VulkanContext>(cfg, (void *)window);

  lvk::HWDeviceDesc device;
  uint32_t numDevices = ctx->queryDevices(lvk::HWDeviceType_Discrete, &device);
  if (numDevices == 0) {
    numDevices = ctx->queryDevices(lvk::HWDeviceType_Integrated, &device);
  }
  if (numDevices == 0) {
    LVK_ASSERT_MSG(numDevices > 0, "GPU is not found");
    return nullptr;
  }

  lvk::Result res = ctx->initContext(device);

  if (!res.isOk()) {
    LVK_ASSERT_MSG(numDevices > 0, "Failed initContext()");
    return nullptr;
  }

  if (width > 0 && height > 0) {
    res = ctx->initSwapchain(width, height);
    if (!res.isOk()) {
      LVK_ASSERT_MSG(numDevices > 0, "Failed to create swapchain");
      return nullptr;
    }
  }

  return std::move(ctx);
}

// #define TRIANGLE_DEMO
//#define TINY_MESH_DEMO
#define TINY_MESH_LARGE_DEMO

#if defined(TRIANGLE_DEMO)

const char* codeVS = R"(
#version 460
layout (location=0) out vec3 color;
const vec2 pos[3] = vec2[3](
	vec2(-0.6, -0.4),
	vec2( 0.6, -0.4),
	vec2( 0.0,  0.6)
);
const vec3 col[3] = vec3[3](
	vec3(1.0, 0.0, 0.0),
	vec3(0.0, 1.0, 0.0),
	vec3(0.0, 0.0, 1.0)
);
void main() {
	gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
	color = col[gl_VertexIndex];
}
)";

const char* codeFS = R"(
#version 460
layout (location=0) in vec3 color;
layout (location=0) out vec4 out_FragColor0;
layout (location=1) out vec4 out_FragColor1;

void main() {
	out_FragColor0 = vec4(color, 1.0);
	out_FragColor1 = vec4(1.0, 1.0, 0.0, 1.0);
};
)";

struct VulkanObjects {
  void init(ANativeWindow* window);
  void createFramebuffer();
  void render();
  lvk::Framebuffer fb_ = {};
  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Triangle_;
  lvk::Holder<lvk::ShaderModuleHandle> vert_;
  lvk::Holder<lvk::ShaderModuleHandle> frag_;
} vk;

void VulkanObjects::init(ANativeWindow* window) {
  ctx_ = createVulkanContext(window, width_, height_, {});

  createFramebuffer();

  vert_ = ctx_->createShaderModule({codeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
  frag_ = ctx_->createShaderModule({codeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});

  renderPipelineState_Triangle_ = ctx_->createRenderPipeline(
    {
      .smVert = vert_,
      .smFrag = frag_,
      .color = {{ctx_->getFormat(fb_.color[0].texture)},
                {ctx_->getFormat(fb_.color[1].texture)},
                {ctx_->getFormat(fb_.color[2].texture)},
                {ctx_->getFormat(fb_.color[3].texture)}},
    },
    nullptr);
  LVK_ASSERT(renderPipelineState_Triangle_.valid());
}

void VulkanObjects::createFramebuffer() {
  lvk::TextureHandle texSwapchain = ctx_->getCurrentSwapchainTexture();

  {
    const lvk::TextureDesc desc = {
      .type = lvk::TextureType_2D,
      .format = ctx_->getFormat(texSwapchain),
      .dimensions = ctx_->getDimensions(texSwapchain),
      .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
    };

    fb_ = {.color = {{.texture = texSwapchain},
                     {.texture = ctx_->createTexture(desc, "Framebuffer C1").release()},
                     {.texture = ctx_->createTexture(desc, "Framebuffer C2").release()},
                     {.texture = ctx_->createTexture(desc, "Framebuffer C3").release()}}};
  }
}

void VulkanObjects::render() {
  if (!ctx_) {
    return;
  }
  if (!width_ || !height_) {
    return;
  }

  fb_.color[0].texture = ctx_->getCurrentSwapchainTexture();

  lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

  // This will clear the framebuffer
  buffer.cmdBeginRendering(
    {.color = {{.loadOp = lvk::LoadOp_Clear, .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}},
               {.loadOp = lvk::LoadOp_Clear, .clearColor = {1.0f, 0.0f, 0.0f, 1.0f}},
               {.loadOp = lvk::LoadOp_Clear, .clearColor = {0.0f, 1.0f, 0.0f, 1.0f}},
               {.loadOp = lvk::LoadOp_Clear, .clearColor = {0.0f, 0.0f, 1.0f, 1.0f}}}},
    fb_);
  {
    buffer.cmdBindRenderPipeline(renderPipelineState_Triangle_);
    buffer.cmdBindViewport({0.0f, 0.0f, (float)width_, (float)height_, 0.0f, +1.0f});
    buffer.cmdBindScissorRect({0, 0, (uint32_t)width_, (uint32_t)height_});
    buffer.cmdPushDebugGroupLabel("Render Triangle", 0xff0000ff);
    buffer.cmdDraw(3);
    buffer.cmdPopDebugGroupLabel();
  }
  buffer.cmdEndRendering();
  ctx_->submit(buffer, fb_.color[0].texture);
}

#elif defined(TINY_MESH_DEMO)

constexpr uint32_t kNumCubes = 16;

std::unique_ptr<lvk::ImGuiRenderer> imgui_;

const char* codeVS = R"(
layout (location=0) out vec3 color;
layout (location=1) out vec2 uv;

struct Vertex {
  float x, y, z;
  float r, g, b;
  vec2 uv;
};

layout(std430, buffer_reference) readonly buffer VertexBuffer {
  Vertex vertices[];
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  uint texture0;
  uint texture1;
  uint sampler0;
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
};

layout(push_constant) uniform constants {
	PerFrame perFrame;
	PerObject perObject;
  VertexBuffer vb;
} pc;

void main() {
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  mat4 model = pc.perObject.model;
  Vertex v = pc.vb.vertices[gl_VertexIndex];
  gl_Position = proj * view * model * vec4(v.x, v.y, v.z, 1.0);
  color = vec3(v.r, v.g, v.b);
  uv = v.uv;
}
)";

const char* codeFS = R"(
layout (location=0) in vec3 color;
layout (location=1) in vec2 uv;
layout (location=0) out vec4 out_FragColor;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  uint texture0;
  uint texture1;
  uint sampler0;
};

layout(push_constant) uniform constants {
	PerFrame perFrame;
} pc;

void main() {
  vec4 t0 = textureBindless2D(pc.perFrame.texture0, pc.perFrame.sampler0, 2.0*uv);
  vec4 t1 = textureBindless2D(pc.perFrame.texture1, pc.perFrame.sampler0, uv);
  out_FragColor = vec4(color * (t0.rgb + t1.rgb), 1.0);
};
)";

using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

vec3 axis_[kNumCubes];

constexpr uint32_t kNumBufferedFrames = 3;

lvk::Framebuffer framebuffer_;
lvk::Holder<lvk::ShaderModuleHandle> vert_;
lvk::Holder<lvk::ShaderModuleHandle> frag_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Mesh_;
lvk::Holder<lvk::BufferHandle> vb0_, ib0_; // buffers for vertices and indices
std::vector<lvk::Holder<lvk::BufferHandle>> ubPerFrame_, ubPerObject_;
lvk::Holder<lvk::TextureHandle> texture0_, texture1_;
lvk::Holder<lvk::SamplerHandle> sampler_;
lvk::RenderPass renderPass_;
lvk::DepthState depthState_;

struct VertexPosUvw {
  vec3 pos;
  vec3 color;
  vec2 uv;
};

struct UniformsPerFrame {
  mat4 proj;
  mat4 view;
  uint32_t texture0;
  uint32_t texture1;
  uint32_t sampler;
};
struct UniformsPerObject {
  mat4 model;
};

const float half = 1.0f;

// UV-mapped cube with indices: 24 vertices, 36 indices
static VertexPosUvw vertexData0[] = {
    // top
    {{-half, -half, +half}, {0.0, 0.0, 1.0}, {0, 0}}, // 0
    {{+half, -half, +half}, {1.0, 0.0, 1.0}, {1, 0}}, // 1
    {{+half, +half, +half}, {1.0, 1.0, 1.0}, {1, 1}}, // 2
    {{-half, +half, +half}, {0.0, 1.0, 1.0}, {0, 1}}, // 3
    // bottom
    {{-half, -half, -half}, {1.0, 1.0, 1.0}, {0, 0}}, // 4
    {{-half, +half, -half}, {0.0, 1.0, 0.0}, {0, 1}}, // 5
    {{+half, +half, -half}, {1.0, 1.0, 0.0}, {1, 1}}, // 6
    {{+half, -half, -half}, {1.0, 0.0, 0.0}, {1, 0}}, // 7
    // left
    {{+half, +half, -half}, {1.0, 1.0, 0.0}, {1, 0}}, // 8
    {{-half, +half, -half}, {0.0, 1.0, 0.0}, {0, 0}}, // 9
    {{-half, +half, +half}, {0.0, 1.0, 1.0}, {0, 1}}, // 10
    {{+half, +half, +half}, {1.0, 1.0, 1.0}, {1, 1}}, // 11
    // right
    {{-half, -half, -half}, {1.0, 1.0, 1.0}, {0, 0}}, // 12
    {{+half, -half, -half}, {1.0, 0.0, 0.0}, {1, 0}}, // 13
    {{+half, -half, +half}, {1.0, 0.0, 1.0}, {1, 1}}, // 14
    {{-half, -half, +half}, {0.0, 0.0, 1.0}, {0, 1}}, // 15
    // front
    {{+half, -half, -half}, {1.0, 0.0, 0.0}, {0, 0}}, // 16
    {{+half, +half, -half}, {1.0, 1.0, 0.0}, {1, 0}}, // 17
    {{+half, +half, +half}, {1.0, 1.0, 1.0}, {1, 1}}, // 18
    {{+half, -half, +half}, {1.0, 0.0, 1.0}, {0, 1}}, // 19
    // back
    {{-half, +half, -half}, {0.0, 1.0, 0.0}, {1, 0}}, // 20
    {{-half, -half, -half}, {1.0, 1.0, 1.0}, {0, 0}}, // 21
    {{-half, -half, +half}, {0.0, 0.0, 1.0}, {0, 1}}, // 22
    {{-half, +half, +half}, {0.0, 1.0, 1.0}, {1, 1}}, // 23
};

static uint16_t indexData[] = {0,  1,  2,  2,  3,  0,  4,  5,  6,  6,  7,  4,
                               8,  9,  10, 10, 11, 8,  12, 13, 14, 14, 15, 12,
                               16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20};

UniformsPerObject perObject[kNumCubes];

static void initIGL(ANativeWindow* window, AAssetManager* assetManager) {
  ctx_ = createVulkanContext(window, width_, height_, {});

  // Vertex buffer, Index buffer and Vertex Input. Buffers are allocated in GPU memory.
  vb0_ = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Storage,
                                .storage = lvk::StorageType_Device,
                                .size = sizeof(vertexData0),
                                .data = vertexData0,
                                .debugName = "Buffer: vertex"},
                               nullptr);
  ib0_ = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Index,
                                .storage = lvk::StorageType_Device,
                                .size = sizeof(indexData),
                                .data = indexData,
                                .debugName = "Buffer: index"},
                               nullptr);
  // create an Uniform buffers to store uniforms for 2 objects
  for (uint32_t i = 0; i != kNumBufferedFrames; i++) {
    ubPerFrame_.push_back(ctx_->createBuffer({.usage = lvk::BufferUsageBits_Uniform,
                                                 .storage = lvk::StorageType_HostVisible,
                                                 .size = sizeof(UniformsPerFrame),
                                                 .debugName = "Buffer: uniforms (per frame)"},
                                                nullptr));
    ubPerObject_.push_back(ctx_->createBuffer({.usage = lvk::BufferUsageBits_Uniform,
                                                  .storage = lvk::StorageType_HostVisible,
                                                  .size = kNumCubes * sizeof(UniformsPerObject),
                                                  .debugName = "Buffer: uniforms (per object)"},
                                                 nullptr));
  }

  //depthState_ = {.compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true};

  {
    const uint32_t texWidth = 256;
    const uint32_t texHeight = 256;
    std::vector<uint32_t> pixels(texWidth * texHeight);
    for (uint32_t y = 0; y != texHeight; y++) {
      for (uint32_t x = 0; x != texWidth; x++) {
        // create a XOR pattern
        pixels[y * texWidth + x] = 0xFF000000 + ((x ^ y) << 16) + ((x ^ y) << 8) + (x ^ y);
      }
    }
    texture0_ = ctx_->createTexture(
        {
            .type = lvk::TextureType_2D,
            .format = lvk::Format_BGRA_UN8,
            .dimensions = {texWidth, texHeight},
            .usage = lvk::TextureUsageBits_Sampled,
            .data = pixels.data(),
            .debugName = "XOR pattern",
        },
        nullptr);
  }
  {
    uint8_t* pixels = nullptr;
    int32_t texWidth = 0;
    int32_t texHeight = 0;
    int32_t channels = 0;
    AAsset * file = AAssetManager_open(assetManager, "wood_polished_01_diff.png", AASSET_MODE_BUFFER);
    if(file) {
    	size_t fileLength = AAsset_getLength(file);
      stbi_uc * temp = (stbi_uc*)malloc(fileLength);
    	memcpy(temp, AAsset_getBuffer(file), fileLength);
      pixels = stbi_load_from_memory(temp,
                                     fileLength,
                                     &texWidth,
                                     &texHeight,
                                     &channels,
                                     4);
      free(temp);
    }

    LVK_ASSERT_MSG(pixels, "Cannot load textures. Run `deploy_content.py` before running this app.");
    if (!pixels) {
      printf("Cannot load textures. Run `deploy_content.py` before running this app.");
      std::terminate();
    }
    texture1_ = ctx_->createTexture(
        {
            .type = lvk::TextureType_2D,
            .format = lvk::Format_RGBA_UN8,
            .dimensions = {(uint32_t)texWidth, (uint32_t)texHeight},
            .usage = lvk::TextureUsageBits_Sampled,
            .data = pixels,
            .debugName = "wood_polished_01_diff.png",
        },
        nullptr);
    stbi_image_free(pixels);
  }

  sampler_ = ctx_->createSampler({.debugName = "Sampler: linear"}, nullptr);

  renderPass_ = {.color = {{
                     .loadOp = lvk::LoadOp_Clear,
                     .storeOp = lvk::StoreOp_Store,
                     .clearColor = {1.0f, 0.0f, 0.0f, 1.0f},
                 }}};
#if TINY_TEST_USE_DEPTH_BUFFER
  renderPass_.depth = {.loadOp = lvk::LoadOp_Clear, .clearDepth = 1.0};
#else
  renderPass_.depth = {.loadOp = lvk::LoadOp_DontCare};
#endif // TINY_TEST_USE_DEPTH_BUFFER

  // initialize random rotation axes for all cubes
  for (uint32_t i = 0; i != kNumCubes; i++) {
    axis_[i] = glm::sphericalRand(1.0f);
  }
}

static void initObjects() {
  if (renderPipelineState_Mesh_.valid()) {
    return;
  }

  vert_ = ctx_->createShaderModule({codeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
  frag_ = ctx_->createShaderModule({codeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});

  renderPipelineState_Mesh_ = ctx_->createRenderPipeline(
      {
          .smVert = vert_,
          .smFrag = frag_,
          .color =
              {
                  {.format = ctx_->getSwapchainFormat()},
              },
          .depthFormat = framebuffer_.depthStencil.texture ? ctx_->getFormat(framebuffer_.depthStencil.texture) : lvk::Format_Invalid,
          .cullMode = lvk::CullMode_Back,
          .frontFaceWinding = lvk::WindingMode_CW,
          .debugName = "Pipeline: mesh",
      },
      nullptr);
}

void render(lvk::TextureHandle nativeDrawable, uint32_t frameIndex) {
  LVK_PROFILER_FUNCTION();

  if (!width_ || !height_) {
    return;
  }

  framebuffer_.color[0].texture = nativeDrawable;

  const float fov = float(45.0f * (M_PI / 180.0f));
  const float aspectRatio = (float)width_ / (float)height_;
  const UniformsPerFrame perFrame = {
      .proj = glm::perspectiveLH(fov, aspectRatio, 0.1f, 500.0f),
      // place a "camera" behind the cubes, the distance depends on the total number of cubes
      .view = glm::translate(mat4(1.0f), vec3(0.0f, 0.0f, sqrtf(kNumCubes / 16) * 20.0f * half)),
      .texture0 = texture0_.index(),
      .texture1 = texture1_.index(),
      .sampler = sampler_.index(),
  };
  ctx_->upload(ubPerFrame_[frameIndex], &perFrame, sizeof(perFrame));

  // rotate cubes around random axes
  float secondsSinceStart = (clock() - start) / float(CLOCKS_PER_SEC);
  for (uint32_t i = 0; i != kNumCubes; i++) {
    const float direction = powf(-1, (float)(i + 1));
    const uint32_t cubesInLine = (uint32_t)sqrt(kNumCubes);
    const vec3 offset = vec3(-1.5f * sqrt(kNumCubes) + 4.0f * (i % cubesInLine),
                             -1.5f * sqrt(kNumCubes) + 4.0f * (i / cubesInLine),
                             0);
    perObject[i].model =
        glm::rotate(glm::translate(mat4(1.0f), offset), direction * secondsSinceStart, axis_[i]);
  }

  ctx_->upload(ubPerObject_[frameIndex], &perObject, sizeof(perObject));

  // Command buffers (1-N per thread): create, submit and forget
  lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

  const lvk::Viewport viewport = {0.0f, 0.0f, (float)width_, (float)height_, 0.0f, +1.0f};
  const lvk::ScissorRect scissor = {0, 0, (uint32_t)width_, (uint32_t)height_};

  // This will clear the framebuffer
  buffer.cmdBeginRendering(renderPass_, framebuffer_);
  {
    buffer.cmdBindRenderPipeline(renderPipelineState_Mesh_);
    buffer.cmdBindViewport(viewport);
    buffer.cmdBindScissorRect(scissor);
    buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
    //buffer.cmdBindDepthState(depthState_);
    buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI16);
    // Draw 2 cubes: we use uniform buffer to update matrices
    for (uint32_t i = 0; i != kNumCubes; i++) {
      struct {
        uint64_t perFrame;
        uint64_t perObject;
        uint64_t vb;
      } bindings = {
          .perFrame = ctx_->gpuAddress(ubPerFrame_[frameIndex]),
          .perObject = ctx_->gpuAddress(ubPerObject_[frameIndex], i * sizeof(UniformsPerObject)),
          .vb = ctx_->gpuAddress(vb0_),
      };
      buffer.cmdPushConstants(bindings);
      buffer.cmdDrawIndexed(3 * 6 * 2);
    }
    buffer.cmdPopDebugGroupLabel();
  }
  imgui_->endFrame(buffer);
  buffer.cmdEndRendering();

  ctx_->submit(buffer, nativeDrawable);
}

#elif defined(TINY_MESH_LARGE_DEMO)

constexpr uint32_t kMeshCacheVersion = 0xC0DE0009;
constexpr int kNumSamplesMSAA = 1;//8;

std::unique_ptr<lvk::ImGuiRenderer> imgui_;

enum GPUTimestamp {
  GPUTimestamp_BeginSceneRendering = 0,
  GPUTimestamp_EndSceneRendering,

  GPUTimestamp_BeginComputePass,
  GPUTimestamp_EndComputePass,

  GPUTimestamp_BeginPresent,
  GPUTimestamp_EndPresent,

  GPUTimestamp_NUM_TIMESTAMPS
};
lvk::Holder<lvk::QueryPoolHandle> queryPoolTimestamps_;
uint64_t pipelineTimestamps[GPUTimestamp_NUM_TIMESTAMPS] = {};
double timestampBeginRendering = 0;
double timestampEndRendering = 0;

const char* kCodeComputeTest = R"(
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (set = 0, binding = 2, rgba8) uniform readonly  image2D kTextures2Din[];
layout (set = 0, binding = 2, rgba8) uniform writeonly image2D kTextures2Dout[];

layout(push_constant) uniform constants {
   uint tex;
   uint width;
   uint height;
} pc;

void main() {
   ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

   if (pos.x < pc.width && pos.y < pc.height) {
     vec4 pixel = imageLoad(kTextures2Din[pc.tex], pos);
     float luminance = dot(pixel, vec4(0.299, 0.587, 0.114, 0.0)); // https://www.w3.org/TR/AERT/#color-contrast
     imageStore(kTextures2Dout[pc.tex], pos, vec4(vec3(luminance), 1.0));
   }
}
)";

const char* kCodeFullscreenVS = R"(
layout (location=0) out vec2 uv;
void main() {
  // generate a triangle covering the entire screen
  uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * vec2(2, -2) + vec2(-1, 1), 0.0, 1.0);
}
)";

const char* kCodeFullscreenFS = R"(
layout (location=0) in vec2 uv;
layout (location=0) out vec4 out_FragColor;

layout(push_constant) uniform constants {
	uint tex;
} pc;

void main() {
  out_FragColor = textureBindless2D(pc.tex, 0, uv);
}
)";

const char* kCodeVS = R"(
layout (location=0) in vec3 pos;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 uv;
layout (location=3) in uint mtlIndex;

struct Material {
   vec4 ambient;
   vec4 diffuse;
   int texAmbient;
   int texDiffuse;
   int texAlpha;
   int padding;
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  uint texSkyboxRadiance;
  uint texSkyboxIrradiance;
  uint texShadow;
  uint sampler0;
  uint samplerShadow0;
  int bDrawNormals;
  int bDebugLines;
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
};

layout(std430, buffer_reference) readonly buffer Materials {
  Material mtl[];
};

layout(push_constant) uniform constants
{
	PerFrame perFrame;
   PerObject perObject;
   Materials materials;
} pc;

// output
struct PerVertex {
  vec3 normal;
  vec2 uv;
  vec4 shadowCoords;
};
layout (location=0) out PerVertex vtx;
layout (location=5) flat out Material mtl;
//

void main() {
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  mat4 model = pc.perObject.model;
  mat4 light = pc.perFrame.light;
  mtl = pc.materials.mtl[mtlIndex];
  gl_Position = proj * view * model * vec4(pos, 1.0);

  // Compute the normal in world-space
  mat3 norm_matrix = transpose(inverse(mat3(model)));
  vtx.normal = normalize(norm_matrix * normal);
  vtx.uv = uv;
  vtx.shadowCoords = light * model * vec4(pos, 1.0);
}
)";

const char* kCodeVS_Wireframe = R"(
layout (location=0) in vec3 pos;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
};

layout(push_constant) uniform constants
{
	PerFrame perFrame;
   PerObject perObject;
} pc;

void main() {
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  mat4 model = pc.perObject.model;
  gl_Position = proj * view * model * vec4(pos, 1.0);
}
)";

const char* kCodeFS_Wireframe = R"(
layout (location=0) out vec4 out_FragColor;

void main() {
  out_FragColor = vec4(1.0);
};
)";

const char* kCodeFS = R"(

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  uint texSkyboxRadiance;
  uint texSkyboxIrradiance;
  uint texShadow;
  uint sampler0;
  uint samplerShadow0;
  int bDrawNormals;
  int bDebugLines;
};

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
};

layout(push_constant) uniform constants
{
	PerFrame perFrame;
} pc;


layout (location=0) in PerVertex vtx;
layout (location=5) flat in Material mtl;

layout (location=0) out vec4 out_FragColor;

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
  bool drawNormals = pc.perFrame.bDrawNormals > 0;
  if (Kd.a < 0.5)
    discard;
  vec3 n = normalize(vtx.normal);
  float NdotL1 = clamp(dot(n, normalize(vec3(-1, 1,+1))), 0.0, 1.0);
  float NdotL2 = clamp(dot(n, normalize(vec3(-1, 1,-1))), 0.0, 1.0);
  float NdotL = 0.5 * (NdotL1+NdotL2);
  // IBL diffuse
  const vec4 f0 = vec4(0.04);
  vec4 diffuse = textureBindlessCube(pc.perFrame.texSkyboxIrradiance, pc.perFrame.sampler0, n) * Kd * (vec4(1.0) - f0);
  out_FragColor = drawNormals ?
    vec4(0.5 * (n+vec3(1.0)), 1.0) :
    Ka + diffuse * shadow(vtx.shadowCoords);
};
)";

const char* kShadowVS = R"(
layout (location=0) in vec3 pos;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  uint texSkyboxRadiance;
  uint texSkyboxIrradiance;
  uint texShadow;
  uint sampler0;
  uint samplerShadow0;
  int bDrawNormals;
  int bDebugLines;
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
};

layout(push_constant) uniform constants
{
	PerFrame perFrame;
	PerObject perObject;
} pc;

void main() {
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  mat4 model = pc.perObject.model;
  gl_Position = proj * view * model * vec4(pos, 1.0);
}
)";

const char* kShadowFS = R"(
void main() {
};
)";

const char* kSkyboxVS = R"(
layout (location=0) out vec3 textureCoords;

const vec3 positions[8] = vec3[8](
	vec3(-1.0,-1.0, 1.0), vec3( 1.0,-1.0, 1.0), vec3( 1.0, 1.0, 1.0), vec3(-1.0, 1.0, 1.0),
	vec3(-1.0,-1.0,-1.0), vec3( 1.0,-1.0,-1.0), vec3( 1.0, 1.0,-1.0), vec3(-1.0, 1.0,-1.0)
);

const int indices[36] = int[36](
	0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7, 4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3
);

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  uint texSkyboxRadiance;
  uint texSkyboxIrradiance;
  uint texShadow;
  uint sampler0;
  uint samplerShadow0;
  int bDrawNormals;
  int bDebugLines;
};

layout(push_constant) uniform constants
{
	PerFrame perFrame;
} pc;

void main() {
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  // discard translation
  view = mat4(view[0], view[1], view[2], vec4(0, 0, 0, 1));
  mat4 transform = proj * view;
  vec3 pos = positions[indices[gl_VertexIndex]];
  gl_Position = (transform * vec4(pos, 1.0)).xyww;

  // skybox
  textureCoords = pos;
}

)";
const char* kSkyboxFS = R"(
layout (location=0) in vec3 textureCoords;
layout (location=0) out vec4 out_FragColor;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  uint texSkyboxRadiance;
  uint texSkyboxIrradiance;
  uint texShadow;
  uint sampler0;
  uint samplerShadow0;
  int bDrawNormals;
  int bDebugLines;
};

layout(push_constant) uniform constants
{
	PerFrame perFrame;
} pc;

void main() {
  out_FragColor = textureBindlessCube(pc.perFrame.texSkyboxRadiance, pc.perFrame.sampler0, textureCoords);
}
)";

using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

constexpr uint32_t kNumBufferedFrames = 3;

lvk::Framebuffer fbMain_; // swapchain
lvk::Framebuffer fbOffscreen_;
lvk::Holder<lvk::TextureHandle> fbOffscreenColor_;
lvk::Holder<lvk::TextureHandle> fbOffscreenDepth_;
lvk::Holder<lvk::TextureHandle> fbOffscreenResolve_;
lvk::Framebuffer fbShadowMap_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshVert_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshWireframeVert_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshWireframeFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smShadowVert_;
lvk::Holder<lvk::ShaderModuleHandle> smShadowFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smFullscreenVert_;
lvk::Holder<lvk::ShaderModuleHandle> smFullscreenFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smSkyboxVert_;
lvk::Holder<lvk::ShaderModuleHandle> smSkyboxFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smGrayscaleComp_;
lvk::Holder<lvk::ComputePipelineHandle> computePipelineState_Grayscale_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Mesh_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_MeshWireframe_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Shadow_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Skybox_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Fullscreen_;
lvk::Holder<lvk::BufferHandle> vb0_, ib0_; // buffers for vertices and indices
lvk::Holder<lvk::BufferHandle> sbMaterials_; // storage buffer for materials
std::vector<lvk::Holder<lvk::BufferHandle>> ubPerFrame_, ubPerFrameShadow_, ubPerObject_;
lvk::Holder<lvk::SamplerHandle> sampler_;
lvk::Holder<lvk::SamplerHandle> samplerShadow_;
lvk::Holder<lvk::TextureHandle> textureDummyWhite_;
lvk::Holder<lvk::TextureHandle> skyboxTextureReference_;
lvk::Holder<lvk::TextureHandle> skyboxTextureIrradiance_;
lvk::RenderPass renderPassOffscreen_;
lvk::RenderPass renderPassMain_;
lvk::RenderPass renderPassShadow_;
lvk::DepthState depthState_;
lvk::DepthState depthStateLEqual_;

// scene navigation
CameraPositioner_FirstPerson positioner_(vec3(-100, 40, -47), vec3(0, 35, 0), vec3(0, 1, 0));
Camera camera_(positioner_);
glm::vec2 mousePos_ = glm::vec2(0.0f);
bool mousePressed_ = false;
bool enableComputePass_ = true;
bool enableWireframe_ = false;

bool isShadowMapDirty_ = true;

struct VertexData {
  vec3 position;
  uint32_t normal; // Int_2_10_10_10_REV
  uint32_t uv; // hvec2
  uint32_t mtlIndex;
};

std::vector<VertexData> vertexData_;
std::vector<uint32_t> indexData_;
std::vector<uint32_t> shapeVertexCnt_;

struct UniformsPerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  uint32_t texSkyboxRadiance = 0;
  uint32_t texSkyboxIrradiance = 0;
  uint32_t texShadow = 0;
  uint32_t sampler = 0;
  uint32_t samplerShadow = 0;
  int bDrawNormals = 0;
  int bDebugLines = 0;
  int padding = 0;
} perFrame_;

struct UniformsPerObject {
  mat4 model;
};
#define MAX_MATERIAL_NAME 128

struct CachedMaterial {
  char name[MAX_MATERIAL_NAME] = {};
  vec3 ambient = vec3(0.0f);
  vec3 diffuse = vec3(0.0f);
  char ambient_texname[MAX_MATERIAL_NAME] = {};
  char diffuse_texname[MAX_MATERIAL_NAME] = {};
  char alpha_texname[MAX_MATERIAL_NAME] = {};
};

// this goes into our GLSL shaders
struct GPUMaterial {
  vec4 ambient = vec4(0.0f);
  vec4 diffuse = vec4(0.0f);
  uint32_t texAmbient = 0;
  uint32_t texDiffuse = 0;
  uint32_t texAlpha = 0;
  uint32_t padding[1];
};

static_assert(sizeof(GPUMaterial) % 16 == 0);

std::vector<CachedMaterial> cachedMaterials_;
std::vector<GPUMaterial> materials_;

struct MaterialTextures {
  lvk::TextureHandle ambient;
  lvk::TextureHandle diffuse;
  lvk::TextureHandle alpha;
};

std::vector<MaterialTextures> textures_; // same indexing as in materials_

struct LoadedImage {
  uint32_t w = 0;
  uint32_t h = 0;
  uint32_t channels = 0;
  uint8_t* pixels = nullptr;
  std::string debugName;
};

struct LoadedMaterial {
  size_t idx = 0;
  LoadedImage ambient;
  LoadedImage diffuse;
  LoadedImage alpha;
};

// file name -> LoadedImage
std::mutex imagesCacheMutex_;
std::unordered_map<std::string, LoadedImage> imagesCache_; // accessible only from the loader pool (multiple threads)
std::unordered_map<std::string, lvk::Holder<lvk::TextureHandle>> texturesCache_; // accessible the main thread
std::vector<LoadedMaterial> loadedMaterials_;
std::mutex loadedMaterialsMutex_;
std::atomic<bool> loaderShouldExit_ = false;
std::atomic<uint32_t> remainingMaterialsToLoad_ = 0;
std::unique_ptr<tf::Executor> loaderPool_ =
  std::make_unique<tf::Executor>(std::max(2u, std::thread::hardware_concurrency() / 2));

void initIGL(ANativeWindow* window, AAssetManager* assetManager) {
  ctx_ = createVulkanContext(window, width_, height_, {});

  {
    const uint32_t pixel = 0xFFFFFFFF;
    textureDummyWhite_ = ctx_->createTexture(
      {
        .type = lvk::TextureType_2D,
        .format = lvk::Format_RGBA_UN8,
        .dimensions = {1, 1},
        .usage = lvk::TextureUsageBits_Sampled,
        .data = &pixel,
        .debugName = "dummy 1x1 (white)",
      },
      nullptr);
  }

  // create an Uniform buffers to store uniforms for 2 objects
  for (uint32_t i = 0; i != kNumBufferedFrames; i++) {
    ubPerFrame_.push_back(ctx_->createBuffer({.usage = lvk::BufferUsageBits_Uniform,
                                               .storage = lvk::StorageType_HostVisible,
                                               .size = sizeof(UniformsPerFrame),
                                               .debugName = "Buffer: uniforms (per frame)"},
                                             nullptr));
    ubPerFrameShadow_.push_back(
      ctx_->createBuffer({.usage = lvk::BufferUsageBits_Uniform,
                           .storage = lvk::StorageType_HostVisible,
                           .size = sizeof(UniformsPerFrame),
                           .debugName = "Buffer: uniforms (per frame shadow)"},
                         nullptr));
    ubPerObject_.push_back(ctx_->createBuffer({.usage = lvk::BufferUsageBits_Uniform,
                                                .storage = lvk::StorageType_HostVisible,
                                                .size = sizeof(UniformsPerObject),
                                                .debugName = "Buffer: uniforms (per object)"},
                                              nullptr));
  }

  depthState_ = {.compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true};
  depthStateLEqual_ = {.compareOp = lvk::CompareOp_LessEqual, .isDepthWriteEnabled = true};

  sampler_ = ctx_->createSampler(
    {
      .mipMap = lvk::SamplerMip_Linear,
      .wrapU = lvk::SamplerWrap_Repeat,
      .wrapV = lvk::SamplerWrap_Repeat,
      .debugName = "Sampler: linear",
    },
    nullptr);
  samplerShadow_ = ctx_->createSampler(
    {
      .wrapU = lvk::SamplerWrap_Clamp,
      .wrapV = lvk::SamplerWrap_Clamp,
      .depthCompareOp = lvk::CompareOp_LessEqual,
      .depthCompareEnabled = true,
      .debugName = "Sampler: shadow",
    },
    nullptr);

  renderPassOffscreen_ = {
    .color = {{
                .loadOp = lvk::LoadOp_Clear,
                .storeOp = kNumSamplesMSAA > 1 ? lvk::StoreOp_MsaaResolve : lvk::StoreOp_Store,
                .clearColor = {0.0f, 0.0f, 0.0f, 1.0f},
              }},
    .depth = {
      .loadOp = lvk::LoadOp_Clear,
      .storeOp = lvk::StoreOp_Store,
      .clearDepth = 1.0f,
    }};

  renderPassMain_ = {
    .color = {{.loadOp = lvk::LoadOp_Clear,
                .storeOp = lvk::StoreOp_Store,
                .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}}},
  };
  renderPassShadow_ = {
    .color = {},
    .depth = {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearDepth = 1.0f},
  };
}

void normalizeName(std::string& name) {
#if defined(__linux__) || defined(__APPLE__) || defined(ANDROID)
  std::replace(name.begin(), name.end(), '\\', '/');
#endif
}

bool loadFromCache(AAssetManager * assetManager) {
  AAsset * cacheFile = AAssetManager_open(assetManager, "cache.data", AASSET_MODE_BUFFER);
  SCOPE_EXIT {
               if (cacheFile) {
                 AAsset_close(cacheFile);
               }
             };
  if (!cacheFile) {
    return false;
  }
#define CHECK_READ(expected, read) \
  if ((read) != (expected)) {      \
    return false;                  \
  }
  uint32_t versionProbe = 0;
  CHECK_READ(sizeof(versionProbe), AAsset_read(cacheFile, &versionProbe, sizeof(versionProbe)));
  if (versionProbe != kMeshCacheVersion) {
    LLOGL("Cache file has wrong version id\n");
    return false;
  }
  uint32_t numMaterials = 0;
  uint32_t numVertices = 0;
  uint32_t numIndices = 0;
  CHECK_READ(sizeof(numMaterials), AAsset_read(cacheFile, &numMaterials, sizeof(numMaterials)));
  CHECK_READ(sizeof(numVertices), AAsset_read(cacheFile, &numVertices, sizeof(numVertices)));
  CHECK_READ(sizeof(numIndices), AAsset_read(cacheFile, &numIndices, sizeof(numIndices)));
  cachedMaterials_.resize(numMaterials);
  vertexData_.resize(numVertices);
  indexData_.resize(numIndices);
  CHECK_READ(sizeof(CachedMaterial) * numMaterials, AAsset_read(cacheFile, cachedMaterials_.data(), sizeof(CachedMaterial) * numMaterials));
  CHECK_READ(sizeof(VertexData) * numVertices, AAsset_read(cacheFile, vertexData_.data(), sizeof(VertexData) * numVertices));
  CHECK_READ(sizeof(uint32_t) * numIndices, AAsset_read(cacheFile, indexData_.data(), sizeof(uint32_t) * numIndices));
#undef CHECK_READ
  return true;
}

bool initModel(AAssetManager * assetManager) {
  if (!loadFromCache(assetManager)) {
    LVK_ASSERT_MSG(false, "Cannot load 3D model");
    return false;
  }

  for (const auto& mtl : cachedMaterials_) {
    materials_.push_back(GPUMaterial{vec4(mtl.ambient, 1.0f),
                                     vec4(mtl.diffuse, 1.0f),
                                     textureDummyWhite_.index(),
                                     textureDummyWhite_.index()});
  }
  sbMaterials_ = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Storage,
                                      .storage = lvk::StorageType_Device,
                                      .size = sizeof(GPUMaterial) * materials_.size(),
                                      .data = materials_.data(),
                                      .debugName = "Buffer: materials"},
                                    nullptr);

  vb0_ = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Vertex,
                              .storage = lvk::StorageType_Device,
                              .size = sizeof(VertexData) * vertexData_.size(),
                              .data = vertexData_.data(),
                              .debugName = "Buffer: vertex"},
                            nullptr);
  ib0_ = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Index,
                              .storage = lvk::StorageType_Device,
                              .size = sizeof(uint32_t) * indexData_.size(),
                              .data = indexData_.data(),
                              .debugName = "Buffer: index"},
                            nullptr);
  return true;
}

void createPipelines() {
  if (renderPipelineState_Mesh_.valid()) {
    return;
  }

  const lvk::VertexInput vdesc = {
    .attributes =
      {
        {.location = 0, .format = lvk::VertexFormat::Float3, .offset = offsetof(VertexData, position)},
        {.location = 1, .format = lvk::VertexFormat::Int_2_10_10_10_REV, .offset = offsetof(VertexData, normal)},
        {.location = 2, .format = lvk::VertexFormat::HalfFloat2, .offset = offsetof(VertexData, uv)},
        {.location = 3, .format = lvk::VertexFormat::UInt1, .offset = offsetof(VertexData, mtlIndex)},
      },
    .inputBindings = {{.stride = sizeof(VertexData)}},
  };

  // shadow
  const lvk::VertexInput vdescs = {
    .attributes = {{.format = lvk::VertexFormat::Float3, .offset = offsetof(VertexData, position)}},
    .inputBindings = {{.stride = sizeof(VertexData)}},
  };

  smMeshVert_ = ctx_->createShaderModule({kCodeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
  smMeshFrag_ = ctx_->createShaderModule({kCodeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});
  smMeshWireframeVert_ = ctx_->createShaderModule({kCodeVS_Wireframe, lvk::Stage_Vert, "Shader Module: main wireframe (vert)"});
  smMeshWireframeFrag_ = ctx_->createShaderModule({kCodeFS_Wireframe, lvk::Stage_Frag, "Shader Module: main wireframe (frag)"});
  smShadowVert_ = ctx_->createShaderModule({kShadowVS, lvk::Stage_Vert, "Shader Module: shadow (vert)"});
  smShadowFrag_ = ctx_->createShaderModule({kShadowFS, lvk::Stage_Frag, "Shader Module: shadow (frag)"});
  smFullscreenVert_ = ctx_->createShaderModule({kCodeFullscreenVS, lvk::Stage_Vert, "Shader Module: fullscreen (vert)"});
  smFullscreenFrag_ = ctx_->createShaderModule({kCodeFullscreenFS, lvk::Stage_Frag, "Shader Module: fullscreen (frag)"});
  smSkyboxVert_ = ctx_->createShaderModule({kSkyboxVS, lvk::Stage_Vert, "Shader Module: skybox (vert)"});
  smSkyboxFrag_ = ctx_->createShaderModule({kSkyboxFS, lvk::Stage_Frag, "Shader Module: skybox (frag)"});

  {
    lvk::RenderPipelineDesc desc = {
      .vertexInput = vdesc,
      .smVert = smMeshVert_,
      .smFrag = smMeshFrag_,
      .color = {{.format = ctx_->getFormat(fbOffscreen_.color[0].texture)}},
      .depthFormat = ctx_->getFormat(fbOffscreen_.depthStencil.texture),
      .cullMode = lvk::CullMode_Back,
      .frontFaceWinding = lvk::WindingMode_CCW,
      .samplesCount = kNumSamplesMSAA,
      .debugName = "Pipeline: mesh",
    };

    renderPipelineState_Mesh_ = ctx_->createRenderPipeline(desc, nullptr);

    desc.polygonMode = lvk::PolygonMode_Line;
    desc.vertexInput = vdescs; // positions-only
    desc.smVert = smMeshWireframeVert_;
    desc.smFrag = smMeshWireframeFrag_;
    desc.debugName = "Pipeline: mesh (wireframe)";
    renderPipelineState_MeshWireframe_ = ctx_->createRenderPipeline(desc, nullptr);
  }

  // shadow
  renderPipelineState_Shadow_ = ctx_->createRenderPipeline(
    lvk::RenderPipelineDesc{
      .vertexInput = vdescs,
      .smVert = smShadowVert_,
      .smFrag = smShadowFrag_,
      .depthFormat = ctx_->getFormat(fbShadowMap_.depthStencil.texture),
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: shadow",
    },
    nullptr);

  // fullscreen
  {
    const lvk::RenderPipelineDesc desc = {
      .smVert = smFullscreenVert_,
      .smFrag = smFullscreenFrag_,
      .color = {{.format = ctx_->getFormat(fbMain_.color[0].texture)}},
      .depthFormat = ctx_->getFormat(fbMain_.depthStencil.texture),
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: fullscreen",
    };
    renderPipelineState_Fullscreen_ = ctx_->createRenderPipeline(desc, nullptr);
  }

  // skybox
  {
    const lvk::RenderPipelineDesc desc = {
      .smVert = smSkyboxVert_,
      .smFrag = smSkyboxFrag_,
      .color = {{
                  .format = ctx_->getFormat(fbOffscreen_.color[0].texture),
                }},
      .depthFormat = ctx_->getFormat(fbOffscreen_.depthStencil.texture),
      .cullMode = lvk::CullMode_Front,
      .frontFaceWinding = lvk::WindingMode_CCW,
      .samplesCount = kNumSamplesMSAA,
      .debugName = "Pipeline: skybox",
    };

    renderPipelineState_Skybox_ = ctx_->createRenderPipeline(desc, nullptr);
  }

  smGrayscaleComp_ = ctx_->createShaderModule({kCodeComputeTest, lvk::Stage_Comp, "Shader Module: grayscale (comp)"});

  computePipelineState_Grayscale_ = ctx_->createComputePipeline({.smComp = smGrayscaleComp_}, nullptr);
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
  const uint32_t w = width_;
  const uint32_t h = height_;
  lvk::TextureDesc descDepth = {
    .type = lvk::TextureType_2D,
    .format = lvk::Format_Z_UN24,
    .dimensions = {w, h},
    .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
    .numMipLevels = lvk::calcNumMipLevels(w, h),
    .debugName = "Offscreen framebuffer (d)",
  };
  if (kNumSamplesMSAA > 1) {
    descDepth.usage = lvk::TextureUsageBits_Attachment;
    descDepth.numSamples = kNumSamplesMSAA;
    descDepth.numMipLevels = 1;
  }

  const uint8_t usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled |
                        lvk::TextureUsageBits_Storage;
  const lvk::Format format = lvk::Format_RGBA_UN8;

  lvk::TextureDesc descColor = {
    .type = lvk::TextureType_2D,
    .format = format,
    .dimensions = {w, h},
    .usage = usage,
    .numMipLevels = lvk::calcNumMipLevels(w, h),
    .debugName = "Offscreen framebuffer (color)",
  };
  if (kNumSamplesMSAA > 1) {
    descColor.usage = lvk::TextureUsageBits_Attachment;
    descColor.numSamples = kNumSamplesMSAA;
    descColor.numMipLevels = 1;
  }

  fbOffscreenColor_ = ctx_->createTexture(descColor);
  fbOffscreenDepth_ = ctx_->createTexture(descDepth);
  lvk::Framebuffer fb = {
    .color = {{.texture = fbOffscreenColor_}},
    .depthStencil = {.texture = fbOffscreenDepth_},
  };

  if (kNumSamplesMSAA > 1) {
    fbOffscreenResolve_ = ctx_->createTexture({.type = lvk::TextureType_2D,
                                                .format = format,
                                                .dimensions = {w, h},
                                                .usage = usage,
                                                .debugName = "Offscreen framebuffer (color resolve)"});
    fb.color[0].resolveTexture = fbOffscreenResolve_;
  }

  fbOffscreen_ = fb;
}

void render(lvk::TextureHandle nativeDrawable, uint32_t frameIndex) {
  LVK_PROFILER_FUNCTION();

  if (!width_ && !height_)
    return;

  timestampBeginRendering = (clock() - start) / float(CLOCKS_PER_SEC);

  fbMain_.color[0].texture = nativeDrawable;

  const float fov = float(45.0f * (M_PI / 180.0f));
  const float aspectRatio = (float)width_ / (float)height_;

  const mat4 shadowProj = glm::perspective(float(60.0f * (M_PI / 180.0f)), 1.0f, 10.0f, 4000.0f);
  const mat4 shadowView = mat4(vec4(0.772608519f, 0.532385886f, -0.345892131f, 0),
                               vec4(0, 0.544812560f, 0.838557839f, 0),
                               vec4(0.634882748f, -0.647876859f, 0.420926809f, 0),
                               vec4(-58.9244843f, -30.4530792f, -508.410126f, 1.0f));
  const mat4 scaleBias = mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0);

  perFrame_ = UniformsPerFrame{
    .proj = glm::perspective(fov, aspectRatio, 0.5f, 500.0f),
    .view = camera_.getViewMatrix(),
    .light = scaleBias * shadowProj * shadowView,
    .texSkyboxRadiance = skyboxTextureReference_.index(),
    .texSkyboxIrradiance = skyboxTextureIrradiance_.index(),
    .texShadow = fbShadowMap_.depthStencil.texture.index(),
    .sampler = sampler_.index(),
    .samplerShadow = samplerShadow_.index(),
    .bDrawNormals = perFrame_.bDrawNormals,
    .bDebugLines = perFrame_.bDebugLines,
  };
  ctx_->upload(ubPerFrame_[frameIndex], &perFrame_, sizeof(perFrame_));

  {
    const UniformsPerFrame perFrameShadow{
      .proj = shadowProj,
      .view = shadowView,
    };
    ctx_->upload(ubPerFrameShadow_[frameIndex], &perFrameShadow, sizeof(perFrameShadow));
  }

  UniformsPerObject perObject;

  perObject.model = glm::scale(mat4(1.0f), vec3(0.05f));

  ctx_->upload(ubPerObject_[frameIndex], &perObject, sizeof(perObject));

  // Command buffers (1-N per thread): create, submit and forget

  // Pass 1: shadows
  if (isShadowMapDirty_) {
    lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

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
        .perFrame = ctx_->gpuAddress(ubPerFrameShadow_[frameIndex]),
        .perObject = ctx_->gpuAddress(ubPerObject_[frameIndex]),
      };
      buffer.cmdPushConstants(bindings);
      buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI32);
      buffer.cmdDrawIndexed(static_cast<uint32_t>(indexData_.size()));
      buffer.cmdPopDebugGroupLabel();
    }
    buffer.cmdEndRendering();
    buffer.transitionToShaderReadOnly(fbShadowMap_.depthStencil.texture);
    ctx_->submit(buffer);
    ctx_->generateMipmap(fbShadowMap_.depthStencil.texture);

    isShadowMapDirty_ = false;
  }

#define GPU_TIMESTAMP(timestamp) buffer.cmdWriteTimestamp(queryPoolTimestamps_, timestamp);

  // Pass 2: mesh
  {
    lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

    buffer.cmdResetQueryPool(queryPoolTimestamps_, 0, GPUTimestamp_NUM_TIMESTAMPS);

    GPU_TIMESTAMP(GPUTimestamp_BeginSceneRendering);

    // This will clear the framebuffer
    buffer.cmdBeginRendering(renderPassOffscreen_, fbOffscreen_);
    {
      // Scene
      buffer.cmdBindRenderPipeline(renderPipelineState_Mesh_);
      buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
      buffer.cmdBindDepthState(depthState_);
      buffer.cmdBindVertexBuffer(0, vb0_, 0);

      struct {
        uint64_t perFrame;
        uint64_t perObject;
        uint64_t materials;
      } bindings = {
        .perFrame = ctx_->gpuAddress(ubPerFrame_[frameIndex]),
        .perObject = ctx_->gpuAddress(ubPerObject_[frameIndex]),
        .materials = ctx_->gpuAddress(sbMaterials_),
      };
      buffer.cmdPushConstants(bindings);
      buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI32);
      buffer.cmdDrawIndexed(static_cast<uint32_t>(indexData_.size()));
      if (enableWireframe_) {
        buffer.cmdBindRenderPipeline(renderPipelineState_MeshWireframe_);
        buffer.cmdDrawIndexed(static_cast<uint32_t>(indexData_.size()));
      }
      buffer.cmdPopDebugGroupLabel();

      // Skybox
      buffer.cmdBindRenderPipeline(renderPipelineState_Skybox_);
      buffer.cmdPushDebugGroupLabel("Render Skybox", 0x00ff00ff);
      buffer.cmdBindDepthState(depthStateLEqual_);
      buffer.cmdDraw(3 * 6 * 2);
      buffer.cmdPopDebugGroupLabel();
    }
    buffer.cmdEndRendering();

    GPU_TIMESTAMP(GPUTimestamp_EndSceneRendering);

    ctx_->submit(buffer);
  }

  // Pass 3: compute shader post-processing
  {
    lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

    GPU_TIMESTAMP(GPUTimestamp_BeginComputePass);

    if (enableComputePass_) {
      lvk::TextureHandle tex = kNumSamplesMSAA > 1 ? fbOffscreen_.color[0].resolveTexture : fbOffscreen_.color[0].texture;

      buffer.cmdBindComputePipeline(computePipelineState_Grayscale_);

      struct {
        uint32_t texture;
        uint32_t width;
        uint32_t height;
      } bindings = {
        .texture = tex.index(),
        .width = (uint32_t)width_,
        .height = (uint32_t)height_,
      };
      buffer.cmdPushConstants(bindings);
      buffer.cmdDispatchThreadGroups(
        {
          .width = 1 + (uint32_t)width_ / 16,
          .height = 1 + (uint32_t)height_ / 16,
          .depth = 1u,
        },
        {
          .textures = {tex},
        });
    }
    GPU_TIMESTAMP(GPUTimestamp_EndComputePass);

    ctx_->submit(buffer);
  }

  // Pass 4: render into the swapchain image
  {
    lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

    GPU_TIMESTAMP(GPUTimestamp_BeginPresent);

    lvk::TextureHandle tex = kNumSamplesMSAA > 1 ? fbOffscreen_.color[0].resolveTexture : fbOffscreen_.color[0].texture;

    // This will clear the framebuffer
    buffer.cmdBeginRendering(renderPassMain_, fbMain_, {.textures = {tex}});
    {
      buffer.cmdBindRenderPipeline(renderPipelineState_Fullscreen_);
      buffer.cmdPushDebugGroupLabel("Swapchain Output", 0xff0000ff);
      //buffer.cmdBindDepthState(depthState_);
      struct {
        uint32_t texture;
      } bindings = {
        .texture = tex.index(),
      };
      buffer.cmdPushConstants(bindings);
      buffer.cmdDraw(3);
      buffer.cmdPopDebugGroupLabel();

      imgui_->endFrame(buffer);
    }
    buffer.cmdEndRendering();

    GPU_TIMESTAMP(GPUTimestamp_EndPresent);

    ctx_->submit(buffer, fbMain_.color[0].texture);
  }

  timestampEndRendering = (clock() - start) / float(CLOCKS_PER_SEC);

  // timestamp stats
  ctx_->getQueryPoolResults(queryPoolTimestamps_,
                            0,
                            LVK_ARRAY_NUM_ELEMENTS(pipelineTimestamps),
                            sizeof(pipelineTimestamps),
                            pipelineTimestamps,
                            sizeof(pipelineTimestamps[0]));
}

LoadedImage loadImage(AAssetManager* assetManager, const char* fileName, int channels) {
  LVK_PROFILER_FUNCTION();

  if (!fileName || !*fileName) {
    return LoadedImage();
  }

  char debugStr[512] = { 0 };

  snprintf(debugStr, sizeof(debugStr)-1, "%s (%i)", fileName, channels);

  const std::string debugName(debugStr);

  {
    std::lock_guard lock(imagesCacheMutex_);

    const auto it = imagesCache_.find(debugName);

    if (it != imagesCache_.end()) {
      LVK_ASSERT(channels == it->second.channels);
      return it->second;
    }
  }

  uint8_t* pixels = nullptr;
  int32_t texWidth = 0;
  int32_t texHeight = 0;
  auto fn = fileName + 3;
  AAsset * file = AAssetManager_open(assetManager, fn, AASSET_MODE_BUFFER);
  if(file) {
    size_t fileLength = AAsset_getLength(file);
    stbi_uc * temp = (stbi_uc*)malloc(fileLength);
    memcpy(temp, AAsset_getBuffer(file), fileLength);
    pixels = stbi_load_from_memory(temp,
                                   fileLength,
                                   &texWidth,
                                   &texHeight,
                                   &channels,
                                   4);
    free(temp);
  } else {
    int a = 1;
    a++;
  }

  if (channels == 3) {
    channels = 4;
  }

  const LoadedImage img = {
    .w = (uint32_t)texWidth,
    .h = (uint32_t)texHeight,
    .channels = (uint32_t)channels,
    .pixels = pixels,
    .debugName = debugName,
  };

  std::lock_guard lock(imagesCacheMutex_);

  imagesCache_[fileName] = img;

  return img;
}

void loadMaterial(AAssetManager* assetManager, size_t i) {
  LVK_PROFILER_FUNCTION();

  SCOPE_EXIT {
               remainingMaterialsToLoad_.fetch_sub(1u, std::memory_order_release);
             };

#define LOAD_TEX(result, tex, channels)                                          \
  const LoadedImage result =                                                     \
      std::string(cachedMaterials_[i].tex).empty()                               \
          ? LoadedImage()                                                        \
          : loadImage(assetManager, cachedMaterials_[i].tex, channels);          \
  if (loaderShouldExit_.load(std::memory_order_acquire)) {                       \
    return;                                                                      \
  }

  LOAD_TEX(ambient, ambient_texname, 4);
  LOAD_TEX(diffuse, diffuse_texname, 4);
  LOAD_TEX(alpha, alpha_texname, 1);

#undef LOAD_TEX

  const LoadedMaterial mtl{i, ambient, diffuse, alpha};

  if (!mtl.ambient.pixels && !mtl.diffuse.pixels) {
    // skip missing textures
    materials_[i].texDiffuse = 0;
  } else {
    std::lock_guard guard(loadedMaterialsMutex_);
    loadedMaterials_.push_back(mtl);
    remainingMaterialsToLoad_.fetch_add(1u, std::memory_order_release);
  }
}

void loadMaterials(AAssetManager* assetManager) {
  stbi_set_flip_vertically_on_load(1);

  remainingMaterialsToLoad_ = (uint32_t)cachedMaterials_.size();

  textures_.resize(cachedMaterials_.size());
  for (size_t i = 0; i != cachedMaterials_.size(); i++) {
    loaderPool_->silent_async([i, assetManager]() { loadMaterial(assetManager, i); });
  }
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

void loadCubemapTexture(AAssetManager* assetManager, const std::string& fileNameKTX,
                        lvk::Holder<lvk::TextureHandle>& tex) {
  LVK_PROFILER_FUNCTION();

  ktx_uint8_t* buf = nullptr;
  ktx_size_t fileLength = 0;
  AAsset * file = AAssetManager_open(assetManager, fileNameKTX.c_str(), AASSET_MODE_BUFFER);
  if(file) {
    fileLength = (ktx_size_t)AAsset_getLength(file);
    buf = (ktx_uint8_t*)malloc(fileLength);
    memcpy(buf, AAsset_getBuffer(file), fileLength);
  } else {
    LVK_ASSERT_MSG(false, "Can't load file");
    return;
  }

  ktxTexture1* texture = nullptr;
  (void)LVK_VERIFY(ktxTexture1_CreateFromMemory(buf, fileLength, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture) == KTX_SUCCESS);
  SCOPE_EXIT {
               ktxTexture_Destroy(ktxTexture(texture));
               free(buf);
             };

  if (!LVK_VERIFY(texture->glInternalformat == GL_RGBA32F)) {
    LVK_ASSERT_MSG(false, "Texture format not supported");
    return;
  }

  const uint32_t width = texture->baseWidth;
  const uint32_t height = texture->baseHeight;

  if (tex.empty()) {
    tex = ctx_->createTexture(
      {
        .type = lvk::TextureType_Cube,
        .format = ktx2iglTextureFormat(texture->glInternalformat),
        .dimensions = {width, height},
        .usage = lvk::TextureUsageBits_Sampled,
        .numMipLevels = lvk::calcNumMipLevels(width, height),
        .data = texture->pData,
        // if compression is enabled, upload all mip-levels
        .dataNumMipLevels = 1u,
        .debugName = fileNameKTX.c_str(),
      },
      nullptr);
  }

  ctx_->generateMipmap(tex);
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

void loadSkyboxTexture(AAssetManager* assetManager) {
  LVK_PROFILER_FUNCTION();

  const std::string skyboxFileName{"immenstadter_horn_2k"};
  const std::string fileNameRefKTX = skyboxFileName + "_ReferenceMap.ktx";
  const std::string fileNameIrrKTX = skyboxFileName + "_IrradianceMap.ktx";

  loadCubemapTexture(assetManager, fileNameRefKTX, skyboxTextureReference_);
  loadCubemapTexture(assetManager, fileNameIrrKTX, skyboxTextureIrradiance_);
}

lvk::Format formatFromChannels(uint32_t channels) {
  if (channels == 1) {
    return lvk::Format_R_UN8;
  }

  if (channels == 4) {
    return lvk::Format_RGBA_UN8;
  }

  return lvk::Format_Invalid;
}

lvk::TextureHandle createTexture(const LoadedImage& img) {
  if (!img.pixels) {
    return {};
  }

  const auto it = texturesCache_.find(img.debugName);

  if (it != texturesCache_.end()) {
    return it->second;
  }

  const void* initialData = img.pixels;
  uint32_t initialDataNumMipLevels = 1u;

  lvk::Holder<lvk::TextureHandle> tex = ctx_->createTexture(
    {
      .type = lvk::TextureType_2D,
      .format = formatFromChannels(img.channels),
      .dimensions = {img.w, img.h},
      .usage = lvk::TextureUsageBits_Sampled,
      .numMipLevels = lvk::calcNumMipLevels(img.w, img.h),
      .data = initialData,
      .dataNumMipLevels = initialDataNumMipLevels,
      .debugName = img.debugName.c_str(),
    },
    nullptr);

  ctx_->generateMipmap(tex);

  lvk::TextureHandle handle = tex;

  texturesCache_[img.debugName] = std::move(tex);

  return handle;
}

void processLoadedMaterials() {
  LoadedMaterial mtl;

  {
    std::lock_guard guard(loadedMaterialsMutex_);
    if (loadedMaterials_.empty()) {
      return;
    } else {
      mtl = loadedMaterials_.back();
      loadedMaterials_.pop_back();
      remainingMaterialsToLoad_.fetch_sub(1u, std::memory_order_release);
    }
  }

  {
    MaterialTextures tex;

    tex.ambient = createTexture(mtl.ambient);
    tex.diffuse = createTexture(mtl.diffuse);
    tex.alpha = createTexture(mtl.alpha);

    // update GPU materials
    materials_[mtl.idx].texAmbient = tex.ambient.index();
    materials_[mtl.idx].texDiffuse = tex.diffuse.index();
    materials_[mtl.idx].texAlpha = tex.alpha.index();
    textures_[mtl.idx] = std::move(tex);
  }
  LVK_ASSERT(materials_[mtl.idx].texAmbient >= 0);
  LVK_ASSERT(materials_[mtl.idx].texDiffuse >= 0);
  LVK_ASSERT(materials_[mtl.idx].texAlpha >= 0);
  ctx_->upload(sbMaterials_, materials_.data(), sizeof(GPUMaterial) * materials_.size());
}

inline ImVec4 toVec4(const vec4& c) {
  return ImVec4(c.x, c.y, c.z, c.w);
}

void showTimeGPU() {
#if defined(LVK_WITH_IMPLOT)
  const double toMS = ctx_->getTimestampPeriodToMs();
  auto getTimespan = [toMS](GPUTimestamp begin) -> double {
    return double(pipelineTimestamps[begin + 1] - pipelineTimestamps[begin]) * toMS;
  };
  struct sTimeStats {
    enum size { kNumTimelines = 5 };
    struct MinMax {
      float vmin = FLT_MAX;
      float vmax = 0.0f;
    };
    double add(uint32_t pass, const char* name, double value) {
      LVK_ASSERT(pass < kNumTimelines);
      names[pass] = name;
      const float prev = timelines[pass].empty() ? (float)value : timelines[pass].back();
      timelines[pass].push_back(0.9 * prev + 0.1 * value);
      if (timelines[pass].size() > 128)
        timelines[pass].erase(timelines[pass].begin());
      avg[pass] = value;
      return (float)value;
    }
    void updateMinMax() {
      for (uint32_t i = 0; i != kNumTimelines; i++) {
        float minT = FLT_MAX;
        float maxT = 0.0f;
        for (float v : timelines[i]) {
          if (v < minT)
            minT = v;
          if (v > maxT)
            maxT = v;
        }
        minmax[i] = {minT, maxT};
      }
    }
    std::vector<float> timelines[kNumTimelines] = {};
    MinMax minmax[kNumTimelines] = {};
    float avg[kNumTimelines] = {};
    const char* names[kNumTimelines] = {};
    const vec4 colors[kNumTimelines] = {LC_Red, LC_Green, LC_Green, LC_LightBlue, LC_Red};
  };
  static sTimeStats stats;

  const double timeScene = stats.add(1, " Scene", getTimespan(GPUTimestamp_BeginSceneRendering));
  const double timeCompute = stats.add(2, " Compute", getTimespan(GPUTimestamp_BeginComputePass));
  const double timePresent = stats.add(3, " Present", getTimespan(GPUTimestamp_BeginPresent));

  const double timeGPU = timeScene + timeCompute + timePresent;
  stats.add(0, "GPU", timeGPU);
  const double timeCPU = stats.add(4, "CPU", (timestampEndRendering - timestampBeginRendering) * 1000);
  stats.updateMinMax();

  char text[128];
  snprintf(text,
           sizeof(text),
           "GPU: %6.02f ms   (Scene: %.02f   Compute: %.02f   Present: %.02f)",
           timeGPU,
           timeScene,
           timeCompute,
           timePresent);

  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                                 ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::SetNextWindowPos({20, height_ * 0.8f}, ImGuiCond_Appearing);
  ImGui::SetNextWindowSize({width_ * 0.4f, 0});
  ImGui::Begin("GPU Stats", nullptr, flags);
  ImGui::SetWindowFontScale(2.0);
  ImGui::Text("%s", text);

  auto Sparkline = [](const char* id, const float* values, int count, float min_v, float max_v, const ImVec4& col, const ImVec2& size) {
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::SetNextAxesLimits(0, count - 1, min_v, max_v, ImGuiCond_Always);
    if (ImPlot::BeginPlot(id, size, ImPlotFlags_CanvasOnly)) {
      ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
      ImPlot::PushStyleColor(ImPlotCol_Line, col);
      ImPlot::PlotLine(id, values, count, 1, 0);
      ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
      ImPlot::PlotShaded(id, values, count, 0, 1, 0);
      ImPlot::PopStyleVar();
      ImPlot::PopStyleColor();
      ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
  };

  auto RowLeadIn = [](const char* stage, float value, const ImVec4& color) {
    ImGui::TableSetColumnIndex(0);
    ImGui::TextColored(color, "%s", stage);
    ImGui::TableSetColumnIndex(1);
    ImGui::TextColored(color, "%6.02f", value);
    ImGui::TableSetColumnIndex(2);
  };

  if (ImGui::BeginTable("##table", 3, ImGuiTableFlags_None, ImVec2(-1, 0))) {
    const ImGuiTableColumnFlags flags = ImGuiTableColumnFlags_NoSort;
    ImGui::TableSetupColumn("Stage", flags);
    ImGui::TableSetupColumn("Time (ms)", flags);
    ImGui::TableSetupColumn("Graph", flags | ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableHeadersRow();
    for (uint32_t i = 0; i != sTimeStats::kNumTimelines; i++) {
      ImGui::TableNextRow();
      const ImVec4 color = toVec4(stats.colors[i]);
      RowLeadIn(stats.names[i], stats.avg[i], color);
      if (stats.avg[i] > 0.01)
        Sparkline("##spark",
                  stats.timelines[i].data(),
                  stats.timelines[i].size(),
                  stats.minmax[i].vmin * 0.8f,
                  stats.minmax[i].vmax * 1.2f,
                  color,
                  ImVec2(-1, 30));
    }

    ImGui::EndTable();
  }

  ImGui::End();
#endif // LVK_WITH_IMPLOT
}

#endif

extern "C" {
void handle_cmd(android_app* app, int32_t cmd) {
  switch (cmd) {
    case APP_CMD_INIT_WINDOW:
      if (app->window != nullptr) {
        width_ = ANativeWindow_getWidth(app->window);
        height_ = ANativeWindow_getHeight(app->window);
#if defined(TRIANGLE_DEMO)
        vk.init(app->window);
#elif defined(TINY_MESH_DEMO)
        initIGL(app->window, app->activity->assetManager);
        initObjects();
        imgui_ = std::make_unique<lvk::ImGuiRenderer>(*ctx_);
#elif defined(TINY_MESH_LARGE_DEMO)
        initIGL(app->window, app->activity->assetManager);
        initModel(app->activity->assetManager);
        loadSkyboxTexture(app->activity->assetManager);
        loadMaterials(app->activity->assetManager);

        fbMain_ = {
          .color = {{.texture = ctx_->getCurrentSwapchainTexture()}},
        };
        createShadowMap();
        createOffscreenFramebuffer();
        createPipelines();

        imgui_ = std::make_unique<lvk::ImGuiRenderer>(*ctx_);

        queryPoolTimestamps_ = ctx_->createQueryPool(GPUTimestamp_NUM_TIMESTAMPS, "queryPoolTimestamps_");
#endif
      }
      break;

    case APP_CMD_TERM_WINDOW:
#if defined(TRIANGLE_DEMO)
      if (ctx_ != nullptr) {
        ctx_->destroy(vk.fb_.color[1].texture);
        ctx_->destroy(vk.fb_.color[2].texture);
        ctx_->destroy(vk.fb_.color[3].texture);
      }
      vk = {};
      ctx_ = nullptr;
#elif defined(TINY_MESH_DEMO)
      imgui_ = nullptr;

      // destroy all the Vulkan stuff before closing the window
      vb0_ = nullptr;
      ib0_ = nullptr;
      ubPerFrame_.clear();
      ubPerObject_.clear();
      vert_ = nullptr;
      frag_ = nullptr;
      renderPipelineState_Mesh_ = nullptr;
      texture0_ = nullptr;
      texture1_ = nullptr;
      sampler_ = nullptr;
      framebuffer_ = {};
      ctx_ = nullptr;
#elif defined(TINY_MESH_LARGE_DEMO)
      loaderShouldExit_.store(true, std::memory_order_release);

      imgui_ = nullptr;
      // destroy all the Vulkan stuff before closing the window
      vb0_ = nullptr;
      ib0_ = nullptr;
      sbMaterials_ = nullptr;
      ubPerFrame_.clear();
      ubPerFrameShadow_.clear();
      ubPerObject_.clear();
      smMeshVert_ = nullptr;
      smMeshFrag_ = nullptr;
      smMeshWireframeVert_ = nullptr;
      smMeshWireframeFrag_ = nullptr;
      smShadowVert_ = nullptr;
      smShadowFrag_ = nullptr;
      smFullscreenVert_ = nullptr;
      smFullscreenFrag_ = nullptr;
      smSkyboxVert_ = nullptr;
      smSkyboxFrag_ = nullptr;
      smGrayscaleComp_ = nullptr;
      renderPipelineState_Mesh_ = nullptr;
      renderPipelineState_MeshWireframe_ = nullptr;
      renderPipelineState_Shadow_ = nullptr;
      renderPipelineState_Skybox_ = nullptr;
      renderPipelineState_Fullscreen_ = nullptr;
      computePipelineState_Grayscale_ = nullptr;
      textureDummyWhite_ = nullptr;
      skyboxTextureReference_ = nullptr;
      skyboxTextureIrradiance_ = nullptr;
      textures_.clear();
      texturesCache_.clear();
      sampler_ = nullptr;
      samplerShadow_ = nullptr;
      ctx_->destroy(fbMain_);
      ctx_->destroy(fbShadowMap_);
      fbOffscreenColor_ = nullptr;
      fbOffscreenDepth_ = nullptr;
      fbOffscreenResolve_ = nullptr;
      queryPoolTimestamps_ = nullptr;
      ctx_ = nullptr;

      printf("Waiting for the loader thread to exit...\n");

      loaderPool_ = nullptr;
#endif
      break;
  }
}

void ResizeCallback(ANativeActivity *activity, ANativeWindow *window){
  width_ = ANativeWindow_getWidth(window);
  height_ = ANativeWindow_getHeight(window);
  if (ctx_) {
    ctx_->recreateSwapchain(width_, height_);
#if defined(TINY_MESH_LARGE_DEMO)
    createOffscreenFramebuffer();
#endif
  }
}

void android_main(android_app* app) {
  minilog::initialize(nullptr, {.threadNames = false});

  app->onAppCmd = handle_cmd;
  app->activity->callbacks->onNativeWindowResized = ResizeCallback;

  auto prevTime = std::chrono::steady_clock::now();
  uint32_t frameIndex = 0;

  int events = 0;
  android_poll_source* source = nullptr;
  do {
    if (ALooper_pollAll(0, nullptr, &events, (void **)&source) >= 0) {
      if (source) {
        source->process(app, source);
      }
    }

#if defined(TRIANGLE_DEMO)
    vk.render();
#elif defined(TINY_MESH_DEMO)
    if (ctx_) {
      framebuffer_ = {
        .color = {{.texture = ctx_->getCurrentSwapchainTexture()}},
      };

      imgui_->beginFrame(framebuffer_);

      ImGui::Begin("Texture Viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
      ImGui::Image(ImTextureID(texture1_.indexAsVoid()), ImVec2(512, 512));
      ImGui::End();

      render(ctx_->getCurrentSwapchainTexture(), frameIndex);
      frameIndex = (frameIndex + 1) % kNumBufferedFrames;
    }
#elif defined(TINY_MESH_LARGE_DEMO)
    if (ctx_) {
      processLoadedMaterials();

      auto newTime = std::chrono::steady_clock::now();
      auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(newTime - prevTime).count();
      prevTime = newTime;
      fps_.tick(delta / 1000.0);
      {
        fbMain_.color[0].texture = ctx_->getCurrentSwapchainTexture();
        imgui_->beginFrame(fbMain_);

        if (uint32_t num = remainingMaterialsToLoad_.load(std::memory_order_acquire)) {
          ImGui::SetNextWindowPos(ImVec2(0, 0));
          ImGui::Begin("Loading...", nullptr,
                       ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoInputs);
          ImGui::ProgressBar(1.0f - float(num) / cachedMaterials_.size(),
                             ImVec2(ImGui::GetIO().DisplaySize.x, 32));
          ImGui::End();
        }
        // a nice FPS counter
        {
          const ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav |
            ImGuiWindowFlags_NoMove;
          const ImGuiViewport *v = ImGui::GetMainViewport();
          LVK_ASSERT(v);
          ImGui::SetNextWindowPos(
            {
              v->WorkPos.x + v->WorkSize.x - 15.0f,
              v->WorkPos.y + 75.0f,
            },
            ImGuiCond_Always,
            {1.0f, 0.0f});
          ImGui::SetNextWindowBgAlpha(0.30f);
          ImGui::SetNextWindowSize(ImVec2(ImGui::CalcTextSize("FPS : _______").x * 3, 0));
          if (ImGui::Begin("##FPS", nullptr, flags)) {
            ImGui::SetWindowFontScale(3.0);
            ImGui::Text("FPS : %i", (int) fps_.getFPS());
            //ImGui::Text("Ms  : %.1f", 1000.0 / fps_.getFPS());
          }
          ImGui::End();
        }

        showTimeGPU();
      }

      positioner_.update(delta, mousePos_, mousePressed_);
      render(ctx_->getCurrentSwapchainTexture(), frameIndex);

      frameIndex = (frameIndex + 1) % kNumBufferedFrames;
    }
#endif
  } while (!app->destroyRequested);
}
}
