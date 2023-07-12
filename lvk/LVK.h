﻿/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <minilog/minilog.h>

#if defined(IGL_WITH_TRACY) && defined(__cplusplus)
#include "tracy/Tracy.hpp"
// predefined RGB colors for "heavy" point-of-interest operations
#define IGL_PROFILER_COLOR_WAIT 0xff0000
#define IGL_PROFILER_COLOR_SUBMIT 0x0000ff
#define IGL_PROFILER_COLOR_PRESENT 0x00ff00
#define IGL_PROFILER_COLOR_CREATE 0xff6600
#define IGL_PROFILER_COLOR_DESTROY 0xffa500
#define IGL_PROFILER_COLOR_TRANSITION 0xffffff
//
#define IGL_PROFILER_FUNCTION() ZoneScoped
#define IGL_PROFILER_FUNCTION_COLOR(color) ZoneScopedC(color)
#define IGL_PROFILER_ZONE(name, color) \
  {                                    \
    ZoneScopedC(color);                \
    ZoneName(name, strlen(name))
#define IGL_PROFILER_ZONE_END() }
#define IGL_PROFILER_THREAD(name) tracy::SetThreadName(name)
#define IGL_PROFILER_FRAME(name) FrameMarkNamed(name)
#else
#define IGL_PROFILER_FUNCTION()
#define IGL_PROFILER_FUNCTION_COLOR(color)
#define IGL_PROFILER_ZONE(name, color) {
#define IGL_PROFILER_ZONE_END() }
#define IGL_PROFILER_THREAD(name)
#define IGL_PROFILER_FRAME(name)
#endif // IGL_WITH_TRACY

#define IGL_ARRAY_NUM_ELEMENTS(x) (sizeof(x) / sizeof((x)[0]))

namespace igl {

bool _IGLVerify(bool cond, const char* file, int line, const char* format, ...);

} // namespace igl

// clang-format off
#if !defined(NDEBUG) && (defined(DEBUG) || defined(_DEBUG) || defined(__DEBUG))
  #define IGL_VERIFY(cond) ::igl::_IGLVerify((cond), __FILE__, __LINE__, #cond)
  #define IGL_ASSERT(cond) (void)IGL_VERIFY(cond)
  #define IGL_ASSERT_MSG(cond, format, ...) (void)::igl::_IGLVerify((cond), __FILE__, __LINE__, (format), ##__VA_ARGS__)
#else
  #define IGL_VERIFY(cond) (cond)
  #define IGL_ASSERT(cond)
  #define IGL_ASSERT_MSG(cond, format, ...)
#endif
// clang-format on

namespace igl {

class ITexture;

enum { IGL_COLOR_ATTACHMENTS_MAX = 4 };

enum class IndexFormat : uint8_t {
  UInt16,
  UInt32,
};

enum class PrimitiveType : uint8_t {
  Point,
  Line,
  LineStrip,
  Triangle,
  TriangleStrip,
};

enum class ColorSpace : uint8_t {
  SRGB_LINEAR,
  SRGB_NONLINEAR,
};

enum class TextureType : uint8_t {
  TwoD,
  ThreeD,
  Cube,
};

enum SamplerFilter : uint8_t { SamplerFilter_Nearest = 0, SamplerFilter_Linear };
enum SamplerMip : uint8_t { SamplerMip_Disabled = 0, SamplerMip_Nearest, SamplerMip_Linear };
enum SamplerWrap : uint8_t { SamplerWrap_Repeat = 0, SamplerWrap_Clamp, SamplerWrap_MirrorRepeat };

enum class HWDeviceType {
  DiscreteGpu = 1,
  ExternalGpu = 2,
  IntegratedGpu = 3,
  SoftwareGpu = 4,
};

struct HWDeviceDesc {
  enum { IGL_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256 };
  uintptr_t guid = 0;
  HWDeviceType type = HWDeviceType::SoftwareGpu;
  char name[IGL_MAX_PHYSICAL_DEVICE_NAME_SIZE] = { 0 };
};

enum StorageType {
  StorageType_Device,
  StorageType_HostVisible,
  StorageType_Memoryless
};

enum CullMode : uint8_t { CullMode_None, CullMode_Front, CullMode_Back };
enum WindingMode : uint8_t { WindingMode_CCW, WindingMode_CW };

struct Color {
  float r;
  float g;
  float b;
  float a;

  Color(float r, float g, float b) : r(r), g(g), b(b), a(1.0f) {}
  Color(float r, float g, float b, float a) : r(r), g(g), b(b), a(a) {}

  const float* toFloatPtr() const {
    return &r;
  }
};

static_assert(sizeof(Color) == 4 * sizeof(float));

struct Result {
  enum class Code {
    Ok,
    ArgumentOutOfRange,
    RuntimeError,
  };

  Code code = Code::Ok;
  const char* message = "";
  explicit Result() = default;
  explicit Result(Code code, const char* message = "") : code(code), message(message) {}

  bool isOk() const {
    return code == Result::Code::Ok;
  }

  static void setResult(Result* outResult, Code code, const char* message = "") {
    if (outResult) {
      outResult->code = code;
      outResult->message = message;
    }
  }

  static void setResult(Result* outResult, const Result& sourceResult) {
    if (outResult) {
      *outResult = sourceResult;
    }
  }
};

struct ScissorRect {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t width = 0;
  uint32_t height = 0;
};

struct Dimensions {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t depth = 0;
};

struct Viewport {
  float x = 0.0f;
  float y = 0.0f;
  float width = 1.0f;
  float height = 1.0f;
  float minDepth = 0.0f;
  float maxDepth = 1.0f;
};

enum CompareOp : uint8_t {
  CompareOp_Never = 0,
  CompareOp_Less,
  CompareOp_Equal,
  CompareOp_LessEqual,
  CompareOp_Greater,
  CompareOp_NotEqual,
  CompareOp_GreaterEqual,
  CompareOp_AlwaysPass
};

enum StencilOp : uint8_t {
  StencilOp_Keep = 0,
  StencilOp_Zero,
  StencilOp_Replace,
  StencilOp_IncrementClamp,
  StencilOp_DecrementClamp,
  StencilOp_Invert,
  StencilOp_IncrementWrap,
  StencilOp_DecrementWrap
};

enum BlendOp : uint8_t {
  BlendOp_Add = 0,
  BlendOp_Subtract,
  BlendOp_ReverseSubtract,
  BlendOp_Min,
  BlendOp_Max
};

enum BlendFactor : uint8_t {
  BlendFactor_Zero = 0,
  BlendFactor_One,
  BlendFactor_SrcColor,
  BlendFactor_OneMinusSrcColor,
  BlendFactor_SrcAlpha,
  BlendFactor_OneMinusSrcAlpha,
  BlendFactor_DstColor,
  BlendFactor_OneMinusDstColor,
  BlendFactor_DstAlpha,
  BlendFactor_OneMinusDstAlpha,
  BlendFactor_SrcAlphaSaturated,
  BlendFactor_BlendColor,
  BlendFactor_OneMinusBlendColor,
  BlendFactor_BlendAlpha,
  BlendFactor_OneMinusBlendAlpha,
  BlendFactor_Src1Color,
  BlendFactor_OneMinusSrc1Color,
  BlendFactor_Src1Alpha,
  BlendFactor_OneMinusSrc1Alpha
};

struct SamplerStateDesc {
  SamplerFilter minFilter = SamplerFilter_Linear;
  SamplerFilter magFilter = SamplerFilter_Linear;
  SamplerMip mipMap = SamplerMip_Disabled;
  SamplerWrap wrapU = SamplerWrap_Repeat;
  SamplerWrap wrapV = SamplerWrap_Repeat;
  SamplerWrap wrapW = SamplerWrap_Repeat;
  CompareOp depthCompareOp = CompareOp_LessEqual;
  uint8_t mipLodMin = 0;
  uint8_t mipLodMax = 15;
  uint8_t maxAnisotropic = 1;
  bool depthCompareEnabled = false;
  const char* debugName = "";
};

struct StencilStateDesc {
  StencilOp stencilFailureOp = StencilOp_Keep;
  StencilOp depthFailureOp = StencilOp_Keep;
  StencilOp depthStencilPassOp = StencilOp_Keep;
  CompareOp stencilCompareOp = CompareOp_AlwaysPass;
  uint32_t readMask = (uint32_t)~0;
  uint32_t writeMask = (uint32_t)~0;
};

struct DepthStencilState {
  CompareOp compareOp = CompareOp_AlwaysPass;
  bool isDepthWriteEnabled = false;
  StencilStateDesc backFaceStencil;
  StencilStateDesc frontFaceStencil;
};

enum PolygonMode : uint8_t {
  PolygonMode_Fill = 0,
  PolygonMode_Line = 1,
};

enum class VertexFormat {
  Invalid = 0,

  Float1,
  Float2,
  Float3,
  Float4,

  Byte1,
  Byte2,
  Byte3,
  Byte4,

  UByte1,
  UByte2,
  UByte3,
  UByte4,

  Short1,
  Short2,
  Short3,
  Short4,

  UShort1,
  UShort2,
  UShort3,
  UShort4,

  Byte2Norm,
  Byte4Norm,

  UByte2Norm,
  UByte4Norm,

  Short2Norm,
  Short4Norm,

  UShort2Norm,
  UShort4Norm,

  Int1,
  Int2,
  Int3,
  Int4,

  UInt1,
  UInt2,
  UInt3,
  UInt4,

  HalfFloat1,
  HalfFloat2,
  HalfFloat3,
  HalfFloat4,

  Int_2_10_10_10_REV,
};

enum TextureFormat : uint8_t {
  Invalid = 0,

  R_UN8,
  R_UI16,
  R_UN16,
  R_F16,
  R_F32,

  RG_UN8,
  RG_UI16,
  RG_UN16,
  RG_F16,
  RG_F32,

  RGBA_UN8,
  RGBA_UI32,
  RGBA_F16,
  RGBA_F32,
  RGBA_SRGB8,

  BGRA_UN8,
  BGRA_SRGB8,

  ETC2_RGB8,
  ETC2_SRGB8,
  BC7_RGBA,

  Z_UN16,
  Z_UN24,
  Z_F32,
  Z_UN24_S_UI8,
};

enum LoadOp : uint8_t {
  LoadOp_Invalid = 0,
  LoadOp_DontCare,
  LoadOp_Load,
  LoadOp_Clear,
  LoadOp_None,
};

enum StoreOp : uint8_t {
  StoreOp_DontCare,
  StoreOp_Store,
  StoreOp_MsaaResolve,
  StoreOp_None,
};

enum class CommandQueueType {
  Compute,
  Graphics,
  Transfer,
};

enum ShaderStage : uint8_t {
  Stage_Vertex = 0,
  Stage_Fragment,
  Stage_Compute,
  kNumShaderStages,
};

struct VertexInput final {
  enum { IGL_VERTEX_ATTRIBUTES_MAX = 16 };
  enum { IGL_VERTEX_BUFFER_MAX = 16 };
  struct VertexAttribute final {
    uint32_t location = 0; // a buffer which contains this attribute stream
    uint32_t binding = 0;
    VertexFormat format = VertexFormat::Invalid; // per-element format
    uintptr_t offset = 0; // an offset where the first element of this attribute stream starts
  } attributes[IGL_VERTEX_ATTRIBUTES_MAX];
  struct VertexInputBinding final {
    uint32_t stride = 0;
  } inputBindings[IGL_VERTEX_BUFFER_MAX];

  uint32_t getNumAttributes() const {
    uint32_t n = 0;
    while (n < IGL_VERTEX_ATTRIBUTES_MAX && attributes[n].format != VertexFormat::Invalid) {
      n++;
    }
    return n;
  }
  uint32_t getNumInputBindings() const {
    uint32_t n = 0;
    while (n < IGL_VERTEX_BUFFER_MAX && inputBindings[n].stride) {
      n++;
    }
    return n;
  }
};

struct ColorAttachment {
  TextureFormat textureFormat = TextureFormat::Invalid;
  bool blendEnabled = false;
  BlendOp rgbBlendOp = BlendOp::BlendOp_Add;
  BlendOp alphaBlendOp = BlendOp::BlendOp_Add;
  BlendFactor srcRGBBlendFactor = BlendFactor_One;
  BlendFactor srcAlphaBlendFactor = BlendFactor_One;
  BlendFactor dstRGBBlendFactor = BlendFactor_Zero;
  BlendFactor dstAlphaBlendFactor = BlendFactor_Zero;
};

using ShaderModuleHandle = uint32_t;

struct ShaderModuleDesc {
  ShaderStage stage = Stage_Fragment;
  const char* data = nullptr;
  size_t dataSize = 0; // if `dataSize` is non-zero, interpret `data` as binary shader data
  const char* entryPoint = "main";
  const char* debugName = "";

  ShaderModuleDesc(const char* source, igl::ShaderStage stage, const char* debugName) :
    stage(stage), data(source), debugName(debugName) {}
  ShaderModuleDesc(const void* data,
                   size_t dataLength,
                   igl::ShaderStage stage,
                   const char* debugName) :
    stage(stage), data(static_cast<const char*>(data)), dataSize(dataLength), debugName(debugName) {
    IGL_ASSERT(dataSize);
  }
};

struct ShaderStages final {
  ShaderStages() = default;
  ShaderStages(ShaderModuleHandle vertexModule, ShaderModuleHandle fragmentModule) {
    modules_[Stage_Vertex] = vertexModule;
    modules_[Stage_Fragment] = fragmentModule;
  }

  explicit ShaderStages(ShaderModuleHandle computeModule) {
    modules_[Stage_Compute] = computeModule;
  }

  ShaderModuleHandle getModule(ShaderStage stage) const {
    IGL_ASSERT(stage < kNumShaderStages);
    return modules_[stage];
  }

  ShaderModuleHandle modules_[kNumShaderStages] = {};
};

struct RenderPipelineDesc final {
  igl::VertexInput vertexInput;
  igl::ShaderStages shaderStages;

  ColorAttachment colorAttachments[IGL_COLOR_ATTACHMENTS_MAX] = {};
  TextureFormat depthAttachmentFormat = TextureFormat::Invalid;
  TextureFormat stencilAttachmentFormat = TextureFormat::Invalid;

  CullMode cullMode = igl::CullMode_None;
  WindingMode frontFaceWinding = igl::WindingMode_CCW;
  PolygonMode polygonMode = igl::PolygonMode_Fill;

  uint32_t samplesCount = 1u;

  const char* debugName = "";

  uint32_t getNumColorAttachments() const {
    uint32_t n = 0;
    while (n < IGL_COLOR_ATTACHMENTS_MAX &&
           colorAttachments[n].textureFormat != TextureFormat::Invalid) {
      n++;
    }
    return n;
  }
};

struct ComputePipelineDesc final {
  ShaderModuleHandle computeShaderModule;
  const char* debugName = "";
};

struct AttachmentDesc final {
  LoadOp loadOp = LoadOp_Invalid;
  StoreOp storeOp = StoreOp_Store;
  uint8_t layer = 0;
  uint8_t level = 0;
  Color clearColor = {0.0f, 0.0f, 0.0f, 0.0f};
  float clearDepth = 1.0f;
  uint32_t clearStencil = 0;
};

struct RenderPass final {
  AttachmentDesc colorAttachments[IGL_COLOR_ATTACHMENTS_MAX] = {};
  AttachmentDesc depthAttachment = {.loadOp = LoadOp_DontCare, .storeOp = StoreOp_DontCare};
  AttachmentDesc stencilAttachment = {.loadOp = LoadOp_Invalid, .storeOp = StoreOp_DontCare};

  uint32_t getNumColorAttachments() const {
    uint32_t n = 0;
    while (n < IGL_COLOR_ATTACHMENTS_MAX && colorAttachments[n].loadOp != LoadOp_Invalid) {
      n++;
    }
    return n;
  }
};

struct Framebuffer final {
  struct AttachmentDesc {
    std::shared_ptr<ITexture> texture;
    std::shared_ptr<ITexture> resolveTexture;
  };

  AttachmentDesc colorAttachments[IGL_COLOR_ATTACHMENTS_MAX] = {};
  AttachmentDesc depthStencilAttachment;

  const char* debugName = "";

  uint32_t getNumColorAttachments() const {
    uint32_t n = 0;
    while (n < IGL_COLOR_ATTACHMENTS_MAX && colorAttachments[n].texture) {
      n++;
    }
    return n;
  }
};

enum BufferUsageBits : uint8_t {
  BufferUsageBits_Index = 1 << 0,
  BufferUsageBits_Vertex = 1 << 1,
  BufferUsageBits_Uniform = 1 << 2,
  BufferUsageBits_Storage = 1 << 3,
  BufferUsageBits_Indirect = 1 << 4,
};

struct BufferDesc final {
  uint8_t usage = 0;
  StorageType storage = StorageType_HostVisible;
  const void* data = nullptr;
  size_t size = 0;
  const char* debugName = "";
};

class IBuffer {
 public:
  virtual ~IBuffer() = default;

  virtual Result upload(const void* data, size_t size, size_t offset = 0) = 0;

  virtual uint8_t* getMappedPtr() const = 0;
  virtual uint64_t gpuAddress(size_t offset = 0) const = 0;

 protected:
  IBuffer() = default;
};

struct TextureRangeDesc {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
  uint32_t width = 1;
  uint32_t height = 1;
  uint32_t depth = 1;
  uint32_t layer = 0;
  uint32_t numLayers = 1;
  uint32_t mipLevel = 0;
  uint32_t numMipLevels = 1;
};

enum TextureUsageBits : uint8_t {
  TextureUsageBits_Sampled = 1 << 0,
  TextureUsageBits_Storage = 1 << 1,
  TextureUsageBits_Attachment = 1 << 2,
};

struct TextureDesc {
  TextureType type = TextureType::TwoD;
  TextureFormat format = TextureFormat::Invalid;

  uint32_t width = 1;
  uint32_t height = 1;
  uint32_t depth = 1;
  uint32_t numLayers = 1;
  uint32_t numSamples = 1;
  uint8_t usage = TextureUsageBits_Sampled;
  uint32_t numMipLevels = 1;
  StorageType storage = StorageType_Device;
  const char* debugName = "";
};

class ISamplerState {
 protected:
  ISamplerState() = default;

 public:
  virtual ~ISamplerState() = default;
  virtual uint32_t getSamplerId() const = 0;
};

class ITexture {
 public:
  explicit ITexture(TextureFormat format) : format_(format) {}
  virtual ~ITexture() = default;

  // data[] contains per-layer mip-stacks
  virtual Result upload(const TextureRangeDesc& range, const void* data[]) const = 0;
  virtual Dimensions getDimensions() const = 0;
  virtual void generateMipmap() const = 0;
  virtual uint32_t getTextureId() const = 0;

  TextureFormat getFormat() const {
    return format_;
  };

 private:
  TextureFormat format_;
};

class IComputePipelineState {
 public:
  virtual ~IComputePipelineState() = default;
};

class IRenderPipelineState {
 public:
  IRenderPipelineState() = default;
  virtual ~IRenderPipelineState() = default;
};

struct Dependencies {
  enum { IGL_MAX_SUBMIT_DEPENDENCIES = 4 };
  ITexture* textures[IGL_MAX_SUBMIT_DEPENDENCIES] = {};
};

class ICommandBuffer {
 public:
  virtual ~ICommandBuffer() = default;

  virtual void transitionToShaderReadOnly(const std::shared_ptr<ITexture>& surface) const = 0;
  virtual void waitUntilCompleted() = 0;

  virtual void cmdPushDebugGroupLabel(const char* label,
                                      const igl::Color& color = igl::Color(1, 1, 1, 1)) const = 0;
  virtual void cmdInsertDebugEventLabel(const char* label,
                                        const igl::Color& color = igl::Color(1, 1, 1, 1)) const = 0;
  virtual void cmdPopDebugGroupLabel() const = 0;

  virtual void cmdBindComputePipelineState(
      const std::shared_ptr<IComputePipelineState>& pipelineState) = 0;
  virtual void cmdDispatchThreadGroups(const Dimensions& threadgroupCount,
                                       const Dependencies& deps = Dependencies()) = 0;

  virtual void cmdBeginRendering(const igl::RenderPass& renderPass,
                                 const igl::Framebuffer& desc) = 0;
  virtual void cmdEndRendering() = 0;

  virtual void cmdBindViewport(const Viewport& viewport) = 0;
  virtual void cmdBindScissorRect(const ScissorRect& rect) = 0;

  virtual void cmdBindRenderPipelineState(
      const std::shared_ptr<IRenderPipelineState>& pipelineState) = 0;
  virtual void cmdBindDepthStencilState(const DepthStencilState& state) = 0;

  virtual void cmdBindVertexBuffer(uint32_t index,
                                   const std::shared_ptr<IBuffer>& buffer,
                                   size_t bufferOffset) = 0;
  virtual void cmdPushConstants(size_t offset, const void* data, size_t length) = 0;

  virtual void cmdDraw(PrimitiveType primitiveType, size_t vertexStart, size_t vertexCount) = 0;
  virtual void cmdDrawIndexed(PrimitiveType primitiveType,
                              size_t indexCount,
                              IndexFormat indexFormat,
                              IBuffer& indexBuffer,
                              size_t indexBufferOffset) = 0;
  virtual void cmdDrawIndirect(PrimitiveType primitiveType,
                               IBuffer& indirectBuffer,
                               size_t indirectBufferOffset,
                               uint32_t drawCount,
                               uint32_t stride = 0) = 0;
  virtual void cmdDrawIndexedIndirect(PrimitiveType primitiveType,
                                      IndexFormat indexFormat,
                                      IBuffer& indexBuffer,
                                      IBuffer& indirectBuffer,
                                      size_t indirectBufferOffset,
                                      uint32_t drawCount,
                                      uint32_t stride = 0) = 0;

  virtual void cmdSetStencilReferenceValues(uint32_t frontValue, uint32_t backValue) = 0;
  virtual void cmdSetBlendColor(Color color) = 0;
  virtual void cmdSetDepthBias(float depthBias, float slopeScale, float clamp) = 0;
};

class IDevice {
 public:
  virtual ~IDevice() = default;

  virtual std::shared_ptr<ICommandBuffer> createCommandBuffer() = 0;

  virtual void submit(const ICommandBuffer& commandBuffer,
                      igl::CommandQueueType queueType,
                      ITexture* present = nullptr) = 0;

  virtual std::unique_ptr<IBuffer> createBuffer(const BufferDesc& desc,
                                                Result* outResult = nullptr) = 0;

  virtual std::shared_ptr<ISamplerState> createSamplerState(const SamplerStateDesc& desc,
                                                            Result* outResult = nullptr) = 0;

  virtual std::shared_ptr<ITexture> createTexture(const TextureDesc& desc,
                                                  const char* debugName = nullptr,
                                                  Result* outResult = nullptr) = 0;

  virtual std::shared_ptr<IComputePipelineState> createComputePipeline(
      const ComputePipelineDesc& desc,
      Result* outResult = nullptr) = 0;

  virtual std::shared_ptr<IRenderPipelineState> createRenderPipeline(
      const RenderPipelineDesc& desc,
      Result* outResult = nullptr) = 0;

  virtual ShaderModuleHandle createShaderModule(const ShaderModuleDesc& desc,
                                                Result* outResult = nullptr) = 0;

  virtual std::shared_ptr<ITexture> getCurrentSwapchainTexture() = 0;

  ShaderStages createShaderStages(const char* vs,
                                  const char* debugNameVS,
                                  const char* fs,
                                  const char* debugNameFS,
                                  Result* outResult = nullptr) {
    auto VS = createShaderModule(igl::ShaderModuleDesc(vs, Stage_Vertex, debugNameVS), outResult);
    auto FS = createShaderModule(igl::ShaderModuleDesc(fs, Stage_Fragment, debugNameFS), outResult);
    return igl::ShaderStages(VS, FS);
  }

 protected:
  IDevice() = default;
};

bool isDepthOrStencilFormat(TextureFormat format);
uint32_t calcNumMipLevels(uint32_t width, uint32_t height);
uint32_t getTextureBytesPerLayer(uint32_t texWidth,
                                 uint32_t texHeight,
                                 uint32_t texDepth,
                                 TextureFormat texFormat,
                                 uint32_t mipLevel);

} // namespace igl
