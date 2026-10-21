/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Packs a SPIR-V graph module written as assembly (SPV_ARM_graph, the TOSA extended instruction set) and the tensors it
 * is bound to into a VGF file (https://github.com/arm/ai-ml-sdk-vgf-library) that `lvk::VgfModel` can load.
 *
 * usage: graph_vgf <graph.spvasm> <resources.bin> <output.vgf>
 *
 * The graph declares its inputs and its outputs as UniformConstant tensor variables at descriptor set 0, and its
 * `OpGraphConstantARM` ids are the indices of the constants below. The resources file describes them:
 *   uint32 magic 'VGFR', uint32 numInputs, uint32 numOutputs, uint32 numConstants
 *   per input, then per output: uint32 binding, uint32 VkFormat, uint32 rank, int64 dimensions[rank]
 *   per constant: uint32 VkFormat, uint32 rank, int64 dimensions[rank], uint64 numBytes, then the bytes
 */

#include <spirv-tools/libspirv.h>
#include <vgf/encoder.hpp>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kMagic = 0x52464756;

struct Resource {
  uint32_t binding = 0;
  uint32_t format = 0;
  std::vector<int64_t> dimensions;
  const char* data = nullptr;
  uint64_t numBytes = 0;
};

std::vector<char> readFile(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::fprintf(stderr, "cannot open %s\n", path);
    return {};
  }
  const std::streamsize size = file.tellg();
  file.seekg(0);
  std::vector<char> bytes((size_t)size);
  if (!file.read(bytes.data(), size)) {
    std::fprintf(stderr, "cannot read %s\n", path);
    return {};
  }
  return bytes;
}

bool buildSpirv(const std::vector<char>& text, std::vector<uint32_t>& spirv) {
  const spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_6);
  spv_binary binary = nullptr;
  spv_diagnostic diagnostic = nullptr;
  const spv_result_t result = spvTextToBinary(context, text.data(), text.size(), &binary, &diagnostic);
  const bool ok = result == SPV_SUCCESS && binary;

  if (ok) {
    spirv.assign(binary->code, binary->code + binary->wordCount);
  } else {
    std::fprintf(stderr,
                 "cannot assemble the graph module at line %zu: %s\n",
                 diagnostic ? diagnostic->position.line + 1 : 0,
                 diagnostic && diagnostic->error ? diagnostic->error : "unknown");
  }

  spvDiagnosticDestroy(diagnostic);
  spvBinaryDestroy(binary);
  spvContextDestroy(context);

  return ok;
}

class Reader final {
 public:
  explicit Reader(const std::vector<char>& blob) : data_(blob.data()), end_(blob.data() + blob.size()) {}

  template<typename T>
  bool read(T& value) {
    if (sizeof(T) > (size_t)(end_ - data_)) {
      return false;
    }
    memcpy(&value, data_, sizeof(T));
    data_ += sizeof(T);
    return true;
  }
  const char* take(uint64_t numBytes) {
    if (numBytes > (uint64_t)(end_ - data_)) {
      return nullptr;
    }
    const char* start = data_;
    data_ += numBytes;
    return start;
  }

 private:
  const char* data_;
  const char* end_;
};

bool readShape(Reader& reader, Resource& resource) {
  uint32_t rank = 0;
  if (!reader.read(resource.format) || !reader.read(rank) || !rank || rank > 8) {
    return false;
  }
  resource.dimensions.resize(rank);
  for (int64_t& d : resource.dimensions) {
    if (!reader.read(d)) {
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <graph.spvasm> <resources.bin> <output.vgf>\n", argv[0]);
    return 1;
  }

  const std::vector<char> text = readFile(argv[1]);
  const std::vector<char> blob = readFile(argv[2]);

  if (text.empty() || blob.empty()) {
    return 1;
  }

  std::vector<uint32_t> spirv;

  if (!buildSpirv(text, spirv)) {
    return 1;
  }

  Reader reader(blob);

  uint32_t header[4] = {};

  for (uint32_t& h : header) {
    if (!reader.read(h)) {
      std::fprintf(stderr, "%s is truncated\n", argv[2]);
      return 1;
    }
  }

  if (header[0] != kMagic || !header[1] || !header[2]) {
    std::fprintf(stderr, "%s is not a resources file\n", argv[2]);
    return 1;
  }

  std::vector<Resource> inputs(header[1]);
  std::vector<Resource> outputs(header[2]);
  std::vector<Resource> constants(header[3]);

  for (std::vector<Resource>* group : {&inputs, &outputs}) {
    for (Resource& r : *group) {
      if (!reader.read(r.binding) || !readShape(reader, r)) {
        std::fprintf(stderr, "%s is truncated\n", argv[2]);
        return 1;
      }
    }
  }

  for (Resource& r : constants) {
    if (!readShape(reader, r) || !reader.read(r.numBytes) || !(r.data = reader.take(r.numBytes))) {
      std::fprintf(stderr, "%s is truncated\n", argv[2]);
      return 1;
    }
  }

  using namespace mlsdk::vgflib;

  constexpr DescriptorType kDescriptorTensor = (DescriptorType)VK_DESCRIPTOR_TYPE_TENSOR_ARM;

  const std::unique_ptr<Encoder> encoder = CreateEncoder(VK_HEADER_VERSION);

  std::vector<BindingSlotRef> inputSlots;
  std::vector<BindingSlotRef> outputSlots;
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;

  for (const Resource& r : inputs) {
    const ResourceRef ref = encoder->AddInputResource(kDescriptorTensor, (FormatType)r.format, r.dimensions, {});
    inputSlots.push_back(encoder->AddBindingSlot(r.binding, ref));
    inputNames.push_back("input" + std::to_string(inputNames.size()));
  }
  for (const Resource& r : outputs) {
    const ResourceRef ref = encoder->AddOutputResource(kDescriptorTensor, (FormatType)r.format, r.dimensions, {});
    outputSlots.push_back(encoder->AddBindingSlot(r.binding, ref));
    outputNames.push_back("output" + std::to_string(outputNames.size()));
  }

  std::vector<ConstantRef> constantRefs;
  constantRefs.reserve(constants.size());

  for (const Resource& r : constants) {
    const ResourceRef ref = encoder->AddConstantResource((FormatType)r.format, r.dimensions, {});
    constantRefs.push_back(encoder->AddConstant(ref, r.data, (size_t)r.numBytes));
  }

  std::vector<BindingSlotRef> slots;
  slots.reserve(inputSlots.size() + outputSlots.size());
  for (const std::vector<BindingSlotRef>* group : {&inputSlots, &outputSlots}) {
    for (const BindingSlotRef& slot : *group) {
      slots.push_back(slot);
    }
  }

  const ModuleRef module = encoder->AddModule(ModuleType::GRAPH, "graph", "main", spirv);
  const DescriptorSetInfoRef descriptorSet = encoder->AddDescriptorSetInfo(slots);

  encoder->AddSegmentInfo(module, "graph", {descriptorSet}, inputSlots, outputSlots, constantRefs);
  encoder->AddModelSequenceInputsOutputs(inputSlots, inputNames, outputSlots, outputNames);
  encoder->Finish();

  std::ofstream output(argv[3], std::ios::binary);

  if (!output || !encoder->WriteTo(output)) {
    std::fprintf(stderr, "cannot write %s\n", argv[3]);
    return 1;
  }

  std::printf("%s: %zu inputs, %zu outputs, %zu constants, %zu SPIR-V words\n",
              argv[3],
              inputs.size(),
              outputs.size(),
              constants.size(),
              spirv.size());

  return 0;
}
