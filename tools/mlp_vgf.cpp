/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Packs a multilayer perceptron quantized to 8 bits into a VGF file (https://github.com/arm/ai-ml-sdk-vgf-library) so
 * that `lvk::VgfModel` can load it: a SPIR-V graph module (SPV_ARM_graph, the TOSA extended instruction set), the
 * weights as graph constants, and the interface of the graph.
 *
 * usage: mlp_vgf <weights.bin> <output.vgf> [<width>x<height>]
 *
 * The graph is TOSA's integer scheme over NHWC tensors [1, height, width, channels]: every layer is an int8 1x1 CONV2D
 * into an int32 sum (a fully connected layer applied to every texel independently), then a RESCALE by
 * `multiplier * 2^-shift` per output channel (`scale32`, single rounding) saturating to int8 around the layer's output
 * zero point, then a 256-entry TABLE for a sigmoid. A ReLU needs no operator: such a layer has a zero point of -128,
 * so anything negative saturates at zero.
 *
 * Without a resolution the model comes out unspecialized, the way Arm publishes its own networks: no tensor carries a
 * shape. Such a model cannot be run as it stands, `VK_ARM_data_graph` requires a shape on every tensor type of a graph
 * (VUID-RuntimeSpirv-pNext-09919), so `tools/make_dfaoit_model.py` shapes it for a resolution first.
 *
 * The weights file is what `tools/train_dfaoit.py` writes:
 *   uint32 magic 'MLPQ', uint32 numInputs, uint32 numLayers, int32 inputZeroPoint
 *   per layer: uint32 numOutputs, uint32 activation (1 ReLU, 2 sigmoid), int32 outputZeroPoint
 *   then, per layer, numOutputs * numInputs int8 weights, numOutputs int32 biases, numOutputs int32 multipliers,
 *   numOutputs int8 shifts, and 256 int8 table entries for a sigmoid
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
#include <unordered_map>
#include <vector>

namespace {

constexpr uint32_t kMagic = 0x51504C4D;
constexpr uint32_t kTableSize = 256;

enum Activation : uint32_t {
  Activation_ReLU = 1,
  Activation_Sigmoid = 2,
};

struct Layer {
  uint32_t numOutputs = 0;
  uint32_t numInputs = 0;
  uint32_t activation = Activation_ReLU;
  int32_t outputZeroPoint = 0;
  const int8_t* weights = nullptr;
  const int32_t* biases = nullptr;
  const int32_t* multipliers = nullptr;
  const int8_t* shifts = nullptr;
  const int8_t* table = nullptr;
};

struct Network {
  uint32_t numInputs = 0;
  int32_t inputZeroPoint = 0;
  std::vector<Layer> layers;
};

class GraphAssembly final {
 public:
  std::string get(const std::string& definition, const char* prefix) {
    const std::unordered_map<std::string, std::string>::const_iterator it = cache_.find(definition);
    if (it != cache_.end()) {
      return it->second;
    }
    const std::string id = std::string("%") + prefix + std::to_string(counter_++);
    declarations_ += id + " = " + definition + "\n";
    cache_[definition] = id;
    return id;
  }
  std::string typeInt32() {
    return get("OpTypeInt 32 0", "t");
  }
  std::string typeInt8() {
    return get("OpTypeInt 8 0", "t");
  }
  std::string constUint(uint32_t v) {
    return get("OpConstant " + typeInt32() + " " + std::to_string(v), "u");
  }
  std::string constInt32(int32_t v) {
    return constUint((uint32_t)v);
  }
  std::string constInt8(int8_t v) {
    return get("OpConstant " + typeInt8() + " " + std::to_string((uint32_t)(uint8_t)v), "c");
  }
  std::string constBool(bool v) {
    return get(std::string(v ? "OpConstantTrue " : "OpConstantFalse ") + get("OpTypeBool", "t"), "b");
  }
  std::string shape(const std::vector<uint32_t>& dims) {
    const std::string type = get("OpTypeArray " + typeInt32() + " " + constUint((uint32_t)dims.size()), "t");
    std::string definition = "OpConstantComposite " + type;
    for (uint32_t d : dims) {
      definition += " " + constUint(d);
    }
    return get(definition, "s");
  }
  std::string typeTensor(const std::string& element, const std::vector<uint32_t>& dims) {
    return get("OpTypeTensorARM " + element + " " + constUint((uint32_t)dims.size()) + " " + shape(dims), "t");
  }
  std::string typeTensorUnshaped(const std::string& element, uint32_t rank) {
    return get("OpTypeTensorARM " + element + " " + constUint(rank), "t");
  }
  std::string constTensor(const std::string& element, const std::vector<std::string>& values) {
    std::string definition = "OpConstantComposite " + typeTensor(element, {(uint32_t)values.size()});
    for (const std::string& v : values) {
      definition += " " + v;
    }
    return get(definition, "k");
  }
  std::string graphConstant(const std::string& element, const std::vector<uint32_t>& dims, uint32_t id) {
    return get("OpGraphConstantARM " + typeTensor(element, dims) + " " + std::to_string(id), "w");
  }
  std::string op(const std::string& type, const std::string& instruction) {
    const std::string id = "%r" + std::to_string(counter_++);
    body_ += id + " = OpExtInst " + type + " %tosa " + instruction + "\n";
    return id;
  }

  const std::string& declarations() const {
    return declarations_;
  }
  const std::string& body() const {
    return body_;
  }

 private:
  uint32_t counter_ = 0;
  std::string declarations_;
  std::string body_;
  std::unordered_map<std::string, std::string> cache_;
};

std::string assemble(const Network& network, uint32_t width, uint32_t height) {
  const bool specialized = width && height;

  GraphAssembly assembly;

  const std::string typeInt32 = assembly.typeInt32();
  const std::string typeInt8 = assembly.typeInt8();
  const std::string zero = assembly.constUint(0);
  const std::string one = assembly.constUint(1);
  const std::string pad = assembly.constTensor(typeInt32, {zero, zero, zero, zero});
  const std::string stride = assembly.constTensor(typeInt32, {one, one});
  const std::string accumulatorInt32 = assembly.constUint(1);
  const std::string roundingSingle = assembly.constUint(1);
  const std::string zeroPointWeights = assembly.constTensor(typeInt8, {assembly.constInt8(0)});
  const std::string zeroPointSum = assembly.constTensor(typeInt32, {zero});

  const auto typeValue = [&](const std::string& element, uint32_t channels) {
    return specialized ? assembly.typeTensor(element, {1, height, width, channels}) : assembly.typeTensorUnshaped(element, 4);
  };

  const std::string typeInput = typeValue(typeInt8, network.numInputs);

  std::string value = "%input";
  std::string type = typeInput;
  std::string zeroPoint = assembly.constTensor(typeInt8, {assembly.constInt8((int8_t)network.inputZeroPoint)});
  uint32_t id = 0;

  for (const Layer& layer : network.layers) {
    const uint32_t idWeights = id++;
    const uint32_t idBiases = id++;

    value = assembly.op(typeValue(typeInt32, layer.numOutputs),
                        "CONV2D " + pad + " " + stride + " " + stride + " " + accumulatorInt32 + " " + assembly.constBool(false) + " " +
                            value + " " + assembly.graphConstant(typeInt8, {layer.numOutputs, 1, 1, layer.numInputs}, idWeights) + " " +
                            assembly.graphConstant(typeInt32, {layer.numOutputs}, idBiases) + " " + zeroPoint + " " + zeroPointWeights);

    std::vector<std::string> multipliers;
    std::vector<std::string> shifts;
    for (uint32_t i = 0; i != layer.numOutputs; i++) {
      multipliers.push_back(assembly.constInt32(layer.multipliers[i]));
      shifts.push_back(assembly.constInt8(layer.shifts[i]));
    }
    zeroPoint = assembly.constTensor(typeInt8, {assembly.constInt8((int8_t)layer.outputZeroPoint)});
    type = typeValue(typeInt8, layer.numOutputs);
    value = assembly.op(type,
                        "RESCALE " + assembly.constBool(true) + " " + roundingSingle + " " + assembly.constBool(true) + " " +
                            assembly.constBool(false) + " " + assembly.constBool(false) + " " + value + " " +
                            assembly.constTensor(typeInt32, multipliers) + " " + assembly.constTensor(typeInt8, shifts) + " " +
                            zeroPointSum + " " + zeroPoint);

    if (layer.activation == Activation_Sigmoid) {
      std::vector<std::string> table;
      for (uint32_t i = 0; i != kTableSize; i++) {
        table.push_back(assembly.constInt8(layer.table[i]));
      }
      value = assembly.op(type, "TABLE " + value + " " + assembly.constTensor(typeInt8, table));
    }
  }

  const std::string typeOutput = type;
  const std::string pointerInput = assembly.get("OpTypePointer UniformConstant " + typeInput, "p");
  const std::string pointerOutput = assembly.get("OpTypePointer UniformConstant " + typeOutput, "p");

  std::string text =
      "OpCapability Shader\nOpCapability Int8\nOpCapability TensorsARM\nOpCapability GraphARM\nOpCapability VulkanMemoryModel\n"
      "OpExtension \"SPV_ARM_tensors\"\nOpExtension \"SPV_ARM_graph\"\n%tosa = OpExtInstImport \"TOSA.001000.1\"\n"
      "OpMemoryModel Logical Vulkan\n"
      "OpDecorate %varInput DescriptorSet 0\nOpDecorate %varInput Binding 0\n"
      "OpDecorate %varOutput DescriptorSet 0\nOpDecorate %varOutput Binding 1\n";
  text += assembly.declarations();
  text += "%varInput = OpVariable " + pointerInput + " UniformConstant\n";
  text += "%varOutput = OpVariable " + pointerOutput + " UniformConstant\n";
  text += "%typeGraph = OpTypeGraphARM 1 " + typeInput + " " + typeOutput + "\n";
  text += "OpGraphEntryPointARM %graph \"main\" %varInput %varOutput\n";
  text += "%graph = OpGraphARM %typeGraph\n";
  text += "%input = OpGraphInputARM " + typeInput + " " + zero + "\n";
  text += assembly.body();
  text += "OpGraphSetOutputARM " + value + " " + zero + "\nOpGraphEndARM\n";

  return text;
}

bool buildSpirv(const std::string& text, std::vector<uint32_t>& spirv) {
  const spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_6);
  spv_binary binary = nullptr;
  spv_diagnostic diagnostic = nullptr;
  const spv_result_t result = spvTextToBinary(context, text.c_str(), text.size(), &binary, &diagnostic);
  const bool ok = result == SPV_SUCCESS && binary;

  if (ok) {
    spirv.assign(binary->code, binary->code + binary->wordCount);
  } else {
    std::fprintf(stderr, "cannot assemble the graph module: %s\n", diagnostic && diagnostic->error ? diagnostic->error : "unknown");
  }

  spvDiagnosticDestroy(diagnostic);
  spvBinaryDestroy(binary);
  spvContextDestroy(context);

  return ok;
}

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

bool parseNetwork(const std::vector<char>& blob, const char* path, Network& network) {
  if (blob.size() < 4 * sizeof(uint32_t)) {
    std::fprintf(stderr, "%s is truncated\n", path);
    return false;
  }

  const uint32_t* header = (const uint32_t*)blob.data();

  if (header[0] != kMagic) {
    std::fprintf(stderr, "%s is not a weights file\n", path);
    return false;
  }

  network.numInputs = header[1];
  network.inputZeroPoint = (int32_t)header[3];

  const uint32_t numLayers = header[2];
  const uint32_t headerWords = 4 + 3 * numLayers;

  if (!network.numInputs || !numLayers || blob.size() < headerWords * sizeof(uint32_t)) {
    std::fprintf(stderr, "%s is truncated\n", path);
    return false;
  }

  const uint32_t* layerHeader = header + 4;

  network.layers.resize(numLayers);

  const char* data = blob.data() + headerWords * sizeof(uint32_t);
  const char* end = blob.data() + blob.size();
  uint32_t layerInputs = network.numInputs;

  const auto take = [&data, end](size_t bytes) -> const char* {
    const char* start = data;
    if (bytes > (size_t)(end - data)) {
      return nullptr;
    }
    data += bytes;
    return start;
  };

  for (uint32_t i = 0; i != numLayers; i++) {
    Layer& layer = network.layers[i];
    layer.numOutputs = layerHeader[3 * i];
    layer.numInputs = layerInputs;
    layer.activation = layerHeader[3 * i + 1];
    layer.outputZeroPoint = (int32_t)layerHeader[3 * i + 2];
    if (!layer.numOutputs || (layer.activation != Activation_ReLU && layer.activation != Activation_Sigmoid)) {
      std::fprintf(stderr, "layer %u is invalid\n", i);
      return false;
    }
    layer.weights = (const int8_t*)take((size_t)layer.numOutputs * layerInputs);
    layer.biases = (const int32_t*)take(layer.numOutputs * sizeof(int32_t));
    layer.multipliers = (const int32_t*)take(layer.numOutputs * sizeof(int32_t));
    layer.shifts = (const int8_t*)take(layer.numOutputs);
    layer.table = layer.activation == Activation_Sigmoid ? (const int8_t*)take(kTableSize) : nullptr;
    if (!layer.weights || !layer.biases || !layer.multipliers || !layer.shifts ||
        (layer.activation == Activation_Sigmoid && !layer.table)) {
      std::fprintf(stderr, "%s is truncated at layer %u\n", path, i);
      return false;
    }
    layerInputs = layer.numOutputs;
  }

  return true;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::fprintf(stderr, "usage: %s <weights.bin> <output.vgf> [<width>x<height>]\n", argv[0]);
    return 1;
  }

  uint32_t width = 0;
  uint32_t height = 0;

  if (argc == 4 && std::sscanf(argv[3], "%ux%u", &width, &height) != 2) {
    std::fprintf(stderr, "`%s` is not <width>x<height>\n", argv[3]);
    return 1;
  }

  const std::vector<char> blob = readFile(argv[1]);

  Network network;

  if (blob.empty() || !parseNetwork(blob, argv[1], network)) {
    return 1;
  }

  std::vector<uint32_t> spirv;

  if (!buildSpirv(assemble(network, width, height), spirv)) {
    return 1;
  }

  using namespace mlsdk::vgflib;

  constexpr FormatType kFormatInt8 = (FormatType)VK_FORMAT_R8_SINT;
  constexpr FormatType kFormatInt32 = (FormatType)VK_FORMAT_R32_SINT;
  constexpr DescriptorType kDescriptorTensor = (DescriptorType)VK_DESCRIPTOR_TYPE_TENSOR_ARM;
  constexpr int64_t kUnknown = -1;

  const std::unique_ptr<Encoder> encoder = CreateEncoder(VK_HEADER_VERSION);

  const std::vector<int64_t> unknownShape = {kUnknown, kUnknown, kUnknown, kUnknown};
  const std::vector<int64_t> inputShape = width ? std::vector<int64_t>{1, height, width, network.numInputs} : unknownShape;
  const std::vector<int64_t> outputShape = width ? std::vector<int64_t>{1, height, width, network.layers.back().numOutputs} : unknownShape;

  const ResourceRef inputRef = encoder->AddInputResource(kDescriptorTensor, kFormatInt8, inputShape, {});
  const ResourceRef outputRef = encoder->AddOutputResource(kDescriptorTensor, kFormatInt8, outputShape, {});

  std::vector<ConstantRef> constantRefs;
  constantRefs.reserve(2 * network.layers.size());

  for (const Layer& layer : network.layers) {
    const std::vector<int64_t> weightShape = {layer.numOutputs, 1, 1, layer.numInputs};
    const ResourceRef weights = encoder->AddConstantResource(kFormatInt8, weightShape, {});
    constantRefs.push_back(encoder->AddConstant(weights, layer.weights, (size_t)layer.numOutputs * layer.numInputs));
    const ResourceRef biases = encoder->AddConstantResource(kFormatInt32, {layer.numOutputs}, {});
    constantRefs.push_back(encoder->AddConstant(biases, layer.biases, (size_t)layer.numOutputs * sizeof(int32_t)));
  }

  const BindingSlotRef inputSlot = encoder->AddBindingSlot(0, inputRef);
  const BindingSlotRef outputSlot = encoder->AddBindingSlot(1, outputRef);

  const ModuleRef module = encoder->AddModule(ModuleType::GRAPH, "mlp", "main", spirv);
  const DescriptorSetInfoRef descriptorSet = encoder->AddDescriptorSetInfo({inputSlot, outputSlot});

  encoder->AddSegmentInfo(module, "mlp", {descriptorSet}, {inputSlot}, {outputSlot}, constantRefs);
  encoder->AddModelSequenceInputsOutputs({inputSlot}, {"input"}, {outputSlot}, {"output"});
  encoder->Finish();

  std::ofstream output(argv[2], std::ios::binary);

  if (!output || !encoder->WriteTo(output)) {
    std::fprintf(stderr, "cannot write %s\n", argv[2]);
    return 1;
  }

  std::printf("%s: %u", argv[2], network.numInputs);
  for (const Layer& layer : network.layers) {
    std::printf(" -> %u", layer.numOutputs);
  }
  if (width) {
    std::printf(", %ux%u\n", width, height);
  } else {
    std::printf(", unspecialized\n");
  }

  return 0;
}
