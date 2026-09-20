/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "SpirvGraphReflection.h"

#include <cstring>
#include <string>
#include <vector>

namespace lvk {

namespace {

constexpr uint32_t kSpvMagicNumber = 0x07230203;
constexpr uint32_t kSpvMaxIdBound = 0x3fffff;
constexpr uint32_t kOpTypeBool = 20;
constexpr uint32_t kOpTypeInt = 21;
constexpr uint32_t kOpTypeFloat = 22;
constexpr uint32_t kOpTypePointer = 32;
constexpr uint32_t kOpConstant = 43;
constexpr uint32_t kOpConstantComposite = 44;
constexpr uint32_t kOpVariable = 59;
constexpr uint32_t kOpDecorate = 71;
constexpr uint32_t kOpTypeTensorARM = 4163;
constexpr uint32_t kOpGraphEntryPointARM = 4182;
constexpr uint32_t kOpGraphARM = 4183;
constexpr uint32_t kOpTypeGraphARM = 4190;
constexpr uint32_t kDecorationBinding = 33;
constexpr uint32_t kDecorationDescriptorSet = 34;

struct TypeInfo {
  uint32_t op = 0;
  uint32_t width = 0;
  uint32_t elementType = 0;
  uint32_t rankId = 0;
  uint32_t shapeId = 0;
  uint32_t pointee = 0;
  uint32_t numInputs = 0;
};

struct IdInfo {
  TypeInfo type;
  const uint32_t* composite = nullptr;
  uint32_t numComposite = 0;
  uint32_t constant = 0;
  uint32_t variableType = 0;
  uint32_t graphType = 0;
  uint32_t binding = 0;
  uint32_t descriptorSet = 0;
  bool hasConstant = false;
  bool hasBinding = false;
};

struct EntryPoint {
  std::string name;
  uint32_t graph = 0;
  std::vector<uint32_t> interfaceVars;
};

} // namespace

bool SpirvGraphReflection::reflect(const void* spirv, size_t size, const char* entryPoint) {
  inputs_.clear();
  outputs_.clear();
  error_.clear();

  const uint32_t* words = static_cast<const uint32_t*>(spirv);
  const size_t numWords = size / sizeof(uint32_t);

  if (!words || numWords < 5 || words[0] != kSpvMagicNumber) {
    error_ = "Not a SPIR-V module";
    return false;
  }
  if (words[3] > kSpvMaxIdBound) {
    error_ = "Malformed SPIR-V module";
    return false;
  }

  std::vector<IdInfo> ids(words[3]);
  std::vector<EntryPoint> entryPoints;

  const auto get = [&ids](uint32_t id) -> IdInfo* { return id < ids.size() ? &ids[id] : nullptr; };

  for (size_t i = 5; i < numWords;) {
    const uint32_t wordCount = words[i] >> 16;
    const uint32_t op = words[i] & 0xffff;
    if (!wordCount || i + wordCount > numWords) {
      error_ = "Malformed SPIR-V module";
      return false;
    }
    const uint32_t* operands = words + i + 1;
    const uint32_t numOperands = wordCount - 1;
    switch (op) {
    case kOpDecorate:
      if (IdInfo* target = numOperands >= 3 ? get(operands[0]) : nullptr) {
        if (operands[1] == kDecorationBinding) {
          target->binding = operands[2];
          target->hasBinding = true;
        } else if (operands[1] == kDecorationDescriptorSet) {
          target->descriptorSet = operands[2];
        }
      }
      break;
    case kOpTypeInt:
    case kOpTypeFloat:
      if (IdInfo* result = numOperands >= 2 ? get(operands[0]) : nullptr) {
        result->type = {.op = op, .width = operands[1]};
      }
      break;
    case kOpTypeBool:
      if (IdInfo* result = numOperands >= 1 ? get(operands[0]) : nullptr) {
        result->type = {.op = op, .width = 8};
      }
      break;
    case kOpTypePointer:
      if (IdInfo* result = numOperands >= 3 ? get(operands[0]) : nullptr) {
        result->type = {.op = op, .pointee = operands[2]};
      }
      break;
    case kOpTypeTensorARM:
      if (IdInfo* result = numOperands >= 2 ? get(operands[0]) : nullptr) {
        result->type = {
            .op = op,
            .elementType = operands[1],
            .rankId = numOperands >= 3 ? operands[2] : 0u,
            .shapeId = numOperands >= 4 ? operands[3] : 0u,
        };
      }
      break;
    case kOpTypeGraphARM:
      if (IdInfo* result = numOperands >= 2 ? get(operands[0]) : nullptr) {
        result->type = {.op = op, .numInputs = operands[1]};
      }
      break;
    case kOpConstant:
      if (IdInfo* result = numOperands >= 3 ? get(operands[1]) : nullptr) {
        result->constant = operands[2];
        result->hasConstant = true;
      }
      break;
    case kOpConstantComposite:
      if (IdInfo* result = numOperands >= 2 ? get(operands[1]) : nullptr) {
        result->composite = operands + 2;
        result->numComposite = numOperands - 2;
      }
      break;
    case kOpVariable:
      if (IdInfo* result = numOperands >= 3 ? get(operands[1]) : nullptr) {
        result->variableType = operands[0];
      }
      break;
    case kOpGraphARM:
      if (IdInfo* result = numOperands >= 2 ? get(operands[1]) : nullptr) {
        result->graphType = operands[0];
      }
      break;
    case kOpGraphEntryPointARM: {
      if (numOperands < 2) {
        break;
      }
      const char* name = reinterpret_cast<const char*>(operands + 1);
      const size_t nameLen = strnlen(name, (numOperands - 1) * sizeof(uint32_t));
      const uint32_t nameWords = (uint32_t)((nameLen + 1 + 3) / 4);
      EntryPoint ep = {.name = std::string(name, nameLen), .graph = operands[0]};
      for (uint32_t k = 1 + nameWords; k < numOperands; k++) {
        ep.interfaceVars.push_back(operands[k]);
      }
      entryPoints.push_back(std::move(ep));
      break;
    }
    default:
      break;
    }
    i += wordCount;
  }

  const EntryPoint* ep = nullptr;
  for (const EntryPoint& candidate : entryPoints) {
    if (!entryPoint || !*entryPoint || candidate.name == entryPoint) {
      ep = &candidate;
      break;
    }
  }
  if (!ep) {
    error_ = "Graph entry point not found in the module: ";
    error_ += entryPoint ? entryPoint : "(null)";
    for (const EntryPoint& candidate : entryPoints) {
      error_ += " (available: " + candidate.name + ")";
    }
    return false;
  }
  const IdInfo* graph = get(ep->graph);
  const IdInfo* graphType = graph ? get(graph->graphType) : nullptr;
  if (!graphType || graphType->type.op != kOpTypeGraphARM) {
    error_ = "Graph entry point without an OpTypeGraphARM type";
    return false;
  }
  const uint32_t numInputs = graphType->type.numInputs;
  if (numInputs > ep->interfaceVars.size()) {
    error_ = "Graph entry point declares fewer interface variables than graph inputs";
    return false;
  }
  for (size_t v = 0; v != ep->interfaceVars.size(); v++) {
    const IdInfo* var = get(ep->interfaceVars[v]);
    const IdInfo* pointerType = var ? get(var->variableType) : nullptr;
    if (!pointerType || pointerType->type.op != kOpTypePointer) {
      error_ = "Graph interface variable is not an OpVariable of pointer type";
      return false;
    }
    const IdInfo* tensorType = get(pointerType->type.pointee);
    if (!tensorType || tensorType->type.op != kOpTypeTensorARM) {
      error_ = "Graph interface variable does not point to an OpTypeTensorARM";
      return false;
    }
    const IdInfo* elementType = get(tensorType->type.elementType);
    if (!elementType || !elementType->type.op) {
      error_ = "Graph tensor with an unknown element type";
      return false;
    }
    if (!var->hasBinding) {
      error_ = "Graph tensor without a Binding decoration";
      return false;
    }
    const IdInfo* rank = get(tensorType->type.rankId);
    Resource res = {
        .descriptorSet = var->descriptorSet,
        .binding = var->binding,
        .isFloat = elementType->type.op == kOpTypeFloat,
        .elementBits = elementType->type.width,
        .rank = rank && rank->hasConstant ? rank->constant : 0u,
    };
    if (res.rank > LVK_TENSOR_MAX_RANK) {
      error_ = "Graph tensor rank is too large";
      return false;
    }
    const IdInfo* shape = get(tensorType->type.shapeId);
    if (shape && shape->composite) {
      if (shape->numComposite != res.rank) {
        error_ = "Graph tensor shape does not match its rank";
        return false;
      }
      res.shaped = true;
      for (uint32_t d = 0; d != res.rank; d++) {
        const IdInfo* dim = get(shape->composite[d]);
        res.dimensions[d] = dim && dim->hasConstant ? (int64_t)dim->constant : -1;
      }
    }
    (v < numInputs ? inputs_ : outputs_).push_back(res);
  }
  return true;
}

const char* SpirvGraphReflection::getError() const {
  return error_.c_str();
}

uint32_t SpirvGraphReflection::getNumInputs() const {
  return (uint32_t)inputs_.size();
}

uint32_t SpirvGraphReflection::getNumOutputs() const {
  return (uint32_t)outputs_.size();
}

const SpirvGraphReflection::Resource& SpirvGraphReflection::getInput(uint32_t index) const {
  LVK_ASSERT(index < inputs_.size());
  return inputs_[index];
}

const SpirvGraphReflection::Resource& SpirvGraphReflection::getOutput(uint32_t index) const {
  LVK_ASSERT(index < outputs_.size());
  return outputs_[index];
}

} // namespace lvk
