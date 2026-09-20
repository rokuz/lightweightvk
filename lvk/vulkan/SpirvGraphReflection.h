/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <lvk/LVK.h>

#include <string>
#include <vector>

namespace lvk {

class SpirvGraphReflection final {
 public:
  struct Resource {
    uint32_t descriptorSet = 0;
    uint32_t binding = 0;
    bool isFloat = false;
    uint32_t elementBits = 0;
    uint32_t rank = 0;
    bool shaped = false;
    int64_t dimensions[LVK_TENSOR_MAX_RANK] = {};
  };

  bool reflect(const void* spirv, size_t size, const char* entryPoint = nullptr);
  const char* getError() const;

  uint32_t getNumInputs() const;
  uint32_t getNumOutputs() const;
  const Resource& getInput(uint32_t index) const;
  const Resource& getOutput(uint32_t index) const;

 private:
  std::vector<Resource> inputs_;
  std::vector<Resource> outputs_;
  std::string error_;
};

} // namespace lvk
