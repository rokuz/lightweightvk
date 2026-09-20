#!/usr/bin/env python3
#
# LightweightVK
#
# Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Teaches SPIRV-Reflect about `OpTypeTensorARM` (SPV_ARM_tensors / VK_ARM_tensors).

Without this, a shader declaring a `tensorARM` descriptor makes spvReflectCreateShaderModule() fail with
SPV_REFLECT_RESULT_ERROR_SPIRV_INVALID_ID_REFERENCE and LVK loses push constants reflection. Applied by `bootstrap.py` as a
post-processing script (see bootstrap-deps.json). Idempotent.
"""

import os
import sys

SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "SPIRV-Reflect"))
HEADER = os.path.join(SRC_DIR, "spirv_reflect.h")
SOURCE = os.path.join(SRC_DIR, "spirv_reflect.c")

MARKER = "SPV_REFLECT_TYPE_FLAG_EXTERNAL_TENSOR_ARM"


def patch(path, replacements):
    with open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    if MARKER in text:
        print(f"{os.path.basename(path)}: already patched")
        return
    crlf = "\r\n" in text
    text = text.replace("\r\n", "\n")
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError(f"{path}: expected exactly one occurrence of:\n{old}")
        text = text.replace(old, new)
    if crlf:
        text = text.replace("\n", "\r\n")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"{os.path.basename(path)}: patched")


HEADER_REPLACEMENTS = [
    (
        "  SPV_REFLECT_TYPE_FLAG_EXTERNAL_ACCELERATION_STRUCTURE = 0x00100000,\n",
        "  SPV_REFLECT_TYPE_FLAG_EXTERNAL_ACCELERATION_STRUCTURE = 0x00100000,\n"
        "  SPV_REFLECT_TYPE_FLAG_EXTERNAL_TENSOR_ARM             = 0x00200000,\n",
    ),
    (
        "  SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR = 1000150000 // = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR\n",
        "  SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR = 1000150000, // = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR\n"
        "  SPV_REFLECT_DESCRIPTOR_TYPE_TENSOR_ARM                 = 1000460000  // = VK_DESCRIPTOR_TYPE_TENSOR_ARM\n",
    ),
]

SOURCE_REPLACEMENTS = [
    (
        "      case SpvOpTypeCooperativeMatrixKHR:\n"
        "      case SpvOpTypeUntypedPointerKHR: {\n"
        "        CHECKED_READU32(p_parser, p_node->word_offset + 1, p_node->result_id);\n"
        "        p_node->is_type = true;\n"
        "      } break;\n",
        "      case SpvOpTypeCooperativeMatrixKHR:\n"
        "      case SpvOpTypeTensorARM:\n"
        "      case SpvOpTypeUntypedPointerKHR: {\n"
        "        CHECKED_READU32(p_parser, p_node->word_offset + 1, p_node->result_id);\n"
        "        p_node->is_type = true;\n"
        "      } break;\n",
    ),
    (
        "      case SpvOpTypeAccelerationStructureKHR: {\n"
        "        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_EXTERNAL_ACCELERATION_STRUCTURE;\n"
        "      } break;\n",
        "      case SpvOpTypeAccelerationStructureKHR: {\n"
        "        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_EXTERNAL_ACCELERATION_STRUCTURE;\n"
        "      } break;\n"
        "\n"
        "      case SpvOpTypeTensorARM: {\n"
        "        p_type->type_flags |= SPV_REFLECT_TYPE_FLAG_EXTERNAL_TENSOR_ARM;\n"
        "      } break;\n",
    ),
    (
        "        case SPV_REFLECT_TYPE_FLAG_EXTERNAL_ACCELERATION_STRUCTURE: {\n"
        "          p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;\n"
        "        } break;\n",
        "        case SPV_REFLECT_TYPE_FLAG_EXTERNAL_ACCELERATION_STRUCTURE: {\n"
        "          p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;\n"
        "        } break;\n"
        "\n"
        "        case SPV_REFLECT_TYPE_FLAG_EXTERNAL_TENSOR_ARM: {\n"
        "          p_descriptor->descriptor_type = SPV_REFLECT_DESCRIPTOR_TYPE_TENSOR_ARM;\n"
        "        } break;\n",
    ),
    (
        "      case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:\n"
        "        p_descriptor->resource_type = SPV_REFLECT_RESOURCE_FLAG_SRV;\n"
        "        break;\n",
        "      case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:\n"
        "        p_descriptor->resource_type = SPV_REFLECT_RESOURCE_FLAG_SRV;\n"
        "        break;\n"
        "      case SPV_REFLECT_DESCRIPTOR_TYPE_TENSOR_ARM:\n"
        "        p_descriptor->resource_type = SPV_REFLECT_RESOURCE_FLAG_UAV;\n"
        "        break;\n",
    ),
]

if __name__ == "__main__":
    if not os.path.isfile(HEADER) or not os.path.isfile(SOURCE):
        print(f"SPIRV-Reflect sources not found in {SRC_DIR}")
        sys.exit(1)
    patch(HEADER, HEADER_REPLACEMENTS)
    patch(SOURCE, SOURCE_REPLACEMENTS)
