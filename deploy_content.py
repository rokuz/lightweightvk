#!/usr/bin/python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys

folder = "third-party"
script = os.path.join(folder, "bootstrap.py")
json = os.path.join(folder, "bootstrap-content.json")
base = os.path.join(folder, "content")

try:
    os.mkdir(base)
except FileExistsError:
    pass

os.system('"{}" {} -b {} --bootstrap-file={}'.format(sys.executable, script, base, json))

model = os.path.join(base, "src", "nss", "2_nss-1920x1080-v1_0_1.vgf")

if not os.path.exists(model):
    if os.system('"{}" "{}" --render 1920x1080'.format(sys.executable, os.path.join("tools", "make_nss_model.py"))):
        print("WARNING: cannot generate the 1920x1080 -> 3840x2160 NSS model; `DEMO_003_NeuralSuperSampling --4k` "
              "will not run. Re-run `tools/make_nss_model.py --render 1920x1080` once the Vulkan SDK and a C++ compiler "
              "are available.")
