#!/usr/bin/python3
# LightweightVK
#
# Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import subprocess
import tarfile

parser = argparse.ArgumentParser(description="Packs the sample content and pushes it to an Android device")
parser.add_argument("--obb", nargs="+", metavar="PACKAGE",
                    help="push as the OBB of these packages (e.g. org.lvk.samples.lvk_001_HelloTriangle) instead of $EXTERNAL_STORAGE/LVK")
args = parser.parse_args()

tar_path = os.path.join("third-party", "content", "archives", "lvk_content.tar")

# source directories and their archive prefixes
paths = [
    ("third-party/content", "content"),
    ("third-party/deps/src/3D-Graphics-Rendering-Cookbook/data", "deps/src/3D-Graphics-Rendering-Cookbook/data"),
    ("third-party/deps/src/ktx-software/tests/srcimages/Iron_Bars", "deps/src/ktx-software/tests/srcimages/Iron_Bars"),
]

# directories excluded by absolute path
exclude_abs = {
    os.path.abspath("third-party/content/archives"),
    os.path.abspath("third-party/content/patches"),
    os.path.abspath("third-party/content/src/cloud"),
    os.path.abspath("third-party/content/src/glTF-Sample-Models"),
    os.path.abspath("third-party/content/src/CT_head"),
}

# directory names excluded everywhere
exclude_names = {".git"}

if not os.path.isfile(tar_path):
    print("Creating {} ...".format(tar_path))
    total_files = 0
    with tarfile.open(tar_path, "w", format=tarfile.GNU_FORMAT) as tf:
        for desktop_path, archive_prefix in paths:
            if not os.path.isdir(desktop_path):
                print("  Warning: {} does not exist, skipping".format(desktop_path))
                continue
            print("  Adding {} ...".format(desktop_path))
            count = 0
            for root, dirs, files in os.walk(desktop_path):
                dirs[:] = [d for d in dirs if d not in exclude_names and os.path.abspath(os.path.join(root, d)) not in exclude_abs]
                for f in files:
                    full_path = os.path.join(root, f)
                    arcname = os.path.join(archive_prefix, os.path.relpath(full_path, desktop_path)).replace("\\", "/")
                    tf.add(full_path, arcname=arcname)
                    count += 1
            total_files += count
            print("    {} files".format(count))
    tar_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print("Created {} ({:.1f} MB, {} files)".format(tar_path, tar_size_mb, total_files))
else:
    tar_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print("{} already exists ({:.1f} MB), skipping creation".format(tar_path, tar_size_mb))

# upload to the device
try:
    result = subprocess.run(["adb", "shell", "echo", "$EXTERNAL_STORAGE"], capture_output=True, text=True)
    external_storage = result.stdout.strip()
except Exception as e:
    print("adb error:", e)
    external_storage = None

if external_storage:
    # the samples look for `main.<versionCode>.<package>.obb`, and `versionCode` is 1 in `android/app/build.gradle.in`
    if args.obb:
        android_tar_paths = ["{}/Android/obb/{}/main.1.{}.obb".format(external_storage, p, p) for p in args.obb]
    else:
        android_tar_paths = [external_storage + "/LVK/lvk_content.tar"]
    for android_tar_path in android_tar_paths:
        print("Uploading to {} ...".format(android_tar_path))
        subprocess.run(["adb", "push", tar_path, android_tar_path])
    print("Completed")
else:
    print("External storage path is not found")
