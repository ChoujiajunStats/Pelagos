#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
source_dir="$root_dir/TP/Localization/okvis2"
apply_patch_once() {
  local patch_file="$1"
  if patch -d "$source_dir" -p1 --batch --force --fuzz=0 --dry-run < "$patch_file" >/dev/null 2>&1; then
    patch -d "$source_dir" -p1 --batch --force --fuzz=0 < "$patch_file"
  elif patch -d "$source_dir" -p1 --batch --force --reverse --fuzz=0 --dry-run < "$patch_file" >/dev/null 2>&1; then
    printf 'Patch already applied: %s\n' "${patch_file##*/}"
  else
    printf 'Patch mismatch: %s\n' "$patch_file" >&2
    patch -d "$source_dir" -p1 --batch --force --fuzz=0 --dry-run < "$patch_file" >&2 || true
    return 1
  fi
}
apply_patch_once "$root_dir/Deployment/contexts/localization/patches/brisk-aarch64.patch"
apply_patch_once "$root_dir/Deployment/contexts/localization/patches/okvis-ros2-broadcaster-lifetime.patch"
cmake -S "$source_dir" -B "$root_dir/build/okvis2" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$root_dir/install/okvis2" \
  -DBUILD_ROS2=ON -DBUILD_APPS=OFF -DBUILD_TESTS=OFF \
  -DUSE_NN=OFF -DUSE_CUDA=OFF -DHAVE_LIBREALSENSE=OFF -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_CXX_FLAGS_RELEASE='-O2 -DNDEBUG'
cmake --build "$root_dir/build/okvis2" -j 2
cmake --install "$root_dir/build/okvis2"
