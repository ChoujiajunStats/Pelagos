#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
build_dir=${DAPENG_NATIVE_BUILD:-$project_root/build/native-navigation}
cmake -S "$project_root/Deployment/contexts/navigation/core" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release -DDAPENG_NAVIGATION_TESTS=ON
cmake --build "$build_dir" -j "${DAPENG_BUILD_JOBS:-2}"
ctest --test-dir "$build_dir" --output-on-failure
