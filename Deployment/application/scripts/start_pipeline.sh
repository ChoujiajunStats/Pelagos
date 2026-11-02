#!/usr/bin/env bash
set -eo pipefail
root_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
exec /usr/bin/python3 -u "$root_dir/Deployment/application/services/run_pipeline.py" "$@"
