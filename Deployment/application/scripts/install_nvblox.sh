#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:---help} != --install ]]; then
  echo 'Usage: bash Deployment/application/scripts/install_nvblox.sh --install'
  echo 'Requires Ubuntu 22.04 arm64, ROS 2 Humble and JetPack/CUDA.'
  exit 0
fi
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
package_dir="$project_root/Deployment/infrastructure/packages/nvblox"
source /etc/os-release
[[ $VERSION_ID == 22.04 && $(dpkg --print-architecture) == arm64 ]] || {
  echo 'Ubuntu 22.04 arm64 required.' >&2; exit 1;
}
[[ -r /opt/ros/humble/setup.bash ]] || { echo 'ROS 2 Humble required.' >&2; exit 1; }
python3 "$project_root/Deployment/tests/unit/verify_delivery.py"
gpgv --keyring "$package_dir/apt_metadata/isaac-ros.gpg" "$package_dir/apt_metadata/InRelease"
sudo install -m 644 "$package_dir/apt_metadata/isaac-ros.gpg" /usr/share/keyrings/dapeng-isaac-ros.gpg
echo 'deb [arch=arm64 signed-by=/usr/share/keyrings/dapeng-isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/release-3 jammy release-3.0' |
  sudo tee /etc/apt/sources.list.d/dapeng-isaac-ros.list >/dev/null
sudo apt-get update
sudo apt-get install "$package_dir"/debs/arm64/*.deb
