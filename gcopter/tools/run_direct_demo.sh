#!/usr/bin/env bash
# Run from any directory; use the checkout's installed packages.
set -eo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if ! command -v ros2 >/dev/null 2>&1; then
  ros_setups=(/opt/ros/*/setup.bash)
  if [[ ${#ros_setups[@]} != 1 || ! -f "${ros_setups[0]}" ]]; then
    echo 'Source your ROS2 setup.bash first (multiple or no installations found).' >&2
    exit 1
  fi
  source "${ros_setups[0]}"
fi
if [[ ! -f "$repo_root/install/setup.bash" ]]; then
  echo 'Build gcopter and mockamap with colcon first; see docs/原始地图交互验证.md.' >&2
  exit 1
fi
source "$repo_root/install/setup.bash"
# Optional unprivileged dependency prefix used on the development machine.
if [[ -d "$repo_root/.scratch/ompl/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$repo_root/.scratch/ompl/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${WAYLAND_DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  export QT_QPA_PLATFORM=xcb
fi
exec ros2 launch gcopter direct_corridor.launch.py "$@"
