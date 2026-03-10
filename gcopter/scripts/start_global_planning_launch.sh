#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

safe_source() {
  set +u
  # shellcheck disable=SC1090
  source "$1"
  set -u
}

find_workspace_setup() {
  local current="${SCRIPT_DIR}"
  while [[ "${current}" != "/" ]]; do
    if [[ -f "${current}/install/setup.bash" ]]; then
      printf '%s\n' "${current}/install/setup.bash"
      return 0
    fi
    current="$(dirname "${current}")"
  done
  return 1
}

if [[ -f /opt/ros/humble/setup.bash ]]; then
  safe_source /opt/ros/humble/setup.bash
fi

if WS_SETUP="$(find_workspace_setup)"; then
  safe_source "${WS_SETUP}"
fi

export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros2_launch_logs}"
mkdir -p "${ROS_LOG_DIR}"

exec ros2 launch gcopter global_planning.launch.py "$@"
