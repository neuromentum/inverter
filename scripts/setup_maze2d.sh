#!/usr/bin/env bash
# Check prerequisites; download native MuJoCo only when explicitly requested.
# Python packages and conda environments are installed separately by the user.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash scripts/setup_maze2d.sh [--download-mujoco]

Checks the native MuJoCo installation and compiler/patchelf availability.
--download-mujoco installs the checksum-verified MuJoCo 2.1.0 Linux archive
only if the destination does not exist. No existing installation is replaced.
Set MUJOCO_PY_MUJOCO_PATH to choose a destination; default: ~/.mujoco/mujoco210.
This script does not create environments, install Python packages, or fetch data.
HELP
  exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--download-mujoco" ) ]]; then
  echo "Usage: bash scripts/setup_maze2d.sh [--download-mujoco]" >&2
  exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "This setup targets Linux x86_64, matching the recorded evaluation platform." >&2
  exit 1
fi
for executable in cc patchelf; do
  if ! command -v "$executable" >/dev/null; then
    echo "Missing $executable; see the system packages in docs/maze2d-environment.md." >&2
    exit 1
  fi
done
inverter_mujoco_dir="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
if [[ ! -f "$inverter_mujoco_dir/bin/libmujoco210.so" ]]; then
  if [[ "${1:-}" != --download-mujoco ]]; then
    echo "Native MuJoCo 2.1.0 is missing at $inverter_mujoco_dir." >&2
    echo "Run again with --download-mujoco to install the official archive." >&2
    exit 1
  fi
  if [[ -e "$inverter_mujoco_dir" ]]; then
    echo "Destination already exists but is incomplete; leaving it unchanged: $inverter_mujoco_dir" >&2
    exit 1
  fi
  inverter_stage="$(mktemp -d)"
  trap 'rm -rf -- "$inverter_stage"' EXIT
  curl --fail --location --proto '=https' --tlsv1.2 \
    'https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz' \
    --output "$inverter_stage/mujoco.tar.gz"
  printf '%s  %s\n' 'a436ca2f4144c38b837205635bbd60ffe1162d5b44c87df22232795978d7d012' "$inverter_stage/mujoco.tar.gz" | sha256sum --check
  tar --extract --gzip --file "$inverter_stage/mujoco.tar.gz" --directory "$inverter_stage"
  mkdir -p -- "$(dirname -- "$inverter_mujoco_dir")"
  mv -- "$inverter_stage/mujoco210" "$inverter_mujoco_dir"
fi
printf 'Native MuJoCo found: %s\n' "$inverter_mujoco_dir"
printf '%s\n' 'Next: activate the dedicated environment and follow docs/maze2d-environment.md.'
