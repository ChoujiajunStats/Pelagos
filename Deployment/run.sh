#!/usr/bin/env bash
set -euo pipefail

deployment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
  cat <<'HELP'
Usage: ./Deployment/run.sh <command> [args]

  mapping [--mapper wavemap|nvblox] [--check-only] [--no-rviz]
  navigation [--check-only] [--no-rviz]
  planner wavestar|omniplanner|stop
  control start|stop|status
  rviz [--mapper auto|wavemap|nvblox]
  build okvis|wavemap|navigation|native|map-export
  check

mapping/navigation/rviz: robot endpoints; planner: path preview only.
control start: existing shadow-mode controller. No automatic ARM.
HELP
}

fail() { echo "ERROR: $*" >&2; usage >&2; exit 2; }
script() {
  local entry="$deployment_dir/$1"
  shift
  [[ -f $entry ]] || { echo "ERROR: Entry not found: $entry" >&2; exit 1; }
  exec bash "$entry" "$@"
}

for argument in "$@"; do
  case "$argument" in -h|--help) usage; exit 0 ;; esac
done

command=${1:-help}
if (($#)); then shift; fi
case "$command" in
  help) usage ;;
  mapping) script application/scripts/start_robot_mapping.sh "$@" ;;
  navigation) script contexts/navigation/scripts/start_robot_navigation.sh "$@" ;;
  rviz) script interfaces/visualization/scripts/open_robot_rviz.sh "$@" ;;
  planner)
    action=${1:-}; if (($#)); then shift; fi
    (($# == 0)) || fail 'Unexpected planner arguments.'
    case "$action" in
      wavestar) script contexts/planning/scripts/start_wavestar_preview.sh ;;
      omniplanner) script contexts/planning/scripts/start_preview.sh ;;
      stop) script contexts/planning/scripts/stop_preview.sh ;;
      *) fail 'Expected planner wavestar|omniplanner|stop.' ;;
    esac ;;
  control)
    action=${1:-}; if (($#)); then shift; fi
    (($# == 0)) || fail 'Unexpected control arguments.'
    case "$action" in
      start) script contexts/control/scripts/start_controller.sh ;;
      stop) script contexts/control/scripts/stop_controller.sh ;;
      status) exec python3 "$deployment_dir/contexts/control/services/controllerctl.py" status ;;
      *) fail 'Expected control start|stop|status.' ;;
    esac ;;
  build)
    component=${1:-}; if (($#)); then shift; fi
    (($# == 0)) || fail 'Unexpected build arguments.'
    case "$component" in
      okvis) script contexts/localization/scripts/build_okvis2.sh ;;
      wavemap) script contexts/mapping/scripts/build_wavemap.sh ;;
      navigation) script contexts/navigation/scripts/build_navigation.sh ;;
      native) script application/scripts/build_native_navigation.sh ;;
      map-export) script contexts/planning/scripts/build_wavemap_export.sh ;;
      *) fail 'Expected build okvis|wavemap|navigation|native|map-export.' ;;
    esac ;;
  check)
    (($# == 0)) || fail 'Unexpected check arguments.'
    exec python3 "$deployment_dir/tests/unit/verify_delivery.py" ;;
  *) fail "Unknown command: $command" ;;
esac
