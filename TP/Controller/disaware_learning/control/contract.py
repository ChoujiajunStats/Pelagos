"""Fixed MANUAL execution and sensor boundary for the standard BlueROV2."""

from __future__ import annotations

import numpy as np


EXECUTION_MODE = "MANUAL"
ACTUATOR_COUNT = 6
HORIZONTAL_ACTUATOR_INDICES = (0, 1, 2, 3)
VERTICAL_ACTUATOR_INDICES = (4, 5)
MANUAL_CONTROL_ORDER = (
    "pitch",
    "roll",
    "throttle",
    "yaw",
    "forward",
    "lateral",
)
MANUAL_INTENT_ORDER = ("surge", "sway", "heave", "yaw")

# The standard frame has no pitch moment. Roll is available through the two
# vertical thrusters; the remaining four axes are directly mixable.
MANUAL_CONTROL_AUTHORITY_MASK = (False, True, True, True, True, True)
MANUAL_CONTROL_SEMANTICS = {
    "pitch": "reserved unactuated axis; always zero on the standard frame",
    "roll": "normalized direct roll input to the ArduSub motor mixer",
    "throttle": "normalized direct heave input to the ArduSub motor mixer",
    "yaw": "normalized direct yaw input to the ArduSub motor mixer",
    "forward": "normalized direct surge input to the ArduSub motor mixer",
    "lateral": "normalized direct sway input to the ArduSub motor mixer",
}

ARDUSUB_NOMINAL_PARAMETERS = {
    "FRAME_CONFIG": 1,
    "MOT_FV_CPLNG_K": 1.0,
    "RC_OVERRIDE_TIME": 1.0,
    "RC_OPTIONS": 0,
    "MOT_PWM_MIN": 1100,
    "MOT_PWM_MAX": 1900,
    "MAV_GCS_SYSID": 250,
    "MAV_GCS_SYSID_HI": 0,
    "RCMAP_PITCH": 1,
    "RCMAP_ROLL": 2,
    "RCMAP_THROTTLE": 3,
    "RCMAP_YAW": 4,
    "RCMAP_FORWARD": 5,
    "RCMAP_LATERAL": 6,
    "GPS1_TYPE": 0,
    "EK3_SRC1_POSXY": 0,
    "EK3_SRC1_VELXY": 0,
    "EK3_SRC1_POSZ": 1,
    "EK3_SRC1_VELZ": 0,
    "EK3_SRC1_YAW": 1,
    "VISO_TYPE": 0,
    **{
        f"RC{channel}_{field}": value
        for channel in range(1, 7)
        for field, value in (
            ("MIN", 1100),
            ("TRIM", 1500),
            ("MAX", 1900),
            ("DZ", 0),
            ("REVERSED", 0),
        )
    },
    **{
        f"SERVO{channel}_{field}": value
        for channel in range(1, 7)
        for field, value in (
            ("FUNCTION", 32 + channel),
            ("MIN", 1100),
            ("TRIM", 1500),
            ("MAX", 1900),
            ("REVERSED", 0),
        )
    },
    "SERVO7_FUNCTION": 0,
    "SERVO8_FUNCTION": 0,
}

ARDUSUB_SITL_PARAMETERS = {
    "SIM_GPS1_ENABLE": 0,
}


def enforce_manual_authority(command: object) -> np.ndarray:
    """Validate one MANUAL command and zero axes without frame authority."""
    values = np.asarray(command, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("command must contain six finite MANUAL axes")
    output = values.copy()
    output[~np.asarray(MANUAL_CONTROL_AUTHORITY_MASK, dtype=bool)] = 0.0
    return output


def intent_to_manual_control(intent: object) -> np.ndarray:
    """Map normalized [surge, sway, heave, yaw] intent to ArduSub RC axes."""
    values = np.asarray(intent, dtype=np.float64)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("intent must contain four finite axes")
    command = np.zeros(6, dtype=np.float64)
    command[2] = values[2]
    command[3] = values[3]
    command[4] = values[0]
    command[5] = values[1]
    return command
