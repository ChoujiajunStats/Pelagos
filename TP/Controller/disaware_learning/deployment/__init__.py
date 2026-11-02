"""Real-vehicle deployment boundaries for the two supported controllers.

The deployment package deliberately sits above the transport layer.  It
turns either controller into the same four-axis command contract and keeps
the six-thruster-to-device protocol mapping explicit.
"""

from disaware_learning.deployment.controllers import (
    DeploymentCommand,
    LatentDeploymentController,
    NmpcDeploymentController,
    build_controller,
)
from disaware_learning.deployment.protocol import (
    DEFAULT_SIX_MOTOR_CHANNELS,
    SixMotorChannelMap,
    command_to_six_motor_normalized,
    normalized_to_protocol_pwm,
    pack_command,
    pack_motor_command,
    pack_heartbeat,
)
from disaware_learning.deployment.transport import RovTcpConfig, RovTcpTransport
from disaware_learning.deployment.observation import (
    ExternalRovObservationAdapter,
    ExternalVioPose,
    RovSensorDecodeConfig,
    RovSensorSample,
    decode_sensor_frame,
)
from disaware_learning.deployment.history import CausalHistoryBuffer, DeploymentHistory

__all__ = [
    "DEFAULT_SIX_MOTOR_CHANNELS",
    "DeploymentCommand",
    "LatentDeploymentController",
    "NmpcDeploymentController",
    "SixMotorChannelMap",
    "build_controller",
    "command_to_six_motor_normalized",
    "pack_heartbeat",
    "pack_command",
    "pack_motor_command",
    "normalized_to_protocol_pwm",
    "RovTcpConfig",
    "RovTcpTransport",
    "ExternalRovObservationAdapter",
    "ExternalVioPose",
    "RovSensorDecodeConfig",
    "RovSensorSample",
    "decode_sensor_frame",
    "CausalHistoryBuffer",
    "DeploymentHistory",
]
