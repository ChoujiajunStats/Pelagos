# Real-vehicle deployment

This package exposes two controller choices behind one four-axis command
contract:

```text
controller = nmpc   # nominal six-thruster model
controller = latent # frozen Stage1 + latent-response JEPA-MPPI
```

Both produce normalized commands in this order:

```text
[surge, sway, heave_up, yaw]
```

The transport and sensor loop must remain outside the controller adapters.
The latent controller requires the complete trained observation/action history
contract and the current NED position; it does not synthesize missing VIO
data.

## Protocol boundary

`protocol.py` packs the vendor's fixed 24-byte motor frame. The six learned
motor outputs are mapped to `L1, L2, L3, R1, R2, R3` by default and `L4/R4`
are sent at neutral `1500`. This default must be verified with a low-power,
disarmed motor test and changed through `SixMotorChannelMap` if the wiring
differs. Motor signs are also configurable there.

`pack_command()` is the complete command bridge: it applies the retained
six-thruster allocation to `[surge, sway, heave_up, yaw]`, converts normalized
motor values to `1000..2000` PWM, and packs the protocol frame. Its default
allocation is the same one used by the trained model; the raw pre-saturation
motor values are available from `command_to_six_motor_normalized()` for a
safety gate or log.

The framing helper is deliberately pure. `transport.py` now provides the
minimum TCP boundary:

1. connection handling for `192.168.1.7:8887`;
2. a heartbeat timer (the device asks for at least every 5 seconds);
3. neutral/disarm output on timeout or disconnect;
4. extraction of complete raw 32-byte sensor frames.

Sensor-frame payload parsing and conversion into the VIO/observation contract
are implemented in `observation.py`. The adapter requires an
`ExternalVioPose` for position, body velocity, and orientation because those
quantities do not exist in the vendor frame. It combines those VIO fields with
the protocol depth/gyro/accelerometer into the exact 16D JEPA observation:

```text
rotation6D + body_velocity + gyro + gravity_compensated_acceleration + relative_depth
```

The protocol document's IMU examples imply packed BCD; the decoder makes the
scales explicit in `RovSensorDecodeConfig`. Verify those scales against one
raw frame before arming. The document's `01 80 00 -> +180.00` example is
internally inconsistent with packed BCD (it decodes as `+80.00`); this must be
resolved with a raw vendor frame before using the gyro in closed loop.

`history.py` provides `CausalHistoryBuffer`. Seed it with a neutral six-motor
action, then call `append_transition()` only after the corresponding action
has actually been executed and the next observation is available. This keeps
the latent controller's action/observation phase causal.

The protocol document does not define enough precision/unit information for
the IMU fields to be safely converted here. Do not feed raw protocol bytes to
JEPA until those units and the VIO coordinate frame have been calibrated.
