"""Version 15 episode schema with enforced pressure-derived depth."""

FORMAT_VERSION = 15
PROTOCOL_ID = "manual_six_thruster_vio_imu_pressure_depth_v5"
# Feature-level contract consumed by Stage 1/2.  The archive keeps the raw
# accelerometer stream, while the loader derives dynamic acceleration by
# adding the gravity vector expressed in the body frame.
OBSERVATION_FEATURE_CONTRACT_ID = (
    "vio_rotation6d_velocity_dynamic_accel_pressure_depth_actuator_modal_v5"
)

GROUND_TRUTH_ARRAYS = (
    "ground_truth_position_ned_m",
    "ground_truth_orientation_body_to_ned_xyzw",
    "ground_truth_velocity_body_frd_m_s",
    "ground_truth_angular_velocity_body_frd_rad_s",
)

PRIVILEGED_ACTUATOR_ARRAYS = (
    "ground_truth_applied_thrust_n",
)

PRIVILEGED_ACTUATOR_NATIVE_ARRAYS = (
    "thruster_state_time",
    "thruster_state_source_stamp_seconds",
    "thruster_setpoint_raw",
    "thruster_rpm_raw",
    "thruster_thrust_n_raw",
    "thruster_torque_nm_raw",
)

DEPLOYABLE_OBSERVATION_ARRAYS = (
    "vio_position_ned_m",
    "vio_orientation_body_to_ned_xyzw",
    "vio_linear_velocity_body_frd_m_s",
    "vio_angular_velocity_body_frd_rad_s",
    "ahrs_orientation_body_to_ned_xyzw",
    "ahrs_angular_velocity_body_frd_rad_s",
    "imu_angular_velocity_body_frd_rad_s",
    "imu_linear_acceleration_body_frd_m_s2",
    "depth_down_m",
    "pressure_pa",
)

ALIGNED_ARRAYS = (
    "time",
    *DEPLOYABLE_OBSERVATION_ARRAYS,
    "vio_valid",
    "vio_age_seconds",
    "ahrs_valid",
    "ahrs_age_seconds",
    "imu_valid",
    "imu_age_seconds",
    "depth_valid",
    "depth_age_seconds",
    "pressure_valid",
    "pressure_age_seconds",
    "manual_control_normalized",
    "manual_control_pwm",
    "manual_control_age_seconds",
    "motor_output_normalized",
    "motor_output_pwm",
    "motor_output_age_seconds",
)

NATIVE_STREAM_ARRAYS = (
    "vio_time",
    "vio_source_stamp_seconds",
    "vio_raw_position_ned_m",
    "vio_raw_orientation_body_to_ned_xyzw",
    "vio_raw_linear_velocity_body_frd_m_s",
    "vio_raw_angular_velocity_body_frd_rad_s",
    "ahrs_time",
    "ahrs_source_stamp_seconds",
    "ahrs_raw_orientation_body_to_ned_xyzw",
    "ahrs_raw_angular_velocity_body_frd_rad_s",
    "imu_time",
    "imu_source_stamp_seconds",
    "imu_raw_angular_velocity_body_frd_rad_s",
    "imu_raw_linear_acceleration_body_frd_m_s2",
    "depth_time",
    "depth_raw_down_m",
    "pressure_time",
    "pressure_source_stamp_seconds",
    "pressure_raw_pa",
    "pressure_raw_variance_pa2",
    "manual_control_time",
    "manual_control_raw_normalized",
    "manual_control_raw_pwm",
    "motor_output_time",
    "motor_output_raw_normalized",
    "motor_output_raw_pwm",
)

REQUIRED_ARRAYS = (*ALIGNED_ARRAYS, *NATIVE_STREAM_ARRAYS)
