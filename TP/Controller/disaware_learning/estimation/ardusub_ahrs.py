"""Publish ArduSub's onboard AHRS attitude as a deployable ROS IMU stream."""

from __future__ import annotations

import math
import os
import time

from rclpy.executors import ExternalShutdownException
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import FluidPressure, Imu
from std_msgs.msg import Float64, Float64MultiArray, UInt8

from disaware_learning.control.contract import ACTUATOR_COUNT
from disaware_learning.control.rc import PWM_MAX, PWM_MIN, pwm_to_normalized

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil  # noqa: E402

PRESSURE_MESSAGE_BY_INSTANCE = {
    0: (
        "SCALED_PRESSURE",
        mavutil.mavlink.MAVLINK_MSG_ID_SCALED_PRESSURE,
    ),
    1: (
        "SCALED_PRESSURE2",
        mavutil.mavlink.MAVLINK_MSG_ID_SCALED_PRESSURE2,
    ),
    2: (
        "SCALED_PRESSURE3",
        mavutil.mavlink.MAVLINK_MSG_ID_SCALED_PRESSURE3,
    ),
}


class ArduSubAhrsBridge(Node):
    """Forward flight-controller AHRS and IMU data, never simulator truth."""

    def __init__(self) -> None:
        super().__init__("ardusub_ahrs_bridge")
        self.declare_parameter("vehicle_name", "bluerov2")
        self.declare_parameter("mavlink_endpoint", "udpin:0.0.0.0:14552")
        self.declare_parameter("source_system", 250)
        self.declare_parameter("source_component", 192)
        self.declare_parameter("message_rate", 50.0)
        self.declare_parameter("imu_message_rate", 50.0)
        self.declare_parameter("pressure_message_rate", 20.0)
        self.declare_parameter("depth_message_rate", 20.0)
        self.declare_parameter("manual_control_message_rate", 50.0)
        self.declare_parameter("motor_output_message_rate", 50.0)
        self.declare_parameter("publish_motor_output", True)
        # -1 follows ArduSub's runtime BARO_PRIMARY selection.  Explicit
        # 0..2 is available for flight-controller builds that do not expose it.
        self.declare_parameter("pressure_instance", -1)
        self.declare_parameter("orientation_std_rad", 0.035)
        self.declare_parameter("angular_velocity_std_rad_s", 0.01)
        self.declare_parameter("pressure_std_pa", 50.0)
        self.declare_parameter("maximum_message_age", 0.25)

        vehicle = str(self.get_parameter("vehicle_name").value).strip("/")
        endpoint = str(self.get_parameter("mavlink_endpoint").value)
        self.message_rate = max(5.0, float(self.get_parameter("message_rate").value))
        self.imu_message_rate = max(
            5.0, float(self.get_parameter("imu_message_rate").value)
        )
        self.pressure_message_rate = max(
            2.0, float(self.get_parameter("pressure_message_rate").value)
        )
        self.depth_message_rate = max(
            2.0, float(self.get_parameter("depth_message_rate").value)
        )
        self.manual_control_message_rate = max(
            2.0,
            float(self.get_parameter("manual_control_message_rate").value),
        )
        self.motor_output_message_rate = max(
            2.0,
            float(self.get_parameter("motor_output_message_rate").value),
        )
        self.publish_motor_output = bool(
            self.get_parameter("publish_motor_output").value
        )
        pressure_instance = int(self.get_parameter("pressure_instance").value)
        if pressure_instance not in {-1, *PRESSURE_MESSAGE_BY_INSTANCE}:
            raise ValueError("pressure_instance must be -1, 0, 1, or 2")
        self.pressure_instance: int | None = (
            None if pressure_instance == -1 else pressure_instance
        )
        self.pressure_message_type: str | None = None
        self.pressure_message_id: int | None = None
        if self.pressure_instance is not None:
            self._set_pressure_instance(self.pressure_instance)
        self.maximum_message_age = max(
            0.05, float(self.get_parameter("maximum_message_age").value)
        )
        orientation_std = max(
            0.0, float(self.get_parameter("orientation_std_rad").value)
        )
        angular_std = max(
            0.0,
            float(self.get_parameter("angular_velocity_std_rad_s").value),
        )
        self.orientation_variance = orientation_std**2
        self.angular_variance = angular_std**2
        self.pressure_variance = (
            max(0.0, float(self.get_parameter("pressure_std_pa").value)) ** 2
        )
        self.publisher = self.create_publisher(Imu, f"/{vehicle}/ardusub/ahrs", 20)
        self.imu_publisher = self.create_publisher(Imu, f"/{vehicle}/ardusub/imu", 20)
        self.pressure_publisher = self.create_publisher(
            FluidPressure, f"/{vehicle}/ardusub/pressure", 20
        )
        self.pressure_instance_publisher = self.create_publisher(
            UInt8, f"/{vehicle}/ardusub/pressure_instance", 10
        )
        self.depth_publisher = self.create_publisher(
            Float64, f"/{vehicle}/ardusub/depth", 20
        )
        self.manual_control_publisher = self.create_publisher(
            Float64MultiArray,
            f"/{vehicle}/ardusub/manual_control",
            20,
        )
        self.manual_control_pwm_publisher = self.create_publisher(
            Float64MultiArray,
            f"/{vehicle}/ardusub/manual_control_pwm",
            20,
        )
        self.motor_output_publisher = self.create_publisher(
            Float64MultiArray,
            f"/{vehicle}/ardusub/motor_output",
            20,
        )
        self.motor_output_pwm_publisher = self.create_publisher(
            Float64MultiArray,
            f"/{vehicle}/ardusub/motor_output_pwm",
            20,
        )
        self.mavlink = mavutil.mavlink_connection(
            endpoint,
            source_system=int(self.get_parameter("source_system").value),
            source_component=int(self.get_parameter("source_component").value),
            dialect="ardupilotmega",
        )
        self.target_ready = False
        self.interval_requested_at = -math.inf
        self.pressure_instance_requested_at = -math.inf
        self.last_attitude_received = -math.inf
        self.last_imu_received = -math.inf
        self.last_pressure_received = -math.inf
        self.last_depth_received = -math.inf
        self.last_manual_control_received = -math.inf
        self.last_motor_output_received = -math.inf
        self.received_count = 0
        self.create_timer(0.005, self._poll)
        self.create_timer(1.0, self._maintain_stream)
        self.get_logger().info(
            "ArduSub AHRS bridge ready: "
            f"endpoint={endpoint}, topic=/{vehicle}/ardusub/ahrs"
        )

    def _poll(self) -> None:
        for _ in range(100):
            message = self.mavlink.recv_match(blocking=False)
            if message is None:
                break
            message_type = message.get_type()
            if message_type == "BAD_DATA":
                continue
            if message_type == "HEARTBEAT":
                self.target_ready = True
            elif message_type == "PARAM_VALUE":
                self._handle_parameter(message)
            elif message_type == "ATTITUDE_QUATERNION":
                self._publish_attitude(message)
            elif message_type == "HIGHRES_IMU":
                self._publish_imu(message)
            elif message_type == self.pressure_message_type:
                self._publish_pressure(message)
            elif message_type == "AHRS2":
                self._publish_depth(message)
            elif message_type == "RC_CHANNELS":
                self._publish_manual_control(message)
            elif message_type == "SERVO_OUTPUT_RAW" and self.publish_motor_output:
                self._publish_motor_output(message)

    def _maintain_stream(self) -> None:
        now = time.monotonic()
        if not self.target_ready:
            return
        if self.pressure_instance is None:
            self._request_pressure_instance(now)
        last_received = [
            self.last_attitude_received,
            self.last_imu_received,
            self.last_depth_received,
            self.last_manual_control_received,
        ]
        if self.publish_motor_output:
            last_received.append(self.last_motor_output_received)
        if self.pressure_message_id is not None:
            last_received.append(self.last_pressure_received)
        if (
            now - self.interval_requested_at >= 5.0
            or max(now - received for received in last_received)
            > self.maximum_message_age
        ):
            requests = [
                (
                    mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION,
                    self.message_rate,
                ),
                (
                    mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU,
                    self.imu_message_rate,
                ),
                (
                    mavutil.mavlink.MAVLINK_MSG_ID_AHRS2,
                    self.depth_message_rate,
                ),
                (
                    mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS,
                    self.manual_control_message_rate,
                ),
            ]
            if self.publish_motor_output:
                requests.append(
                    (
                        mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW,
                        self.motor_output_message_rate,
                    )
                )
            if self.pressure_message_id is not None:
                requests.append((self.pressure_message_id, self.pressure_message_rate))
            for message_id, rate in requests:
                self.mavlink.mav.command_long_send(
                    self.mavlink.target_system,
                    self.mavlink.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    message_id,
                    int(round(1_000_000.0 / rate)),
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            self.interval_requested_at = now

    def _request_pressure_instance(self, now: float) -> None:
        if now - self.pressure_instance_requested_at < 2.0:
            return
        self.mavlink.mav.param_request_read_send(
            self.mavlink.target_system,
            self.mavlink.target_component,
            b"BARO_PRIMARY",
            -1,
        )
        self.pressure_instance_requested_at = now

    def _handle_parameter(self, message) -> None:
        if self.pressure_instance is not None:
            return
        name = message.param_id
        if isinstance(name, bytes):
            name = name.decode("ascii", errors="ignore")
        if str(name).rstrip("\x00") != "BARO_PRIMARY":
            return
        value = float(message.param_value)
        instance = int(round(value))
        if (
            not math.isfinite(value)
            or not math.isclose(value, instance, abs_tol=1e-3)
            or instance not in PRESSURE_MESSAGE_BY_INSTANCE
        ):
            self.get_logger().error(
                "BARO_PRIMARY is not a supported MAVLink pressure instance: " f"{value}"
            )
            return
        self._set_pressure_instance(instance)

    def _set_pressure_instance(self, instance: int) -> None:
        self.pressure_instance = instance
        (
            self.pressure_message_type,
            self.pressure_message_id,
        ) = PRESSURE_MESSAGE_BY_INSTANCE[instance]
        if hasattr(self, "interval_requested_at"):
            self.interval_requested_at = -math.inf
            self.get_logger().info(
                "ArduSub pressure source selected: "
                f"BARO_PRIMARY={instance}, "
                f"message={self.pressure_message_type}"
            )

    def _publish_attitude(self, message) -> None:
        values = (
            float(message.q2),
            float(message.q3),
            float(message.q4),
            float(message.q1),
            float(message.rollspeed),
            float(message.pitchspeed),
            float(message.yawspeed),
        )
        if not all(math.isfinite(value) for value in values):
            return
        norm = math.sqrt(sum(value * value for value in values[:4]))
        if norm <= 1e-9:
            return
        output = Imu()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = "base_link_frd"
        output.orientation.x = values[0] / norm
        output.orientation.y = values[1] / norm
        output.orientation.z = values[2] / norm
        output.orientation.w = values[3] / norm
        output.angular_velocity.x = values[4]
        output.angular_velocity.y = values[5]
        output.angular_velocity.z = values[6]
        for index in (0, 4, 8):
            output.orientation_covariance[index] = self.orientation_variance
            output.angular_velocity_covariance[index] = self.angular_variance
        output.linear_acceleration_covariance[0] = -1.0
        self.publisher.publish(output)
        self.last_attitude_received = time.monotonic()
        self.received_count += 1

    def _publish_imu(self, message) -> None:
        values = (
            float(message.xgyro),
            float(message.ygyro),
            float(message.zgyro),
            float(message.xacc),
            float(message.yacc),
            float(message.zacc),
        )
        if not all(math.isfinite(value) for value in values):
            return
        output = Imu()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = "base_link_frd"
        output.orientation_covariance[0] = -1.0
        output.angular_velocity.x = values[0]
        output.angular_velocity.y = values[1]
        output.angular_velocity.z = values[2]
        output.linear_acceleration.x = values[3]
        output.linear_acceleration.y = values[4]
        output.linear_acceleration.z = values[5]
        for index in (0, 4, 8):
            output.angular_velocity_covariance[index] = self.angular_variance
        self.imu_publisher.publish(output)
        self.last_imu_received = time.monotonic()

    def _publish_pressure(self, message) -> None:
        pressure_pa = float(message.press_abs) * 100.0
        if not math.isfinite(pressure_pa) or pressure_pa <= 0.0:
            return
        output = FluidPressure()
        output.header.stamp = self.get_clock().now().to_msg()
        assert self.pressure_instance is not None
        output.header.frame_id = f"ardusub_baro_{self.pressure_instance}"
        output.fluid_pressure = pressure_pa
        output.variance = self.pressure_variance
        self.pressure_publisher.publish(output)
        self.pressure_instance_publisher.publish(UInt8(data=self.pressure_instance))
        self.last_pressure_received = time.monotonic()

    def _publish_depth(self, message) -> None:
        """Publish ArduSub's pressure-referenced altitude as NED depth."""
        depth_down_m = -float(message.altitude)
        if not math.isfinite(depth_down_m):
            return
        self.depth_publisher.publish(Float64(data=depth_down_m))
        self.last_depth_received = time.monotonic()

    @staticmethod
    def _pwm_values(message, prefix: str, count: int) -> list[float] | None:
        values = [
            float(getattr(message, f"{prefix}{index}_raw", 0.0))
            for index in range(1, count + 1)
        ]
        if not all(PWM_MIN <= value <= PWM_MAX for value in values):
            return None
        return values

    def _publish_manual_control(self, message) -> None:
        pwm = self._pwm_values(message, "chan", 8)
        if pwm is None:
            return
        normalized = pwm_to_normalized(pwm[:6])
        self.manual_control_publisher.publish(
            Float64MultiArray(data=normalized.tolist())
        )
        self.manual_control_pwm_publisher.publish(Float64MultiArray(data=pwm))
        self.last_manual_control_received = time.monotonic()

    def _publish_motor_output(self, message) -> None:
        pwm = self._pwm_values(message, "servo", ACTUATOR_COUNT)
        if pwm is None:
            return
        normalized = pwm_to_normalized(pwm)
        self.motor_output_publisher.publish(Float64MultiArray(data=normalized.tolist()))
        self.motor_output_pwm_publisher.publish(Float64MultiArray(data=pwm))
        self.last_motor_output_received = time.monotonic()

    def destroy_node(self) -> bool:
        try:
            self.mavlink.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = ArduSubAhrsBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
