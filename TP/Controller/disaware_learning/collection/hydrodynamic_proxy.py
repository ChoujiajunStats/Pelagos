"""Small independent underwater plant used for cheap controller ranking.

This module deliberately does not import a JEPA model.  It is a deterministic,
low-order six-thruster plant: motor lag, modal wrench gains, current-relative
drag, buoyancy, and a small ballast-induced attitude coupling.  Its purpose is
to rank controller/checkpoint candidates before expensive Stonefish tests, not
to claim hydrodynamic fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from disaware_learning.collection.physics_domain import PhysicsDomain


def _skew(value: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(value, dtype=np.float64)
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _so3_exp(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    theta = float(np.linalg.norm(value))
    if theta < 1e-9:
        return np.eye(3) + _skew(value)
    axis = value / theta
    k = _skew(axis)
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def _manual_mixer(command: np.ndarray, roll_command: float) -> np.ndarray:
    """Local copy of the fixed six-thruster allocation, without JEPA imports."""
    surge, sway, heave, yaw = command
    roll = float(roll_command)
    return np.asarray(
        (
            yaw - surge + sway,
            -yaw - surge - sway,
            -yaw + surge + sway,
            yaw + surge - sway,
            roll - heave,
            -roll - heave,
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class HydrodynamicProxyConfig:
    """Numerical coefficients for the deliberately conservative proxy."""

    nominal_mass_kg: float = 11.189
    nominal_inertia_kg_m2: tuple[float, float, float] = (0.45, 0.55, 0.65)
    linear_acceleration_gain_m_s2: tuple[float, float, float] = (0.95, 0.90, 0.82)
    angular_acceleration_gain_rad_s2: tuple[float, float, float] = (0.30, 0.30, 0.42)
    yaw_drag: float = 0.32
    attitude_damping: float = 0.75
    max_speed_m_s: float = 0.80
    max_angular_speed_rad_s: float = 1.20
    max_position_m: float = 100.0

    def __post_init__(self) -> None:
        if self.nominal_mass_kg <= 0.0 or any(v <= 0.0 for v in self.nominal_inertia_kg_m2):
            raise ValueError("proxy mass and inertia must be positive")
        if any(v <= 0.0 for v in self.linear_acceleration_gain_m_s2):
            raise ValueError("linear gains must be positive")
        if any(v <= 0.0 for v in self.angular_acceleration_gain_rad_s2):
            raise ValueError("angular gains must be positive")
        if self.yaw_drag <= 0.0 or self.attitude_damping <= 0.0:
            raise ValueError("proxy damping must be positive")


@dataclass
class ProxyState:
    position_ned_m: np.ndarray
    rotation_body_to_ned: np.ndarray
    body_velocity_frd_m_s: np.ndarray
    body_omega_frd_rad_s: np.ndarray
    body_acceleration_frd_m_s2: np.ndarray
    motor_state: np.ndarray
    depth_down_m: float
    previous_acceleration_frd_m_s2: np.ndarray

    @classmethod
    def initial(cls, position_ned_m: np.ndarray = (0.0, 0.0, 3.0)) -> "ProxyState":
        position = np.asarray(position_ned_m, dtype=np.float64).copy()
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("initial position must contain three finite values")
        zero = np.zeros(3, dtype=np.float64)
        return cls(
            position_ned_m=position,
            rotation_body_to_ned=np.eye(3, dtype=np.float64),
            body_velocity_frd_m_s=zero.copy(),
            body_omega_frd_rad_s=zero.copy(),
            body_acceleration_frd_m_s2=zero.copy(),
            motor_state=np.zeros(6, dtype=np.float64),
            depth_down_m=float(position[2]),
            previous_acceleration_frd_m_s2=zero.copy(),
        )


class HydrodynamicProxy:
    """Independent first-order modal underwater dynamics."""

    def __init__(
        self,
        domain: PhysicsDomain,
        *,
        config: HydrodynamicProxyConfig = HydrodynamicProxyConfig(),
        dt_seconds: float = 0.2,
        initial_position_ned_m: np.ndarray = (0.0, 0.0, 3.0),
    ) -> None:
        if dt_seconds <= 0.0 or not math.isfinite(float(dt_seconds)):
            raise ValueError("dt_seconds must be positive and finite")
        self.domain = domain
        self.config = config
        self.dt_seconds = float(dt_seconds)
        self.state = ProxyState.initial(initial_position_ned_m)
        self.time_seconds = 0.0
        self.last_command = np.zeros(4, dtype=np.float64)
        self.last_motor_command = np.zeros(6, dtype=np.float64)

    def reset(self, position_ned_m: np.ndarray = (0.0, 0.0, 3.0)) -> None:
        self.state = ProxyState.initial(position_ned_m)
        self.time_seconds = 0.0
        self.last_command.fill(0.0)
        self.last_motor_command.fill(0.0)

    def _mass(self) -> float:
        return max(1.0, self.config.nominal_mass_kg + self.domain.payload_mass_delta_kg)

    def _step_motor(self, target: np.ndarray) -> None:
        tau = max(float(self.domain.rotor_tau_seconds), self.dt_seconds * 0.15)
        alpha = 1.0 - math.exp(-self.dt_seconds / tau)
        self.state.motor_state += alpha * (target - self.state.motor_state)

    def step(self, command: np.ndarray, *, roll_command: float = 0.0) -> dict[str, Any]:
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (4,) or not np.all(np.isfinite(command)):
            raise ValueError("proxy command must contain four finite values")
        command = np.clip(command, -1.0, 1.0)
        target_motor = np.clip(_manual_mixer(command, roll_command), -1.0, 1.0)
        self._step_motor(target_motor)
        motor = self.state.motor_state
        gain = np.asarray(self.domain.thruster_gain, dtype=np.float64)
        realized = motor * gain
        modal = np.array(
            (
                (-realized[0] - realized[1] + realized[2] + realized[3]) / 4.0,
                (realized[0] - realized[1] + realized[2] - realized[3]) / 4.0,
                -(realized[4] + realized[5]) / 2.0,
                (realized[0] - realized[1] - realized[2] + realized[3]) / 4.0,
                (realized[4] - realized[5]) / 2.0,
            ),
            dtype=np.float64,
        )
        mass = self._mass()
        rotation = self.state.rotation_body_to_ned
        current_ned = np.asarray(
            self.domain.external_current_velocity_ned_m_s(self.time_seconds),
            dtype=np.float64,
        )
        current_body = rotation.T @ current_ned
        relative_velocity = self.state.body_velocity_frd_m_s - current_body
        viscous = np.asarray(self.domain.viscous_drag_coefficient_xyz, dtype=np.float64)
        quadratic = np.asarray(self.domain.quadratic_drag_coefficient_xyz, dtype=np.float64)
        drag = viscous * relative_velocity + quadratic * np.abs(relative_velocity) * relative_velocity
        accel = (
            np.asarray(self.config.linear_acceleration_gain_m_s2)
            * modal[:3]
            * self.domain.global_thrust_gain
            * (self.config.nominal_mass_kg / mass)
            - drag / mass
        )
        # NED +z is down; positive buoyancy therefore accelerates in -z.
        accel[2] -= float(self.domain.net_buoyancy_force_n) / mass
        # A ballast offset produces a deliberately small, observable attitude
        # coupling, while keeping the proxy numerically well behaved.
        offset = np.asarray(self.domain.ballast_offset_xyz_m, dtype=np.float64)
        torque = np.cross(offset, modal[:3]) * 0.10
        omega = self.state.body_omega_frd_rad_s
        angular_gain = np.asarray(self.config.angular_acceleration_gain_rad_s2)
        # The standard six-thruster frame has roll authority but no direct
        # pitch actuator; pitch changes only through the small ballast torque.
        angular_accel = angular_gain * np.array((modal[4], 0.0, modal[3]))
        angular_accel += torque / np.asarray(self.config.nominal_inertia_kg_m2)
        angular_accel -= self.config.attitude_damping * omega
        angular_accel[2] -= self.config.yaw_drag * omega[2]
        dt = self.dt_seconds
        self.state.previous_acceleration_frd_m_s2 = self.state.body_acceleration_frd_m_s2.copy()
        self.state.body_acceleration_frd_m_s2 = accel
        self.state.body_velocity_frd_m_s += accel * dt
        self.state.body_velocity_frd_m_s = np.clip(
            self.state.body_velocity_frd_m_s,
            -self.config.max_speed_m_s,
            self.config.max_speed_m_s,
        )
        self.state.body_omega_frd_rad_s += angular_accel * dt
        self.state.body_omega_frd_rad_s = np.clip(
            self.state.body_omega_frd_rad_s,
            -self.config.max_angular_speed_rad_s,
            self.config.max_angular_speed_rad_s,
        )
        self.state.rotation_body_to_ned = self.state.rotation_body_to_ned @ _so3_exp(
            self.state.body_omega_frd_rad_s * dt
        )
        self.state.position_ned_m += rotation @ self.state.body_velocity_frd_m_s * dt
        if np.linalg.norm(self.state.position_ned_m) > self.config.max_position_m:
            self.state.position_ned_m *= self.config.max_position_m / np.linalg.norm(self.state.position_ned_m)
        self.state.depth_down_m = float(self.state.position_ned_m[2])
        self.time_seconds += dt
        self.last_command = command.copy()
        self.last_motor_command = target_motor.copy()
        return self.observation()

    def observation(self) -> dict[str, Any]:
        state = self.state
        return {
            "rotation6d": np.concatenate((state.rotation_body_to_ned[:, 0], state.rotation_body_to_ned[:, 1])).astype(np.float32),
            "body_velocity_frd_m_s": state.body_velocity_frd_m_s.astype(np.float32).copy(),
            "body_omega_frd_rad_s": state.body_omega_frd_rad_s.astype(np.float32).copy(),
            # The archive/runtime IMU field is specific force.  Keep the
            # dynamic acceleration available separately for diagnostics; the
            # JEPA loader performs the same gravity compensation as hardware.
            "body_acceleration_frd_m_s2": state.body_acceleration_frd_m_s2.astype(np.float32).copy(),
            "imu_acceleration_frd_m_s2": (
                state.body_acceleration_frd_m_s2
                - state.rotation_body_to_ned.T @ np.asarray((0.0, 0.0, 9.80665))
            ).astype(np.float32),
            "depth_down_m": state.depth_down_m,
            "position_ned_m": state.position_ned_m.astype(np.float32).copy(),
            "motor_output_normalized": state.motor_state.astype(np.float32).copy(),
            "modal_action": _motor_to_modal(self.last_motor_command).astype(np.float32),
            "time_seconds": self.time_seconds,
        }


def observation_vector(observation: dict[str, Any]) -> np.ndarray:
    """Convert proxy output into the 16D JEPA physical observation contract."""
    return np.concatenate(
        (
            np.asarray(observation["rotation6d"], dtype=np.float32),
            np.asarray(observation["body_velocity_frd_m_s"], dtype=np.float32),
            np.asarray(observation["body_omega_frd_rad_s"], dtype=np.float32),
            np.asarray(observation["imu_acceleration_frd_m_s2"], dtype=np.float32),
            np.asarray([observation["depth_down_m"]], dtype=np.float32),
        )
    ).astype(np.float32)


def modal_action_vector(observation: dict[str, Any]) -> np.ndarray:
    """Return the five-channel modal action used by the retained Stage 2."""
    motor = np.asarray(observation["motor_output_normalized"], dtype=np.float32)
    return _motor_to_modal(motor).astype(np.float32)


def _motor_to_modal(motor: np.ndarray) -> np.ndarray:
    motor = np.asarray(motor, dtype=np.float64)
    if motor.shape != (6,):
        raise ValueError("motor output must contain six values")
    m1, m2, m3, m4, m5, m6 = motor
    return np.asarray(
        ((-m1 - m2 + m3 + m4) / 4.0,
         (m1 - m2 + m3 - m4) / 4.0,
         -(m5 + m6) / 2.0,
         (m1 - m2 - m3 + m4) / 4.0,
         (m5 - m6) / 2.0), dtype=np.float32
    )
