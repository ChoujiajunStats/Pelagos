"""Nominal nonlinear MPC for BlueROV2 trajectory tracking.

The controller is intentionally independent of JEPA.  It is a benchmark and
data-collection controller whose model contains the nominal body-frame
velocity response, NED pose kinematics, yaw dynamics, and the six-thruster
MANUAL mixer.  A short direct multiple-shooting problem is solved with SciPy
SLSQP at each receding-horizon update.

The optimization variable is a sequence of normalized four-axis MANUAL
intents in ``[surge, sway, heave_up, yaw]`` order.  The plant state is
``[p_n, p_e, p_d, yaw, v_x, v_y, v_z, yaw_rate]``.  Body ``+z`` is down,
whereas MANUAL ``heave_up`` is positive up, so the heave input is negated in
the nominal dynamics model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from disaware_learning.control.body_twist_pid import quaternion_rpy, wrap_angle
from disaware_learning.estimation.vio_state import VioState


COMMAND_ORDER = ("surge", "sway", "heave_up", "yaw")
STATE_WIDTH = 8
COMMAND_WIDTH = 4
IMPLEMENTATION_ID = "nominal_bluerov2_nmpc_v1"
CAUSAL_SIMULATED_VIO_SOURCE = "simulated_stereo_vio_slam_from_gt"
_FORBIDDEN_RUNTIME_STATE_MARKERS = (
    "stonefish",
    "ground_truth",
    "truth",
    "simulation_truth",
    "raw_gt",
    "oracle",
    "physics_domain",
)


def _tuple(values: object, width: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {width} finite values") from error
    if len(result) != width or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {width} finite values")
    return result


def _wrap_array(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


@dataclass(frozen=True)
class NominalNmpcConfig:
    """Nominal model and solver settings.

    The first three ``control_gain`` values are steady-state body speed per
    unit normalized MANUAL command. The yaw value maps the mixer-level
    signed-square yaw wrench to steady-state yaw rate. Stonefish's fluid
    thrust model is quadratic, so a constant yaw-rate-per-command gain is not
    valid when surge and sway already load the horizontal thrusters.
    """

    sample_time_seconds: float = 0.10
    horizon_steps: int = 12
    control_hold_steps: int = 2
    # The Stonefish six-thruster nominal heave response is higher than the
    # initial model. Keep the default aligned with the calibrated config so
    # callers that do not load JSON get the same vertical scale.
    control_gain: tuple[float, ...] = (0.50, 0.455, 0.49, 7.10)
    time_constant_seconds: tuple[float, ...] = (2.0, 2.0, 2.5, 0.54)
    manual_limit: tuple[float, ...] = (0.48, 0.48, 0.45, 0.35)
    command_slew_per_second: tuple[float, ...] = (1.20, 1.20, 1.00, 0.10)
    position_weight: tuple[float, ...] = (8.0, 8.0, 10.0)
    position_scale_m: tuple[float, ...] = (0.50, 0.50, 0.30)
    velocity_weight: tuple[float, ...] = (3.0, 3.0, 4.0)
    velocity_scale_m_s: tuple[float, ...] = (0.15, 0.15, 0.12)
    yaw_weight: float = 5.0
    yaw_scale_rad: float = 0.35
    yaw_rate_weight: float = 2.0
    yaw_rate_scale_rad_s: float = 0.30
    command_weight: float = 0.04
    command_rate_weight: float = 0.25
    command_curvature_weight: float = 0.10
    motor_soft_limit: float = 0.90
    motor_hard_limit: float = 1.0
    motor_barrier_weight: float = 8.0
    terminal_weight: float = 2.0
    maximum_tilt_rad: float = math.radians(25.0)
    maximum_iterations: int = 80
    solver_ftol: float = 1e-5
    acceptance_tolerance: float = 1e-5
    fallback_on_solver_failure: bool = True

    def __post_init__(self) -> None:
        for name, width in (
            ("control_gain", 4),
            ("time_constant_seconds", 4),
            ("manual_limit", 4),
            ("command_slew_per_second", 4),
            ("position_weight", 3),
            ("position_scale_m", 3),
            ("velocity_weight", 3),
            ("velocity_scale_m_s", 3),
        ):
            values = _tuple(getattr(self, name), width, name)
            object.__setattr__(self, name, values)
        if any(value <= 0.0 for value in self.control_gain):
            raise ValueError("control_gain must be positive")
        if any(value <= 0.0 for value in self.time_constant_seconds):
            raise ValueError("time_constant_seconds must be positive")
        if any(value <= 0.0 or value > 1.0 for value in self.manual_limit):
            raise ValueError("manual_limit must be in (0, 1]")
        if any(value <= 0.0 for value in self.command_slew_per_second):
            raise ValueError("command_slew_per_second must be positive")
        if any(value <= 0.0 for value in self.position_scale_m):
            raise ValueError("position_scale_m must be positive")
        if any(value <= 0.0 for value in self.velocity_scale_m_s):
            raise ValueError("velocity_scale_m_s must be positive")
        for name in (
            "position_weight",
            "velocity_weight",
            "yaw_weight",
            "yaw_rate_weight",
            "command_weight",
            "command_rate_weight",
            "command_curvature_weight",
            "motor_barrier_weight",
            "terminal_weight",
            "acceptance_tolerance",
        ):
            value = getattr(self, name)
            if any(item < 0.0 for item in (value if isinstance(value, tuple) else (value,))):
                raise ValueError(f"{name} must be nonnegative")
        if (
            self.sample_time_seconds <= 0.0
            or self.horizon_steps < 2
            or self.control_hold_steps < 1
            or self.horizon_steps % self.control_hold_steps
        ):
            raise ValueError("invalid NMPC time horizon")
        if not 0.0 < self.motor_soft_limit <= self.motor_hard_limit <= 1.0:
            raise ValueError("motor limits must satisfy 0 < soft <= hard <= 1")
        if self.yaw_scale_rad <= 0.0 or self.yaw_rate_scale_rad_s <= 0.0:
            raise ValueError("yaw tracking scales must be positive")
        if not 0.0 < self.maximum_tilt_rad < math.pi / 2.0:
            raise ValueError("maximum_tilt_rad is invalid")
        if self.maximum_iterations < 1 or self.solver_ftol <= 0.0:
            raise ValueError("invalid solver settings")
        for name in (
            "sample_time_seconds",
            "yaw_weight",
            "yaw_scale_rad",
            "yaw_rate_weight",
            "yaw_rate_scale_rad_s",
            "command_weight",
            "command_rate_weight",
            "command_curvature_weight",
            "motor_soft_limit",
            "motor_hard_limit",
            "motor_barrier_weight",
            "terminal_weight",
            "maximum_tilt_rad",
            "solver_ftol",
            "acceptance_tolerance",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")

    def as_dict(self) -> dict[str, Any]:
        return {"implementation_id": IMPLEMENTATION_ID, **asdict(self)}


def load_nominal_nmpc_config(path: str | Path) -> NominalNmpcConfig:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("nominal NMPC config must contain a JSON object")
    implementation = payload.pop("implementation_id", IMPLEMENTATION_ID)
    if implementation != IMPLEMENTATION_ID:
        raise ValueError("nominal NMPC implementation_id is incompatible")
    unknown = set(payload) - set(NominalNmpcConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown nominal NMPC config keys: {sorted(unknown)}")
    return NominalNmpcConfig(**payload)


@dataclass(frozen=True)
class NmpcReference:
    """Reference values at the current state and every future shooting node."""

    position_ned_m: np.ndarray
    body_velocity_frd_m_s: np.ndarray
    yaw_rad: np.ndarray
    yaw_rate_rad_s: np.ndarray
    # This is a command/planner reference, never a measured future state.
    source: str = "planner_reference"

    @classmethod
    def from_arrays(
        cls,
        position_ned_m: np.ndarray,
        body_velocity_frd_m_s: np.ndarray,
        yaw_rad: np.ndarray,
        yaw_rate_rad_s: np.ndarray,
        *,
        source: str = "planner_reference",
    ) -> "NmpcReference":
        position = np.asarray(position_ned_m, dtype=np.float64)
        velocity = np.asarray(body_velocity_frd_m_s, dtype=np.float64)
        yaw = np.asarray(yaw_rad, dtype=np.float64)
        yaw_rate = np.asarray(yaw_rate_rad_s, dtype=np.float64)
        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError("position_ned_m must be [nodes, 3]")
        nodes = position.shape[0]
        if (
            velocity.shape != (nodes, 3)
            or yaw.shape != (nodes,)
            or yaw_rate.shape != (nodes,)
            or nodes < 2
            or not all(np.all(np.isfinite(value)) for value in (position, velocity, yaw, yaw_rate))
        ):
            raise ValueError("NMPC reference arrays have invalid shape or values")
        source_value = str(source).strip()
        if not source_value:
            raise ValueError("NMPC reference source is required")
        return cls(position.copy(), velocity.copy(), yaw.copy(), yaw_rate.copy(), source_value)

    @classmethod
    def body_twist(
        cls,
        body_twist_reference: np.ndarray,
        position_ned_m: np.ndarray,
        yaw_preview_rad: np.ndarray,
        *,
        source: str = "planner_reference",
    ) -> "NmpcReference":
        twist = np.asarray(body_twist_reference, dtype=np.float64)
        if twist.ndim != 2 or twist.shape[1] != 4:
            raise ValueError("body_twist_reference must be [nodes, 4]")
        return cls.from_arrays(
            position_ned_m,
            twist[:, :3],
            yaw_preview_rad,
            twist[:, 3],
            source=source,
        )

    def window(self, start: int, nodes: int) -> "NmpcReference":
        """Return a padded horizon window for an offline reference bank."""
        if start < 0 or nodes < 2:
            raise ValueError("invalid reference window")
        indices = np.minimum(np.arange(start, start + nodes), len(self.yaw_rad) - 1)
        return NmpcReference.from_arrays(
            self.position_ned_m[indices],
            self.body_velocity_frd_m_s[indices],
            self.yaw_rad[indices],
            self.yaw_rate_rad_s[indices],
            source=self.source,
        )


@dataclass(frozen=True)
class NmpcResult:
    command: np.ndarray
    planned_command: np.ndarray
    predicted_state: np.ndarray
    accepted: bool
    reason: str
    diagnostics: dict[str, Any]


def _state_from_vio(state: VioState) -> np.ndarray:
    roll, pitch, yaw = quaternion_rpy(state.orientation_body_to_ned_xyzw)
    del roll, pitch
    return np.concatenate(
        (
            state.position_ned_m,
            np.asarray((yaw,), dtype=np.float64),
            state.linear_velocity_body_frd_m_s,
            np.asarray((state.angular_velocity_body_frd_rad_s[2],), dtype=np.float64),
        )
    )


class NominalNmpc:
    """Short-horizon nominal nonlinear model predictive controller."""

    def __init__(self, config: NominalNmpcConfig = NominalNmpcConfig()) -> None:
        self.config = config
        self._previous_plan = np.zeros((config.horizon_steps, COMMAND_WIDTH), dtype=np.float64)

    @property
    def knot_count(self) -> int:
        return self.config.horizon_steps // self.config.control_hold_steps

    @property
    def control_update_period_seconds(self) -> float:
        return self.config.sample_time_seconds * self.config.control_hold_steps

    def reset(self, command: np.ndarray | None = None) -> None:
        values = np.zeros(COMMAND_WIDTH, dtype=np.float64) if command is None else np.asarray(command, dtype=np.float64)
        if values.shape != (COMMAND_WIDTH,) or not np.all(np.isfinite(values)):
            raise ValueError("reset command must contain four finite values")
        self._previous_plan[:] = np.clip(values, -np.asarray(self.config.manual_limit), np.asarray(self.config.manual_limit))

    def _control_target_response(self, command: np.ndarray) -> np.ndarray:
        gain = np.asarray(self.config.control_gain, dtype=np.float64)
        horizontal_motor = self._mixer_raw(command)[:4]
        yaw_sign = np.asarray((1.0, -1.0, -1.0, 1.0), dtype=np.float64)
        yaw_wrench = float(
            np.mean(
                yaw_sign
                * np.sign(horizontal_motor)
                * np.square(horizontal_motor)
            )
        )
        # Body FRD +z is down; MANUAL throttle/heave_up is positive up.
        return np.asarray(
            (
                gain[0] * command[0],
                gain[1] * command[1],
                -gain[2] * command[2],
                gain[3] * yaw_wrench,
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _mixer_raw(command: np.ndarray) -> np.ndarray:
        surge, sway, heave, yaw = np.asarray(command, dtype=np.float64).T
        return np.stack(
            (
                yaw - surge + sway,
                -yaw - surge - sway,
                -yaw + surge + sway,
                yaw + surge - sway,
                -heave,
                -heave,
            ),
            axis=-1,
        )

    def envelope_saturation_metrics(
        self, command_plan: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Return explicit command and motor envelope occupancy metrics.

        The public MANUAL saturation gate only sees the final six-channel
        command.  NMPC also has a tighter per-axis command envelope and a
        soft motor envelope, so those must be audited independently rather
        than collapsed into the global MANUAL flag.
        """
        commands = np.asarray(command_plan, dtype=np.float64)
        if commands.ndim == 1:
            commands = commands[None, :]
        if commands.ndim != 2 or commands.shape[1] != COMMAND_WIDTH:
            raise ValueError("command_plan must be [N, 4]")
        if not np.all(np.isfinite(commands)):
            raise ValueError("command_plan must be finite")
        limits = np.asarray(self.config.manual_limit, dtype=np.float64)
        raw_motor = self._mixer_raw(commands)
        tolerance = 1e-7
        return {
            "command_envelope_saturation_fraction": np.mean(
                np.abs(commands) >= limits[None, :] - tolerance, axis=0
            ),
            "motor_soft_envelope_saturation_fraction": np.mean(
                np.abs(raw_motor) >= self.config.motor_soft_limit - tolerance, axis=0
            ),
            "motor_hard_envelope_saturation_fraction": np.mean(
                np.abs(raw_motor) >= self.config.motor_hard_limit - tolerance, axis=0
            ),
            "maximum_raw_motor": np.asarray([np.max(np.abs(raw_motor))]),
        }

    def rollout(
        self,
        state: np.ndarray,
        command_plan: np.ndarray,
    ) -> np.ndarray:
        state_value = np.asarray(state, dtype=np.float64)
        commands = np.asarray(command_plan, dtype=np.float64)
        if state_value.shape != (STATE_WIDTH,) or commands.shape != (self.config.horizon_steps, COMMAND_WIDTH):
            raise ValueError("invalid state or command plan shape")
        dt = self.config.sample_time_seconds
        tau = np.asarray(self.config.time_constant_seconds, dtype=np.float64)
        output = np.empty((self.config.horizon_steps + 1, STATE_WIDTH), dtype=np.float64)
        output[0] = state_value
        for index, command in enumerate(commands):
            previous = output[index]
            yaw = previous[3]
            body_velocity = previous[4:7]
            angular_rate = previous[7]
            target_response = self._control_target_response(command)
            c = math.cos(yaw)
            s = math.sin(yaw)
            next_velocity = body_velocity + dt * (
                target_response[:3] - body_velocity
            ) / tau[:3]
            next_rate = angular_rate + dt * (
                target_response[3] - angular_rate
            ) / tau[3]
            world_velocity = np.asarray(
                (c * body_velocity[0] - s * body_velocity[1], s * body_velocity[0] + c * body_velocity[1], body_velocity[2]),
                dtype=np.float64,
            )
            next_state = previous.copy()
            next_state[:3] = previous[:3] + dt * world_velocity
            next_state[3] = wrap_angle(yaw + dt * angular_rate)
            next_state[4:7] = next_velocity
            next_state[7] = next_rate
            output[index + 1] = next_state
        return output

    def _expand_knots(self, knots: np.ndarray) -> np.ndarray:
        values = np.asarray(knots, dtype=np.float64)
        if values.shape != (self.knot_count, COMMAND_WIDTH):
            raise ValueError("NMPC knots have an invalid shape")
        return np.repeat(values, self.config.control_hold_steps, axis=0)

    def _project_knots(self, knots: np.ndarray, previous_command: np.ndarray) -> np.ndarray:
        limits = np.asarray(self.config.manual_limit, dtype=np.float64)
        slew = (
            np.asarray(self.config.command_slew_per_second, dtype=np.float64)
            * self.control_update_period_seconds
        )
        output = np.empty_like(knots)
        prior = np.asarray(previous_command, dtype=np.float64)
        for index, target in enumerate(knots):
            bounded = np.clip(target, -limits, limits)
            desired = prior + np.clip(bounded - prior, -slew, slew)
            if np.max(np.abs(self._mixer_raw(desired))) > self.config.motor_hard_limit:
                lower, upper = 0.0, 1.0
                for _ in range(24):
                    middle = 0.5 * (lower + upper)
                    candidate = prior + middle * (desired - prior)
                    if np.max(np.abs(self._mixer_raw(candidate))) <= self.config.motor_hard_limit:
                        lower = middle
                    else:
                        upper = middle
                desired = prior + lower * (desired - prior)
            output[index] = desired
            prior = output[index]
        return output

    def _objective(
        self,
        flat_plan: np.ndarray,
        state: np.ndarray,
        reference: NmpcReference,
        previous_command: np.ndarray,
    ) -> float:
        knots = flat_plan.reshape(self.knot_count, COMMAND_WIDTH)
        plan = self._expand_knots(knots)
        predicted = self.rollout(state, plan)
        cfg = self.config
        pos_error = predicted[1:, :3] - reference.position_ned_m[1:]
        velocity_error = predicted[1:, 4:7] - reference.body_velocity_frd_m_s[1:]
        yaw_error = _wrap_array(predicted[1:, 3] - reference.yaw_rad[1:])
        yaw_rate_error = predicted[1:, 7] - reference.yaw_rate_rad_s[1:]
        p_scale = np.asarray(cfg.position_scale_m, dtype=np.float64)
        v_scale = np.asarray(cfg.velocity_scale_m_s, dtype=np.float64)
        position_cost = np.mean((pos_error / p_scale) ** 2 @ np.asarray(cfg.position_weight))
        velocity_cost = np.mean((velocity_error / v_scale) ** 2 @ np.asarray(cfg.velocity_weight))
        yaw_cost = cfg.yaw_weight * float(np.mean((yaw_error / cfg.yaw_scale_rad) ** 2))
        yaw_rate_cost = cfg.yaw_rate_weight * float(
            np.mean((yaw_rate_error / cfg.yaw_rate_scale_rad_s) ** 2)
        )
        command_cost = cfg.command_weight * float(np.mean(plan**2))
        command_rate = np.diff(np.vstack((previous_command, plan)), axis=0) / cfg.sample_time_seconds
        rate_cost = cfg.command_rate_weight * float(np.mean(command_rate**2))
        curvature = np.diff(plan, n=2, axis=0)
        curvature_cost = cfg.command_curvature_weight * float(np.mean(curvature**2))
        raw_motor = self._mixer_raw(plan)
        excess = np.maximum(np.abs(raw_motor) - cfg.motor_soft_limit, 0.0)
        motor_cost = cfg.motor_barrier_weight * float(np.mean(excess**2))
        terminal = cfg.terminal_weight * float(
            np.mean((pos_error[-1] / p_scale) ** 2 @ np.asarray(cfg.position_weight))
            + np.mean((velocity_error[-1] / v_scale) ** 2 @ np.asarray(cfg.velocity_weight))
            + cfg.yaw_weight * (yaw_error[-1] / cfg.yaw_scale_rad) ** 2
        )
        return float(position_cost + velocity_cost + yaw_cost + yaw_rate_cost + command_cost + rate_cost + curvature_cost + motor_cost + terminal)

    def _constraints(self, previous_command: np.ndarray) -> tuple[LinearConstraint, LinearConstraint]:
        horizon = self.knot_count
        width = COMMAND_WIDTH
        matrix = np.zeros(((horizon) * width, horizon * width), dtype=np.float64)
        for index in range(horizon):
            for axis in range(width):
                row = index * width + axis
                matrix[row, index * width + axis] = 1.0
                if index:
                    matrix[row, (index - 1) * width + axis] = -1.0
        lower = (
            -np.asarray(self.config.command_slew_per_second)
            * self.control_update_period_seconds
        )
        upper = -lower
        lower_all = np.tile(lower, horizon)
        upper_all = np.tile(upper, horizon)
        if np.any(np.abs(previous_command) > np.asarray(self.config.manual_limit) + 1e-9):
            raise ValueError("previous command exceeds NMPC bounds")
        # The first difference is u_0 - previous_command, represented by a
        # constant shift in the lower/upper constraint bounds.
        lower_all[:width] += previous_command
        upper_all[:width] += previous_command
        slew_constraint = LinearConstraint(matrix, lower_all, upper_all)
        mixer = np.asarray(
            (
                (-1.0, 1.0, 0.0, 1.0),
                (-1.0, -1.0, 0.0, -1.0),
                (1.0, 1.0, 0.0, -1.0),
                (1.0, -1.0, 0.0, 1.0),
                (0.0, 0.0, -1.0, 0.0),
                (0.0, 0.0, -1.0, 0.0),
            ),
            dtype=np.float64,
        )
        mixer_matrix = np.zeros((horizon * 6, horizon * width), dtype=np.float64)
        for index in range(horizon):
            mixer_matrix[index * 6 : (index + 1) * 6, index * width : (index + 1) * width] = mixer
        motor_limit = np.full(horizon * 6, self.config.motor_hard_limit, dtype=np.float64)
        mixer_constraint = LinearConstraint(mixer_matrix, -motor_limit, motor_limit)
        return slew_constraint, mixer_constraint

    def optimize(
        self,
        state: VioState,
        reference: NmpcReference,
        *,
        previous_command: np.ndarray | None = None,
    ) -> NmpcResult:
        started = time.perf_counter()
        if not isinstance(state, VioState):
            raise TypeError("runtime NMPC state must use the deployable VioState boundary")
        if not isinstance(reference, NmpcReference):
            raise TypeError("runtime NMPC reference must use NmpcReference")
        source_lower = state.source.casefold()
        if (
            state.source != CAUSAL_SIMULATED_VIO_SOURCE
            and any(marker in source_lower for marker in _FORBIDDEN_RUNTIME_STATE_MARKERS)
        ):
            raise ValueError(f"NMPC runtime state source is privileged: {state.source}")
        reference_source_lower = reference.source.casefold()
        if any(marker in reference_source_lower for marker in _FORBIDDEN_RUNTIME_STATE_MARKERS):
            raise ValueError(
                f"NMPC reference source is privileged: {reference.source}"
            )
        state_vector = _state_from_vio(state)
        if state_vector.shape != (STATE_WIDTH,) or not np.all(np.isfinite(state_vector)):
            raise ValueError("state must be a finite NMPC state")
        if len(reference.yaw_rad) != self.config.horizon_steps + 1:
            raise ValueError("reference must contain horizon_steps + 1 nodes")
        prior = self._previous_plan[-1] if previous_command is None else np.asarray(previous_command, dtype=np.float64)
        if prior.shape != (COMMAND_WIDTH,) or not np.all(np.isfinite(prior)):
            raise ValueError("previous_command must be a finite four-vector")
        limits = np.asarray(self.config.manual_limit, dtype=np.float64)
        if np.any(np.abs(prior) > limits + 1e-8):
            raise ValueError("previous_command exceeds manual limits")
        previous_motor_peak = float(np.max(np.abs(self._mixer_raw(prior))))
        previous_command_projected = False
        if previous_motor_peak > self.config.motor_hard_limit:
            # The external MANUAL slew limiter can introduce a tiny numerical
            # mixer overshoot between two individually feasible NMPC commands.
            # Project the model's starting point back into the hard envelope
            # instead of terminating the rollout.
            prior = prior * (self.config.motor_hard_limit / previous_motor_peak)
            previous_command_projected = True
        roll, pitch, _ = quaternion_rpy(state.orientation_body_to_ned_xyzw)
        tilt = math.hypot(roll, pitch)
        if tilt > self.config.maximum_tilt_rad:
            neutral_knots = self._project_knots(
                np.zeros((self.knot_count, COMMAND_WIDTH), dtype=np.float64), prior
            )
            neutral_plan = self._expand_knots(neutral_knots)
            prediction = self.rollout(state_vector, neutral_plan)
            self._previous_plan[:] = neutral_plan
            return NmpcResult(
                command=neutral_plan[0].copy(),
                planned_command=neutral_plan,
                predicted_state=prediction,
                accepted=False,
                reason="tilt_gate_fallback",
                diagnostics={
                    "implementation_id": IMPLEMENTATION_ID,
                    "solver": "not_run",
                    "solver_success": False,
                    "tilt_rad": tilt,
                    "maximum_tilt_rad": self.config.maximum_tilt_rad,
                    "solve_time_seconds": float(time.perf_counter() - started),
                    "state_source": state.source,
                    "reference_source": reference.source,
                },
            )
        seed_knots = self._project_knots(
            self._previous_plan[:: self.config.control_hold_steps], prior
        )
        if np.allclose(seed_knots, 0.0):
            yaw_wrench = (
                reference.yaw_rate_rad_s[1] / self.config.control_gain[3]
            )
            desired = np.asarray(
                (
                    reference.body_velocity_frd_m_s[1, 0] / self.config.control_gain[0],
                    reference.body_velocity_frd_m_s[1, 1] / self.config.control_gain[1],
                    -reference.body_velocity_frd_m_s[1, 2] / self.config.control_gain[2],
                    math.copysign(math.sqrt(abs(yaw_wrench)), yaw_wrench),
                ),
                dtype=np.float64,
            )
            seed_knots = self._project_knots(
                np.repeat(desired[None, :], self.knot_count, axis=0), prior
            )
        seed_knots = np.asarray(seed_knots, dtype=np.float64)
        bounds = [
            (-limit, limit)
            for _ in range(self.knot_count)
            for limit in limits
        ]
        baseline_cost = self._objective(
            seed_knots.reshape(-1), state_vector, reference, prior
        )
        result = minimize(
            self._objective,
            seed_knots.reshape(-1),
            args=(state_vector, reference, prior),
            method="SLSQP",
            bounds=bounds,
            constraints=self._constraints(prior),
            options={"maxiter": self.config.maximum_iterations, "ftol": self.config.solver_ftol, "disp": False},
        )
        solver_knots = (
            np.asarray(result.x, dtype=np.float64).reshape(self.knot_count, COMMAND_WIDTH)
            if np.all(np.isfinite(result.x))
            else seed_knots
        )
        solver_knots = self._project_knots(solver_knots, prior)
        solver_cost = self._objective(
            solver_knots.reshape(-1), state_vector, reference, prior
        )
        solver_ok = bool(result.success and np.isfinite(solver_cost))
        accepted = solver_ok and solver_cost < baseline_cost - self.config.acceptance_tolerance
        if not solver_ok and not self.config.fallback_on_solver_failure:
            raise RuntimeError(f"NMPC solver failed: {result.message}")
        selected_knots = solver_knots if accepted else seed_knots
        selected_plan = self._expand_knots(selected_knots)
        selected_cost = solver_cost if accepted else baseline_cost
        selected_prediction = self.rollout(state_vector, selected_plan)
        shift = self.config.control_hold_steps
        self._previous_plan[:-shift] = selected_plan[shift:]
        self._previous_plan[-shift:] = selected_plan[-1]
        reason = "accepted" if accepted else ("solver_failed_fallback" if not solver_ok else "baseline_not_worse")
        diagnostics = {
            "implementation_id": IMPLEMENTATION_ID,
            "solver": "scipy_slsqp",
            "solver_success": solver_ok,
            "solver_message": str(result.message),
            "solver_iterations": int(getattr(result, "nit", 0)),
            "baseline_cost": float(baseline_cost),
            "solver_cost": float(solver_cost),
            "selected_cost": float(selected_cost),
            "predicted_improvement": float(baseline_cost - selected_cost),
            "maximum_raw_motor": float(np.max(np.abs(self._mixer_raw(selected_plan)))),
            "maximum_command": float(np.max(np.abs(selected_plan))),
            "control_knot_count": self.knot_count,
            "control_update_period_seconds": self.control_update_period_seconds,
            "solve_time_seconds": float(time.perf_counter() - started),
            "state_source": state.source,
            "reference_source": reference.source,
            "previous_command_projected": previous_command_projected,
            "privileged_state_rejected": True,
        }
        diagnostics.update(
            {
                name: values.tolist()
                for name, values in self.envelope_saturation_metrics(
                    selected_plan
                ).items()
                if name != "maximum_raw_motor"
            }
        )
        return NmpcResult(
            command=selected_plan[0].copy(),
            planned_command=selected_plan.copy(),
            predicted_state=selected_prediction,
            accepted=accepted,
            reason=reason,
            diagnostics=diagnostics,
        )


__all__ = [
    "COMMAND_ORDER",
    "IMPLEMENTATION_ID",
    "NmpcReference",
    "NmpcResult",
    "NominalNmpc",
    "NominalNmpcConfig",
    "load_nominal_nmpc_config",
]
