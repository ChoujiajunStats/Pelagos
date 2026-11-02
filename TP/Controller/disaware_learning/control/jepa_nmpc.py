"""Differentiable JEPA nonlinear MPC for BlueROV2 trajectory tracking.

The controller optimizes complete four-axis MANUAL command knots.  Commands
are mapped through the six-thruster allocation into the five actuator-modal
coordinates used to train Stage 2.  The frozen history encoder is evaluated
once per control tick; gradients flow through the actuator model, nominal
hydrodynamics, learned context-conditioned residual, and physical rollout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
from scipy.optimize import LinearConstraint, minimize

try:
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - optional controller dependency.
    raise ImportError("JEPA-NMPC requires PyTorch") from error

from disaware_learning.control.jepa_mppi import (
    JepaMppiConfig,
    ReferenceConditionedJepaMppi,
    ReferenceTrajectory,
    standard_bluerov2_manual_mixer,
)
from disaware_learning.learning.motor_jepa_data import (
    FEATURE_CONTRACT_ID,
    Standardizer,
)
from disaware_learning.learning.stage2 import (
    IMPLEMENTATION as STAGE2_IMPLEMENTATION,
    load_stage2_checkpoint,
)

IMPLEMENTATION_ID = "context_hybrid_jepa_nmpc_stage2_modal4d_v2"
COMMAND_ORDER = ("surge", "sway", "heave_up", "yaw")


@dataclass(frozen=True)
class JepaNmpcConfig(JepaMppiConfig):
    """JEPA trajectory cost, physical envelopes, and deterministic solver."""

    # Sampling-only inherited fields are kept at their minimum valid values;
    # JepaNmpc never reads them during optimization.
    sample_count: int = 8
    target_effective_sample_size: float = 8.0
    iterations: int = 1
    command_hold_steps: int = 3
    # Use the complete three-second Stage-2 rollout.  The previous 1.2 s
    # decision window ended before the identified translational time constants
    # (roughly 2-3 s), so the optimizer could not act on delayed effects.
    optimization_horizon_seconds: float = 3.0
    deadline_seconds: float = 0.18

    motor_hard_limit: float = 1.0
    # Three SQP iterations capture most of the local improvement on the real
    # checkpoint while keeping a warm GPU call below the 5 Hz planning tick.
    maximum_iterations: int = 3
    maximum_objective_evaluations: int = 2
    solver_ftol: float = 1e-4
    acceptance_tolerance: float = 1e-5
    constraint_tolerance: float = 2e-5
    accept_best_feasible_on_solver_failure: bool = True
    # Strict model ablation: retain the same NMPC problem while disabling only
    # the learned Stage-2 residual on top of nominal hydrodynamics.
    include_learned_residual: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.motor_hard_limit <= 1.0:
            raise ValueError("motor_hard_limit must be in (0, 1]")
        if self.motor_soft_limit > self.motor_hard_limit:
            raise ValueError("motor soft limit cannot exceed hard limit")
        if self.maximum_iterations < 1 or self.maximum_objective_evaluations < 1:
            raise ValueError("solver iteration/evaluation budgets must be positive")
        for name in ("solver_ftol", "constraint_tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not math.isfinite(self.acceptance_tolerance)
            or self.acceptance_tolerance < 0.0
        ):
            raise ValueError("acceptance_tolerance must be finite and nonnegative")

    def as_dict(self) -> dict[str, Any]:
        return {"implementation_id": IMPLEMENTATION_ID, **asdict(self)}


def load_jepa_nmpc_config(path: str | Path) -> JepaNmpcConfig:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("JEPA-NMPC config must contain a JSON object")
    payload = dict(payload)
    implementation = payload.pop("implementation_id", IMPLEMENTATION_ID)
    if implementation != IMPLEMENTATION_ID:
        raise ValueError(f"unexpected JEPA-NMPC implementation: {implementation}")
    expected = set(JepaNmpcConfig.__dataclass_fields__)
    unknown = set(payload) - expected
    if unknown:
        raise ValueError(f"unknown JEPA-NMPC config keys: {sorted(unknown)}")
    return JepaNmpcConfig(**payload)


@dataclass(frozen=True)
class JepaNmpcResult:
    command: np.ndarray
    planned_command: np.ndarray
    nominal_command: np.ndarray
    predicted_physical_observation: np.ndarray
    predicted_position_delta_ned_m: np.ndarray | None
    accepted: bool
    reason: str
    diagnostics: dict[str, Any]


class _DeadlineExceeded(RuntimeError):
    pass


class _EvaluationBudgetExceeded(RuntimeError):
    pass


class JepaNmpc(ReferenceConditionedJepaMppi):
    """Single-shooting nonlinear MPC through the retained Stage-2 model."""

    def __init__(
        self,
        model: nn.Module,
        standardizer: Standardizer,
        *,
        history_steps: int,
        model_horizon_steps: int,
        sample_rate_hz: float,
        config: JepaNmpcConfig = JepaNmpcConfig(),
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            model,
            standardizer,
            history_steps=history_steps,
            model_horizon_steps=model_horizon_steps,
            sample_rate_hz=sample_rate_hz,
            config=config,
            device=device,
        )
        if model_horizon_steps % config.command_hold_steps:
            raise ValueError(
                "JEPA-NMPC model horizon must be divisible by command_hold_steps"
            )
        if self.optimization_knot_count > self.knot_count:
            raise ValueError("optimization horizon exceeds the model horizon")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        config: JepaNmpcConfig = JepaNmpcConfig(),
        device: str | torch.device = "cpu",
    ) -> "JepaNmpc":
        checkpoint_path = Path(checkpoint_path).expanduser()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("feature_contract_id") != FEATURE_CONTRACT_ID:
            raise ValueError("checkpoint does not use the retained feature contract")
        if checkpoint.get("implementation") != STAGE2_IMPLEMENTATION:
            raise ValueError(
                "checkpoint is not the retained physics-integrated Stage 2"
            )
        readiness = checkpoint.get("training_readiness")
        if not isinstance(readiness, Mapping) or readiness.get("passed") is not True:
            raise ValueError("checkpoint failed Stage-2 training readiness")
        model, checkpoint = load_stage2_checkpoint(checkpoint_path, device=device)
        return cls(
            model,
            Standardizer.from_dict(checkpoint["standardizer"]),
            history_steps=int(checkpoint["history_steps"]),
            model_horizon_steps=int(checkpoint["horizon_steps"]),
            sample_rate_hz=float(checkpoint["sample_rate_hz"]),
            config=config,
            device=device,
        )

    def _expand_decision(self, decision: Tensor) -> Tensor:
        if decision.shape != (self.optimization_knot_count, 4):
            raise ValueError("JEPA-NMPC decision has an invalid shape")
        if self.optimization_knot_count == self.knot_count:
            return decision
        tail = decision[-1:].expand(self.knot_count - self.optimization_knot_count, -1)
        return torch.cat((decision, tail), dim=0)

    def _decision_selection_matrix(self) -> np.ndarray:
        selection = np.zeros(
            (self.knot_count * 4, self.optimization_knot_count * 4),
            dtype=np.float64,
        )
        for knot in range(self.knot_count):
            source = min(knot, self.optimization_knot_count - 1)
            selection[knot * 4 : (knot + 1) * 4, source * 4 : (source + 1) * 4] = (
                np.eye(4)
            )
        return selection

    def _decision_bounds(self) -> list[tuple[float, float]]:
        lower = -np.asarray(self.config.manual_limit, dtype=np.float64)
        upper = np.asarray(self.config.manual_limit, dtype=np.float64)
        if self.action_dim == 5:
            trust = np.asarray(self.config.modal_action_trust_region_std)
            modal_lower = (
                self._action_mean.detach().cpu().numpy()
                - trust * self._action_std.detach().cpu().numpy()
            )
            modal_upper = (
                self._action_mean.detach().cpu().numpy()
                + trust * self._action_std.detach().cpu().numpy()
            )
            lower = np.maximum(lower, modal_lower[:4])
            upper = np.minimum(upper, modal_upper[:4])
        if np.any(lower >= upper):
            raise ValueError("JEPA model support does not overlap command bounds")
        return [
            (float(lower[axis]), float(upper[axis]))
            for _ in range(self.optimization_knot_count)
            for axis in range(4)
        ]

    def _linear_constraints(
        self,
        previous_command: np.ndarray,
        roll_plan: np.ndarray,
    ) -> tuple[LinearConstraint, LinearConstraint]:
        selection = self._decision_selection_matrix()
        difference = np.zeros((self.knot_count * 4, self.knot_count * 4))
        for knot in range(self.knot_count):
            difference[knot * 4 : (knot + 1) * 4, knot * 4 : (knot + 1) * 4] = np.eye(4)
            if knot:
                difference[
                    knot * 4 : (knot + 1) * 4,
                    (knot - 1) * 4 : knot * 4,
                ] = -np.eye(4)
        durations = np.full(
            self.knot_count,
            self.config.command_hold_steps / self.sample_rate_hz,
            dtype=np.float64,
        )
        durations[0] = self.config.execution_hold_steps / self.sample_rate_hz
        maximum = (
            durations[:, None]
            * np.asarray(self.config.command_slew_per_second, dtype=np.float64)[None]
        )
        slew_lower = -maximum.reshape(-1)
        slew_upper = maximum.reshape(-1)
        slew_lower[:4] += previous_command
        slew_upper[:4] += previous_command
        slew_constraint = LinearConstraint(
            difference @ selection, slew_lower, slew_upper
        )

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
        allocation = np.zeros((self.knot_count * 6, self.knot_count * 4))
        offset = np.zeros((self.knot_count, 6), dtype=np.float64)
        offset[:, 4] = roll_plan
        offset[:, 5] = -roll_plan
        for knot in range(self.knot_count):
            allocation[
                knot * 6 : (knot + 1) * 6,
                knot * 4 : (knot + 1) * 4,
            ] = mixer
        hard = float(self.config.motor_hard_limit)
        motor_constraint = LinearConstraint(
            allocation @ selection,
            (-hard - offset).reshape(-1),
            (hard - offset).reshape(-1),
        )
        return slew_constraint, motor_constraint

    def _is_feasible(
        self,
        plan: Tensor,
        previous_command: Tensor,
        roll_plan: Tensor,
    ) -> bool:
        tolerance = self.config.constraint_tolerance
        limits = torch.as_tensor(self.config.manual_limit, device=self.device)
        if bool((plan.abs() > limits + tolerance).any()):
            return False
        prior = torch.cat((previous_command.view(1, 4), plan[:-1]), dim=0)
        durations = plan.new_full(
            (self.knot_count,),
            self.config.command_hold_steps / self.sample_rate_hz,
        )
        durations[0] = self.config.execution_hold_steps / self.sample_rate_hz
        slew = torch.as_tensor(self.config.command_slew_per_second, device=self.device)
        if bool(((plan - prior).abs() > durations[:, None] * slew + tolerance).any()):
            return False
        _motor, raw = standard_bluerov2_manual_mixer(plan, roll_plan)
        if bool((raw.abs() > self.config.motor_hard_limit + tolerance).any()):
            return False
        if self.action_dim == 5:
            modal = torch.cat((plan, roll_plan.unsqueeze(-1)), dim=-1)
            trust = torch.as_tensor(
                self.config.modal_action_trust_region_std, device=self.device
            )
            lower = self._action_mean - trust * self._action_std
            upper = self._action_mean + trust * self._action_std
            if bool(((modal < lower - tolerance) | (modal > upper + tolerance)).any()):
                return False
        return bool(torch.isfinite(plan).all())

    def warmup(
        self,
        observation_history: np.ndarray,
        action_history: np.ndarray,
        previous_command: np.ndarray,
        reference: ReferenceTrajectory | None = None,
        current_position_ned_m: np.ndarray | None = None,
        roll_command: float | np.ndarray = 0.0,
    ) -> float:
        """Warm the exact forward/backward path used by deterministic NMPC."""
        started = time.perf_counter()
        previous = np.asarray(previous_command, dtype=np.float32)
        self.reset(previous)
        self.optimize(
            observation_history,
            action_history,
            (
                reference
                if reference is not None
                else ReferenceTrajectory.body_twist(np.zeros(4, dtype=np.float32))
            ),
            previous,
            current_position_ned_m=current_position_ned_m,
            roll_command=roll_command,
        )
        # Warmup must not seed deployment with a solution computed from the
        # repeated synthetic history used by the rollout adapter.
        self.reset(previous)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - started

    def optimize(
        self,
        observation_history: np.ndarray,
        action_history: np.ndarray,
        reference: ReferenceTrajectory,
        previous_command: np.ndarray,
        *,
        current_position_ned_m: np.ndarray | None = None,
        roll_command: float | np.ndarray = 0.0,
    ) -> JepaNmpcResult:
        started = time.perf_counter()
        previous_values = np.asarray(previous_command, dtype=np.float32)
        if previous_values.shape != (4,) or not np.all(np.isfinite(previous_values)):
            raise ValueError("previous command must contain four finite axes")
        previous = torch.as_tensor(previous_values, device=self.device)
        observation, action, current = self._prepare_history(
            observation_history, action_history
        )
        prepared_reference = self._prepare_reference(reference)
        roll_plan = self._prepare_roll_plan(roll_command)
        roll_values = roll_plan.detach().cpu().numpy().astype(np.float64)
        if current_position_ned_m is None:
            if "position" in prepared_reference:
                raise ValueError("position tracking requires the current NED position")
            position = torch.zeros(3, dtype=torch.float32, device=self.device)
        else:
            position_values = np.asarray(current_position_ned_m, dtype=np.float32)
            if position_values.shape != (3,) or not np.all(
                np.isfinite(position_values)
            ):
                raise ValueError(
                    "current NED position must contain three finite values"
                )
            position = torch.as_tensor(position_values, device=self.device)
        position = position.view(1, 1, 3)

        previous_model_action = action[-1] * self._action_std + self._action_mean
        previous_modal_action = (
            previous_model_action if self.action_dim == 5 else previous
        )
        fallback_full = self._initial_plan(previous, previous_modal_action)
        seed_full = self._reference_action_seed(
            prepared_reference,
            fallback_full,
            previous,
            previous_modal_action,
        )
        seed_decision = seed_full[: self.optimization_knot_count].clone()
        seed_full = self._expand_decision(seed_decision)
        if not self._is_feasible(
            fallback_full, previous, roll_plan
        ) or not self._is_feasible(seed_full, previous, roll_plan):
            # Projection may preserve an out-of-support fixed roll command.
            raise ValueError(
                "initial JEPA-NMPC plan is outside the model/actuator envelope"
            )

        with torch.no_grad():
            prediction_cache = self._prepare_prediction_cache(observation, action)
        if prediction_cache is not None:
            prediction_cache = {
                name: value.detach() for name, value in prediction_cache.items()
            }

        best_plan = seed_full.detach().clone()
        best_cost = math.inf
        best_gradient_norm = math.inf
        best_components: dict[str, Tensor] = {}
        best_prediction: Tensor | None = None
        best_raw: Tensor | None = None
        best_position_delta: Tensor | None = None
        evaluation_count = 0
        cache_x: np.ndarray | None = None
        cache_value = math.inf
        cache_gradient = np.zeros(self.optimization_knot_count * 4, dtype=np.float64)

        def value_and_gradient(flat: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal best_plan, best_cost, best_gradient_norm, evaluation_count
            nonlocal best_components, best_prediction, best_raw, best_position_delta
            nonlocal cache_x, cache_value, cache_gradient
            if cache_x is not None and np.array_equal(flat, cache_x):
                return cache_value, cache_gradient.copy()
            if evaluation_count >= self.config.maximum_objective_evaluations:
                raise _EvaluationBudgetExceeded
            if (
                self.config.deadline_seconds > 0.0
                and time.perf_counter() - started > self.config.deadline_seconds
            ):
                raise _DeadlineExceeded
            decision = torch.as_tensor(
                flat, dtype=torch.float32, device=self.device
            ).reshape(self.optimization_knot_count, 4)
            decision.requires_grad_(True)
            plan = self._expand_decision(decision)
            (
                cost,
                components,
                prediction,
                _motor,
                raw,
                position_delta,
                _support_diagnostics,
            ) = self._evaluate(
                observation,
                action,
                plan.unsqueeze(0),
                prepared_reference,
                position,
                previous,
                current,
                roll_plan,
                prediction_cache,
            )
            scalar = cost[0]
            gradient = torch.autograd.grad(scalar, decision)[0]
            evaluation_count += 1
            value = float(scalar.detach().cpu())
            gradient_np = gradient.detach().cpu().numpy().astype(np.float64).reshape(-1)
            if not math.isfinite(value) or not np.all(np.isfinite(gradient_np)):
                value = 1e12
                gradient_np = np.zeros_like(gradient_np)
            elif (
                self._is_feasible(plan.detach(), previous, roll_plan)
                and value < best_cost
            ):
                best_plan = plan.detach().clone()
                best_cost = value
                best_gradient_norm = float(np.linalg.norm(gradient_np))
                best_components = {
                    name: component.detach().clone()
                    for name, component in components.items()
                }
                best_prediction = prediction.detach().clone()
                best_raw = raw.detach().clone()
                best_position_delta = (
                    None if position_delta is None else position_delta.detach().clone()
                )
            cache_x = np.asarray(flat, dtype=np.float64).copy()
            cache_value = value
            cache_gradient = gradient_np.copy()
            return value, gradient_np

        seed_flat = seed_decision.detach().cpu().numpy().astype(np.float64).reshape(-1)
        seed_cost = math.inf
        seed_gradient = np.zeros_like(seed_flat)
        baseline_cost = math.inf
        baseline_components: dict[str, Tensor] = {}
        baseline_prediction: Tensor | None = None
        baseline_raw: Tensor | None = None
        baseline_position_delta: Tensor | None = None
        solver_result = None
        deadline_exceeded = False
        evaluation_budget_exhausted = False
        try:
            decision = seed_decision.detach().clone().requires_grad_(True)
            seed_plan = self._expand_decision(decision)
            paired_plan = torch.stack((fallback_full, seed_plan), dim=0)
            (
                paired_cost,
                paired_components,
                paired_prediction,
                _paired_motor,
                paired_raw,
                paired_position_delta,
                _paired_support_diagnostics,
            ) = self._evaluate(
                observation,
                action,
                paired_plan,
                prepared_reference,
                position,
                previous,
                current,
                roll_plan,
                prediction_cache,
            )
            gradient = torch.autograd.grad(paired_cost[1], decision)[0]
            baseline_cost = float(paired_cost[0].detach().cpu())
            seed_cost = float(paired_cost[1].detach().cpu())
            seed_gradient = (
                gradient.detach().cpu().numpy().astype(np.float64).reshape(-1)
            )
            if not math.isfinite(baseline_cost):
                raise ValueError("JEPA-NMPC fallback plan has non-finite model cost")
            baseline_components = {
                name: value[0:1].detach().clone()
                for name, value in paired_components.items()
            }
            baseline_prediction = paired_prediction[0:1].detach().clone()
            baseline_raw = paired_raw[0:1].detach().clone()
            baseline_position_delta = (
                None
                if paired_position_delta is None
                else paired_position_delta[0:1].detach().clone()
            )
            evaluation_count += 1
            if math.isfinite(seed_cost) and np.all(np.isfinite(seed_gradient)):
                best_plan = seed_plan.detach().clone()
                best_cost = seed_cost
                best_gradient_norm = float(np.linalg.norm(seed_gradient))
                best_components = {
                    name: value[1:2].detach().clone()
                    for name, value in paired_components.items()
                }
                best_prediction = paired_prediction[1:2].detach().clone()
                best_raw = paired_raw[1:2].detach().clone()
                best_position_delta = (
                    None
                    if paired_position_delta is None
                    else paired_position_delta[1:2].detach().clone()
                )
            cache_x = seed_flat.copy()
            cache_value = seed_cost
            cache_gradient = seed_gradient.copy()
            constraints = self._linear_constraints(
                previous_values.astype(np.float64), roll_values
            )
            solver_result = minimize(
                lambda values: value_and_gradient(values)[0],
                seed_flat,
                jac=lambda values: value_and_gradient(values)[1],
                method="SLSQP",
                bounds=self._decision_bounds(),
                constraints=constraints,
                options={
                    "maxiter": self.config.maximum_iterations,
                    "ftol": self.config.solver_ftol,
                    "disp": False,
                },
            )
            if np.all(np.isfinite(solver_result.x)):
                value_and_gradient(np.asarray(solver_result.x, dtype=np.float64))
        except _DeadlineExceeded:
            deadline_exceeded = True
        except _EvaluationBudgetExceeded:
            evaluation_budget_exhausted = True

        elapsed = time.perf_counter() - started
        if self.config.deadline_seconds > 0.0:
            deadline_exceeded = bool(
                deadline_exceeded or elapsed > self.config.deadline_seconds
            )
        required = max(
            self.config.minimum_predicted_improvement,
            self.config.minimum_relative_improvement * max(abs(baseline_cost), 1e-9),
            self.config.acceptance_tolerance,
        )
        improvement = baseline_cost - best_cost
        solver_success = bool(solver_result is not None and solver_result.success)
        solver_acceptable = (
            solver_success or self.config.accept_best_feasible_on_solver_failure
        )
        accepted = bool(
            solver_acceptable
            and not deadline_exceeded
            and math.isfinite(best_cost)
            and improvement >= required
        )
        selected = best_plan if accepted else fallback_full
        reason = "accepted"
        if deadline_exceeded:
            reason = "deadline_exceeded_fallback"
        elif not solver_acceptable:
            reason = "solver_failed_fallback"
        elif not accepted:
            reason = "no_verified_improvement"

        if accepted:
            if best_prediction is None or best_raw is None:
                raise RuntimeError("accepted JEPA-NMPC plan has no cached rollout")
            selected_cost = best_cost
            selected_components = best_components
            selected_prediction = best_prediction
            selected_raw = best_raw
            selected_position_delta = best_position_delta
        else:
            if baseline_prediction is None or baseline_raw is None:
                raise RuntimeError("JEPA-NMPC fallback has no cached rollout")
            (
                selected_components,
                selected_prediction,
                selected_raw,
                selected_position_delta,
            ) = (
                baseline_components,
                baseline_prediction,
                baseline_raw,
                baseline_position_delta,
            )
            selected_cost = baseline_cost
        elapsed = time.perf_counter() - started
        if (
            accepted
            and self.config.deadline_seconds > 0.0
            and elapsed > self.config.deadline_seconds
        ):
            deadline_exceeded = True
            accepted = False
            reason = "deadline_exceeded_fallback"
            selected = fallback_full
            selected_cost = baseline_cost
            selected_components = baseline_components
            selected_prediction = baseline_prediction
            selected_raw = baseline_raw
            selected_position_delta = baseline_position_delta
        self.nominal_command.copy_(self._shift_plan(selected))
        diagnostics: dict[str, Any] = {
            "implementation_id": IMPLEMENTATION_ID,
            "stage2_implementation": STAGE2_IMPLEMENTATION,
            "learned_residual_enabled": self.config.include_learned_residual,
            "solver": "scipy_slsqp_exact_jepa_gradient",
            "seed_source": "projected_reference_action_seed",
            "solver_success": solver_success,
            "solver_message": (
                str(solver_result.message) if solver_result is not None else reason
            ),
            "solver_iterations": int(getattr(solver_result, "nit", 0)),
            "objective_evaluations": evaluation_count,
            "evaluation_budget_exhausted": evaluation_budget_exhausted,
            "seed_cost": seed_cost,
            "nominal_cost": baseline_cost,
            "best_feasible_cost": best_cost,
            "selected_cost": selected_cost,
            "predicted_improvement": baseline_cost - selected_cost,
            "required_improvement": required,
            "seed_gradient_norm": float(np.linalg.norm(seed_gradient)),
            "best_gradient_norm": best_gradient_norm,
            "maximum_raw_motor": float(selected_raw.abs().amax().cpu()),
            "optimization_knot_count": self.optimization_knot_count,
            "model_knot_count": self.knot_count,
            "model_horizon_steps": self.model_horizon_steps,
            "model_sample_rate_hz": self.sample_rate_hz,
            "timing_seconds": elapsed,
            "deadline_exceeded": deadline_exceeded,
            "accepted": accepted,
            "cost_components": {
                name: float(value[0].cpu())
                for name, value in selected_components.items()
            },
        }
        return JepaNmpcResult(
            command=selected[0].cpu().numpy().copy(),
            planned_command=selected.cpu().numpy().copy(),
            nominal_command=fallback_full.cpu().numpy().copy(),
            predicted_physical_observation=selected_prediction[0].cpu().numpy().copy(),
            predicted_position_delta_ned_m=(
                None
                if selected_position_delta is None
                else selected_position_delta[0].cpu().numpy().copy()
            ),
            accepted=accepted,
            reason=reason,
            diagnostics=diagnostics,
        )


__all__ = [
    "COMMAND_ORDER",
    "IMPLEMENTATION_ID",
    "JepaNmpc",
    "JepaNmpcConfig",
    "JepaNmpcResult",
    "ReferenceTrajectory",
    "load_jepa_nmpc_config",
]
