"""Deterministic JEPA residual shooting controller.

This controller is deliberately separate from the historical MPPI
implementation.  It evaluates a small, deterministic population of complete
MANUAL plans through the Stage-2 v6 model, scores the resulting physical
rollouts, and executes only the first command of a plan that was evaluated
verbatim.  The nominal shifted plan is always candidate zero and is the hard
fallback.

The implementation keeps the deployable boundary causal: only the VINS/IMU/
depth observation history and measured post-mixer motor history are passed to
Stage 2.  Ground truth can be used by the collection process for an offline
report, never by this controller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

try:
    import torch
    from torch import Tensor
    from torch.quasirandom import SobolEngine
except ImportError as error:  # pragma: no cover - optional controller dependency.
    raise ImportError("JEPA shooting MPC requires PyTorch") from error

from disaware_learning.control.jepa_mppi import (
    IMPLEMENTATION_ID as MPPI_IMPLEMENTATION_ID,
    JepaMppiConfig,
    ReferenceConditionedJepaMppi,
    ReferenceTrajectory,
    standard_bluerov2_manual_mixer,
)
from disaware_learning.learning.motor_jepa_data import (
    ACTION_CONTRACT_ID,
    FEATURE_CONTRACT_ID,
    IMU_ACCEL_SLICE,
    OBSERVATION_LABELS,
    ROTATION_6D_SLICE,
    VIO_VELOCITY_SLICE,
    YAW_RATE_INDEX,
)
from disaware_learning.learning.stage2 import (
    IMPLEMENTATION as STAGE2_IMPLEMENTATION,
    rotation6d_to_matrix,
)


IMPLEMENTATION_ID = "latent_residual_shooting_mpc_stage2_v6"
CONTROLLER_TYPE = "latent_residual_shooting_mpc"


@dataclass(frozen=True)
class LatentResidualShootingConfig(JepaMppiConfig):
    """Small deterministic candidate bank and verification policy."""

    # The inherited MPPI fields are retained for the common runtime contract;
    # only the fields below determine this optimizer's candidate population.
    candidate_count: int = 32
    blend_fractions: tuple[float, ...] = (0.25, 0.50, 0.75)
    fast_residual_scale: tuple[float, ...] = (0.045, 0.045, 0.055, 0.012)
    hold_residual_scale: tuple[float, ...] = (0.025, 0.025, 0.035, 0.006)
    fast_time_constant_seconds: float = 0.45
    hold_time_constant_seconds: float = 1.20
    # Candidate commands are built from future reference samples so the
    # learned rollout can choose an anticipatory plan for the slow underwater
    # plant.  Each profile is ordered surge/sway/heave/yaw.  These are proposal
    # locations, not fixed compensation: Stage 2 still evaluates every plan.
    reference_lead_profiles_seconds: tuple[tuple[float, ...], ...] = (
        (0.80, 0.80, 1.20, 0.20),
        (1.20, 1.20, 1.60, 0.40),
        (1.60, 1.60, 2.00, 0.40),
    )
    # The asynchronous planner result is applied about one 5 Hz model step
    # after its context was sampled. The rollout adapter advances the target
    # horizon by this amount while retaining causal state history.
    planner_application_preview_seconds: float = 0.20
    # A candidate is accepted only when its learned trajectory improves the
    # nominal learned trajectory.  A tiny tolerance avoids noise-driven flips.
    acceptance_epsilon: float = 1e-5
    direct_sign_tolerance: float = 0.25
    support_margin: float = 1e-5
    nominal_crosscheck_interval: int = 5
    capture_candidates: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.candidate_count < 8:
            raise ValueError("candidate_count must be at least eight")
        if not self.blend_fractions or any(
            not 0.0 < float(value) < 1.0 for value in self.blend_fractions
        ):
            raise ValueError("blend fractions must be strictly between zero and one")
        for name in ("fast_residual_scale", "hold_residual_scale"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 4 or any(value < 0.0 or not math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain four nonnegative finite values")
            object.__setattr__(self, name, values)
        for name in ("fast_time_constant_seconds", "hold_time_constant_seconds"):
            value = float(getattr(self, name))
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be positive and finite")
        profiles: list[tuple[float, ...]] = []
        for profile in self.reference_lead_profiles_seconds:
            values = tuple(float(value) for value in profile)
            if len(values) != 4 or any(
                value < 0.0 or not math.isfinite(value) for value in values
            ):
                raise ValueError(
                    "reference lead profiles must contain four finite "
                    "nonnegative values"
                )
            profiles.append(values)
        object.__setattr__(self, "reference_lead_profiles_seconds", tuple(profiles))
        if (
            self.planner_application_preview_seconds < 0.0
            or not math.isfinite(self.planner_application_preview_seconds)
        ):
            raise ValueError(
                "planner_application_preview_seconds must be finite and nonnegative"
            )
        for name in ("acceptance_epsilon", "support_margin"):
            value = float(getattr(self, name))
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0.0 <= self.direct_sign_tolerance <= 1.0:
            raise ValueError("direct_sign_tolerance must be in [0, 1]")
        if self.nominal_crosscheck_interval < 1:
            raise ValueError("nominal_crosscheck_interval must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {"implementation_id": IMPLEMENTATION_ID, **asdict(self)}


def load_latent_residual_shooting_config(
    path: str | Path,
) -> LatentResidualShootingConfig:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("shooting MPC config must contain a JSON object")
    payload = dict(payload)
    implementation = payload.pop("implementation_id", IMPLEMENTATION_ID)
    if implementation != IMPLEMENTATION_ID:
        raise ValueError(f"unexpected shooting MPC implementation: {implementation}")
    expected = set(LatentResidualShootingConfig.__dataclass_fields__)
    unknown = set(payload) - expected
    if unknown:
        raise ValueError(f"unknown shooting MPC config keys: {sorted(unknown)}")
    return LatentResidualShootingConfig(**payload)


@dataclass(frozen=True)
class LatentResidualShootingResult:
    """Runtime result matching the collection adapter's prediction contract."""

    command: np.ndarray
    planned_command: np.ndarray
    nominal_command: np.ndarray
    predicted_physical_observation: np.ndarray
    predicted_position_delta_ned_m: np.ndarray | None
    accepted: bool
    reason: str
    diagnostics: dict[str, Any]


class LatentResidualShootingMpc(ReferenceConditionedJepaMppi):
    """Stage-2 v6 deterministic residual shooting controller."""

    IMPLEMENTATION_ID = IMPLEMENTATION_ID

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._optimization_count = 0

    def reset(self, command: np.ndarray | None = None) -> None:
        super().reset(command)
        self._optimization_count = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        config: LatentResidualShootingConfig = LatentResidualShootingConfig(),
        device: str | torch.device = "cpu",
    ) -> "LatentResidualShootingMpc":
        # Reuse the strict Stage-2/action/time contract checks in the base
        # constructor, but instantiate this class so optimize is replaced.
        base = super().from_checkpoint(checkpoint_path, config=config, device=device)
        return cls(
            base.model,
            base.standardizer,
            history_steps=base.history_steps,
            model_horizon_steps=base.model_horizon_steps,
            sample_rate_hz=base.sample_rate_hz,
            config=config,
            device=device,
        )

    def _candidate_plans(
        self,
        nominal: Tensor,
        reference_seed: Tensor,
        lead_seeds: list[tuple[tuple[float, ...], Tensor]],
        previous: Tensor,
        previous_modal_action: Tensor,
        roll_plan: Tensor,
    ) -> tuple[Tensor, list[str], np.ndarray]:
        """Create a reproducible low-dimensional fast/hold candidate bank."""
        config = self.config
        candidates: list[Tensor] = [nominal, reference_seed]
        sources = ["shifted_nominal", "reference_action_seed"]
        for fraction in config.blend_fractions:
            candidates.append(nominal + float(fraction) * (reference_seed - nominal))
            sources.append(f"nominal_reference_blend_{fraction:.2f}")

        for profile, seed in lead_seeds:
            if len(candidates) >= config.candidate_count:
                break
            candidates.append(seed)
            profile_name = "_".join(f"{value:.1f}" for value in profile)
            sources.append(f"reference_phase_lead_{profile_name}s")

        remaining = max(0, config.candidate_count - len(candidates))
        # Two coefficients per axis describe a short corrective impulse and a
        # slower hold.  Sobol is deterministic here, so a failed run can be
        # replayed exactly from its seed and diagnostics.
        engine = SobolEngine(dimension=8, scramble=True, seed=config.seed)
        if remaining:
            unit = engine.draw(remaining).to(
                device=self.device, dtype=nominal.dtype
            )
            coefficients = unit.mul(2.0).sub(1.0)
        else:
            coefficients = nominal.new_empty((0, 8))
        steps = torch.arange(self.knot_count, device=self.device, dtype=nominal.dtype)
        time_axis = steps / self.sample_rate_hz
        fast = torch.exp(-time_axis / config.fast_time_constant_seconds)
        hold = torch.exp(-time_axis / config.hold_time_constant_seconds)
        fast = fast / fast[:1].clamp_min(1e-6)
        hold = hold / hold[:1].clamp_min(1e-6)
        fast_scale = nominal.new_tensor(config.fast_residual_scale)
        hold_scale = nominal.new_tensor(config.hold_residual_scale)
        perturbations = (
            coefficients[:, :4].unsqueeze(1) * fast.view(1, -1, 1) * fast_scale
            + coefficients[:, 4:].unsqueeze(1) * hold.view(1, -1, 1) * hold_scale
        )
        if remaining:
            candidates.extend((reference_seed.unsqueeze(0) + perturbations).unbind(0))
            sources.extend(["sobol_fast_hold"] * remaining)
        desired = torch.stack(candidates, dim=0)
        projected = self._project_commands(
            desired,
            previous,
            previous_modal_action,
            roll_plan,
        )
        return projected, sources, coefficients.detach().cpu().numpy()

    def _lead_reference_for_seed(
        self,
        reference: Mapping[str, tuple[Tensor, Tensor]],
        profile_seconds: tuple[float, ...],
    ) -> dict[str, tuple[Tensor, Tensor]]:
        """Advance twist targets per axis while preserving the scoring target."""

        shifted = dict(reference)

        def advance(
            values: Tensor,
            mask: Tensor,
            lead_seconds: tuple[float, ...],
        ) -> tuple[Tensor, Tensor]:
            output = values.clone()
            output_mask = mask.clone()
            count = len(values)
            for axis, seconds in enumerate(lead_seconds):
                lead = min(int(round(seconds * self.sample_rate_hz)), count - 1)
                if lead <= 0:
                    continue
                output[:, axis] = torch.cat(
                    (values[lead:, axis], values[-1:, axis].expand(lead)), dim=0
                )
                output_mask[:, axis] = torch.cat(
                    (mask[lead:, axis], mask[-1:, axis].expand(lead)), dim=0
                )
            return output, output_mask

        if "body_velocity" in reference:
            shifted["body_velocity"] = advance(
                *reference["body_velocity"], profile_seconds[:3]
            )
        if "yaw_rate" in reference:
            shifted["yaw_rate"] = advance(
                *reference["yaw_rate"], profile_seconds[3:]
            )
        return shifted

    def _predict_variant(
        self,
        observation: Tensor,
        action: Tensor,
        command: Tensor,
        roll_plan: Tensor,
        prediction_cache: Mapping[str, Tensor] | None,
        *,
        include_residual: bool,
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]:
        motor, raw_motor = standard_bluerov2_manual_mixer(
            command, roll_plan.unsqueeze(0).expand(len(command), -1)
        )
        _, standardized = self._model_action_sequence(motor, action)
        observation_batch = observation.unsqueeze(0).expand(len(command), -1, -1)
        action_batch = action.unsqueeze(0).expand(len(command), -1, -1)
        context = None
        if prediction_cache is not None:
            context = prediction_cache["backbone_context"].expand(len(command), -1)
        prediction = self.model.rollout_from_context(
            observation_batch,
            action_batch,
            standardized,
            context=context,
            include_residual=include_residual,
        )
        physical = prediction["predicted_future_physical_observation"]
        if physical.shape != (len(command), self.model_horizon_steps, len(OBSERVATION_LABELS)):
            raise ValueError("Stage-2 physical rollout shape mismatch")
        position_delta = prediction.get("predicted_position_delta_ned_m")
        if position_delta is None:
            position_delta = torch.cumsum(
                (rotation6d_to_matrix(physical[..., ROTATION_6D_SLICE])
                 @ physical[..., VIO_VELOCITY_SLICE].unsqueeze(-1)).squeeze(-1)
                / self.sample_rate_hz,
                dim=1,
            )
        return prediction, physical, motor, raw_motor, position_delta

    def _cost_variant(
        self,
        observation: Tensor,
        action: Tensor,
        command: Tensor,
        reference: Mapping[str, tuple[Tensor, Tensor]],
        position: Tensor,
        previous: Tensor,
        current_observation: Tensor,
        roll_plan: Tensor,
        prediction_cache: Mapping[str, Tensor] | None,
        *,
        include_residual: bool,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor], Tensor, Tensor, Tensor]:
        prediction, physical, motor, raw_motor, position_delta = self._predict_variant(
            observation, action, command, roll_plan, prediction_cache,
            include_residual=include_residual,
        )
        cost, components = self._trajectory_cost(
            physical, position_delta, motor, raw_motor, command, reference,
            position, previous, current_observation,
        )
        support_components, support = self._model_support_terms(motor, action)
        components.update(support_components)
        return cost + sum(support_components.values()), components, support, physical, position_delta, prediction

    def _direct_sign_ok(
        self,
        physical: Tensor,
        reference: Mapping[str, tuple[Tensor, Tensor]],
    ) -> tuple[bool, dict[str, float]]:
        checks: dict[str, float] = {}
        if "body_velocity" in reference:
            target, mask = reference["body_velocity"]
            response = physical[..., VIO_VELOCITY_SLICE]
            for axis, name in enumerate(("surge", "sway", "heave")):
                active = mask[:, axis] > 0.5
                if not bool(active.any()):
                    continue
                target_axis = target[:, axis][active]
                response_axis = response[:, :, axis][:, active].mean(dim=0)
                denom = target_axis.square().sum().clamp_min(1e-6)
                alignment = (target_axis * response_axis).sum() / denom
                checks[name] = float(alignment.detach().cpu())
                if float(alignment.detach().cpu()) < -self.config.direct_sign_tolerance:
                    return False, checks
        if "yaw_rate" in reference:
            target, mask = reference["yaw_rate"]
            active = mask[:, 0] > 0.5
            if bool(active.any()):
                target_axis = target[:, 0][active]
                response_axis = physical[..., YAW_RATE_INDEX][:, active].mean(dim=0)
                alignment = (
                    (target_axis * response_axis).sum()
                    / target_axis.square().sum().clamp_min(1e-6)
                )
                checks["yaw"] = float(alignment.detach().cpu())
                if float(alignment.detach().cpu()) < -self.config.direct_sign_tolerance:
                    return False, checks
        return True, checks

    @torch.inference_mode()
    def optimize(
        self,
        observation_history: np.ndarray,
        action_history: np.ndarray,
        reference: ReferenceTrajectory,
        previous_command: np.ndarray,
        *,
        current_position_ned_m: np.ndarray | None = None,
        roll_command: float | np.ndarray = 0.0,
        capture_candidates: bool | None = None,
    ) -> LatentResidualShootingResult:
        started = time.perf_counter()
        capture = (
            self.config.capture_candidates
            if capture_candidates is None
            else bool(capture_candidates)
        )
        timings: dict[str, float] = {}
        previous_values = np.asarray(previous_command, dtype=np.float32)
        if previous_values.shape != (4,) or not np.all(np.isfinite(previous_values)):
            raise ValueError("previous command must contain four finite axes")
        previous = torch.as_tensor(previous_values, device=self.device)
        observation, action, current = self._prepare_history(observation_history, action_history)
        previous_model_action = action[-1] * self._action_std + self._action_mean
        previous_modal_action = previous_model_action if self.action_dim == 5 else previous
        prepared_reference = self._prepare_reference(reference)
        roll_plan = self._prepare_roll_plan(roll_command)
        if current_position_ned_m is None:
            if "position" in prepared_reference:
                raise ValueError("position tracking requires current NED position")
            position = torch.zeros(3, device=self.device)
        else:
            position = torch.as_tensor(np.asarray(current_position_ned_m, dtype=np.float32), device=self.device)
            if position.shape != (3,) or not torch.isfinite(position).all():
                raise ValueError("current NED position must contain three finite values")
        nominal = self._initial_plan(previous, previous_modal_action, roll_plan)
        reference_seed = self._reference_action_seed(
            prepared_reference, nominal, previous, previous_modal_action, roll_plan
        )
        timings["candidate_generation_start"] = time.perf_counter()
        lead_seeds = [
            (
                profile,
                self._reference_action_seed(
                    self._lead_reference_for_seed(prepared_reference, profile),
                    nominal,
                    previous,
                    previous_modal_action,
                    roll_plan,
                ),
            )
            for profile in self.config.reference_lead_profiles_seconds
        ]
        candidates, sources, coefficients = self._candidate_plans(
            nominal,
            reference_seed,
            lead_seeds,
            previous,
            previous_modal_action,
            roll_plan,
        )
        timings["candidate_generation"] = time.perf_counter() - timings.pop("candidate_generation_start")
        prediction_cache = self._prepare_prediction_cache(observation, action)
        timings["context_cache"] = time.perf_counter() - started - sum(timings.values())
        learned = self._cost_variant(
            observation, action, candidates, prepared_reference, position, previous,
            current, roll_plan, prediction_cache, include_residual=True,
        )
        learned_costs, learned_components, learned_support, learned_physical, learned_pos, learned_output = learned
        timings["learned_rollout_and_cost"] = (
            time.perf_counter() - started - sum(timings.values())
        )
        finite = torch.isfinite(learned_costs)
        feasible = learned_support["feasible"] & finite
        selected_index = 0
        if bool(feasible.any()):
            masked = torch.where(feasible, learned_costs, torch.full_like(learned_costs, float("inf")))
            selected_index = int(torch.argmin(masked).item())
        crosscheck_due = capture or (
            self._optimization_count % self.config.nominal_crosscheck_interval == 0
        )
        if crosscheck_due:
            if capture:
                crosscheck_indices = torch.arange(len(candidates), device=self.device)
            else:
                crosscheck_indices = torch.as_tensor(
                    sorted({0, selected_index}), dtype=torch.long, device=self.device
                )
            nominal_only = self._cost_variant(
                observation,
                action,
                candidates[crosscheck_indices],
                prepared_reference,
                position,
                previous,
                current,
                roll_plan,
                prediction_cache,
                include_residual=False,
            )
            (
                nominal_costs,
                nominal_components,
                _,
                nominal_physical,
                _,
                _,
            ) = nominal_only
            baseline_crosscheck_index = int(
                torch.nonzero(crosscheck_indices == 0, as_tuple=False)[0, 0].item()
            )
            selected_crosscheck_index = int(
                torch.nonzero(
                    crosscheck_indices == selected_index, as_tuple=False
                )[0, 0].item()
            )
        else:
            crosscheck_indices = torch.empty(
                0, dtype=torch.long, device=self.device
            )
            nominal_costs = learned_costs.new_empty(0)
            nominal_components = {}
            nominal_physical = None
            baseline_crosscheck_index = -1
            selected_crosscheck_index = -1
        timings["nominal_crosscheck_rollout_and_cost"] = (
            time.perf_counter() - started - sum(timings.values())
        )
        selected_cost = float(learned_costs[selected_index].detach().cpu())
        baseline_cost = float(learned_costs[0].detach().cpu())
        improvement = baseline_cost - selected_cost
        candidate_sign_ok, sign_checks = self._direct_sign_ok(
            learned_physical[selected_index : selected_index + 1], prepared_reference
        )
        selected_finite = bool(torch.isfinite(learned_physical[selected_index]).all())
        selected_position_finite = bool(torch.isfinite(learned_pos[selected_index]).all())
        support_ok = bool(learned_support["feasible"][selected_index])
        motor_ok = bool(torch.max(torch.abs(standard_bluerov2_manual_mixer(
            candidates[selected_index:selected_index + 1], roll_plan.unsqueeze(0)
        )[1])) <= 1.0 + self.config.support_margin)
        deadline_ok = (
            self.config.deadline_seconds <= 0.0
            or (time.perf_counter() - started) <= self.config.deadline_seconds
        )
        accepted = (
            selected_index != 0
            and math.isfinite(selected_cost)
            and math.isfinite(baseline_cost)
            and improvement > self.config.acceptance_epsilon
            and candidate_sign_ok
            and selected_finite
            and selected_position_finite
            and support_ok
            and motor_ok
            and deadline_ok
        )
        selected = candidates[selected_index] if accepted else nominal
        selected_index_effective = selected_index if accepted else 0
        selected_physical = learned_physical[selected_index_effective]
        selected_position = learned_pos[selected_index_effective]
        if accepted:
            reason = "accepted"
        elif not deadline_ok:
            reason = "deadline_exceeded"
        elif not selected_finite or not selected_position_finite:
            reason = "nonfinite_prediction"
        elif not support_ok:
            reason = "model_support_violation"
        elif not motor_ok:
            reason = "motor_envelope_violation"
        elif not candidate_sign_ok:
            reason = "direct_sign_violation"
        else:
            reason = "no_predicted_improvement"
        timings["total"] = time.perf_counter() - started
        diagnostics: dict[str, Any] = {
            "implementation_id": IMPLEMENTATION_ID,
            "parent_implementation_id": MPPI_IMPLEMENTATION_ID,
            "stage2_implementation": STAGE2_IMPLEMENTATION,
            "feature_contract_id": FEATURE_CONTRACT_ID,
            "action_contract_id": ACTION_CONTRACT_ID,
            # Stable top-level fields consumed by the common Stonefish rollout.
            "nominal_cost": baseline_cost,
            "selected_cost": selected_cost if accepted else baseline_cost,
            "predicted_improvement": improvement if accepted else 0.0,
            "input_contract": {
                "observation_shape": tuple(observation.shape),
                "action_history_shape": tuple(action.shape),
                "future_action_shape": (len(candidates), self.model_horizon_steps, self.action_dim),
                "observation_labels": OBSERVATION_LABELS,
                "gt_used_for_feedback": False,
            },
            "history": {
                "current_observation": current.detach().cpu().numpy(),
                "previous_command": previous.detach().cpu().numpy(),
                "previous_modal_action": previous_modal_action.detach().cpu().numpy(),
            },
            "context": {
                "cache_used": prediction_cache is not None,
                "state_latent_norm": float(prediction_cache["backbone_context"][..., : self.model.backbone.config.latent_dim].norm(dim=-1).mean().detach().cpu()) if prediction_cache is not None else math.nan,
                "belief_latent_norm": float(prediction_cache["backbone_context"][..., self.model.backbone.config.latent_dim :].norm(dim=-1).mean().detach().cpu()) if prediction_cache is not None else math.nan,
                "domain_mass_scale": learned_output.get("domain_mass_scale", torch.tensor(float("nan"), device=self.device))[selected_index_effective].detach().cpu().numpy(),
                "domain_control_scale": learned_output.get("domain_control_scale", torch.tensor(float("nan"), device=self.device))[selected_index_effective].detach().cpu().numpy(),
                "domain_damping_scale": learned_output.get("domain_damping_scale", torch.tensor(float("nan"), device=self.device))[selected_index_effective].detach().cpu().numpy(),
                "domain_actuator_time_constant_scale": learned_output.get("domain_actuator_time_constant_scale", torch.tensor(float("nan"), device=self.device))[selected_index_effective].detach().cpu().numpy(),
            },
            "candidate_generation": {
                "candidate_count": len(candidates),
                "sources": sources,
                "sobol_coefficients": coefficients,
                "reference_lead_profiles_seconds": (
                    self.config.reference_lead_profiles_seconds
                ),
                "selected_index": selected_index_effective,
                "proposed_selected_index": selected_index,
                "selected_source": sources[selected_index_effective],
                "proposed_selected_source": sources[selected_index],
            },
            "model_rollout": {
                "nominal_crosscheck_completed": crosscheck_due,
                "nominal_crosscheck_interval": self.config.nominal_crosscheck_interval,
                "learned_minus_nominal_velocity_rms": float((learned_physical[crosscheck_indices] - nominal_physical)[..., VIO_VELOCITY_SLICE].square().mean().sqrt().detach().cpu()) if nominal_physical is not None else math.nan,
                "learned_minus_nominal_yaw_rate_rms": float((learned_physical[crosscheck_indices] - nominal_physical)[..., YAW_RATE_INDEX].square().mean().sqrt().detach().cpu()) if nominal_physical is not None else math.nan,
                "selected_residual_rate_rms": float(learned_output.get("residual_rate_body", torch.zeros(1, device=self.device))[selected_index_effective].square().mean().sqrt().detach().cpu()) if isinstance(learned_output, Mapping) else math.nan,
                "selected_residual_action_wrench_rms": float(learned_output.get("residual_action_wrench_body", torch.zeros(1, device=self.device))[selected_index_effective].square().mean().sqrt().detach().cpu()) if isinstance(learned_output, Mapping) else math.nan,
                "baseline_nominal_only_cost": float(nominal_costs[baseline_crosscheck_index].detach().cpu()) if crosscheck_due else math.nan,
                "selected_nominal_only_cost": float(nominal_costs[selected_crosscheck_index].detach().cpu()) if crosscheck_due else math.nan,
                "nominal_crosscheck_candidate_indices": crosscheck_indices.detach().cpu().numpy(),
                "selected_learned_cost": selected_cost,
            },
            "cost": {
                "baseline_cost": baseline_cost,
                "selected_cost": selected_cost if accepted else baseline_cost,
                "predicted_improvement": improvement if accepted else 0.0,
                "learned_components": {name: float(value[selected_index_effective].detach().cpu()) for name, value in learned_components.items()},
                "baseline_components": {name: float(value[0].detach().cpu()) for name, value in learned_components.items()},
                "baseline_nominal_only_components": {name: float(value[baseline_crosscheck_index].detach().cpu()) for name, value in nominal_components.items()} if crosscheck_due else {},
                "selected_nominal_only_components": {name: float(value[selected_crosscheck_index].detach().cpu()) for name, value in nominal_components.items()} if crosscheck_due else {},
            },
            "verification": {
                "finite_prediction": selected_finite,
                "finite_position_prediction": selected_position_finite,
                "model_support": support_ok,
                "motor_envelope": motor_ok,
                "direct_sign_ok": candidate_sign_ok,
                "direct_sign_alignment": sign_checks,
                "deadline_ok": deadline_ok,
                "accepted": accepted,
                "reason": reason,
            },
            "timing_seconds": timings["total"],
            "timing_breakdown_seconds": timings,
        }
        if capture:
            diagnostics["candidate_plans"] = candidates.detach().cpu().numpy()
            diagnostics["candidate_learned_cost"] = learned_costs.detach().cpu().numpy()
            diagnostics["candidate_nominal_only_cost"] = nominal_costs.detach().cpu().numpy()
            diagnostics["candidate_learned_components"] = {name: value.detach().cpu().numpy() for name, value in learned_components.items()}
            diagnostics["candidate_nominal_components"] = {name: value.detach().cpu().numpy() for name, value in nominal_components.items()}
            diagnostics["candidate_support_feasible"] = learned_support["feasible"].detach().cpu().numpy()
            diagnostics["candidate_predicted_position_delta_ned_m"] = learned_pos.detach().cpu().numpy()
        self._optimization_count += 1
        self.nominal_command.copy_(self._shift_plan(selected))
        return LatentResidualShootingResult(
            command=selected[0].detach().cpu().numpy().copy(),
            planned_command=selected.detach().cpu().numpy().copy(),
            nominal_command=nominal.detach().cpu().numpy().copy(),
            predicted_physical_observation=selected_physical.detach().cpu().numpy().copy(),
            predicted_position_delta_ned_m=selected_position.detach().cpu().numpy().copy(),
            accepted=accepted,
            reason=reason,
            diagnostics=diagnostics,
        )


__all__ = [
    "IMPLEMENTATION_ID",
    "CONTROLLER_TYPE",
    "LatentResidualShootingConfig",
    "LatentResidualShootingMpc",
    "LatentResidualShootingResult",
    "ReferenceTrajectory",
    "load_latent_residual_shooting_config",
]
