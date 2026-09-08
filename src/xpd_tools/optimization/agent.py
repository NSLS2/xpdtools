"""Blop optimizer and Queue Server integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from ax.api.protocols import IMetric
from blop.ax import (
    AxOptimizer,
    Objective,
    OutcomeConstraint,
    RangeDOF,
    to_ax_objective_str,
)
from blop.protocols import QueueserverOptimizationProblem
from blop.queueserver import QueueserverClient, QueueserverOptimizationRunner

from .evaluation import (
    PdfEvaluationMode,
    PdfFitConfig,
    PdfReferenceConfig,
    XrayUvvisEvaluation,
)


def build_dofs(use_oam: bool = False) -> list[RangeDOF]:
    """Build the supported precursor-flow degrees of freedom."""
    if use_oam:
        return [
            RangeDOF(
                name="infusion_rate_CsPb",
                bounds=(5, 200),
                parameter_type="float",
            ),
            RangeDOF(
                name="infusion_rate_Br",
                bounds=(5, 250),
                parameter_type="float",
            ),
            RangeDOF(
                name="infusion_rate_Cl",
                bounds=(5, 200),
                parameter_type="float",
            ),
            RangeDOF(
                name="infusion_rate_OAm",
                bounds=(0, 70),
                parameter_type="float",
            ),
        ]
    return [
        RangeDOF(
            name="infusion_rate_CsPb",
            bounds=(10, 200),
            parameter_type="float",
        ),
        RangeDOF(
            name="infusion_rate_Br",
            bounds=(5, 200),
            parameter_type="float",
        ),
        RangeDOF(
            name="infusion_rate_I2",
            bounds=(0, 200),
            parameter_type="float",
        ),
    ]


def build_outcome_constraints(
    peak_target: float = 660,
    peak_tolerance: float = 5,
) -> list[OutcomeConstraint]:
    """Constrain the fitted PL peak to the requested target window."""
    peak = IMetric(name="Peak")
    return [
        OutcomeConstraint(f"p >= {peak_target - peak_tolerance}", p=peak),
        OutcomeConstraint(f"p <= {peak_target + peak_tolerance}", p=peak),
    ]


def build_objectives(
    pdf_fit_config: PdfFitConfig,
    pdf_references: PdfReferenceConfig,
) -> list[Objective]:
    """Build optical and phase objectives from one shared PDF schema."""
    prefix = (
        "pdf_fit_corr_"
        if pdf_fit_config.mode is PdfEvaluationMode.PDF_FIT_OBJECTIVES
        else "corr_"
    )
    return [
        Objective(name="log_FWHM", minimize=True),
        Objective(name="log_PLQY", minimize=False),
        Objective(name="peak_distance", minimize=True),
        *(
            Objective(name=f"{prefix}{phase.name}", minimize=phase.minimize)
            for phase in pdf_references.phases
        ),
    ]


def pdf_tracking_metrics(
    pdf_fit_config: PdfFitConfig,
    pdf_references: PdfReferenceConfig,
) -> tuple[str, ...]:
    """Return configured PDF metrics recorded outside the objective set."""
    if pdf_fit_config.mode is PdfEvaluationMode.PDF_FIT_OBJECTIVES:
        return tuple(f"corr_{phase.name}" for phase in pdf_references.phases)
    if pdf_fit_config.mode is PdfEvaluationMode.RAW_OBJECTIVES_PDF_FIT_TRACKED:
        return tuple(f"pdf_fit_corr_{phase.name}" for phase in pdf_references.phases)
    return ()


def load_historical_data(
    path: str | Path,
    dof_names: Sequence[str],
    pdf_fit_config: PdfFitConfig,
    pdf_references: PdfReferenceConfig,
    *,
    peak_target: float = 660,
    r2_min: float = 0.70,
) -> list[dict[str, float]]:
    """Load headered historical observations using the live metric schema."""
    frame = pd.read_csv(path)
    phase_objectives = [
        objective.name
        for objective in build_objectives(pdf_fit_config, pdf_references)[3:]
    ]
    required = {*dof_names, "Peak", "FWHM", "PLQY", *phase_objectives}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"historical data is missing columns: {', '.join(missing)}")
    if "r_2" in frame.columns:
        frame = frame[frame["r_2"] >= r2_min]

    points: list[dict[str, float]] = []
    for _, row in frame.iterrows():
        values = cast(dict[str, Any], row.to_dict())
        peak = float(values["Peak"])
        fwhm = float(values["FWHM"])
        plqy = float(values["PLQY"])
        if not np.isfinite(peak):
            raise ValueError("historical Peak values must be finite")
        if not np.isfinite(fwhm) or fwhm <= 0:
            fwhm = 1000.0
        if not np.isfinite(plqy) or plqy <= 0:
            plqy = 1e-10

        point = {name: float(values[name]) for name in dof_names}
        point.update(
            {
                "Peak": peak,
                "peak_distance": abs(peak - peak_target),
                "log_FWHM": float(np.log(fwhm)),
                "log_PLQY": float(np.log(plqy)),
            }
        )
        for name in phase_objectives:
            value = float(values[name])
            if not np.isfinite(value):
                raise ValueError(f"historical {name} values must be finite")
            point[name] = value
        points.append(point)
    return points


def build_optimizer(
    *,
    pdf_references: PdfReferenceConfig,
    peak_target: float = 660,
    peak_tolerance: float = 5,
    use_oam: bool = False,
    pdf_fit_config: PdfFitConfig | None = None,
    checkpoint_path: str | Path | None = None,
    history_path: str | Path | None = None,
    r2_min: float = 0.70,
) -> AxOptimizer:
    """Build and optionally seed an Ax optimizer for this workflow."""
    fit_config = pdf_fit_config or PdfFitConfig()
    dofs = build_dofs(use_oam)
    objectives = build_objectives(fit_config, pdf_references)
    constraints = build_outcome_constraints(peak_target, peak_tolerance)
    optimizer = AxOptimizer(
        parameters=[dof.to_ax_parameter_config() for dof in dofs],
        objective=to_ax_objective_str(objectives),
        outcome_constraints=[constraint.ax_constraint for constraint in constraints],
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
    )
    tracking_metrics = pdf_tracking_metrics(fit_config, pdf_references)
    if tracking_metrics:
        optimizer.ax_client.configure_tracking_metrics(tracking_metrics)
    if history_path is not None:
        optimizer.ingest(
            load_historical_data(
                history_path,
                [dof.parameter_name for dof in dofs],
                fit_config,
                pdf_references,
                peak_target=peak_target,
                r2_min=r2_min,
            )
        )
    return optimizer


class XrayUvvisQueueAgent:
    """One-point facade over Blop's Queue Server optimization runner."""

    def __init__(
        self,
        runner: QueueserverOptimizationRunner,
        queueserver_client: QueueserverClient,
    ) -> None:
        self._runner = runner
        self._queueserver_client = queueserver_client
        self._future: Future[Any] | None = None

    @property
    def optimizer(self) -> AxOptimizer:
        """The configured optimizer."""
        return cast(AxOptimizer, self._runner.optimization_problem.optimizer)

    @property
    def current_iteration(self) -> int:
        """The runner's current iteration count."""
        return self._runner.current_iteration

    def run(
        self,
        iterations: int = 1,
        checkpoint_interval: int | None = None,
    ) -> Future[Any]:
        """Run sequential one-point optimization iterations."""
        future = self._runner.run(
            iterations=iterations,
            num_points=1,
            checkpoint_interval=checkpoint_interval,
        )
        self._future = future
        return future

    def submit_suggestion(self, suggestion: Mapping[str, Any]) -> Future[Any]:
        """Submit exactly one caller-provided suggestion."""
        future = self._runner.submit_suggestions([dict(suggestion)])
        self._future = future
        return future

    def close(self) -> None:
        """Stop the document listener when no acquisition is active."""
        if self._future is not None and not self._future.done():
            raise RuntimeError("cannot close queue agent during an active acquisition")
        self._queueserver_client.stop_listener()


def _required_plan_kwargs(
    acquisition_plan_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Copy plan kwargs and require a scattering-producing X-ray run."""
    plan_kwargs = deepcopy(dict(acquisition_plan_kwargs or {}))
    xray_config = dict(plan_kwargs.get("xray_config", {}))
    if xray_config.get("do_xray") is False:
        raise ValueError("xray_config.do_xray cannot be False for optimization")
    if "stream_name" in xray_config and xray_config["stream_name"] != "scattering":
        raise ValueError(
            "xray_config.stream_name must be 'scattering' for optimization"
        )
    xray_config["do_xray"] = True
    xray_config["stream_name"] = "scattering"
    plan_kwargs["xray_config"] = xray_config
    return plan_kwargs


def build_queue_agent(
    evaluation: XrayUvvisEvaluation,
    queueserver_client: QueueserverClient,
    *,
    sensor_names: Sequence[str] = ("qepro",),
    acquisition_plan_name: str = "xray_uvvis_acquire",
    acquisition_plan_kwargs: Mapping[str, Any] | None = None,
    peak_tolerance: float = 5,
    use_oam: bool = False,
    checkpoint_path: str | Path | None = None,
    history_path: str | Path | None = None,
    r2_min: float = 0.70,
) -> XrayUvvisQueueAgent:
    """Build a queue agent; Blop starts the supplied client's listener."""
    optimizer = build_optimizer(
        pdf_references=evaluation.pdf_references,
        peak_target=evaluation.peak_target,
        peak_tolerance=peak_tolerance,
        use_oam=use_oam,
        pdf_fit_config=evaluation.pdf_fit_config,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        r2_min=r2_min,
    )
    problem = QueueserverOptimizationProblem(
        optimizer=optimizer,
        actuators=(),
        sensors=tuple(sensor_names),
        evaluation_function=evaluation,
        acquisition_plan=acquisition_plan_name,
        acquisition_plan_kwargs=_required_plan_kwargs(acquisition_plan_kwargs),
    )
    runner = QueueserverOptimizationRunner(problem, queueserver_client)
    return XrayUvvisQueueAgent(runner, queueserver_client)
