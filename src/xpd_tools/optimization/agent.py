"""Blop optimizer and Queue Server integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ax.api.protocols import IMetric
from blop.ax import Objective, OutcomeConstraint, RangeDOF
from blop.ax.queueserver_agent import QueueserverAgent

from .evaluation import XrayUvvisEvaluation

# TODO: These may change and should be configurable.
_STANDARD_DOFS: tuple[tuple[str, tuple[float, float]], ...] = (
    ("infusion_rate_CsPb", (10, 200)),
    ("infusion_rate_Br", (5, 200)),
    ("infusion_rate_I2", (0, 200)),
)


def build_queue_agent(
    evaluation: XrayUvvisEvaluation,
    re_manager_api: Any,
    document_dispatcher: Any,  # WARN: Future release of blop will remove this
    *,
    peak_tolerance: float = 5,
    checkpoint_path: str | Path | None = None,
) -> QueueserverAgent:
    """Build the standard three-flow Queue Server optimization agent."""
    dofs = [
        RangeDOF(name=name, bounds=bounds, parameter_type="float")
        for name, bounds in _STANDARD_DOFS
    ]
    metric_prefix = "corr_" if evaluation.pdf_mode == "raw" else "pdf_fit_corr_"

    # TODO: Too many competing objectives may be very hard to optimize.
    # Almost any direction sampled will be a hyper-volume (pareto) improvement.
    # Should prefer some linear combination of these with pre-defined, configurable
    # weights.
    objectives = [
        Objective(name="log_FWHM", minimize=True),
        Objective(name="log_PLQY", minimize=False),
        Objective(name="peak_distance", minimize=True),
        *(
            Objective(
                name=f"{metric_prefix}{phase.name}",
                minimize=phase.minimize,
            )
            for phase in evaluation.phases
        ),
    ]
    target = evaluation.peak_target
    peak = IMetric(name="Peak")
    agent = QueueserverAgent(
        re_manager_api,
        document_dispatcher,
        sensors=(),
        dofs=dofs,
        objectives=objectives,
        evaluation_function=evaluation,
        acquisition_plan="xray_uvvis_acquire",
        outcome_constraints=(
            OutcomeConstraint(f"p >= {target - peak_tolerance:g}", p=peak),
            OutcomeConstraint(f"p <= {target + peak_tolerance:g}", p=peak),
        ),
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
    )
    if evaluation.pdf_mode == "fit":
        agent.ax_client.configure_tracking_metrics(
            tuple([f"corr_{phase.name}" for phase in evaluation.phases])
        )
    return agent
