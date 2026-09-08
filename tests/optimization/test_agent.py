from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import xpd_tools.optimization.agent as agent_module
from xpd_tools.optimization.agent import (
    XrayUvvisQueueAgent,
    build_dofs,
    build_objectives,
    build_outcome_constraints,
    build_queue_agent,
    load_historical_data,
    pdf_tracking_metrics,
)
from xpd_tools.optimization.evaluation import (
    DEFAULT_PLQY_PARAMS,
    PdfEvaluationMode,
    PdfFitConfig,
    PdfPhaseReference,
    PdfReferenceConfig,
    XrayUvvisEvaluation,
)


def _references(tmp_path: Path) -> PdfReferenceConfig:
    wanted = tmp_path / "wanted.gr"
    impurity = tmp_path / "impurity.gr"
    wanted.write_text("2 1\n3 2\n")
    impurity.write_text("2 2\n3 1\n")
    return PdfReferenceConfig(
        (
            PdfPhaseReference("Wanted", wanted, False),
            PdfPhaseReference("Impurity", impurity, True),
        )
    )


def test_build_dofs_supports_both_precursor_sets() -> None:
    standard = build_dofs()
    oam = build_dofs(use_oam=True)

    assert [(dof.parameter_name, dof.bounds) for dof in standard] == [
        ("infusion_rate_CsPb", (10, 200)),
        ("infusion_rate_Br", (5, 200)),
        ("infusion_rate_I2", (0, 200)),
    ]
    assert [(dof.parameter_name, dof.bounds) for dof in oam] == [
        ("infusion_rate_CsPb", (5, 200)),
        ("infusion_rate_Br", (5, 250)),
        ("infusion_rate_Cl", (5, 200)),
        ("infusion_rate_OAm", (0, 70)),
    ]


def test_dynamic_objectives_tracking_and_constraints(tmp_path: Path) -> None:
    references = _references(tmp_path)
    fitted = PdfFitConfig(PdfEvaluationMode.PDF_FIT_OBJECTIVES)
    tracked = PdfFitConfig(PdfEvaluationMode.RAW_OBJECTIVES_PDF_FIT_TRACKED)
    raw = PdfFitConfig(PdfEvaluationMode.RAW_ONLY)

    assert [
        (item.name, item.minimize) for item in build_objectives(fitted, references)
    ] == [
        ("log_FWHM", True),
        ("log_PLQY", False),
        ("peak_distance", True),
        ("pdf_fit_corr_Wanted", False),
        ("pdf_fit_corr_Impurity", True),
    ]
    assert [item.name for item in build_objectives(raw, references)][-2:] == [
        "corr_Wanted",
        "corr_Impurity",
    ]
    assert pdf_tracking_metrics(fitted, references) == (
        "corr_Wanted",
        "corr_Impurity",
    )
    assert pdf_tracking_metrics(tracked, references) == (
        "pdf_fit_corr_Wanted",
        "pdf_fit_corr_Impurity",
    )
    assert pdf_tracking_metrics(raw, references) == ()
    assert [str(item) for item in build_outcome_constraints(650, 4)] == [
        "Peak >= 646",
        "Peak <= 654",
    ]


def test_headered_history_conversion_and_validation(tmp_path: Path) -> None:
    references = _references(tmp_path)
    path = tmp_path / "history.csv"
    path.write_text(
        "infusion_rate_CsPb,infusion_rate_Br,infusion_rate_I2,"
        "Peak,FWHM,PLQY,corr_Wanted,corr_Impurity,r_2\n"
        "10,20,30,670,0,-1,0.8,0.2,0.9\n"
        "11,21,31,650,30,0.2,0.7,0.3,0.5\n"
    )

    points = load_historical_data(
        path,
        ["infusion_rate_CsPb", "infusion_rate_Br", "infusion_rate_I2"],
        PdfFitConfig(PdfEvaluationMode.RAW_ONLY),
        references,
        peak_target=650,
        r2_min=0.7,
    )

    assert len(points) == 1
    assert points[0]["peak_distance"] == 20
    assert points[0]["log_FWHM"] == pytest.approx(np.log(1000))
    assert points[0]["log_PLQY"] == pytest.approx(np.log(1e-10))
    assert points[0]["corr_Wanted"] == 0.8

    path.write_text("infusion_rate_CsPb,Peak,FWHM,PLQY\n10,660,20,0.1\n")
    with pytest.raises(ValueError, match="corr_Impurity.*corr_Wanted"):
        load_historical_data(
            path,
            ["infusion_rate_CsPb"],
            PdfFitConfig(PdfEvaluationMode.RAW_ONLY),
            references,
        )


def test_plan_kwargs_are_deep_copied_and_xray_is_required() -> None:
    supplied = {
        "flow_config": {"nested": ["original"]},
        "xray_config": {
            "exposure": 7.0,
            "frame_acq_time": 0.25,
            "no_dark": False,
        },
    }

    merged = agent_module._required_plan_kwargs(supplied)

    assert merged["xray_config"] == {
        "exposure": 7.0,
        "frame_acq_time": 0.25,
        "no_dark": False,
        "do_xray": True,
        "stream_name": "scattering",
    }
    merged["flow_config"]["nested"].append("changed")
    assert supplied["flow_config"]["nested"] == ["original"]
    with pytest.raises(ValueError, match="do_xray"):
        agent_module._required_plan_kwargs({"xray_config": {"do_xray": False}})
    with pytest.raises(ValueError, match="stream_name"):
        agent_module._required_plan_kwargs({"xray_config": {"stream_name": "primary"}})


class _FakeRunner:
    def __init__(self) -> None:
        self.current_iteration = 4
        self.future: Future[Any] = Future()
        self.run_kwargs: dict[str, Any] | None = None
        self.submitted: list[dict[str, Any]] | None = None

    def run(self, **kwargs: Any) -> Future[Any]:
        self.run_kwargs = kwargs
        return self.future

    def submit_suggestions(self, suggestions: list[dict[str, Any]]) -> Future[Any]:
        self.submitted = suggestions
        return self.future


class _FakeQueueClient:
    def __init__(self) -> None:
        self.listener_callback: Any = None
        self.stop_count = 0

    def start_listener(self, on_stop: Any) -> None:
        self.listener_callback = on_stop

    def stop_listener(self) -> None:
        self.stop_count += 1


def test_queue_agent_forces_one_point_and_closes_completed_listener() -> None:
    runner = _FakeRunner()
    client = _FakeQueueClient()
    agent = XrayUvvisQueueAgent(cast(Any, runner), cast(Any, client))

    assert agent.run(iterations=3) is runner.future
    assert runner.run_kwargs == {
        "iterations": 3,
        "num_points": 1,
        "checkpoint_interval": None,
    }
    assert agent.submit_suggestion({"x": 1}) is runner.future
    assert runner.submitted == [{"x": 1}]
    with pytest.raises(RuntimeError, match="active acquisition"):
        agent.close()
    runner.future.set_result(None)
    agent.close()
    assert client.stop_count == 1


def test_build_queue_agent_uses_injected_client_without_writes(
    tmp_path: Path,
) -> None:
    references = _references(tmp_path)
    evaluation = XrayUvvisEvaluation(
        object(),
        object(),
        DEFAULT_PLQY_PARAMS,
        references,
        pdf_fit_config=PdfFitConfig(PdfEvaluationMode.RAW_ONLY),
    )
    client = _FakeQueueClient()
    checkpoint = tmp_path / "checkpoint.json"
    supplied = {"xray_config": {"exposure": 3.0, "no_dark": False}}

    agent = build_queue_agent(
        evaluation,
        cast(Any, client),
        sensor_names=("qepro",),
        acquisition_plan_kwargs=supplied,
        checkpoint_path=checkpoint,
    )

    assert client.listener_callback is not None
    assert not checkpoint.exists()
    problem = agent._runner.optimization_problem
    assert problem.actuators == ()
    assert problem.sensors == ("qepro",)
    assert problem.acquisition_plan == "xray_uvvis_acquire"
    assert problem.acquisition_plan_kwargs is not None
    assert problem.acquisition_plan_kwargs["xray_config"] == {
        "exposure": 3.0,
        "no_dark": False,
        "do_xray": True,
        "stream_name": "scattering",
    }
    assert supplied == {"xray_config": {"exposure": 3.0, "no_dark": False}}
    agent.close()
