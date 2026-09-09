from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from blop.ax.queueserver_agent import QueueserverAgent

from xpd_tools.optimization.agent import build_queue_agent
from xpd_tools.optimization.evaluation import XrayUvvisEvaluation


@patch("blop.ax.queueserver_agent.QueueserverOptimizationRunner")
@patch("blop.ax.queueserver_agent.QueueserverClient")
def test_build_queue_agent_exposes_standard_raw_and_fit_schema(
    client_type: Any,
    runner_type: Any,
    tmp_path: Path,
    reference_config_factory: Any,
) -> None:
    reference_path = reference_config_factory(
        phases=(("Wanted", False), ("Impurity", True))
    )
    checkpoint = tmp_path / "checkpoint.json"
    expected_parameters = [
        ("infusion_rate_CsPb", 10.0, 200.0),
        ("infusion_rate_Br", 5.0, 200.0),
        ("infusion_rate_I2", 0.0, 200.0),
    ]
    modes = {
        "raw": (
            "-log_FWHM, log_PLQY, -peak_distance, corr_Wanted, -corr_Impurity",
            [],
        ),
        "fit": (
            "-log_FWHM, log_PLQY, -peak_distance, "
            "pdf_fit_corr_Wanted, -pdf_fit_corr_Impurity",
            ["corr_Wanted", "corr_Impurity"],
        ),
    }
    manager, dispatcher = object(), object()

    built_agents: list[QueueserverAgent] = []
    for mode, (objective, tracking_metrics) in modes.items():
        evaluation = XrayUvvisEvaluation(
            object(),
            object(),
            reference_path,
            pdf_mode=cast(Any, mode),
            peak_target=650,
        )
        agent = build_queue_agent(
            evaluation,
            manager,
            dispatcher,
            peak_tolerance=4,
            checkpoint_path=checkpoint,
        )

        built_agents.append(agent)
        assert isinstance(agent, QueueserverAgent)
        experiment = cast(Any, agent.ax_client._experiment)
        assert [
            (name, parameter.lower, parameter.upper)
            for name, parameter in experiment.search_space.parameters.items()
        ] == expected_parameters
        assert experiment.optimization_config.objective.expression == objective
        assert [
            str(constraint)
            for constraint in experiment.optimization_config.outcome_constraints
        ] == [
            "OutcomeConstraint(Peak >= 646)",
            "OutcomeConstraint(Peak <= 654)",
        ]
        assert [
            metric.name for metric in experiment.tracking_metrics
        ] == tracking_metrics
        assert agent.checkpoint_path == str(checkpoint)
        assert not checkpoint.exists()

        problem = agent.to_optimization_problem()
        assert problem.actuators == []
        assert problem.sensors == ()
        assert problem.acquisition_plan == "xray_uvvis_acquire"
        assert problem.acquisition_plan_kwargs == {}

    assert client_type.call_count == 2
    assert all(
        call.args == (manager, dispatcher) for call in client_type.call_args_list
    )
    assert runner_type.call_count == 2
    agent = built_agents[-1]

    runner = runner_type.return_value
    future = object()
    runner.run.return_value = future
    assert agent.run(iterations=3, n_points=2) is future
    runner.run.assert_called_once_with(
        iterations=3, num_points=2, checkpoint_interval=None
    )
    suggestions = [{"x": 1}, {"x": 2}]
    agent.submit_suggestions(suggestions)
    runner.submit_suggestions.assert_called_once_with(suggestions)
    agent.stop()
    runner.stop.assert_called_once_with()
