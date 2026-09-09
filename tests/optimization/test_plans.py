from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest
from bluesky.run_engine import RunEngine
from bluesky.utils import FailedStatus, RunEngineInterrupted
from ophyd import Signal

from xpd_tools.optimization.plans import (
    DilutionStage,
    FlowSource,
    QualityPolicy,
    WashCycle,
    XraySettings,
    create_xray_uvvis_plan,
)


def _source(name: str, pump: Any) -> FlowSource:
    return FlowSource(
        dof=f"infusion_rate_{name}",
        pump=pump,
        precursor=f"precursor-{name}",
        sample_label=name,
    )


def test_quality_gate_xray_and_canonical_metadata(
    RE: RunEngine,
    documents: list[tuple[str, dict[str, Any]]],
    fake_qepro: Any,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
    fake_area_detector: Any,
    wavelength: np.ndarray,
    bad_spectrum: np.ndarray,
    good_spectrum: np.ndarray,
    plan_context_factory: Any,
) -> None:
    absorbance = (
        0.0001 * wavelength
        + 0.2
        + 0.25 * np.exp(-((wavelength - 365) ** 2) / (2 * 12**2))
    )
    fake_qepro.spectra = [bad_spectrum, good_spectrum, absorbance]
    wrapped: list[bool] = []

    def wrap(plan: Any, no_dark: bool):
        wrapped.append(no_dark)
        return plan

    pump = fake_pumps["dds2_p1"]
    context = plan_context_factory(
        sources=(_source("CsPb", pump),),
        quality=QualityPolicy(
            enabled=True,
            good_batches=1,
            max_bad_batches=2,
            absorbance_shots=1,
            fluorescence_shots=1,
        ),
        xray=XraySettings(exposure=0.25, frame_acq_time=0.1, no_dark=False),
        wrap_xray_run=wrap,
    )
    plan = create_xray_uvvis_plan(context)
    metadata = {
        "sample_name": "caller-value",
        "flow_config": "obsolete",
        "blop_correlation_uid": "correlation",
        "blop_suggestions": [{"_id": 7}],
    }
    safe_at_stop: list[tuple[str, str, int]] = []

    def record_stop(name: str, doc: dict[str, Any]) -> None:
        if name == "stop":
            safe_at_stop.append(
                cast(
                    tuple[str, str, int],
                    tuple(signal.get() for signal in optical_signals),
                )
            )

    token = RE.subscribe(record_stop)
    try:
        result = RE(
            plan(
                [{"_id": 7, "infusion_rate_CsPb": 25}],
                [],
                md=metadata,
            )
        )
    finally:
        RE.unsubscribe(token)

    start = next(doc for name, doc in documents if name == "start")
    descriptors = {
        doc["uid"]: doc["name"] for name, doc in documents if name == "descriptor"
    }
    quality_events = [
        doc
        for name, doc in documents
        if name == "event" and descriptors[doc["descriptor"]] == "fluorescence_quality"
    ]
    stream_counts = {
        stream: sum(
            name == "event" and descriptors.get(doc["descriptor"]) == stream
            for name, doc in documents
            if name == "event"
        )
        for stream in ("fluorescence", "absorbance", "scattering")
    }

    assert plan.__name__ == "xray_uvvis_acquire"
    assert cast(Any, result).plan_result == start["uid"]
    assert [event["data"]["verdict"] for event in quality_events] == ["bad", "good"]
    assert [event["data"]["n_events_in_batch"] for event in quality_events] == [1, 1]
    assert stream_counts == {"fluorescence": 2, "absorbance": 1, "scattering": 1}
    assert wrapped == [False]
    assert fake_area_detector.images_per_set.get() == 3
    assert start["sp_num_frames"] == 3
    assert start["sample_name"] == "CsPb_025"
    assert start["sample_type"] == "CsPb_025"
    assert start["blop_correlation_uid"] == "correlation"
    assert start["blop_suggestions"] == [{"_id": 7}]
    assert start["dof_names"] == ["infusion_rate_CsPb"]
    assert start["precursors"] == ["precursor-CsPb"]
    assert start["pumps"] == ["dds2_p1"]
    assert start["detectors"] == ["QEPro", "xray_detector"]
    assert start["use_good_bad"] is True
    assert start["xray_uvvis_config"]["sources"][0]["pump"] == "dds2_p1"
    assert start["xray_uvvis_config"]["xray"]["exposure"] == 0.25
    assert "flow_config" not in start
    assert pump.status.get() == "Stopped"
    assert safe_at_stop == [("Low", "Low", 20)]


@pytest.mark.parametrize(
    "suggestions",
    [
        [],
        [{"infusion_rate_CsPb": 10}, {"infusion_rate_CsPb": 20}],
        [{"_id": 1}],
        [{"infusion_rate_unknown": 10}],
        [{"infusion_rate_CsPb": -1}],
        [{"infusion_rate_CsPb": np.nan}],
        [{"infusion_rate_CsPb": 0}],
        [{"infusion_rate_CsPb": 10, "extra": 1}],
    ],
)
def test_preflight_errors_before_device_messages(
    suggestions: list[dict[str, Any]],
    fake_pumps: Mapping[str, Any],
    fake_area_detector: Any,
    plan_context_factory: Any,
) -> None:
    pump = fake_pumps["dds2_p1"]
    plan = create_xray_uvvis_plan(
        plan_context_factory(
            sources=(_source("CsPb", pump),),
        )
    )

    with pytest.raises(ValueError):
        next(plan(suggestions, []))
    assert pump.stop_count == 0
    assert fake_area_detector.images_per_set.get() == 1


def test_factory_validates_static_context(
    fake_pumps: Mapping[str, Any], plan_context_factory: Any
) -> None:
    source = _source("CsPb", fake_pumps["dds2_p1"])
    with pytest.raises(ValueError, match="unique"):
        create_xray_uvvis_plan(plan_context_factory(sources=(source, source)))


def test_both_dilution_positions_use_total_source_rate(
    RE: RunEngine,
    fake_pumps: Mapping[str, Any],
    plan_context_factory: Any,
) -> None:
    source = fake_pumps["dds2_p1"]
    before = fake_pumps["dds1_p1"]
    after = fake_pumps["ultra2"]
    context = plan_context_factory(
        sources=(_source("CsPb", source),),
        dilutions=(
            DilutionStage(
                pump=before,
                ratio=2,
                position="before_equilibrium",
                syringe_ml=20,
                material="plastic_BD",
                target_ml=20,
            ),
            DilutionStage(
                pump=after,
                ratio=3,
                position="after_equilibrium",
                syringe_ml=100,
                material="steel",
                target_ml=100,
            ),
        ),
    )

    RE(create_xray_uvvis_plan(context)([{"infusion_rate_CsPb": 20}], []))

    assert source.configurations[0]["infuse_rate"] == 20
    assert before.configurations[0]["infuse_rate"] == 40
    assert after.configurations[0]["infuse_rate"] == 60
    assert before.configurations[0]["infuse_unit"] == "ul/min"
    assert after.configurations[0]["target_unit"] == "ml"
    assert (source.status.get(), before.status.get(), after.status.get()) == (
        "Stopped",
        "Stopped",
        "Stopped",
    )


def test_partial_start_failure_cleans_only_successfully_started_pumps(
    RE: RunEngine,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
    plan_context_factory: Any,
) -> None:
    first = fake_pumps["dds2_p1"]
    second = fake_pumps["dds2_p2"]
    second.fail_start = True
    context = plan_context_factory(
        sources=(_source("A", first), _source("B", second)),
    )

    with pytest.raises(RuntimeError, match="failed to start"):
        RE(
            create_xray_uvvis_plan(context)(
                [{"infusion_rate_A": 10, "infusion_rate_B": 20}], []
            )
        )

    assert first.start_count == 1
    assert first.stop_count == 2
    assert first.status.get() == "Stopped"
    assert second.start_count == 1
    assert second.stop_count == 1
    assert tuple(signal.get() for signal in optical_signals) == ("Low", "Low", 20)


def test_wash_failure_closes_devices_safely(
    RE: RunEngine,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
    plan_context_factory: Any,
) -> None:
    source = fake_pumps["dds2_p1"]
    wash = fake_pumps["ultra1"]
    wash.fail_start = True
    context = plan_context_factory(
        sources=(_source("CsPb", source),),
        wash_cycles=(WashCycle(pump=wash, rate_ul_min=500, duration_sec=0),),
    )

    with pytest.raises(RuntimeError, match="failed to start"):
        RE(create_xray_uvvis_plan(context)([{"infusion_rate_CsPb": 10}], []))

    assert source.status.get() == "Stopped"
    assert wash.start_count == 1
    assert tuple(signal.get() for signal in optical_signals) == ("Low", "Low", 20)


def test_scattering_failure_closes_shutter_and_pumps(
    RE: RunEngine,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
    fake_area_detector: Any,
    plan_context_factory: Any,
) -> None:
    source = fake_pumps["dds2_p1"]
    fake_area_detector.fail_trigger = True
    wrapped: list[bool] = []

    def wrap(plan: Any, no_dark: bool):
        wrapped.append(no_dark)
        return plan

    context = plan_context_factory(
        sources=(_source("CsPb", source),),
        wrap_xray_run=wrap,
    )

    with pytest.raises(FailedStatus):
        RE(create_xray_uvvis_plan(context)([{"infusion_rate_CsPb": 10}], []))

    assert wrapped == [True]
    assert source.status.get() == "Stopped"
    assert optical_signals[2].get() == 20


def test_cleanup_aggregates_failures_and_continues_safe_actions(
    RE: RunEngine,
    fake_qepro: Any,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
    plan_context_factory: Any,
) -> None:
    first = fake_pumps["dds2_p1"]
    second = fake_pumps["dds2_p2"]
    first.fail_stop_call = 2
    fake_qepro.fail_on_trigger = 1
    context = plan_context_factory(
        sources=(_source("A", first), _source("B", second)),
    )

    with pytest.raises(ExceptionGroup, match="acquisition cleanup failed") as exc:
        RE(
            create_xray_uvvis_plan(context)(
                [{"infusion_rate_A": 10, "infusion_rate_B": 20}], []
            )
        )

    assert isinstance(exc.value.__cause__, FailedStatus)
    assert first.stop_count == 2
    assert second.stop_count == 2
    assert second.status.get() == "Stopped"
    assert tuple(signal.get() for signal in optical_signals) == ("Low", "Low", 20)


def test_cleanup_runs_when_plan_is_cancelled(
    RE: RunEngine,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
    plan_context_factory: Any,
) -> None:
    pump = fake_pumps["dds2_p1"]
    context = plan_context_factory(
        sources=(_source("CsPb", pump),),
        mixer_lengths_cm=(1.0,),
        residence_time_ratio=1.0,
    )
    pause = threading.Timer(0.1, RE.request_pause)
    pause.start()
    try:
        with pytest.raises(RunEngineInterrupted):
            RE(create_xray_uvvis_plan(context)([{"infusion_rate_CsPb": 10}], []))
        RE.abort()
    finally:
        pause.cancel()

    assert pump.status.get() == "Stopped"
    assert tuple(signal.get() for signal in optical_signals) == ("Low", "Low", 20)
