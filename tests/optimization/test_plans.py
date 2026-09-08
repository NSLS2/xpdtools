from __future__ import annotations

import inspect
import threading
from collections.abc import Mapping
from typing import Any, cast

import pytest
from bluesky.run_engine import RunEngine
from bluesky.utils import FailedStatus, RunEngineInterrupted
from ophyd import Signal

from xpd_tools.optimization.plans import XrayUvvisPlanContext, create_xray_uvvis_plan


def _flow_config(names: tuple[str, ...]) -> dict[str, Any]:
    standard_pumps = {
        "CsPb": "dds2_p1",
        "Br": "dds2_p2",
        "I2": "dds3_p1",
    }
    return {
        "precursor_prefix_list": list(names),
        "precursor_list": [f"precursor-{name}" for name in names],
        "syringe_list": [50] * len(names),
        "syringe_mater_list": ["steel"] * len(names),
        "target_vol_list": ["30 ml"] * len(names),
        "set_target_list": [True] * len(names),
        "dof_to_pump": {
            f"infusion_rate_{name}": standard_pumps.get(name, f"pump-{name}")
            for name in names
        },
        "post_dilute": False,
        "mixer_lengths_cm": [0.0],
        "resident_t_ratio": 0.0,
    }


def _disabled_options() -> dict[str, dict[str, Any]]:
    return {
        "xray_config": {"do_xray": False},
        "wash_config": {"do_wash": False},
        "quality_config": {"use_good_bad": False, "num_abs": 1, "num_flu": 1},
    }


def _context(
    fake_qepro: Any,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
    *,
    xray_detector: Any = None,
    wrap_xray_run: Any = None,
) -> XrayUvvisPlanContext:
    led, uv_shutter, fast_shutter = optical_signals
    return XrayUvvisPlanContext(
        fake_qepro,
        led,
        uv_shutter,
        fast_shutter,
        fake_pumps,
        xray_detector,
        wrap_xray_run,
    )


def test_factory_name_and_signature(
    fake_qepro: Any,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
) -> None:
    plan = create_xray_uvvis_plan(_context(fake_qepro, fake_pumps, optical_signals))

    assert plan.__name__ == "xray_uvvis_acquire"
    assert list(inspect.signature(plan).parameters) == [
        "suggestions",
        "actuators",
        "sensors",
        "md",
        "flow_config",
        "xray_config",
        "wash_config",
        "quality_config",
    ]


def test_quality_gate_streams_and_canonical_metadata(
    RE: RunEngine,
    documents: list[tuple[str, dict[str, Any]]],
    fake_qepro: Any,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
    bad_spectrum: Any,
    good_spectrum: Any,
) -> None:
    fake_qepro.spectra = [bad_spectrum, good_spectrum, good_spectrum]
    plan = create_xray_uvvis_plan(_context(fake_qepro, fake_pumps, optical_signals))
    metadata = {
        "sample_name": "caller-value",
        "blop_correlation_uid": "correlation",
        "blop_suggestions": [{"_id": 7}],
    }

    result = RE(
        plan(
            [{"_id": 7, "infusion_rate_CsPb": 25}],
            [],
            md=metadata,
            flow_config=_flow_config(("CsPb",)),
            xray_config={"do_xray": False},
            wash_config={"do_wash": False},
            quality_config={
                "use_good_bad": True,
                "good_target": 1,
                "max_bad": 2,
                "num_abs": 1,
                "num_flu": 1,
            },
        )
    )

    start = next(doc for name, doc in documents if name == "start")
    descriptors = {
        doc["uid"]: doc["name"] for name, doc in documents if name == "descriptor"
    }
    quality_events = [
        doc
        for name, doc in documents
        if name == "event" and descriptors[doc["descriptor"]] == "fluorescence_quality"
    ]
    fluorescence_events = [
        doc
        for name, doc in documents
        if name == "event" and descriptors[doc["descriptor"]] == "fluorescence"
    ]
    led, uv_shutter, fast_shutter = optical_signals
    pump = fake_pumps["dds2_p1"]

    assert cast(Any, result).plan_result == start["uid"]
    assert [event["data"]["verdict"] for event in quality_events] == ["bad", "good"]
    assert len(fluorescence_events) == 2
    assert [event["data"]["n_events_in_batch"] for event in quality_events] == [1, 1]
    assert start["sample_name"] == "CsPb_025"
    assert start["blop_correlation_uid"] == "correlation"
    assert start["blop_suggestions"] == [{"_id": 7}]
    assert start["detectors"] == ["QEPro"]
    assert start["quality_thresholds"]["key_height"] == 2000
    assert pump.status.get() == "Stopped"
    assert (led.get(), uv_shutter.get(), fast_shutter.get()) == ("Low", "Low", 20)


@pytest.mark.parametrize(
    "suggestions",
    [
        [],
        [{"infusion_rate_CsPb": 10}, {"infusion_rate_CsPb": 20}],
        [{"_id": 1}],
        [{"infusion_rate_unknown": 10}],
        [{"infusion_rate_CsPb": -1}],
        [{"infusion_rate_CsPb": 0}],
    ],
)
def test_preflight_errors_before_device_messages(
    suggestions: list[dict[str, Any]],
    fake_qepro: Any,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
) -> None:
    plan = create_xray_uvvis_plan(_context(fake_qepro, fake_pumps, optical_signals))
    generator = plan(
        suggestions,
        [],
        flow_config=_flow_config(("CsPb",)),
        **_disabled_options(),
    )

    with pytest.raises(ValueError):
        next(generator)
    assert fake_pumps["dds2_p1"].stop_count == 0


def test_disabled_optional_paths_do_not_require_devices(
    RE: RunEngine,
    fake_qepro: Any,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
) -> None:
    plan = create_xray_uvvis_plan(
        _context(
            fake_qepro,
            {"dds2_p1": fake_pumps["dds2_p1"]},
            optical_signals,
        )
    )

    RE(
        plan(
            [{"infusion_rate_CsPb": 10}],
            [],
            flow_config=_flow_config(("CsPb",)),
            **_disabled_options(),
        )
    )

    assert fake_pumps["dds2_p1"].start_count == 1


def test_xray_wrapper_and_detector_metadata(
    RE: RunEngine,
    documents: list[tuple[str, dict[str, Any]]],
    fake_qepro: Any,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
    fake_area_detector: Any,
) -> None:
    wrapped: list[bool] = []

    def wrap(plan: Any, no_dark: bool):
        wrapped.append(no_dark)
        return plan

    plan = create_xray_uvvis_plan(
        _context(
            fake_qepro,
            {"dds2_p1": fake_pumps["dds2_p1"]},
            optical_signals,
            xray_detector=fake_area_detector,
            wrap_xray_run=wrap,
        )
    )
    RE(
        plan(
            [{"infusion_rate_CsPb": 10}],
            [],
            flow_config=_flow_config(("CsPb",)),
            xray_config={
                "do_xray": True,
                "exposure": 0.25,
                "frame_acq_time": 0.1,
                "no_dark": False,
            },
            wash_config={"do_wash": False},
            quality_config={"use_good_bad": False, "num_abs": 1, "num_flu": 1},
        )
    )

    start = next(doc for name, doc in documents if name == "start")
    assert wrapped == [False]
    assert fake_area_detector.images_per_set.get() == 3
    assert start["detectors"] == ["QEPro", "xray_detector"]
    assert start["sp_num_frames"] == 3
    assert any(
        doc["name"] == "scattering" for name, doc in documents if name == "descriptor"
    )


def test_partial_start_and_inner_failure_cleanup(
    RE: RunEngine,
    fake_qepro: Any,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
) -> None:
    first = fake_pumps["dds2_p1"]
    second = fake_pumps["dds2_p2"]
    second.fail_start = True
    plan = create_xray_uvvis_plan(
        _context(
            fake_qepro,
            {"pump-A": first, "pump-B": second},
            optical_signals,
        )
    )

    with pytest.raises(RuntimeError, match="failed to start"):
        RE(
            plan(
                [{"infusion_rate_A": 10, "infusion_rate_B": 20}],
                [],
                flow_config=_flow_config(("A", "B")),
                **_disabled_options(),
            )
        )

    assert first.start_count == 1
    assert first.stop_count == 2
    assert first.status.get() == "Stopped"
    assert second.start_count == 1
    assert second.stop_count == 1
    assert tuple(signal.get() for signal in optical_signals) == ("Low", "Low", 20)


def test_wash_failure_and_scattering_failure_close_safely(
    RE: RunEngine,
    fake_qepro: Any,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
    fake_area_detector: Any,
) -> None:
    synthesis = fake_pumps["dds2_p1"]
    wash = fake_pumps["ultra1"]
    wash.fail_start = True
    plan = create_xray_uvvis_plan(
        _context(
            fake_qepro,
            {"dds2_p1": synthesis, "wash": wash},
            optical_signals,
        )
    )
    with pytest.raises(RuntimeError, match="failed to start"):
        RE(
            plan(
                [{"infusion_rate_CsPb": 10}],
                [],
                flow_config=_flow_config(("CsPb",)),
                xray_config={"do_xray": False},
                wash_config={
                    "do_wash": True,
                    "pump_names": ["wash"],
                    "syringe_list": [50],
                    "rate_list": ["1 ml/hr"],
                    "duration_sec": 0,
                    "syringe_mater_list": ["steel"],
                    "target_vol_list": ["30 ml"],
                    "set_target_list": [False],
                },
                quality_config={"use_good_bad": False, "num_abs": 1, "num_flu": 1},
            )
        )
    assert synthesis.status.get() == "Stopped"
    assert wash.start_count == 1
    assert tuple(signal.get() for signal in optical_signals) == ("Low", "Low", 20)

    wash.fail_start = False
    fake_area_detector.fail_trigger = True
    wrapped: list[bool] = []

    def wrap(inner: Any, no_dark: bool):
        wrapped.append(no_dark)
        return inner

    plan = create_xray_uvvis_plan(
        _context(
            fake_qepro,
            {"dds2_p1": synthesis},
            optical_signals,
            xray_detector=fake_area_detector,
            wrap_xray_run=wrap,
        )
    )
    with pytest.raises(FailedStatus):
        RE(
            plan(
                [{"infusion_rate_CsPb": 10}],
                [],
                flow_config=_flow_config(("CsPb",)),
                xray_config={"do_xray": True},
                wash_config={"do_wash": False},
                quality_config={"use_good_bad": False, "num_abs": 1, "num_flu": 1},
            )
        )
    assert wrapped == [True]
    assert synthesis.status.get() == "Stopped"
    assert optical_signals[2].get() == 20


def test_cleanup_runs_when_plan_is_cancelled(
    RE: RunEngine,
    fake_qepro: Any,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
) -> None:
    pump = fake_pumps["dds2_p1"]
    plan = create_xray_uvvis_plan(
        _context(
            fake_qepro,
            {"dds2_p1": pump},
            optical_signals,
        )
    )
    flow = _flow_config(("CsPb",))
    flow["mixer_lengths_cm"] = [1.0]
    flow["resident_t_ratio"] = 1.0
    pause = threading.Timer(0.1, RE.request_pause)
    pause.start()
    try:
        with pytest.raises(RunEngineInterrupted):
            RE(
                plan(
                    [{"infusion_rate_CsPb": 10}],
                    [],
                    flow_config=flow,
                    xray_config={"do_xray": False},
                    wash_config={"do_wash": False},
                    quality_config={
                        "use_good_bad": False,
                        "num_abs": 1,
                        "num_flu": 2,
                    },
                )
            )
        RE.abort()
    finally:
        pause.cancel()

    assert pump.status.get() == "Stopped"
    assert tuple(signal.get() for signal in optical_signals) == ("Low", "Low", 20)
