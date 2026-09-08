from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
from bluesky import plan_stubs as bps
from bluesky.run_engine import RunEngine
from ophyd import Signal

from xpd_tools.optimization.evaluation import (
    DEFAULT_PLQY_PARAMS,
    PdfEvaluationMode,
    PdfFitConfig,
    PdfReferenceConfig,
    XrayUvvisEvaluation,
)
from xpd_tools.optimization.plans import XrayUvvisPlanContext, create_xray_uvvis_plan


@dataclass
class _ArrayField:
    values: Any


class _Stream:
    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self.data = data

    def read(self) -> dict[str, _ArrayField]:
        return {name: _ArrayField(values) for name, values in self.data.items()}


class _Run:
    def __init__(
        self,
        metadata: Mapping[str, Any],
        streams: Mapping[str, _Stream],
    ) -> None:
        self.metadata = {"start": metadata}
        self.streams = streams

    def __getitem__(self, name: str) -> _Stream:
        return self.streams[name]


class _Keys(tuple[str, ...]):
    def last(self) -> str:
        return self[-1]


class _Catalog:
    def __init__(self, runs: Mapping[str, _Run]) -> None:
        self.runs = dict(runs)

    def __getitem__(self, uid: Hashable) -> _Run:
        return self.runs[str(uid)]

    def search(self, query: Any) -> _Catalog:
        return self

    def keys(self) -> _Keys:
        return _Keys(self.runs)


def _run_from_documents(
    documents: Sequence[tuple[str, Mapping[str, Any]]], uid: str
) -> _Run:
    start = next(
        doc for name, doc in documents if name == "start" and doc["uid"] == uid
    )
    descriptors = {
        doc["uid"]: doc["name"]
        for name, doc in documents
        if name == "descriptor" and doc["run_start"] == uid
    }
    events: dict[str, list[Mapping[str, Any]]] = {
        stream_name: [] for stream_name in descriptors.values()
    }
    for name, doc in documents:
        if name == "event" and doc["descriptor"] in descriptors:
            events[descriptors[doc["descriptor"]]].append(doc["data"])
    streams = {
        stream_name: _Stream(
            {
                field: np.asarray([event[field] for event in stream_events])
                for field in stream_events[0]
            }
        )
        for stream_name, stream_events in events.items()
        if stream_events
    }
    return _Run(start, streams)


def test_simulated_acquisition_evaluates_external_references(
    RE: RunEngine,
    documents: list[tuple[str, dict[str, Any]]],
    fake_qepro: Any,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
    wavelength: np.ndarray,
    good_spectrum: np.ndarray,
    reference_config_factory: Any,
) -> None:
    absorbance = (
        0.0001 * wavelength
        + 0.2
        + 0.25 * np.exp(-((wavelength - 365) ** 2) / (2 * 12**2))
    )
    fake_qepro.spectra = [good_spectrum, good_spectrum, absorbance, absorbance]

    class _UnusedXray:
        @property
        def name(self) -> str:
            raise AssertionError("disabled X-ray detector was accessed")

    def unused_wrapper(plan: Any, no_dark: bool):
        raise AssertionError("disabled X-ray wrapper was called")

    led, uv_shutter, fast_shutter = optical_signals
    plan = create_xray_uvvis_plan(
        XrayUvvisPlanContext(
            fake_qepro,
            led,
            uv_shutter,
            fast_shutter,
            {"dds2_p1": fake_pumps["dds2_p1"]},
            _UnusedXray(),
            unused_wrapper,
        )
    )
    flow = {
        "precursor_prefix_list": ["CsPb"],
        "precursor_list": ["CsPbOA"],
        "syringe_list": [50],
        "syringe_mater_list": ["steel"],
        "target_vol_list": ["30 ml"],
        "set_target_list": [True],
        "dof_to_pump": {"infusion_rate_CsPb": "dds2_p1"},
        "post_dilute": False,
        "mixer_lengths_cm": [0.0],
        "resident_t_ratio": 0.0,
    }
    acquisition = RE(
        plan(
            [{"_id": 7, "infusion_rate_CsPb": 25}],
            [],
            md={
                "blop_correlation_uid": "correlation-7",
                "blop_suggestions": [{"_id": 7}],
            },
            flow_config=flow,
            xray_config={"do_xray": False},
            wash_config={"do_wash": False},
            quality_config={
                "use_good_bad": False,
                "num_abs": 2,
                "num_flu": 2,
            },
        )
    )
    raw_uid = cast(Any, acquisition).plan_result

    radial = np.linspace(1.0, 25.0, 241)
    profile = np.sin(radial)
    gr_r = Signal(name="gr_r", value=radial)
    gr_g = Signal(name="gr_G", value=profile)

    def sandbox_plan():
        sandbox_uid = yield from bps.open_run(md={"original_run_uid": raw_uid})
        yield from bps.trigger_and_read(cast(Any, [gr_r, gr_g]), name="scattering")
        yield from bps.close_run()
        return sandbox_uid

    sandbox_uid = cast(Any, RE(sandbox_plan())).plan_result
    raw_run = _run_from_documents(documents, raw_uid)
    sandbox_run = _run_from_documents(documents, sandbox_uid)
    references = PdfReferenceConfig.from_json(
        reference_config_factory(include_cif=False)
    )
    evaluator = XrayUvvisEvaluation(
        _Catalog({raw_uid: raw_run}),
        _Catalog({sandbox_uid: sandbox_run}),
        DEFAULT_PLQY_PARAMS,
        references,
        pdf_fit_config=PdfFitConfig(PdfEvaluationMode.RAW_ONLY),
        max_retries=1,
        retry_delay=0,
    )

    outcome = evaluator(raw_uid, [{"_id": 7}])[0]

    assert raw_uid == raw_run.metadata["start"]["uid"]
    assert raw_run.metadata["start"]["blop_correlation_uid"] == "correlation-7"
    assert raw_run.metadata["start"]["quality_config"]["num_flu"] == 2
    assert raw_run.metadata["start"]["quality_config"]["num_abs"] == 2
    assert fake_qepro.trigger_count == 4
    assert fake_pumps["dds2_p1"].status.get() == "Stopped"
    assert (led.get(), uv_shutter.get(), fast_shutter.get()) == ("Low", "Low", 20)
    assert outcome["_id"] == 7
    assert outcome["peak_distance"] == pytest.approx(0, abs=1e-6)
    assert set(outcome) >= {
        "Peak",
        "peak_distance",
        "log_FWHM",
        "log_PLQY",
        "corr_Target",
        "_id",
    }
    assert all(np.isfinite(value) for key, value in outcome.items() if key != "_id")
