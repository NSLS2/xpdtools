from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest
from bluesky import plan_stubs as bps
from bluesky.run_engine import RunEngine
from ophyd import Signal

from xpd_tools.optimization.evaluation import XrayUvvisEvaluation
from xpd_tools.optimization.plans import QualityPolicy, create_xray_uvvis_plan


def test_simulated_acquisition_evaluates_external_references(
    RE: RunEngine,
    documents: list[tuple[str, dict[str, Any]]],
    fake_qepro: Any,
    fake_pumps: Mapping[str, Any],
    optical_signals: tuple[Signal, Signal, Signal],
    wavelength: np.ndarray,
    good_spectrum: np.ndarray,
    reference_config_factory: Any,
    plan_context_factory: Any,
    tiled_fakes: Any,
) -> None:
    absorbance = (
        0.0001 * wavelength
        + 0.2
        + 0.25 * np.exp(-((wavelength - 365) ** 2) / (2 * 12**2))
    )
    fake_qepro.spectra = [good_spectrum, good_spectrum, absorbance, absorbance]
    context = plan_context_factory(
        quality=QualityPolicy(
            enabled=True,
            good_batches=1,
            max_bad_batches=2,
            absorbance_shots=2,
            fluorescence_shots=2,
        )
    )
    plan = create_xray_uvvis_plan(context)
    acquisition = RE(
        plan(
            [
                {
                    "_id": 7,
                    "infusion_rate_CsPb": 25,
                    "infusion_rate_Br": 15,
                    "infusion_rate_I2": 5,
                }
            ],
            [],
            md={
                "blop_correlation_uid": "correlation-7",
                "blop_suggestions": [{"_id": 7}],
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
    raw_run = tiled_fakes.run_from_documents(documents, raw_uid)
    sandbox_run = tiled_fakes.run_from_documents(documents, sandbox_uid)
    evaluator = XrayUvvisEvaluation(
        tiled_fakes.Catalog({raw_uid: raw_run}),
        tiled_fakes.Catalog({sandbox_uid: sandbox_run}),
        reference_config_factory(include_cif=False),
        pdf_mode="raw",
        max_retries=1,
        retry_delay=0,
    )

    outcome = evaluator(raw_uid, [{"_id": 7}])[0]

    metadata = raw_run.metadata["start"]
    assert raw_uid == metadata["uid"]
    assert metadata["blop_correlation_uid"] == "correlation-7"
    assert metadata["blop_suggestions"] == [{"_id": 7}]
    assert metadata["infuse_rates"] == [25.0, 15.0, 5.0]
    assert metadata["xray_uvvis_config"]["quality"]["fluorescence_shots"] == 2
    assert metadata["xray_uvvis_config"]["quality"]["absorbance_shots"] == 2
    assert set(raw_run.streams) >= {
        "fluorescence",
        "absorbance",
        "fluorescence_quality",
        "scattering",
    }
    assert len(raw_run.streams["fluorescence"].data["QEPro_output"].values) == 2
    assert len(raw_run.streams["absorbance"].data["QEPro_output"].values) == 2
    assert fake_qepro.trigger_count == 4
    assert all(pump.status.get() == "Stopped" for pump in fake_pumps.values())
    led, uv_shutter, fast_shutter = optical_signals
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
    assert "pdf_fit_corr_Target" not in outcome
    assert all(np.isfinite(value) for key, value in outcome.items() if key != "_id")
