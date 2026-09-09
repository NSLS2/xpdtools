from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xpd_tools.optimization.evaluation import XrayUvvisEvaluation


def _catalogs(
    tiled: Any,
    wavelength: np.ndarray,
    fluorescence: np.ndarray,
    *,
    quality: dict[str, Any] | None = None,
    use_good_bad: bool = False,
    fluorescence_failures: int = 0,
    sandbox_failures: int = 0,
    pdf: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[Any, Any, Any, Any]:
    absorbance = (
        0.0001 * wavelength
        + 0.2
        + 0.25 * np.exp(-((wavelength - 365) ** 2) / (2 * 12**2))
    )
    dataset = lambda values: {  # noqa: E731
        "QEPro_x_axis": np.broadcast_to(wavelength, np.shape(values)),
        "QEPro_output": values,
    }
    fluorescence_stream = tiled.Stream(
        dataset(fluorescence), failures=fluorescence_failures
    )
    absorbance_stream = tiled.Stream(dataset(absorbance))
    streams = {"fluorescence": fluorescence_stream, "absorbance": absorbance_stream}
    if quality is not None:
        streams["fluorescence_quality"] = tiled.Stream(quality)
    raw = tiled.Catalog({"uid": tiled.Run(streams, use_good_bad=use_good_bad)})

    radial, profile = pdf or (
        np.linspace(1.0, 25.0, 241),
        np.sin(np.linspace(1.0, 25.0, 241)),
    )
    scattering = tiled.Stream({"gr_r": radial, "gr_G": profile})
    sandbox = tiled.Catalog(
        {"sandbox": tiled.Run({"scattering": scattering})},
        search_failures=sandbox_failures,
    )
    return raw, sandbox, fluorescence_stream, absorbance_stream


def _reference_json(path: Path, gr_path: Path, cif_path: Path) -> Path:
    phase = {
        "name": "Target",
        "gr_path": str(gr_path),
        "cif_path": str(cif_path),
        "minimize": False,
    }
    path.write_text(json.dumps({"schema_version": 1, "phases": [phase]}))
    return path


def test_reference_configuration_resolves_paths_and_modes(
    tmp_path: Path, reference_config_factory: Any
) -> None:
    config_path = reference_config_factory(
        phases=(("Wanted", False), ("Impurity", True))
    )
    evaluator = XrayUvvisEvaluation(object(), object(), config_path)

    assert evaluator.pdf_mode == "fit"
    assert evaluator.peak_target == 660
    assert [(phase.name, phase.minimize) for phase in evaluator.phases] == [
        ("Wanted", False),
        ("Impurity", True),
    ]
    assert evaluator.phases[0].gr_path == tmp_path / "Wanted.gr"
    assert evaluator.phases[0].cif_path == tmp_path / "Wanted.cif"

    no_cif = reference_config_factory(include_cif=False)
    assert (
        XrayUvvisEvaluation(object(), object(), no_cif, pdf_mode="raw").pdf_mode
        == "raw"
    )
    with pytest.raises(ValueError, match="Target.*cif_path"):
        XrayUvvisEvaluation(object(), object(), no_cif)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update(schema_version=2), "schema_version"),
        (lambda data: data.update(extra=True), "unknown fields"),
        (lambda data: data["phases"][0].pop("minimize"), "missing fields"),
        (lambda data: data["phases"][0].update(name="not-valid!"), "name"),
        (
            lambda data: data["phases"][0].update(constraint_profile="unknown"),
            "constraint_profile",
        ),
        (
            lambda data: data["phases"].append(dict(data["phases"][0])),
            "duplicated",
        ),
        (lambda data: data["phases"][0].update(gr_path="missing.gr"), "gr_path"),
    ],
)
def test_reference_configuration_rejects_invalid_schema(
    reference_config_factory: Any, mutation: Any, message: str
) -> None:
    config_path = reference_config_factory(include_cif=False)
    data = json.loads(config_path.read_text())
    mutation(data)
    config_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        XrayUvvisEvaluation(object(), object(), config_path, pdf_mode="raw")


def test_two_field_reads_preserve_successes_across_retries(
    tiled_fakes: Any,
    wavelength: np.ndarray,
    good_spectrum: np.ndarray,
    reference_config_factory: Any,
) -> None:
    raw, sandbox, fluorescence, absorbance = _catalogs(
        tiled_fakes,
        wavelength,
        good_spectrum,
        fluorescence_failures=1,
        sandbox_failures=1,
    )
    evaluator = XrayUvvisEvaluation(
        raw,
        sandbox,
        reference_config_factory(include_cif=False),
        pdf_mode="raw",
        max_retries=3,
        retry_delay=0,
    )
    outcome = evaluator("uid", [{"_id": 7}])[0]

    assert outcome["_id"] == 7
    assert outcome["corr_Target"] == pytest.approx(1.0)
    assert (fluorescence.read_count, absorbance.read_count, sandbox.search_count) == (
        2,
        1,
        2,
    )


def test_schema_errors_are_immediate_and_access_errors_retain_context(
    tiled_fakes: Any,
    wavelength: np.ndarray,
    good_spectrum: np.ndarray,
    reference_config_factory: Any,
) -> None:
    config = reference_config_factory(include_cif=False)
    raw, sandbox, fluorescence, _ = _catalogs(tiled_fakes, wavelength, good_spectrum)
    fluorescence.data.pop("QEPro_output")
    evaluator = XrayUvvisEvaluation(
        raw, sandbox, config, pdf_mode="raw", max_retries=2, retry_delay=0
    )
    with pytest.raises(ValueError, match="QEPro_output"):
        evaluator("uid", [{"_id": 1}])
    assert fluorescence.read_count == 1

    raw, sandbox, fluorescence, _ = _catalogs(tiled_fakes, wavelength, good_spectrum)
    raw.runs["uid"].streams.pop("absorbance")
    evaluator = XrayUvvisEvaluation(
        raw, sandbox, config, pdf_mode="raw", max_retries=2, retry_delay=0
    )
    with pytest.raises(RuntimeError, match="Missing: absorbance stream") as exc:
        evaluator("uid", [{"_id": 1}])
    assert isinstance(exc.value.__cause__, KeyError)
    assert fluorescence.read_count == 1


def test_quality_batches_exactly_partition_and_filter_events(
    tiled_fakes: Any,
    wavelength: np.ndarray,
    good_spectrum: np.ndarray,
    reference_config_factory: Any,
) -> None:
    other_peak = 5000 * np.exp(-((wavelength - 610) ** 2) / (2 * 20**2))
    quality = {
        "verdict": np.array(["bad", "good"]),
        "n_events_in_batch": np.array([1, 1]),
    }
    raw, sandbox, _, _ = _catalogs(
        tiled_fakes,
        wavelength,
        np.vstack([other_peak, good_spectrum]),
        quality=quality,
        use_good_bad=True,
    )
    config = reference_config_factory(include_cif=False)
    evaluator = XrayUvvisEvaluation(raw, sandbox, config, pdf_mode="raw")
    assert evaluator("uid", [{"_id": 2}])[0]["Peak"] == pytest.approx(660)

    quality_stream = raw.runs["uid"].streams["fluorescence_quality"]
    quality_stream.data["n_events_in_batch"].values = np.array([1, 0])
    with pytest.raises(ValueError, match="exactly partition"):
        evaluator("uid", [{"_id": 2}])


def test_no_peak_outcomes_are_finite_penalties(
    tiled_fakes: Any, wavelength: np.ndarray, reference_config_factory: Any
) -> None:
    raw, sandbox, _, _ = _catalogs(tiled_fakes, wavelength, np.zeros_like(wavelength))
    evaluator = XrayUvvisEvaluation(
        raw,
        sandbox,
        reference_config_factory(include_cif=False),
        pdf_mode="raw",
        peak_target=650,
    )
    outcome = evaluator("uid", [{"_id": 7}])[0]

    assert outcome["_id"] == 7
    assert outcome["Peak"] == 0.0
    assert outcome["peak_distance"] == 650
    assert outcome["log_FWHM"] == pytest.approx(np.log(1000.0))
    assert outcome["log_PLQY"] == pytest.approx(np.log(1e-10))
    assert all(np.isfinite(value) for key, value in outcome.items() if key != "_id")


def test_fit_failure_wraps_original_error(
    tmp_path: Path,
    tiled_fakes: Any,
    wavelength: np.ndarray,
    good_spectrum: np.ndarray,
) -> None:
    radial = np.linspace(1.0, 25.0, 241)
    gr_path, cif_path = tmp_path / "phase.gr", tmp_path / "invalid.cif"
    np.savetxt(gr_path, np.column_stack((radial, np.sin(radial))))
    cif_path.write_text("not a CIF")
    config = _reference_json(tmp_path / "references.json", gr_path, cif_path)
    raw, sandbox, _, _ = _catalogs(tiled_fakes, wavelength, good_spectrum)

    with pytest.raises(RuntimeError, match="PDF fitting failed for uid='uid'") as exc:
        XrayUvvisEvaluation(raw, sandbox, config)("uid", [{"_id": 3}])
    assert exc.value.__cause__ is not None


@pytest.mark.timeout(60)
def test_external_target_pdf_fit_is_finite(
    tmp_path: Path,
    tiled_fakes: Any,
    wavelength: np.ndarray,
    good_spectrum: np.ndarray,
) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    radial, profile = np.loadtxt(fixtures / "target.gr", unpack=True)
    config = _reference_json(
        tmp_path / "references.json", fixtures / "target.gr", fixtures / "target.cif"
    )
    raw, sandbox, _, _ = _catalogs(
        tiled_fakes, wavelength, good_spectrum, pdf=(radial, profile)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outcome = XrayUvvisEvaluation(raw, sandbox, config)("uid", [{"_id": 9}])[0]

    assert outcome["corr_Target"] == pytest.approx(1.0)
    assert -1 <= outcome["pdf_fit_corr_Target"] <= 1
