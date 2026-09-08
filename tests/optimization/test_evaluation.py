from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from xpd_tools.optimization.agent import build_objectives
from xpd_tools.optimization.evaluation import (
    DEFAULT_PLQY_PARAMS,
    QEPRO_FIELDS,
    PdfEvaluationMode,
    PdfFitConfig,
    PdfPhaseReference,
    PdfReferenceConfig,
    XrayUvvisEvaluation,
    read_qepro_stream,
)


@dataclass
class _Field:
    values: Any


class _Stream:
    def __init__(self, data: dict[str, Any], *, failures: int = 0) -> None:
        self.data = data
        self.failures = failures
        self.read_count = 0

    def read(self) -> dict[str, Any]:
        self.read_count += 1
        if self.read_count <= self.failures:
            raise OSError("stream is not ready")
        return self.data


class _Run:
    def __init__(self, streams: dict[str, _Stream], *, use_good_bad: bool = False):
        self.streams = streams
        self.metadata = {"start": {"use_good_bad": use_good_bad}}

    def __getitem__(self, name: str) -> _Stream:
        return self.streams[name]


class _RawClient:
    def __init__(self, run: _Run) -> None:
        self.run = run

    def __getitem__(self, uid: object) -> _Run:
        return self.run


def _qepro_dataset() -> dict[str, _Field]:
    return {field: _Field(np.array([1])) for field in QEPRO_FIELDS}


def _raw_evaluator(reference_path: Path, raw_client: Any = None) -> XrayUvvisEvaluation:
    references = PdfReferenceConfig(
        (PdfPhaseReference("Target", reference_path, False),)
    )
    return XrayUvvisEvaluation(
        raw_client,
        object(),
        DEFAULT_PLQY_PARAMS,
        references,
        pdf_fit_config=PdfFitConfig(PdfEvaluationMode.RAW_ONLY),
        max_retries=3,
        retry_delay=0,
        sleep=lambda _: None,
    )


def test_reference_config_resolves_external_paths(tmp_path: Path) -> None:
    relative_gr = tmp_path / "relative.gr"
    absolute_gr = tmp_path / "absolute.gr"
    relative_cif = tmp_path / "relative.cif"
    absolute_cif = tmp_path / "absolute.cif"
    for path in (relative_gr, absolute_gr, relative_cif, absolute_cif):
        path.write_text("fixture")
    config_path = tmp_path / "references.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phases": [
                    {
                        "name": "Wanted",
                        "gr_path": relative_gr.name,
                        "cif_path": relative_cif.name,
                        "minimize": False,
                    },
                    {
                        "name": "Impurity",
                        "gr_path": str(absolute_gr),
                        "cif_path": str(absolute_cif),
                        "minimize": True,
                    },
                ],
            }
        )
    )

    config = PdfReferenceConfig.from_json(config_path)

    assert config.phases[0].gr_path == relative_gr
    assert config.phases[0].cif_path == relative_cif
    assert config.phases[1].gr_path == absolute_gr
    assert config.phases[1].cif_path == absolute_cif
    assert [phase.minimize for phase in config.phases] == [False, True]
    objectives = build_objectives(PdfFitConfig(), config)
    assert [(objective.name, objective.minimize) for objective in objectives[-2:]] == [
        ("pdf_fit_corr_Wanted", False),
        ("pdf_fit_corr_Impurity", True),
    ]

    duplicate = json.loads(config_path.read_text())
    duplicate["phases"][1]["name"] = "Wanted"
    config_path.write_text(json.dumps(duplicate))
    with pytest.raises(ValueError, match="duplicated"):
        PdfReferenceConfig.from_json(config_path)

    duplicate["phases"][1]["name"] = "Impurity"
    duplicate["phases"][0]["gr_path"] = "missing.gr"
    config_path.write_text(json.dumps(duplicate))
    with pytest.raises(ValueError, match=r"phases\[0\]\.gr_path.*missing\.gr"):
        PdfReferenceConfig.from_json(config_path)


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
    ],
)
def test_reference_config_rejects_invalid_schema(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    gr_path = tmp_path / "phase.gr"
    gr_path.write_text("2 1\n3 2\n")
    data = {
        "schema_version": 1,
        "phases": [{"name": "Phase", "gr_path": "phase.gr", "minimize": False}],
    }
    mutation(data)
    config_path = tmp_path / "references.json"
    config_path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match=message):
        PdfReferenceConfig.from_json(config_path)


def test_raw_mode_allows_missing_cif_but_fit_modes_require_it(
    reference_config_factory: Any,
) -> None:
    references = PdfReferenceConfig.from_json(
        reference_config_factory(include_cif=False)
    )
    XrayUvvisEvaluation(
        object(),
        object(),
        DEFAULT_PLQY_PARAMS,
        references,
        pdf_fit_config=PdfFitConfig(PdfEvaluationMode.RAW_ONLY),
    )
    with pytest.raises(ValueError, match="Target.*cif_path"):
        XrayUvvisEvaluation(
            object(),
            object(),
            DEFAULT_PLQY_PARAMS,
            references,
            pdf_fit_config=PdfFitConfig(PdfEvaluationMode.PDF_FIT_OBJECTIVES),
        )


def test_read_qepro_stream_requires_all_fields(tmp_path: Path) -> None:
    dataset = _qepro_dataset()
    run = _Run({"fluorescence": _Stream(dataset)})

    values, metadata = read_qepro_stream(_RawClient(run), "uid", "fluorescence")

    assert tuple(values) == QEPRO_FIELDS
    assert metadata is run.metadata["start"]
    dataset.pop("QEPro_dark")
    with pytest.raises(ValueError, match="QEPro_dark"):
        read_qepro_stream(_RawClient(run), "uid", "fluorescence")


def test_tiled_retries_preserve_successful_reads(tmp_path: Path) -> None:
    fluorescence = _Stream(_qepro_dataset(), failures=1)
    absorbance = _Stream(_qepro_dataset())
    run = _Run({"fluorescence": fluorescence, "absorbance": absorbance})
    reference = tmp_path / "phase.gr"
    reference.write_text("2 1\n3 2\n")
    evaluator = _raw_evaluator(reference, _RawClient(run))

    evaluator._read_tiled_data("uid")

    assert fluorescence.read_count == 2
    assert absorbance.read_count == 1


def test_quality_batches_must_partition_events(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reference = tmp_path / "phase.gr"
    reference.write_text("2 1\n3 2\n")
    evaluator = _raw_evaluator(reference)
    fluorescence = {"QEPro_output": np.zeros((3, 4)), "other": np.arange(3)}

    with pytest.raises(ValueError, match="exactly partition"):
        evaluator._filter_fl_to_good_batches(
            fluorescence,
            [{"verdict": "good", "n_events_in_batch": 2}],
        )

    filtered = evaluator._filter_fl_to_good_batches(
        fluorescence,
        [
            {"verdict": "good", "n_events_in_batch": 1},
            {"verdict": "bad", "n_events_in_batch": 2},
        ],
    )
    np.testing.assert_array_equal(filtered["other"], [0])

    with caplog.at_level("WARNING"):
        unfiltered = evaluator._filter_fl_to_good_batches(
            fluorescence,
            [{"verdict": "BAD", "n_events_in_batch": 3}],
        )
    assert unfiltered is fluorescence
    assert (
        caplog.messages.count(
            "No good PL batches found; using all 3 fluorescence events"
        )
        == 1
    )


def test_single_shot_pl_includes_event_zero(
    tmp_path: Path,
    wavelength: np.ndarray,
    good_spectrum: np.ndarray,
) -> None:
    reference = tmp_path / "phase.gr"
    reference.write_text("2 1\n3 2\n")
    evaluator = _raw_evaluator(reference)

    peak, fwhm, integral, r_squared, has_peak = evaluator._process_pl(
        {
            "QEPro_x_axis": wavelength[np.newaxis, :],
            "QEPro_output": good_spectrum[np.newaxis, :],
        }
    )

    assert has_peak
    assert peak == pytest.approx(660)
    assert fwhm > 0
    assert integral > 0
    assert r_squared > 0.99


def test_dynamic_modes_and_fit_failure_policy(tmp_path: Path) -> None:
    radial = np.linspace(1, 25, 241)
    profile = np.sin(radial)
    reference = tmp_path / "phase.gr"
    np.savetxt(reference, np.column_stack((radial, profile)))
    cif = tmp_path / "phase.cif"
    cif.write_text("placeholder")
    references = PdfReferenceConfig(
        (PdfPhaseReference("Dynamic", reference, True, cif),)
    )
    data = {"gr_r": radial, "gr_G": profile}

    raw = XrayUvvisEvaluation(
        object(),
        object(),
        DEFAULT_PLQY_PARAMS,
        references,
        pdf_fit_config=PdfFitConfig(PdfEvaluationMode.RAW_ONLY),
    )
    assert raw._process_pdf(data, uid="uid") == pytest.approx({"corr_Dynamic": 1.0})

    tracked = XrayUvvisEvaluation(
        object(),
        object(),
        DEFAULT_PLQY_PARAMS,
        references,
        pdf_fit_config=PdfFitConfig(PdfEvaluationMode.RAW_OBJECTIVES_PDF_FIT_TRACKED),
    )
    tracked_for_patch = cast(Any, tracked)
    tracked_for_patch._fit_pdf_correlations = lambda pdf_data: (_ for _ in ()).throw(
        ValueError("fit failed")
    )
    assert tracked._process_pdf(data, uid="uid") == pytest.approx({"corr_Dynamic": 1.0})

    strict = XrayUvvisEvaluation(
        object(),
        object(),
        DEFAULT_PLQY_PARAMS,
        references,
        pdf_fit_config=PdfFitConfig(PdfEvaluationMode.PDF_FIT_OBJECTIVES),
    )
    strict_for_patch = cast(Any, strict)
    strict_for_patch._fit_pdf_correlations = tracked_for_patch._fit_pdf_correlations
    with pytest.raises(RuntimeError, match="PDF fitting failed for uid='uid'") as exc:
        strict._process_pdf(data, uid="uid")
    assert isinstance(exc.value.__cause__, ValueError)


def test_no_peak_outcomes_are_finite(tmp_path: Path) -> None:
    reference = tmp_path / "phase.gr"
    reference.write_text("2 1\n3 2\n")
    evaluator = _raw_evaluator(reference)
    evaluator_for_patch = cast(Any, evaluator)
    evaluator_for_patch._read_tiled_data = lambda uid: ({}, {}, {}, None)
    evaluator_for_patch._process_pl = lambda fluorescence: (
        0.0,
        1000.0,
        0.0,
        0.0,
        False,
    )
    evaluator_for_patch._process_absorbance = lambda absorbance: (
        np.array([365.0]),
        np.array([0.0]),
    )
    evaluator_for_patch._read_pdfstream_data = lambda uid: {}
    evaluator_for_patch._process_pdf = lambda pdf_data, *, uid: {"corr_Target": 0.25}

    outcome = evaluator("uid", [{"_id": 7}])[0]

    assert outcome["_id"] == 7
    assert outcome["Peak"] == 0.0
    assert outcome["peak_distance"] == 660.0
    assert outcome["log_FWHM"] == pytest.approx(np.log(1000.0))
    assert outcome["log_PLQY"] == pytest.approx(np.log(1e-10))
    assert all(np.isfinite(value) for key, value in outcome.items() if key != "_id")


@pytest.mark.timeout(60)
def test_external_target_pdf_fit_is_finite() -> None:
    fixture_directory = Path(__file__).parent / "fixtures"
    radial, profile = np.loadtxt(fixture_directory / "target.gr", unpack=True)
    references = PdfReferenceConfig(
        (
            PdfPhaseReference(
                "Target",
                fixture_directory / "target.gr",
                False,
                fixture_directory / "target.cif",
            ),
        )
    )
    evaluator = XrayUvvisEvaluation(
        object(),
        object(),
        DEFAULT_PLQY_PARAMS,
        references,
        pdf_fit_config=PdfFitConfig(rmax=20),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evaluator._fit_pdf_correlations({"gr_r": radial, "gr_G": profile})

    assert -1 <= result["pdf_fit_corr_Target"] <= 1
