"""X-ray and UV-Vis optimization evaluation."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

import numpy as np
from tiled.queries import Eq

from .analysis import (
    calculate_plqy,
    classify_pl,
    correct_absorbance,
    fit_pl_spectrum,
    pearson_profile,
    select_pl_spectra,
)


class PdfEvaluationMode(StrEnum):
    """Select which PDF correlations participate in optimization."""

    RAW_ONLY = "raw_only"
    PDF_FIT_OBJECTIVES = "pdf_fit_objectives"
    RAW_OBJECTIVES_PDF_FIT_TRACKED = "raw_objectives_pdf_fit_tracked"


@dataclass(frozen=True)
class PdfFitConfig:
    """Configure PDF correlation and optional pdffit2 refinement."""

    mode: PdfEvaluationMode | str = PdfEvaluationMode.PDF_FIT_OBJECTIVES
    qmax: float = 18.0
    rmax: float = 120.0
    qdamp: float = 0.031
    qbroad: float = 0.032
    fix_apd: bool = True
    toler: float = 0.000001

    def __post_init__(self) -> None:
        """Normalize a string mode to its enum value."""
        object.__setattr__(self, "mode", PdfEvaluationMode(self.mode))


@dataclass(frozen=True)
class PdfPhaseReference:
    """External PDF reference inputs and optimization direction for one phase."""

    name: str
    gr_path: Path
    minimize: bool
    cif_path: Path | None = None
    constraint_profile: Literal["none", "cs_pb_br3"] = "none"


_PHASE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_ROOT_FIELDS = frozenset({"schema_version", "phases"})
_PHASE_FIELDS = frozenset(
    {"name", "gr_path", "cif_path", "minimize", "constraint_profile"}
)
_PHASE_REQUIRED_FIELDS = frozenset({"name", "gr_path", "minimize"})


@dataclass(frozen=True)
class PdfReferenceConfig:
    """Ordered PDF phase references loaded from an external configuration."""

    phases: tuple[PdfPhaseReference, ...]

    def __post_init__(self) -> None:
        """Enforce phase identity and required GR-file invariants."""
        object.__setattr__(self, "phases", tuple(self.phases))
        if not self.phases:
            raise ValueError("phases must contain at least one PDF reference")

        names: set[str] = set()
        for index, phase in enumerate(self.phases):
            field = f"phases[{index}]"
            if not _PHASE_NAME.fullmatch(phase.name):
                raise ValueError(
                    f"{field}.name must match {_PHASE_NAME.pattern!r}: {phase.name!r}"
                )
            if phase.name in names:
                raise ValueError(f"{field}.name is duplicated: {phase.name!r}")
            names.add(phase.name)
            if not isinstance(phase.minimize, bool):
                raise ValueError(f"{field}.minimize must be a boolean")
            if phase.constraint_profile not in {"none", "cs_pb_br3"}:
                raise ValueError(
                    f"{field}.constraint_profile is unsupported: "
                    f"{phase.constraint_profile!r}"
                )
            if not phase.gr_path.is_file():
                raise ValueError(f"{field}.gr_path is not a file: {phase.gr_path}")

    @classmethod
    def from_json(cls, path: str | Path) -> PdfReferenceConfig:
        """Load and validate a version-1 external reference configuration."""
        config_path = Path(path).expanduser().resolve()
        try:
            document = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {config_path}: {exc.msg}") from exc

        if not isinstance(document, dict):
            raise ValueError(f"configuration root must be an object: {config_path}")
        _require_exact_fields(document, _ROOT_FIELDS, _ROOT_FIELDS, "configuration")
        if type(document["schema_version"]) is not int:
            raise ValueError("schema_version must be the integer 1")
        if document["schema_version"] != 1:
            raise ValueError(
                f"schema_version is unsupported: {document['schema_version']!r}"
            )
        raw_phases = document["phases"]
        if not isinstance(raw_phases, list) or not raw_phases:
            raise ValueError("phases must be a non-empty list")

        phases: list[PdfPhaseReference] = []
        for index, raw_phase in enumerate(raw_phases):
            field = f"phases[{index}]"
            if not isinstance(raw_phase, dict):
                raise ValueError(f"{field} must be an object")
            _require_exact_fields(
                raw_phase,
                _PHASE_FIELDS,
                _PHASE_REQUIRED_FIELDS,
                field,
            )
            name = raw_phase["name"]
            if not isinstance(name, str):
                raise ValueError(f"{field}.name must be a string")
            minimize = raw_phase["minimize"]
            if not isinstance(minimize, bool):
                raise ValueError(f"{field}.minimize must be a boolean")
            constraint_profile = raw_phase.get("constraint_profile", "none")
            if constraint_profile not in {"none", "cs_pb_br3"}:
                raise ValueError(
                    f"{field}.constraint_profile is unsupported: {constraint_profile!r}"
                )
            gr_path = _reference_path(
                raw_phase["gr_path"], config_path.parent, f"{field}.gr_path"
            )
            cif_value = raw_phase.get("cif_path")
            cif_path = (
                None
                if cif_value is None
                else _reference_path(
                    cif_value,
                    config_path.parent,
                    f"{field}.cif_path",
                    require_file=False,
                )
            )
            phases.append(
                PdfPhaseReference(
                    name=name,
                    gr_path=gr_path,
                    cif_path=cif_path,
                    minimize=minimize,
                    constraint_profile=constraint_profile,
                )
            )
        return cls(tuple(phases))


def _require_exact_fields(
    value: dict[str, Any],
    allowed: frozenset[str],
    required: frozenset[str],
    field: str,
) -> None:
    """Reject unknown and missing JSON object fields."""
    unknown = sorted(value.keys() - allowed)
    if unknown:
        raise ValueError(f"{field} has unknown fields: {', '.join(unknown)}")
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"{field} is missing fields: {', '.join(missing)}")


def _reference_path(
    value: object,
    config_directory: Path,
    field: str,
    *,
    require_file: bool = True,
) -> Path:
    """Validate and resolve one configured filesystem path."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_directory / candidate
    if require_file and not candidate.is_file():
        raise ValueError(f"{field} is not a file: {candidate}")
    return candidate


QEProFields = tuple[str, ...]
QEPRO_FIELDS: QEProFields = (
    "QEPro_x_axis",
    "QEPro_output",
    "QEPro_sample",
    "QEPro_dark",
    "QEPro_reference",
    "QEPro_spectrum_type",
    "QEPro_integration_time",
    "QEPro_num_spectra",
    "QEPro_buff_capacity",
)


class _TiledAccessError(RuntimeError):
    """Mark an exception raised while looking up or reading Tiled data."""


def _read_stream_dataset(
    client: Any, uid: Hashable, stream_name: str
) -> tuple[Any, Mapping[str, Any]]:
    """Read one stream while distinguishing access from schema failures."""
    try:
        run = client[uid]
        dataset = run[stream_name].read()
        metadata = run.metadata["start"]
    except Exception as exc:
        raise _TiledAccessError(
            f"failed to read stream {stream_name!r} for uid={uid!r}"
        ) from exc
    return dataset, metadata


def read_qepro_stream(
    client: Any,
    uid: Hashable,
    stream_name: str,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    """Read one QEPro stream and return its nine fields plus start metadata."""
    dataset, metadata = _read_stream_dataset(client, uid, stream_name)
    missing = [field for field in QEPRO_FIELDS if field not in dataset]
    if missing:
        raise ValueError(
            f"QEPro stream {stream_name!r} is missing fields: {', '.join(missing)}"
        )
    values = {field: np.asarray(dataset[field].values) for field in QEPRO_FIELDS}
    return values, metadata


def _write_oxidation_free_cif(cif_path: Path, output_directory: Path) -> Path:
    """Write a temporary CIF without oxidation states for diffpy."""
    from pymatgen.io.cif import CifParser, CifWriter

    structure = CifParser(str(cif_path)).parse_structures(primitive=True)[0]
    structure.remove_oxidation_states()
    output_path = output_directory / f"{cif_path.stem}_pym.cif"
    CifWriter(structure, symprec=0.1).write_file(str(output_path))
    return output_path


def _set_cs_pb_br3_constraints(
    pdf_fit: Any,
    *,
    phase_index: int = 1,
    fix_apd: bool = True,
) -> None:
    """Apply the established CsPbBr3 pdffit2 constraints."""
    pdf_fit.setphase(phase_index)
    for axis, parameter in enumerate((11, 12, 13), start=1):
        pdf_fit.constrain(pdf_fit.lat(axis), f"@{parameter}")
        pdf_fit.setpar(parameter, pdf_fit.lat(axis))

    pdf_fit.constrain("pscale", "@111")
    pdf_fit.setpar(111, 1.0)
    pdf_fit.constrain(pdf_fit.delta2, "@122")
    pdf_fit.setpar(122, 6.87)
    pdf_fit.fixpar(122)
    pdf_fit.constrain(pdf_fit.spdiameter, "@133")
    pdf_fit.setpar(133, 80)

    for atom_range, parameter, value in (
        (range(1, 5), 101, 0.029385),
        (range(5, 9), 102, 0.027296),
        (range(9, 17), 103, 0.041577),
        (range(17, 21), 104, 0.028164),
    ):
        for atom_index in atom_range:
            pdf_fit.constrain(pdf_fit.u11(atom_index), f"@{parameter}")
            pdf_fit.constrain(pdf_fit.u22(atom_index), f"@{parameter}")
            pdf_fit.constrain(pdf_fit.u33(atom_index), f"@{parameter}")
        pdf_fit.setpar(parameter, value)
        if fix_apd:
            pdf_fit.fixpar(parameter)


DEFAULT_TILED_PROFILE = "xpd"
DEFAULT_SANDBOX_URI = "https://tiled.nsls2.bnl.gov"
SANDBOX_CATALOG = "xpd/sandbox"
DEFAULT_PLQY_PARAMS = (1, "quinine", 365, 0.06, 1.2e6, 1.33, 0.546)

logger = logging.getLogger(__name__)


class XrayUvvisEvaluation:
    """Evaluate optical spectra and PDF data for one acquisition run."""

    def __init__(
        self,
        tiled_client: Any,
        sandbox_client: Any,
        plqy_params: Sequence[float | str],
        pdf_references: PdfReferenceConfig,
        *,
        key_height: float = 200,
        distance: int = 100,
        height: float = 50,
        percent_range_pl: tuple[float, float] = (40, 100),
        percent_range_abs: tuple[float, float] = (10, 70),
        peak_target: float = 660,
        pdf_fit_config: PdfFitConfig | None = None,
        max_retries: int = 10,
        retry_delay: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.tiled_client = tiled_client
        self.sandbox_client = sandbox_client
        self.plqy_params = tuple(plqy_params)
        self.pdf_references = pdf_references
        self.key_height = key_height
        self.distance = distance
        self.height = height
        self.percent_range_pl = percent_range_pl
        self.percent_range_abs = percent_range_abs
        self.peak_target = peak_target
        self.pdf_fit_config = pdf_fit_config or PdfFitConfig()
        if max_retries < 1:
            raise ValueError("max_retries must be at least one")
        if retry_delay < 0:
            raise ValueError("retry_delay cannot be negative")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.sleep = sleep
        self._validate_fit_references()

    def _validate_fit_references(self) -> None:
        """Require one existing CIF per phase when PDF fitting is enabled."""
        if self.pdf_fit_config.mode is PdfEvaluationMode.RAW_ONLY:
            return
        for phase in self.pdf_references.phases:
            if phase.cif_path is None:
                raise ValueError(
                    f"phase {phase.name!r} requires cif_path for "
                    f"PDF mode {PdfEvaluationMode(self.pdf_fit_config.mode).value!r}"
                )
            if not phase.cif_path.is_file():
                raise ValueError(
                    f"phase {phase.name!r} cif_path is not a file: {phase.cif_path}"
                )

    def _read_tiled_data(
        self, uid: Hashable
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        Mapping[str, Any],
        list[dict[str, Any]] | None,
    ]:
        """Read required raw streams, retaining successes between retries."""
        fluorescence: dict[str, np.ndarray] | None = None
        absorbance: dict[str, np.ndarray] | None = None
        metadata: Mapping[str, Any] | None = None
        quality: list[dict[str, Any]] | None = None
        use_good_bad: bool | None = None
        last_access_error: BaseException | None = None

        for attempt in range(self.max_retries):
            if fluorescence is None:
                try:
                    fluorescence, metadata = read_qepro_stream(
                        self.tiled_client, uid, "fluorescence"
                    )
                    use_good_bad = bool(metadata.get("use_good_bad", False))
                except _TiledAccessError as exc:
                    last_access_error = exc.__cause__ or exc
                    logger.warning(
                        "Failed to read fluorescence stream for uid=%r "
                        "(attempt %d/%d): %r",
                        uid,
                        attempt + 1,
                        self.max_retries,
                        last_access_error,
                    )

            if absorbance is None:
                try:
                    absorbance, absorbance_metadata = read_qepro_stream(
                        self.tiled_client, uid, "absorbance"
                    )
                    if metadata is None:
                        metadata = absorbance_metadata
                        use_good_bad = bool(metadata.get("use_good_bad", False))
                except _TiledAccessError as exc:
                    last_access_error = exc.__cause__ or exc
                    logger.warning(
                        "Failed to read absorbance stream for uid=%r "
                        "(attempt %d/%d): %r",
                        uid,
                        attempt + 1,
                        self.max_retries,
                        last_access_error,
                    )

            if use_good_bad and quality is None:
                try:
                    quality = self._read_quality_stream(uid)
                except _TiledAccessError as exc:
                    last_access_error = exc.__cause__ or exc
                    logger.warning(
                        "Failed to read fluorescence_quality stream for uid=%r "
                        "(attempt %d/%d): %r",
                        uid,
                        attempt + 1,
                        self.max_retries,
                        last_access_error,
                    )

            ready = (
                fluorescence is not None
                and absorbance is not None
                and metadata is not None
                and use_good_bad is not None
                and (not use_good_bad or quality is not None)
            )
            if ready:
                return (
                    cast(dict[str, np.ndarray], fluorescence),
                    cast(dict[str, np.ndarray], absorbance),
                    cast(Mapping[str, Any], metadata),
                    quality,
                )
            if attempt + 1 < self.max_retries:
                self.sleep(self.retry_delay)

        missing: list[str] = []
        if metadata is None or use_good_bad is None:
            missing.append("run metadata")
        if fluorescence is None:
            missing.append("fluorescence stream")
        if absorbance is None:
            missing.append("absorbance stream")
        if use_good_bad and quality is None:
            missing.append("fluorescence_quality stream")
        message = (
            f"Failed to read required Tiled data for uid={uid!r} after "
            f"{self.max_retries} attempts. Missing: {', '.join(missing)}."
        )
        raise RuntimeError(message) from last_access_error

    def _read_quality_stream(self, uid: Hashable) -> list[dict[str, Any]]:
        """Read and convert the per-batch fluorescence quality stream."""
        dataset, _ = _read_stream_dataset(
            self.tiled_client, uid, "fluorescence_quality"
        )
        required = ("verdict", "n_events_in_batch")
        missing = [field for field in required if field not in dataset]
        if missing:
            raise ValueError(
                "fluorescence_quality stream is missing fields: " + ", ".join(missing)
            )
        verdicts = np.asarray(dataset["verdict"].values)
        counts = np.asarray(dataset["n_events_in_batch"].values)
        if verdicts.shape != counts.shape:
            raise ValueError(
                "fluorescence_quality verdict and event-count shapes differ"
            )
        return [
            {"verdict": str(verdict), "n_events_in_batch": int(count)}
            for verdict, count in zip(verdicts, counts, strict=True)
        ]

    def _filter_fl_to_good_batches(
        self,
        fluorescence: dict[str, np.ndarray],
        batch_info: Sequence[Mapping[str, Any]],
    ) -> dict[str, np.ndarray]:
        """Keep events from batches whose verdict is exactly ``good``."""
        output = np.asarray(fluorescence["QEPro_output"])
        event_count = 1 if output.ndim == 1 else output.shape[0]
        counts = [int(batch["n_events_in_batch"]) for batch in batch_info]
        if any(count < 0 for count in counts) or sum(counts) != event_count:
            raise ValueError(
                "fluorescence quality batch counts must exactly partition "
                f"{event_count} events; received {counts}"
            )

        selected_indices: list[int] = []
        cursor = 0
        for batch, count in zip(batch_info, counts, strict=True):
            if batch["verdict"] == "good":
                selected_indices.extend(range(cursor, cursor + count))
            cursor += count

        if not selected_indices:
            logger.warning(
                "No good PL batches found; using all %d fluorescence events",
                event_count,
            )
            return fluorescence

        indices = np.asarray(selected_indices, dtype=np.intp)
        logger.info(
            "Keeping %d/%d fluorescence events from good batches",
            indices.size,
            event_count,
        )
        filtered: dict[str, np.ndarray] = {}
        for field, values in fluorescence.items():
            array = np.asarray(values)
            filtered[field] = (
                array[indices]
                if array.ndim >= 1 and array.shape[0] == event_count
                else array
            )
        return filtered

    def _process_pl(
        self, fluorescence: Mapping[str, np.ndarray]
    ) -> tuple[float, float, float, float, bool]:
        """Classify all events, average selected good spectra, and fit one peak."""
        intensities = np.asarray(fluorescence["QEPro_output"], dtype=float)
        wavelengths = np.asarray(fluorescence["QEPro_x_axis"], dtype=float)
        if intensities.ndim == 1:
            intensities = intensities[np.newaxis, :]
        if wavelengths.ndim == 1:
            wavelengths = np.broadcast_to(wavelengths, intensities.shape)
        if intensities.ndim != 2 or wavelengths.shape != intensities.shape:
            raise ValueError(
                "QEPro_x_axis and QEPro_output must contain aligned spectra"
            )

        good_indices = [
            index
            for index, (x_row, y_row) in enumerate(
                zip(wavelengths, intensities, strict=True)
            )
            if classify_pl(
                x_row,
                y_row,
                key_height=self.key_height,
                distance=self.distance,
                height=self.height,
            ).is_good
        ]
        if not good_indices:
            return 0.0, 1000.0, 0.0, 0.0, False

        good_wavelengths = wavelengths[good_indices]
        good_spectra = intensities[good_indices]
        selected = select_pl_spectra(
            good_wavelengths,
            good_spectra,
            percent_range=self.percent_range_pl,
        )
        averaged = np.mean(selected, axis=0)
        fit_wavelength = good_wavelengths[0]
        classification = classify_pl(
            fit_wavelength,
            averaged,
            key_height=self.key_height,
            distance=self.distance,
            height=self.height,
        )
        if not classification.is_good:
            return 0.0, 1000.0, 0.0, 0.0, False
        peak, fwhm, integral, r_squared = fit_pl_spectrum(
            fit_wavelength,
            averaged,
            classification,
        )
        return peak, fwhm, integral, r_squared, True

    def _process_absorbance(
        self, absorbance: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Percentile-filter and baseline-correct absorbance spectra."""
        return correct_absorbance(
            absorbance["QEPro_x_axis"],
            absorbance["QEPro_output"],
            percent_range=self.percent_range_abs,
        )

    def _compute_plqy(
        self,
        absorbance: np.ndarray,
        wavelength: np.ndarray,
        pl_integral: float,
    ) -> float:
        """Calculate PLQY at the configured excitation wavelength."""
        if len(self.plqy_params) != 7:
            raise ValueError("plqy_params must contain exactly seven values")
        (
            _,
            reference_type,
            excitation_wavelength,
            absorbance_reference,
            pl_integral_reference,
            refractive_index_reference,
            plqy_reference,
        ) = self.plqy_params
        excitation_index = int(
            np.abs(wavelength - float(excitation_wavelength)).argmin()
        )
        return calculate_plqy(
            float(absorbance[excitation_index]),
            pl_integral,
            1.506,
            reference_type=str(reference_type),
            absorbance_reference=float(absorbance_reference),
            pl_integral_reference=float(pl_integral_reference),
            refractive_index_reference=float(refractive_index_reference),
            plqy_reference=float(plqy_reference),
        )

    def _read_pdfstream_data(self, uid: Hashable) -> dict[str, np.ndarray]:
        """Read the correlated pdfstream G(r) stream with bounded retries."""
        last_access_error: BaseException | None = None
        for attempt in range(self.max_retries):
            try:
                matches = self.sandbox_client.search(Eq("start.original_run_uid", uid))
                key = matches.keys().last()
                dataset = matches[key]["scattering"].read()
            except Exception as exc:
                last_access_error = exc
                logger.warning(
                    "Failed to read pdfstream data for uid=%r (attempt %d/%d): %r",
                    uid,
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
            else:
                missing = [field for field in ("gr_r", "gr_G") if field not in dataset]
                if missing:
                    raise ValueError(
                        "scattering stream is missing fields: " + ", ".join(missing)
                    )
                return {
                    field: np.asarray(dataset[field].values).squeeze()
                    for field in ("gr_r", "gr_G")
                }
            if attempt + 1 < self.max_retries:
                self.sleep(self.retry_delay)

        raise RuntimeError(
            "Could not read pdfstream scattering data with "
            f"original_run_uid={uid!r} after {self.max_retries} attempts."
        ) from last_access_error

    def _raw_pdf_correlations(
        self, pdf_data: Mapping[str, np.ndarray]
    ) -> dict[str, float]:
        """Compute one raw G(r) correlation per configured phase."""
        results: dict[str, float] = {}
        for phase in self.pdf_references.phases:
            reference_r, reference_g = np.loadtxt(
                phase.gr_path,
                usecols=(0, 1),
                unpack=True,
            )
            results[f"corr_{phase.name}"] = pearson_profile(
                pdf_data["gr_r"],
                pdf_data["gr_G"],
                reference_r,
                reference_g,
            )
        return results

    def _fit_pdf_correlations(
        self, pdf_data: Mapping[str, np.ndarray]
    ) -> dict[str, float]:
        """Refine and correlate each configured phase using pdffit2."""
        from diffpy.pdffit2 import PdfFit
        from diffpy.structure import loadStructure

        experimental_r = np.asarray(pdf_data["gr_r"], dtype=float)
        experimental_g = np.asarray(pdf_data["gr_G"], dtype=float)
        finite = np.isfinite(experimental_r) & np.isfinite(experimental_g)
        if np.count_nonzero(finite) < 2:
            raise ValueError("not enough finite PDF points to fit G(r)")

        fit_rmax = min(
            self.pdf_fit_config.rmax,
            float(np.max(experimental_r[finite])),
        )
        if fit_rmax <= 2.5:
            raise ValueError(
                f"PDF fit rmax must be greater than 2.5 A; measured rmax is {fit_rmax}"
            )
        fit_config = replace(self.pdf_fit_config, rmax=fit_rmax)

        results: dict[str, float] = {}
        with TemporaryDirectory() as directory:
            output_directory = Path(directory)
            measured_path = output_directory / "measured.gr"
            np.savetxt(
                measured_path,
                np.column_stack((experimental_r[finite], experimental_g[finite])),
                fmt="%.10g %.10g",
            )
            for phase in self.pdf_references.phases:
                if phase.cif_path is None:
                    raise ValueError(f"phase {phase.name!r} requires cif_path")
                clean_cif = _write_oxidation_free_cif(phase.cif_path, output_directory)
                structure = loadStructure(str(clean_cif))
                structure.Uisoequiv = 0.04
                structure.title = clean_cif.stem

                pdf_fit: Any = PdfFit()
                pdf_fit.read_data(
                    str(measured_path),
                    "X",
                    fit_config.qmax,
                    fit_config.qdamp,
                )
                pdf_fit.add_structure(structure)
                if phase.constraint_profile == "cs_pb_br3":
                    _set_cs_pb_br3_constraints(
                        pdf_fit,
                        fix_apd=fit_config.fix_apd,
                    )
                pdf_fit.constrain(pdf_fit.dscale, "@902")
                pdf_fit.setpar(902, 1.0)
                pdf_fit.setvar(pdf_fit.qdamp, fit_config.qdamp)
                pdf_fit.setvar(pdf_fit.qbroad, fit_config.qbroad)
                pdf_fit.pdfrange(1, 2.5, fit_config.rmax)
                pdf_fit.refine(toler=fit_config.toler)
                results[f"pdf_fit_corr_{phase.name}"] = pearson_profile(
                    experimental_r,
                    experimental_g,
                    np.asarray(pdf_fit.getR()),
                    np.asarray(pdf_fit.getpdf_fit()),
                )
        return results

    def _process_pdf(
        self,
        pdf_data: Mapping[str, np.ndarray],
        *,
        uid: Hashable,
    ) -> dict[str, float]:
        """Compute PDF metrics using the configured fitting failure policy."""
        results = self._raw_pdf_correlations(pdf_data)
        if self.pdf_fit_config.mode is PdfEvaluationMode.RAW_ONLY:
            return results
        try:
            results.update(self._fit_pdf_correlations(pdf_data))
        except Exception as exc:
            if self.pdf_fit_config.mode is PdfEvaluationMode.PDF_FIT_OBJECTIVES:
                raise RuntimeError(f"PDF fitting failed for uid={uid!r}") from exc
            logger.warning(
                "PDF fitting failed for uid=%r; returning raw metrics only",
                uid,
                exc_info=True,
            )
        return results

    def __call__(
        self,
        uid: Hashable,
        suggestions: Sequence[Mapping[str, Any]],
    ) -> Sequence[Mapping[str, Any]]:
        """Evaluate a run and return finite outcomes for each suggestion."""
        fluorescence, absorbance, _metadata, batch_info = self._read_tiled_data(uid)
        if batch_info is not None:
            fluorescence = self._filter_fl_to_good_batches(
                fluorescence,
                batch_info,
            )

        peak, fwhm, pl_integral, _r_squared, has_peak = self._process_pl(fluorescence)
        wavelength, corrected_absorbance = self._process_absorbance(absorbance)
        if has_peak:
            if not np.isfinite(peak):
                raise ValueError(f"fitted Peak is not finite for uid={uid!r}")
            if not np.isfinite(fwhm) or fwhm <= 0:
                raise ValueError(
                    f"fitted FWHM is not positive and finite for uid={uid!r}"
                )
            plqy = self._compute_plqy(
                corrected_absorbance,
                wavelength,
                pl_integral,
            )
        else:
            peak = 0.0
            fwhm = 1000.0
            plqy = 1e-10

        if not np.isfinite(plqy) or plqy <= 0:
            plqy = 1e-10
        pdf_metrics = self._process_pdf(self._read_pdfstream_data(uid), uid=uid)
        for name, value in pdf_metrics.items():
            if not np.isfinite(value):
                raise ValueError(f"PDF correlation {name!r} is not finite")

        outcomes = {
            "Peak": float(peak),
            "peak_distance": float(abs(self.peak_target - peak)),
            "log_FWHM": float(np.log(fwhm)),
            "log_PLQY": float(np.log(plqy)),
            **pdf_metrics,
        }
        return [{**outcomes, "_id": suggestion["_id"]} for suggestion in suggestions]
