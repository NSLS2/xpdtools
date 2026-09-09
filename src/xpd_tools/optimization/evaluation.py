"""X-ray and UV-Vis optimization evaluation."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, TypeVar, cast

import numpy as np
from tiled.queries import Eq

from .analysis import (
    analyze_pl_spectra,
    calculate_plqy,
    correct_absorbance,
    pearson_profile,
)

logger = logging.getLogger(__name__)

_PHASE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_ROOT_FIELDS = frozenset({"schema_version", "phases"})
_PHASE_FIELDS = frozenset(
    {"name", "gr_path", "cif_path", "minimize", "constraint_profile"}
)
_PHASE_REQUIRED_FIELDS = frozenset({"name", "gr_path", "minimize"})
_QEPRO_FIELDS = ("QEPro_x_axis", "QEPro_output")
_PDF_QMAX = 18.0
_PDF_RMAX = 120.0
_PDF_RMIN = 2.5
_PDF_QDAMP = 0.031
_PDF_QBROAD = 0.032
_PDF_TOLERANCE = 1e-6
_T = TypeVar("_T")


@dataclass(frozen=True)
class _PdfPhaseReference:
    """Validated external PDF inputs for one phase."""

    name: str
    gr_path: Path
    minimize: bool
    cif_path: Path | None = None
    constraint_profile: Literal["none", "cs_pb_br3"] = "none"


@dataclass(frozen=True)
class PlqyReference:
    """Reference values used for relative PLQY calculation."""

    reference_type: str = "quinine"
    excitation_wavelength_nm: float = 365.0
    absorbance: float = 0.06
    pl_integral: float = 1.2e6
    refractive_index: float = 1.33
    plqy: float = 0.546
    solvent_refractive_index: float = 1.506


def _require_exact_fields(
    value: dict[str, Any],
    allowed: frozenset[str],
    required: frozenset[str],
    field: str,
) -> None:
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
    require_file: bool,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_directory / candidate
    if require_file and not candidate.is_file():
        raise ValueError(f"{field} is not a file: {candidate}")
    return candidate


def _load_pdf_references(path: str | Path) -> tuple[_PdfPhaseReference, ...]:
    """Load and validate version-1 external PDF phase references."""
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

    names: set[str] = set()
    phases: list[_PdfPhaseReference] = []
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
        if not _PHASE_NAME.fullmatch(name):
            raise ValueError(
                f"{field}.name must match {_PHASE_NAME.pattern!r}: {name!r}"
            )
        if name in names:
            raise ValueError(f"{field}.name is duplicated: {name!r}")
        names.add(name)

        minimize = raw_phase["minimize"]
        if not isinstance(minimize, bool):
            raise ValueError(f"{field}.minimize must be a boolean")
        constraint_profile = raw_phase.get("constraint_profile", "none")
        if constraint_profile not in {"none", "cs_pb_br3"}:
            raise ValueError(
                f"{field}.constraint_profile is unsupported: {constraint_profile!r}"
            )

        gr_path = _reference_path(
            raw_phase["gr_path"],
            config_path.parent,
            f"{field}.gr_path",
            require_file=True,
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
            _PdfPhaseReference(
                name=name,
                gr_path=gr_path,
                minimize=minimize,
                cif_path=cif_path,
                constraint_profile=cast(
                    Literal["none", "cs_pb_br3"], constraint_profile
                ),
            )
        )
    return tuple(phases)


class _TiledAccessError(RuntimeError):
    """Mark an exception raised while looking up or reading Tiled data."""


def _read_stream_dataset(
    client: Any, uid: Hashable, stream_name: str
) -> tuple[Any, Mapping[str, Any]]:
    try:
        run = client[uid]
        dataset = run[stream_name].read()
        metadata = run.metadata["start"]
    except Exception as exc:
        raise _TiledAccessError(
            f"failed to read stream {stream_name!r} for uid={uid!r}"
        ) from exc
    return dataset, metadata


def _read_qepro_stream(
    client: Any,
    uid: Hashable,
    stream_name: str,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    dataset, metadata = _read_stream_dataset(client, uid, stream_name)
    missing = [field for field in _QEPRO_FIELDS if field not in dataset]
    if missing:
        raise ValueError(
            f"QEPro stream {stream_name!r} is missing fields: {', '.join(missing)}"
        )
    return (
        {field: np.asarray(dataset[field].values) for field in _QEPRO_FIELDS},
        metadata,
    )


def _write_oxidation_free_cif(cif_path: Path, output_directory: Path) -> Path:
    """Write a temporary CIF without oxidation states for diffpy."""
    from pymatgen.io.cif import CifParser, CifWriter

    structure = CifParser(str(cif_path)).parse_structures(primitive=True)[0]
    structure.remove_oxidation_states()
    output_path = output_directory / f"{cif_path.stem}_pym.cif"
    CifWriter(structure, symprec=0.1).write_file(str(output_path))
    return output_path


def _set_cs_pb_br3_constraints(pdf_fit: Any, *, phase_index: int = 1) -> None:
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
        pdf_fit.fixpar(parameter)


class XrayUvvisEvaluation:
    """Evaluate optical spectra and PDF data for one acquisition run."""

    def __init__(
        self,
        tiled_client: Any,
        sandbox_client: Any,
        pdf_references: str | Path,
        *,
        pdf_mode: Literal["raw", "fit"] = "fit",
        plqy: PlqyReference = PlqyReference(),  # ruff:ignore[function-call-in-default-argument]
        peak_target: float = 660,
        max_retries: int = 10,
        retry_delay: float = 2.0,
    ) -> None:
        if pdf_mode not in {"raw", "fit"}:
            raise ValueError("pdf_mode must be either 'raw' or 'fit'")
        if max_retries < 1:
            raise ValueError("max_retries must be at least one")
        if retry_delay < 0:
            raise ValueError("retry_delay cannot be negative")

        phases = _load_pdf_references(pdf_references)
        if pdf_mode == "fit":
            for phase in phases:
                if phase.cif_path is None:
                    raise ValueError(
                        f"phase {phase.name!r} requires cif_path for PDF mode 'fit'"
                    )
                if not phase.cif_path.is_file():
                    raise ValueError(
                        f"phase {phase.name!r} cif_path is not a file: {phase.cif_path}"
                    )

        self.tiled_client = tiled_client
        self.sandbox_client = sandbox_client
        self._pdf_mode: Literal["raw", "fit"] = pdf_mode
        self._phases = phases
        self._plqy = plqy
        self._peak_target = peak_target
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    @property
    def pdf_mode(self) -> Literal["raw", "fit"]:
        """PDF metrics used as optimization objectives."""
        return self._pdf_mode

    @property
    def phases(self) -> tuple[_PdfPhaseReference, ...]:
        """Validated PDF phase references in configured order."""
        return self._phases

    @property
    def peak_target(self) -> float:
        """Target fluorescence peak wavelength in nanometers."""
        return self._peak_target

    def _retry_access(
        self,
        uid: Hashable,
        operation: str,
        read: Callable[[], _T | None],
        missing: Callable[[], Sequence[str]],
    ) -> _T:
        last_access_error: BaseException | None = None
        for attempt in range(self._max_retries):
            try:
                result = read()
            except _TiledAccessError as exc:
                last_access_error = exc.__cause__ or exc
                logger.warning(
                    "Failed to read %s for uid=%r (attempt %d/%d): %r",
                    operation,
                    uid,
                    attempt + 1,
                    self._max_retries,
                    last_access_error,
                )
            else:
                if result is not None:
                    return result
            if attempt + 1 < self._max_retries:
                time.sleep(self._retry_delay)

        missing_items = tuple(missing())
        suffix = f" Missing: {', '.join(missing_items)}." if missing_items else ""
        raise RuntimeError(
            f"Failed to read {operation} for uid={uid!r} after "
            f"{self._max_retries} attempts.{suffix}"
        ) from last_access_error

    def _read_tiled_data(
        self, uid: Hashable
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        Mapping[str, Any],
        list[dict[str, Any]] | None,
    ]:
        """Read required raw streams, retaining successes between retries."""
        state: dict[str, Any] = {}

        def read() -> (
            tuple[
                dict[str, np.ndarray],
                dict[str, np.ndarray],
                Mapping[str, Any],
                list[dict[str, Any]] | None,
            ]
            | None
        ):
            errors: list[_TiledAccessError] = []
            for key, stream_name in (
                ("fluorescence", "fluorescence"),
                ("absorbance", "absorbance"),
            ):
                if key in state:
                    continue
                try:
                    values, metadata = _read_qepro_stream(
                        self.tiled_client, uid, stream_name
                    )
                except _TiledAccessError as exc:
                    errors.append(exc)
                else:
                    state[key] = values
                    state.setdefault("metadata", metadata)

            metadata = state.get("metadata")
            if metadata is not None:
                state["use_good_bad"] = bool(metadata.get("use_good_bad", False))
            if state.get("use_good_bad") and "quality" not in state:
                try:
                    state["quality"] = self._read_quality_stream(uid)
                except _TiledAccessError as exc:
                    errors.append(exc)

            if (
                "fluorescence" in state
                and "absorbance" in state
                and "metadata" in state
                and "use_good_bad" in state
                and (not state["use_good_bad"] or "quality" in state)
            ):
                return (
                    state["fluorescence"],
                    state["absorbance"],
                    state["metadata"],
                    state.get("quality"),
                )
            if errors:
                raise errors[-1]
            return None

        def missing() -> list[str]:
            names: list[str] = []
            if "metadata" not in state:
                names.append("run metadata")
            if "fluorescence" not in state:
                names.append("fluorescence stream")
            if "absorbance" not in state:
                names.append("absorbance stream")
            if state.get("use_good_bad") and "quality" not in state:
                names.append("fluorescence_quality stream")
            return names

        return self._retry_access(uid, "required Tiled data", read, missing)

    def _read_quality_stream(self, uid: Hashable) -> list[dict[str, Any]]:
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

    @staticmethod
    def _filter_fl_to_good_batches(
        fluorescence: dict[str, np.ndarray],
        batch_info: Sequence[Mapping[str, Any]],
    ) -> dict[str, np.ndarray]:
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
        return {
            field: array[indices]
            if (array := np.asarray(values)).ndim >= 1 and array.shape[0] == event_count
            else array
            for field, values in fluorescence.items()
        }

    def _read_pdfstream_data(self, uid: Hashable) -> dict[str, np.ndarray]:
        """Read the correlated pdfstream G(r) stream with bounded retries."""

        def read() -> dict[str, np.ndarray]:
            try:
                matches = self.sandbox_client.search(Eq("start.original_run_uid", uid))
                key = matches.keys().last()
                dataset = matches[key]["scattering"].read()
            except Exception as exc:
                raise _TiledAccessError(
                    f"failed to read pdfstream data for uid={uid!r}"
                ) from exc
            missing = [field for field in ("gr_r", "gr_G") if field not in dataset]
            if missing:
                raise ValueError(
                    "scattering stream is missing fields: " + ", ".join(missing)
                )
            return {
                field: np.asarray(dataset[field].values).squeeze()
                for field in ("gr_r", "gr_G")
            }

        return self._retry_access(
            uid,
            "pdfstream scattering data",
            read,
            lambda: ("scattering stream",),
        )

    def _raw_pdf_correlations(
        self, pdf_data: Mapping[str, np.ndarray]
    ) -> dict[str, float]:
        results: dict[str, float] = {}
        for phase in self._phases:
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

        fit_rmax = min(_PDF_RMAX, float(np.max(experimental_r[finite])))
        if fit_rmax <= _PDF_RMIN:
            raise ValueError(
                f"PDF fit rmax must be greater than {_PDF_RMIN} A; "
                f"measured rmax is {fit_rmax}"
            )

        results: dict[str, float] = {}
        with TemporaryDirectory() as directory:
            output_directory = Path(directory)
            measured_path = output_directory / "measured.gr"
            np.savetxt(
                measured_path,
                np.column_stack((experimental_r[finite], experimental_g[finite])),
                fmt="%.10g %.10g",
            )
            for phase in self._phases:
                if phase.cif_path is None:
                    raise RuntimeError(f"phase {phase.name!r} has no CIF path")
                clean_cif = _write_oxidation_free_cif(phase.cif_path, output_directory)
                structure = loadStructure(str(clean_cif))
                structure.Uisoequiv = 0.04
                structure.title = clean_cif.stem

                pdf_fit: Any = PdfFit()
                pdf_fit.read_data(
                    str(measured_path),
                    "X",
                    _PDF_QMAX,
                    _PDF_QDAMP,
                )
                pdf_fit.add_structure(structure)
                if phase.constraint_profile == "cs_pb_br3":
                    _set_cs_pb_br3_constraints(pdf_fit)
                pdf_fit.constrain(pdf_fit.dscale, "@902")
                pdf_fit.setpar(902, 1.0)
                pdf_fit.setvar(pdf_fit.qdamp, _PDF_QDAMP)
                pdf_fit.setvar(pdf_fit.qbroad, _PDF_QBROAD)
                pdf_fit.pdfrange(1, _PDF_RMIN, fit_rmax)
                pdf_fit.refine(toler=_PDF_TOLERANCE)
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
        results = self._raw_pdf_correlations(pdf_data)
        if self._pdf_mode == "raw":
            return results
        try:
            results.update(self._fit_pdf_correlations(pdf_data))
        except Exception as exc:
            raise RuntimeError(f"PDF fitting failed for uid={uid!r}") from exc
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

        pl_result = analyze_pl_spectra(
            fluorescence["QEPro_x_axis"], fluorescence["QEPro_output"]
        )
        wavelength, corrected_absorbance = correct_absorbance(
            absorbance["QEPro_x_axis"], absorbance["QEPro_output"]
        )
        if pl_result is None:
            peak = 0.0
            peak_distance = self._peak_target
            fwhm = 1000.0
            plqy = 1e-10
        else:
            peak, fwhm, pl_integral, _r_squared = pl_result
            if not np.isfinite(peak):
                raise ValueError(f"fitted Peak is not finite for uid={uid!r}")
            if not np.isfinite(fwhm) or fwhm <= 0:
                raise ValueError(
                    f"fitted FWHM is not positive and finite for uid={uid!r}"
                )
            excitation_index = int(
                np.abs(wavelength - self._plqy.excitation_wavelength_nm).argmin()
            )
            plqy = calculate_plqy(
                float(corrected_absorbance[excitation_index]),
                pl_integral,
                self._plqy.solvent_refractive_index,
                reference_type=self._plqy.reference_type,
                absorbance_reference=self._plqy.absorbance,
                pl_integral_reference=self._plqy.pl_integral,
                refractive_index_reference=self._plqy.refractive_index,
                plqy_reference=self._plqy.plqy,
            )
            peak_distance = abs(self._peak_target - peak)

        if not np.isfinite(plqy) or plqy <= 0:
            plqy = 1e-10
        pdf_metrics = self._process_pdf(self._read_pdfstream_data(uid), uid=uid)
        for name, value in pdf_metrics.items():
            if not np.isfinite(value):
                raise ValueError(f"PDF correlation {name!r} is not finite")

        outcomes = {
            "Peak": float(peak),
            "peak_distance": float(peak_distance),
            "log_FWHM": float(np.log(fwhm)),
            "log_PLQY": float(np.log(plqy)),
            **pdf_metrics,
        }
        return [{**outcomes, "_id": suggestion["_id"]} for suggestion in suggestions]
