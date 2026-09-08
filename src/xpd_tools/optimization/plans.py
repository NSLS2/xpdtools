"""Worker-side X-ray and UV-Vis acquisition plans."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import ceil, pi
from typing import Any, TypedDict, cast
from uuid import uuid4

import numpy as np
from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.callbacks import CallbackBase
from event_model import Event, EventDescriptor
from ophyd import Signal

from .analysis import classify_pl

logger = logging.getLogger(__name__)


class FlowConfig(TypedDict, total=False):
    """JSON-serializable synthesis-flow configuration."""

    syringe_list: list[float]
    syringe_mater_list: list[str]
    target_vol_list: list[str]
    set_target_list: list[bool]
    rate_unit: str
    mixer_lengths_cm: list[float]
    resident_t_ratio: float
    precursor_list: list[str]
    precursor_prefix_list: list[str]
    post_dilute: bool
    post_dilute_ratio: list[float]
    post_dilute_wait_sec: float
    dof_to_pump: dict[str, str]
    dilute_pump_name: list[str]


class XrayConfig(TypedDict, total=False):
    """JSON-serializable X-ray acquisition configuration."""

    do_xray: bool
    exposure: float
    frame_acq_time: float
    stream_name: str
    no_dark: bool


class WashConfig(TypedDict, total=False):
    """JSON-serializable wash-loop configuration."""

    do_wash: bool
    pump_names: list[str]
    syringe_list: list[float]
    rate_list: list[str]
    duration_sec: float
    syringe_mater_list: list[str]
    target_vol_list: list[str]
    set_target_list: list[bool]


class QualityThresholds(TypedDict, total=False):
    """PL quality-classification thresholds."""

    key_height: float
    height: float
    distance: int
    c2_c3: bool
    threshold: tuple[float, float, float]
    integration_bounds: tuple[float, float, float]


class QualityConfig(TypedDict, total=False):
    """PL quality-gating and UV-Vis shot-count configuration."""

    use_good_bad: bool
    good_target: int
    max_bad: int
    num_abs: int
    num_flu: int
    thresholds: QualityThresholds


DEFAULT_FLOW_CONFIG: FlowConfig = {
    "syringe_list": [50, 50, 50],
    "syringe_mater_list": ["steel", "steel", "steel"],
    "target_vol_list": ["30 ml", "30 ml", "30 ml"],
    "set_target_list": [True, True, True],
    "rate_unit": "ul/min",
    "mixer_lengths_cm": [30.0],
    "resident_t_ratio": 1.0,
    "precursor_list": ["CsPbOA", "TOABr", "ZnI2"],
    "precursor_prefix_list": ["CsPb", "Br", "I2"],
    "post_dilute": True,
    "post_dilute_ratio": [1.0, 1.0],
    "post_dilute_wait_sec": 30,
    "dof_to_pump": {
        "infusion_rate_CsPb": "dds2_p1",
        "infusion_rate_Br": "dds2_p2",
        "infusion_rate_I2": "dds3_p1",
    },
    "dilute_pump_name": ["dds1_p1", "ultra2"],
}

DEFAULT_XRAY_CONFIG: XrayConfig = {
    "do_xray": True,
    "exposure": 5.0,
    "frame_acq_time": 0.2,
    "stream_name": "scattering",
    "no_dark": True,
}

DEFAULT_WASH_CONFIG: WashConfig = {
    "do_wash": True,
    "pump_names": ["ultra1"],
    "syringe_list": [50],
    "rate_list": ["500 ul/min"],
    "duration_sec": 60,
    "syringe_mater_list": ["steel"],
    "target_vol_list": ["30 ml"],
    "set_target_list": [False],
}

DEFAULT_THRESHOLDS: QualityThresholds = {
    "key_height": 2000,
    "height": 30,
    "distance": 30,
    "c2_c3": False,
    "threshold": (560, 100000, 200000),
    "integration_bounds": (340, 400, 800),
}

DEFAULT_QUALITY_CONFIG: QualityConfig = {
    "use_good_bad": True,
    "good_target": 3,
    "max_bad": 3,
    "num_abs": 10,
    "num_flu": 10,
    "thresholds": DEFAULT_THRESHOLDS,
}


@dataclass(frozen=True)
class XrayUvvisPlanContext:
    """Hardware and X-ray wrapper owned by a bound acquisition plan."""

    qepro: Any
    led: Any
    uv_shutter: Any
    fast_shutter: Any
    pumps: Mapping[str, Any]
    xray_detector: Any | None = None
    wrap_xray_run: Callable[[Any, bool], Any] | None = None


@dataclass(frozen=True)
class _QualitySignals:
    """Signals emitted for one quality decision."""

    batch_index: Signal
    verdict: Signal
    peak_wavelength_nm: Signal
    n_good_total: Signal
    n_bad_total: Signal
    n_events_in_batch: Signal

    @property
    def all(self) -> tuple[Signal, ...]:
        """All quality-event signals in document order."""
        return (
            self.batch_index,
            self.verdict,
            self.peak_wavelength_nm,
            self.n_good_total,
            self.n_bad_total,
            self.n_events_in_batch,
        )


def _new_quality_signals() -> _QualitySignals:
    """Create the six event signals owned by one bound plan."""
    return _QualitySignals(
        batch_index=Signal(name="batch_index", value=0),
        verdict=Signal(name="verdict", value="bad"),
        peak_wavelength_nm=Signal(name="peak_wavelength_nm", value=-1.0),
        n_good_total=Signal(name="n_good_total", value=0),
        n_bad_total=Signal(name="n_bad_total", value=0),
        n_events_in_batch=Signal(name="n_events_in_batch", value=0),
    )


class PLQualityMonitor(CallbackBase):
    """Classify the final QEPro spectrum in each fluorescence batch."""

    def __init__(
        self,
        qepro: Any,
        stream_name: str = "fluorescence",
        thresholds: Mapping[str, Any] | None = None,
        *,
        signals: _QualitySignals | None = None,
    ) -> None:
        super().__init__()
        self.stream_name = stream_name
        self.x_field = qepro.x_axis.name
        self.y_field = qepro.output.name
        self.signals = signals or _new_quality_signals()
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._target_descriptors: set[str] = set()
        self._latest_spectrum: tuple[np.ndarray, np.ndarray] | None = None
        self._batch_event_count = 0
        self.good_count = 0
        self.bad_count = 0
        self.batch_index = 0
        self.batch_results: list[dict[str, Any]] = []

    def descriptor(self, doc: EventDescriptor) -> EventDescriptor:
        if doc.get("name") == self.stream_name:
            self._target_descriptors.add(str(doc["uid"]))
        return doc

    def event(self, doc: Event) -> Event:
        if doc["descriptor"] not in self._target_descriptors:
            return doc
        data = doc["data"]
        if self.x_field not in data or self.y_field not in data:
            return doc
        self._latest_spectrum = (
            np.asarray(data[self.x_field]),
            np.asarray(data[self.y_field]),
        )
        self._batch_event_count += 1
        return doc

    def finalize_batch(self) -> dict[str, Any]:
        """Classify and reset the most recently completed fluorescence batch."""
        if self._latest_spectrum is None:
            raise RuntimeError("fluorescence batch produced no matching QEPro event")
        wavelength, intensity = self._latest_spectrum
        classification = classify_pl(
            wavelength,
            intensity,
            key_height=self.thresholds["key_height"],
            height=self.thresholds["height"],
            distance=self.thresholds["distance"],
            c2_c3=self.thresholds["c2_c3"],
            threshold=self.thresholds["threshold"],
            integration_bounds=self.thresholds["integration_bounds"],
        )
        if classification.is_good:
            self.good_count += 1
        else:
            self.bad_count += 1
        result = {
            "batch_index": self.batch_index,
            "verdict": "good" if classification.is_good else "bad",
            "peak_wavelength_nm": classification.peak_wavelength_nm,
            "n_good_total": self.good_count,
            "n_bad_total": self.bad_count,
            "n_events_in_batch": self._batch_event_count,
        }
        self.batch_results.append(result)
        self.batch_index += 1
        self._batch_event_count = 0
        self._latest_spectrum = None
        return result


@dataclass(frozen=True)
class _ResolvedAcquisition:
    """Purely resolved configuration used by the acquisition generator."""

    flow: dict[str, Any]
    xray: dict[str, Any]
    wash: dict[str, Any]
    quality: dict[str, Any]
    thresholds: dict[str, Any]
    dof_names: tuple[str, ...]
    rates: tuple[float, ...]
    prefixes: tuple[str, ...]
    precursors: tuple[str, ...]
    synthesis_pumps: tuple[Any, ...]
    dilution_pumps: tuple[Any, ...]
    wash_pumps: tuple[Any, ...]


def _merge_config(
    defaults: Mapping[str, Any], overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Deep-copy defaults and caller overrides into an isolated mapping."""
    merged = deepcopy(dict(defaults))
    merged.update(deepcopy(dict(overrides or {})))
    return merged


def _resolve_pumps(
    names: Sequence[str],
    available: Mapping[str, Any],
    *,
    field: str,
) -> tuple[Any, ...]:
    """Resolve configured pump names only through the supplied context."""
    pumps: list[Any] = []
    for name in names:
        if name not in available:
            raise ValueError(f"{field} references unknown pump {name!r}")
        pumps.append(available[name])
    return tuple(pumps)


def _require_vector_coverage(
    config: Mapping[str, Any],
    fields: Sequence[str],
    count: int,
    *,
    owner: str,
) -> None:
    """Require every per-pump configuration vector to cover all pumps."""
    for field in fields:
        if len(config[field]) < count:
            raise ValueError(
                f"{owner}.{field} has {len(config[field])} entries for {count} pumps"
            )


def _preflight(
    context: XrayUvvisPlanContext,
    suggestions: Sequence[Mapping[str, Any]],
    flow_config: Mapping[str, Any] | None,
    xray_config: Mapping[str, Any] | None,
    wash_config: Mapping[str, Any] | None,
    quality_config: Mapping[str, Any] | None,
) -> _ResolvedAcquisition:
    """Resolve and validate all configuration before emitting a device message."""
    if len(suggestions) != 1 or not suggestions[0]:
        raise ValueError("xray_uvvis_acquire requires exactly one nonempty suggestion")

    flow = _merge_config(DEFAULT_FLOW_CONFIG, flow_config)
    xray = _merge_config(DEFAULT_XRAY_CONFIG, xray_config)
    wash = _merge_config(DEFAULT_WASH_CONFIG, wash_config)
    quality = _merge_config(DEFAULT_QUALITY_CONFIG, quality_config)
    thresholds = _merge_config(
        DEFAULT_THRESHOLDS,
        quality.get("thresholds"),
    )
    quality["thresholds"] = thresholds

    suggestion = suggestions[0]
    prefixes = tuple(flow["precursor_prefix_list"])
    known_dofs = {
        f"infusion_rate_{prefix}": index for index, prefix in enumerate(prefixes)
    }
    supplied_dofs = {name for name in suggestion if name.startswith("infusion_rate_")}
    unknown_dofs = sorted(supplied_dofs - known_dofs.keys())
    if unknown_dofs:
        raise ValueError(f"unknown infusion DOFs: {', '.join(unknown_dofs)}")
    dof_names = tuple(name for name in known_dofs if name in supplied_dofs)
    if not dof_names:
        raise ValueError("suggestion contains no configured infusion DOFs")

    indices = tuple(known_dofs[name] for name in dof_names)
    largest_index = max(indices) + 1
    _require_vector_coverage(
        flow,
        (
            "syringe_list",
            "syringe_mater_list",
            "target_vol_list",
            "set_target_list",
            "precursor_list",
        ),
        largest_index,
        owner="flow_config",
    )
    dof_to_pump = flow["dof_to_pump"]
    unmapped = [name for name in dof_names if name not in dof_to_pump]
    if unmapped:
        raise ValueError(f"unmapped infusion DOFs: {', '.join(unmapped)}")

    rates = tuple(float(suggestion[name]) for name in dof_names)
    if any(not np.isfinite(rate) or rate < 0 for rate in rates):
        raise ValueError("infusion rates must be finite and non-negative")
    if not any(rate > 0 for rate in rates):
        raise ValueError("at least one infusion rate must be greater than zero")
    rate_unit = str(flow["rate_unit"])
    for index, rate in zip(indices, rates, strict=True):
        normalized_rate = _rate_to_ul_per_minute(rate, rate_unit)
        if not np.isfinite(normalized_rate) or normalized_rate < 0:
            raise ValueError("infusion rates must be finite and non-negative")
        _parse_volume(str(flow["target_vol_list"][index]))
    mixer_lengths = tuple(float(value) for value in flow["mixer_lengths_cm"])
    if len(mixer_lengths) not in {1, 2} or any(
        not np.isfinite(length) or length < 0 for length in mixer_lengths
    ):
        raise ValueError("mixer_lengths_cm must contain one or two non-negative values")
    if (
        not np.isfinite(float(flow["resident_t_ratio"]))
        or float(flow["resident_t_ratio"]) < 0
    ):
        raise ValueError("flow_config.resident_t_ratio must be non-negative")

    pump_names = tuple(str(dof_to_pump[name]) for name in dof_names)
    synthesis_pumps = _resolve_pumps(
        pump_names,
        context.pumps,
        field="flow_config.dof_to_pump",
    )
    selected_prefixes = tuple(prefixes[index] for index in indices)
    precursors = tuple(flow["precursor_list"][index] for index in indices)

    dilution_pumps: tuple[Any, ...] = ()
    if flow["post_dilute"]:
        dilution_names = tuple(flow["dilute_pump_name"])
        if not dilution_names:
            raise ValueError("post_dilute requires at least one dilution pump")
        _require_vector_coverage(
            flow,
            ("post_dilute_ratio",),
            len(dilution_names),
            owner="flow_config",
        )
        dilution_pumps = _resolve_pumps(
            dilution_names,
            context.pumps,
            field="flow_config.dilute_pump_name",
        )

        dilution_ratios = [
            float(value) for value in flow["post_dilute_ratio"][: len(dilution_pumps)]
        ]
        if any(not np.isfinite(value) or value < 0 for value in dilution_ratios):
            raise ValueError("post_dilute_ratio values must be finite and non-negative")
    wash_pumps: tuple[Any, ...] = ()
    if wash["do_wash"]:
        wash_names = tuple(wash["pump_names"])
        _require_vector_coverage(
            wash,
            (
                "syringe_list",
                "rate_list",
                "syringe_mater_list",
                "target_vol_list",
                "set_target_list",
            ),
            len(wash_names),
            owner="wash_config",
        )
        wash_pumps = _resolve_pumps(
            wash_names,
            context.pumps,
            field="wash_config.pump_names",
        )

        for rate, target in zip(
            wash["rate_list"],
            wash["target_vol_list"],
            strict=True,
        ):
            normalized_rate = _rate_to_ul_per_minute(rate, str(flow["rate_unit"]))
            if not np.isfinite(normalized_rate) or normalized_rate < 0:
                raise ValueError("wash rates must be finite and non-negative")
            _parse_volume(str(target))
    if xray["do_xray"]:
        if context.xray_detector is None:
            raise ValueError("xray_detector is required when do_xray is true")
        if context.wrap_xray_run is None:
            raise ValueError("wrap_xray_run is required when do_xray is true")
        if float(xray["exposure"]) <= 0:
            raise ValueError("xray_config.exposure must be positive")
        if float(xray["frame_acq_time"]) <= 0:
            raise ValueError("xray_config.frame_acq_time must be positive")

    if int(quality["num_abs"]) < 1 or int(quality["num_flu"]) < 1:
        raise ValueError("quality shot counts must be positive")
    if float(wash["duration_sec"]) < 0:
        raise ValueError("wash_config.duration_sec cannot be negative")
    if float(flow["post_dilute_wait_sec"]) < 0:
        raise ValueError("flow_config.post_dilute_wait_sec cannot be negative")
    for field in ("good_target", "max_bad"):
        if int(quality[field]) < 0:
            raise ValueError(f"quality_config.{field} cannot be negative")
    if int(thresholds["distance"]) < 1:
        raise ValueError("quality_config.thresholds.distance must be positive")
    if len(thresholds["threshold"]) != 3:
        raise ValueError(
            "quality_config.thresholds.threshold must contain three values"
        )
    if len(thresholds["integration_bounds"]) != 3:
        raise ValueError(
            "quality_config.thresholds.integration_bounds must contain three values"
        )

    return _ResolvedAcquisition(
        flow=flow,
        xray=xray,
        wash=wash,
        quality=quality,
        thresholds=thresholds,
        dof_names=dof_names,
        rates=rates,
        prefixes=selected_prefixes,
        precursors=precursors,
        synthesis_pumps=synthesis_pumps,
        dilution_pumps=dilution_pumps,
        wash_pumps=wash_pumps,
    )


def _rate_to_ul_per_minute(value: float | int | str, default_unit: str) -> float:
    """Convert a configured pump rate to microliters per minute."""
    if isinstance(value, str):
        parts = value.split()
        if len(parts) != 2:
            raise ValueError(f"invalid rate {value!r}")
        magnitude = float(parts[0])
        unit = parts[1]
    else:
        magnitude = float(value)
        unit = default_unit
    try:
        volume_unit, time_unit = unit.lower().split("/", maxsplit=1)
        volume_factor = {"pl": 1e-6, "nl": 1e-3, "ul": 1.0, "ml": 1e3}[volume_unit]
        time_minutes = {"sec": 1 / 60, "min": 1.0, "hr": 60.0}[time_unit]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"unsupported flow-rate unit {unit!r}") from exc
    return magnitude * volume_factor / time_minutes


def _parse_volume(value: str) -> tuple[float, str]:
    """Parse a configured ``value unit`` volume."""
    parts = value.split()
    if len(parts) != 2:
        raise ValueError(f"invalid target volume {value!r}")
    return float(parts[0]), parts[1]


def _configure_pump(
    pump: Any,
    *,
    syringe_size: float,
    syringe_material: str,
    set_target: bool,
    target_volume: str,
    rate: float | int | str,
    rate_unit: str,
):
    """Configure one pump through its Bluesky plan interface."""
    normalized_rate = _rate_to_ul_per_minute(rate, rate_unit)
    if normalized_rate == 0:
        return
    target_value, target_unit = _parse_volume(target_volume)
    yield from pump.set_infuse2(
        syringe_size,
        syringe_material=syringe_material,
        set_target=set_target,
        target_vol=target_value,
        target_unit=target_unit,
        infuse_rate=normalized_rate,
        infuse_unit="ul/min",
    )


def _start_pump(pump: Any):
    """Start one pump through its Bluesky plan interface."""
    yield from pump.infuse_pump2()


def _stop_pump(pump: Any):
    """Stop one pump through its Bluesky plan interface."""
    yield from pump.stop_pump2()


def _unique_devices(devices: Sequence[Any]) -> list[Any]:
    """Deduplicate device objects while retaining first-seen order."""
    seen: set[int] = set()
    unique: list[Any] = []
    for device in devices:
        identity = id(device)
        if identity not in seen:
            seen.add(identity)
            unique.append(device)
    return unique


def _wait_for_equilibrium(
    pumps: Sequence[Any],
    mixer_lengths_cm: Sequence[float],
    *,
    ratio: float,
    tubing_id_mm: float = 1.016,
):
    """Wait for configured mixer residence time using live pump readbacks."""
    if len(mixer_lengths_cm) == 2:
        stages = (
            (float(mixer_lengths_cm[0]), pumps[:2]),
            (float(mixer_lengths_cm[1]), pumps),
        )
    elif len(mixer_lengths_cm) == 1:
        stages = ((float(mixer_lengths_cm[0]), pumps),)
    else:
        raise ValueError("mixer_lengths_cm must contain one or two lengths")

    residence_seconds = 0.0
    for length_cm, stage_pumps in stages:
        total_rate = 0.0
        for pump in stage_pumps:
            rate = yield from bps.rd(pump.read_infuse_rate)
            unit = yield from bps.rd(pump.read_infuse_rate_unit)
            status = yield from bps.rd(pump.status)
            if status == "Infusing":
                total_rate += _rate_to_ul_per_minute(float(rate), str(unit))
        if total_rate <= 0:
            raise RuntimeError("active mixer flow must be greater than zero")
        mixer_volume_ul = pi * (tubing_id_mm / 2) ** 2 * length_cm * 10
        residence_seconds += 60 * mixer_volume_ul / total_rate
    yield from bps.sleep(residence_seconds * ratio)


def measure_absorbance(
    qepro: Any,
    led: Any,
    uv_shutter: Any,
    n_shots: int,
    *,
    stream: str = "absorbance",
    settle_sec: float = 2,
):
    """Configure the optical path and collect absorbance events."""
    state = (
        (yield from bps.rd(led)),
        (yield from bps.rd(uv_shutter)),
        (yield from bps.rd(qepro.correction)),
        (yield from bps.rd(qepro.spectrum_type)),
    )
    if state != ("Low", "High", "Reference", "Absorbtion"):
        yield from bps.mv(
            qepro.correction,
            "Reference",
            qepro.spectrum_type,
            "Absorbtion",
            led,
            "Low",
            uv_shutter,
            "High",
        )
        yield from bps.sleep(settle_sec)
    for _ in range(n_shots):
        yield from bps.trigger_and_read([qepro], name=stream)


def measure_pl(
    qepro: Any,
    led: Any,
    uv_shutter: Any,
    n_shots: int,
    *,
    stream: str = "fluorescence",
    settle_sec: float = 2,
):
    """Configure the optical path and collect fluorescence events."""
    state = (
        (yield from bps.rd(led)),
        (yield from bps.rd(uv_shutter)),
        (yield from bps.rd(qepro.correction)),
        (yield from bps.rd(qepro.spectrum_type)),
    )
    if state != ("High", "Low", "Dark", "Corrected Sample"):
        yield from bps.mv(
            qepro.correction,
            "Dark",
            qepro.spectrum_type,
            "Corrected Sample",
            led,
            "High",
            uv_shutter,
            "Low",
        )
        yield from bps.sleep(settle_sec)
    for _ in range(n_shots):
        yield from bps.trigger_and_read([qepro], name=stream)


def prepare_xray_detector(
    detector: Any,
    exposure: float,
    frame_acq_time: float,
):
    """Configure an area detector and return scan-plan metadata."""
    yield from bps.mv(detector.cam.acquire_time, frame_acq_time)
    acquisition_time = float((yield from bps.rd(detector.cam.acquire_time)))
    if acquisition_time <= 0:
        raise ValueError("detector acquisition time must be positive")
    frame_count = max(1, int(ceil(exposure / acquisition_time)))
    if hasattr(detector, "images_per_set"):
        yield from bps.mv(detector.images_per_set, frame_count)
    computed_exposure = frame_count * acquisition_time
    plan_metadata = {
        "time_per_frame": acquisition_time,
        "num_frames": frame_count,
        "requested_exposure": exposure,
        "computed_exposure": computed_exposure,
        "type": "generator",
        "uid": str(uuid4()),
        "plan_name": "trigger",
    }
    return {
        "sp_time_per_frame": acquisition_time,
        "sp_num_frames": frame_count,
        "sp_requested_exposure": exposure,
        "sp_computed_exposure": computed_exposure,
        "sp_type": "bps.trigger",
        "sp_uid": str(uuid4()),
        "sp_plan_name": "trigger",
        "sp_detector": detector.name,
        "sp": plan_metadata,
    }


def measure_scattering(
    detector: Any,
    fast_shutter: Any,
    *,
    stream_name: str = "scattering",
):
    """Collect one scattering event and always close the fast shutter."""

    def acquire():
        yield from bps.mv(fast_shutter, -20)
        yield from bps.trigger_and_read([detector], name=stream_name)

    return (yield from bpp.finalize_wrapper(acquire(), bps.mv(fast_shutter, 20)))


def _emit_quality_event(signals: _QualitySignals, result: Mapping[str, Any]):
    """Emit one fluorescence-quality event."""
    peak_wavelength = float(result["peak_wavelength_nm"])
    if not np.isfinite(peak_wavelength):
        peak_wavelength = -1.0
    yield from bps.mv(
        signals.batch_index,
        int(result["batch_index"]),
        signals.verdict,
        result["verdict"],
        signals.peak_wavelength_nm,
        peak_wavelength,
        signals.n_good_total,
        int(result["n_good_total"]),
        signals.n_bad_total,
        int(result["n_bad_total"]),
        signals.n_events_in_batch,
        int(result["n_events_in_batch"]),
    )
    yield from bps.create(name="fluorescence_quality")
    for signal in signals.all:
        yield from bps.read(cast(Any, signal))
    yield from bps.save()


def _measure_pl_with_quality_gate(
    context: XrayUvvisPlanContext,
    monitor: PLQualityMonitor | None,
    *,
    num_flu: int,
    good_target: int,
    max_bad: int,
):
    """Collect PL batches until the configured quality limit is reached."""
    yield from measure_pl(
        context.qepro,
        context.led,
        context.uv_shutter,
        num_flu,
    )
    if monitor is None:
        return
    yield from _emit_quality_event(monitor.signals, monitor.finalize_batch())
    while monitor.good_count < good_target and monitor.bad_count < max_bad:
        yield from measure_pl(
            context.qepro,
            context.led,
            context.uv_shutter,
            num_flu,
        )
        yield from _emit_quality_event(monitor.signals, monitor.finalize_batch())


def _sample_name(rates: Sequence[float], prefixes: Sequence[str]) -> str:
    """Build the established rate-and-precursor sample name."""
    return "_".join(
        component
        for prefix, rate in zip(prefixes, rates, strict=True)
        for component in (prefix, f"{int(rate):03d}")
    )


def _device_name(device: Any) -> str:
    """Return a device name suitable for run metadata."""
    return str(device.name)


def _build_run_metadata(
    context: XrayUvvisPlanContext,
    resolved: _ResolvedAcquisition,
    supplied: Mapping[str, Any] | None,
    detector_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge caller metadata before authoritative derived metadata."""
    metadata = deepcopy(dict(supplied or {}))
    metadata.update(detector_metadata)
    detectors = [_device_name(context.qepro)]
    if resolved.xray["do_xray"]:
        detectors.append(_device_name(context.xray_detector))
    sample_name = _sample_name(resolved.rates, resolved.prefixes)
    metadata.update(
        {
            "sample_type": sample_name,
            "sample_name": sample_name,
            "infuse_rates": list(resolved.rates),
            "dof_names": list(resolved.dof_names),
            "precursors": list(resolved.precursors),
            "pumps": [_device_name(pump) for pump in resolved.synthesis_pumps],
            "detectors": detectors,
            "flow_config": deepcopy(resolved.flow),
            "xray_config": deepcopy(resolved.xray),
            "wash_config": deepcopy(resolved.wash),
            "quality_config": deepcopy(resolved.quality),
            "quality_thresholds": deepcopy(resolved.thresholds),
            "use_good_bad": bool(resolved.quality["use_good_bad"]),
        }
    )
    return metadata


def _configure_group(
    pumps: Sequence[Any],
    rates: Sequence[float | int | str],
    *,
    syringe_sizes: Sequence[float],
    syringe_materials: Sequence[str],
    set_targets: Sequence[bool],
    target_volumes: Sequence[str],
    rate_unit: str,
):
    """Configure each nonzero-rate pump in sequence."""
    for pump, rate, size, material, set_target, target in zip(
        pumps,
        rates,
        syringe_sizes,
        syringe_materials,
        set_targets,
        target_volumes,
        strict=True,
    ):
        yield from _configure_pump(
            pump,
            syringe_size=float(size),
            syringe_material=str(material),
            set_target=bool(set_target),
            target_volume=str(target),
            rate=rate,
            rate_unit=rate_unit,
        )


def _start_group(
    pumps: Sequence[Any],
    rates: Sequence[float | int | str],
    *,
    rate_unit: str,
    started: list[Any],
):
    """Start each nonzero-rate pump and record each successful start."""
    for pump, rate in zip(pumps, rates, strict=True):
        if _rate_to_ul_per_minute(rate, rate_unit) == 0:
            continue
        yield from _start_pump(pump)
        started.append(pump)


def _stop_running(pumps: Sequence[Any], started: list[Any]):
    """Stop selected running pumps and remove successful stops from tracking."""
    for pump in _unique_devices(pumps):
        yield from _stop_pump(pump)
        started[:] = [running for running in started if running is not pump]


def _cleanup_devices(context: XrayUvvisPlanContext, started: Sequence[Any]):
    """Attempt every pump and optical safe-state action before raising failures."""
    errors: list[Exception] = []
    for pump in reversed(_unique_devices(started)):
        try:
            yield from _stop_pump(pump)
        except Exception as exc:
            logger.exception("Failed to stop pump %s", _device_name(pump))
            errors.append(exc)
    for signal, value, label in (
        (context.led, "Low", "LED"),
        (context.uv_shutter, "Low", "UV shutter"),
        (context.fast_shutter, 20, "fast shutter"),
    ):
        try:
            yield from bps.abs_set(signal, value, wait=True)
        except Exception as exc:
            logger.exception("Failed to place %s in its safe state", label)
            errors.append(exc)
    if errors:
        raise ExceptionGroup("acquisition cleanup failed", errors)


def _with_safe_cleanup(plan: Any, cleanup: Callable[[], Any]):
    """Run cleanup and explicitly chain failures from acquisition errors."""
    try:
        result = yield from plan
    except Exception as primary_error:
        try:
            yield from cleanup()
        except Exception as cleanup_error:
            raise cleanup_error from primary_error
        raise
    else:
        yield from cleanup()
        return result


def create_xray_uvvis_plan(context: XrayUvvisPlanContext) -> Callable[..., Any]:
    """Bind hardware once and return the Queue Server acquisition plan."""
    quality_signals = _new_quality_signals()

    def xray_uvvis_acquire(
        suggestions: Sequence[Mapping[str, Any]],
        actuators: Sequence[Any],
        sensors: Sequence[Any] | None = None,
        md: Mapping[str, Any] | None = None,
        *,
        flow_config: Mapping[str, Any] | None = None,
        xray_config: Mapping[str, Any] | None = None,
        wash_config: Mapping[str, Any] | None = None,
        quality_config: Mapping[str, Any] | None = None,
    ):
        """Acquire one correlated UV-Vis and optional X-ray optimization run."""
        del actuators, sensors
        resolved = _preflight(
            context,
            suggestions,
            flow_config,
            xray_config,
            wash_config,
            quality_config,
        )

        detector_metadata: Mapping[str, Any] = {}
        if resolved.xray["do_xray"]:
            detector_metadata = yield from prepare_xray_detector(
                context.xray_detector,
                float(resolved.xray["exposure"]),
                float(resolved.xray["frame_acq_time"]),
            )
        run_metadata = _build_run_metadata(
            context,
            resolved,
            md,
            detector_metadata,
        )

        monitor = (
            PLQualityMonitor(
                context.qepro,
                thresholds=resolved.thresholds,
                signals=quality_signals,
            )
            if resolved.quality["use_good_bad"]
            else None
        )
        started: list[Any] = []

        def cleanup():
            yield from _cleanup_devices(context, started)

        def acquisition():
            all_pumps = _unique_devices(
                [
                    *resolved.synthesis_pumps,
                    *resolved.dilution_pumps,
                    *resolved.wash_pumps,
                ]
            )
            for pump in all_pumps:
                yield from _stop_pump(pump)

            indices = [
                resolved.flow["precursor_prefix_list"].index(prefix)
                for prefix in resolved.prefixes
            ]
            yield from _configure_group(
                resolved.synthesis_pumps,
                resolved.rates,
                syringe_sizes=[resolved.flow["syringe_list"][i] for i in indices],
                syringe_materials=[
                    resolved.flow["syringe_mater_list"][i] for i in indices
                ],
                set_targets=[resolved.flow["set_target_list"][i] for i in indices],
                target_volumes=[resolved.flow["target_vol_list"][i] for i in indices],
                rate_unit=str(resolved.flow["rate_unit"]),
            )
            yield from _start_group(
                resolved.synthesis_pumps,
                resolved.rates,
                rate_unit=str(resolved.flow["rate_unit"]),
                started=started,
            )

            total_rate = sum(rate for rate in resolved.rates if rate > 0)
            dilution_rates = [
                total_rate * float(ratio)
                for ratio in resolved.flow["post_dilute_ratio"][
                    : len(resolved.dilution_pumps)
                ]
            ]
            pre_dilution_count = max(0, len(resolved.dilution_pumps) - 1)
            if pre_dilution_count:
                pre_pumps = resolved.dilution_pumps[:pre_dilution_count]
                pre_rates = dilution_rates[:pre_dilution_count]
                yield from _configure_group(
                    pre_pumps,
                    pre_rates,
                    syringe_sizes=[20] * pre_dilution_count,
                    syringe_materials=["plastic_BD"] * pre_dilution_count,
                    set_targets=[True] * pre_dilution_count,
                    target_volumes=["20 ml"] * pre_dilution_count,
                    rate_unit="ul/min",
                )
                yield from _start_group(
                    pre_pumps,
                    pre_rates,
                    rate_unit="ul/min",
                    started=started,
                )

            yield from _wait_for_equilibrium(
                resolved.synthesis_pumps,
                resolved.flow["mixer_lengths_cm"],
                ratio=float(resolved.flow["resident_t_ratio"]),
            )

            if resolved.dilution_pumps:
                post_pumps = resolved.dilution_pumps[-1:]
                post_rates = dilution_rates[-1:]
                yield from _configure_group(
                    post_pumps,
                    post_rates,
                    syringe_sizes=[100],
                    syringe_materials=["steel"],
                    set_targets=[True],
                    target_volumes=["100 ml"],
                    rate_unit="ul/min",
                )
                yield from _start_group(
                    post_pumps,
                    post_rates,
                    rate_unit="ul/min",
                    started=started,
                )
                yield from bps.sleep(float(resolved.flow["post_dilute_wait_sec"]))

            yield from _measure_pl_with_quality_gate(
                context,
                monitor,
                num_flu=int(resolved.quality["num_flu"]),
                good_target=int(resolved.quality["good_target"]),
                max_bad=int(resolved.quality["max_bad"]),
            )
            yield from measure_absorbance(
                context.qepro,
                context.led,
                context.uv_shutter,
                int(resolved.quality["num_abs"]),
            )
            yield from bps.mv(context.led, "Low", context.uv_shutter, "Low")

            if resolved.xray["do_xray"]:
                scattering = measure_scattering(
                    context.xray_detector,
                    context.fast_shutter,
                    stream_name=str(resolved.xray["stream_name"]),
                )
                wrapper = context.wrap_xray_run
                if wrapper is None:
                    raise RuntimeError("X-ray wrapper disappeared after preflight")
                yield from wrapper(
                    scattering,
                    bool(resolved.xray["no_dark"]),
                )

            yield from _stop_running(tuple(started), started)

            if resolved.wash_pumps:
                wash_count = len(resolved.wash_pumps)
                yield from _configure_group(
                    resolved.wash_pumps,
                    resolved.wash["rate_list"][:wash_count],
                    syringe_sizes=resolved.wash["syringe_list"][:wash_count],
                    syringe_materials=resolved.wash["syringe_mater_list"][
                        :wash_count
                    ],
                    set_targets=resolved.wash["set_target_list"][:wash_count],
                    target_volumes=resolved.wash["target_vol_list"][:wash_count],
                    rate_unit=str(resolved.flow["rate_unit"]),
                )
                yield from _start_group(
                    resolved.wash_pumps,
                    resolved.wash["rate_list"][:wash_count],
                    rate_unit=str(resolved.flow["rate_unit"]),
                    started=started,
                )
                yield from bps.sleep(float(resolved.wash["duration_sec"]))

        stage_devices = [context.qepro]
        if resolved.xray["do_xray"]:
            stage_devices.append(context.xray_detector)
        plan = bpp.stage_wrapper(acquisition(), stage_devices)
        plan = bpp.baseline_wrapper(plan, list(resolved.synthesis_pumps))
        plan = _with_safe_cleanup(plan, cleanup)
        if monitor is not None:
            plan = bpp.subs_wrapper(
                plan,
                {"descriptor": [monitor], "event": [monitor]},
            )
        plan = bpp.run_wrapper(plan, md=run_metadata)
        plan = bpp.set_run_key_wrapper(plan, "xray_uvvis_acquire")
        return (yield from plan)

    return xray_uvvis_acquire
