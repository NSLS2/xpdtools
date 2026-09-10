"""Worker-side X-ray and UV-Vis acquisition plans."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
from math import ceil, pi
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np
from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from ophyd import Signal

from .analysis import classify_pl

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class FlowSource:
    """One precursor pump controlled by an optimization degree of freedom."""

    dof: str
    pump: Any
    precursor: str
    sample_label: str
    syringe_ml: float = 50.0
    material: str = "steel"
    target_ml: float = 30.0
    set_target: bool = True


@dataclass(frozen=True, kw_only=True)
class DilutionStage:
    """A dilution pump started before or after mixer equilibration."""

    pump: Any
    ratio: float
    position: Literal["before_equilibrium", "after_equilibrium"]
    syringe_ml: float
    material: str
    target_ml: float
    set_target: bool = True
    wait_sec: float = 0.0


@dataclass(frozen=True, kw_only=True)
class WashCycle:
    """One sequential wash-pump cycle."""

    pump: Any
    rate_ul_min: float = 500.0
    duration_sec: float = 60.0
    syringe_ml: float = 50.0
    material: str = "steel"
    target_ml: float = 30.0
    set_target: bool = False


@dataclass(frozen=True, kw_only=True)
class QualityPolicy:
    """Fluorescence quality-gating and UV-Vis shot policy."""

    enabled: bool = True
    good_batches: int = 3
    max_bad_batches: int = 3
    absorbance_shots: int = 10
    fluorescence_shots: int = 10


@dataclass(frozen=True, kw_only=True)
class XraySettings:
    """X-ray detector acquisition settings."""

    exposure: float = 5.0
    frame_acq_time: float = 0.2
    no_dark: bool = True


@dataclass(frozen=True, kw_only=True)
class XrayUvvisPlanContext:
    """Static hardware and process configuration for one bound plan."""

    qepro: Any
    led: Any
    uv_shutter: Any
    fast_shutter: Any
    xray_detector: Any
    wrap_xray_run: Callable[[Any, bool], Any]
    sources: tuple[FlowSource, ...]
    dilutions: tuple[DilutionStage, ...]
    wash_cycles: tuple[WashCycle, ...]
    mixer_lengths_cm: tuple[float, ...] = (30.0,)
    residence_time_ratio: float = 1.0
    quality: QualityPolicy = QualityPolicy()
    xray: XraySettings = XraySettings()


def _new_quality_signals() -> dict[str, Signal]:
    return {
        name: Signal(name=name, value=value)
        for name, value in (
            ("batch_index", 0),
            ("verdict", "bad"),
            ("peak_wavelength_nm", -1.0),
            ("n_good_total", 0),
            ("n_bad_total", 0),
            ("n_events_in_batch", 0),
        )
    }


def _require_finite_nonnegative(value: float, field: str) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and non-negative")


def _validate_context(context: XrayUvvisPlanContext) -> None:
    """Validate static process configuration before constructing a plan."""
    if context.xray_detector is None:
        raise ValueError("xray_detector is required")
    if context.wrap_xray_run is None:
        raise ValueError("wrap_xray_run is required")
    if not context.sources:
        raise ValueError("sources must contain at least one flow source")

    dofs = [source.dof for source in context.sources]
    if any(not dof or not dof.startswith("infusion_rate_") for dof in dofs):
        raise ValueError("source DOFs must be nonempty infusion_rate_* names")
    if len(set(dofs)) != len(dofs):
        raise ValueError("source DOFs must be unique")

    if len(context.mixer_lengths_cm) not in {1, 2}:
        raise ValueError("mixer_lengths_cm must contain one or two values")
    for index, length in enumerate(context.mixer_lengths_cm):
        _require_finite_nonnegative(length, f"mixer_lengths_cm[{index}]")
    _require_finite_nonnegative(context.residence_time_ratio, "residence_time_ratio")

    for index, stage in enumerate(context.dilutions):
        if stage.position not in {"before_equilibrium", "after_equilibrium"}:
            raise ValueError(f"dilutions[{index}].position is unsupported")
        _require_finite_nonnegative(stage.ratio, f"dilutions[{index}].ratio")
        _require_finite_nonnegative(stage.wait_sec, f"dilutions[{index}].wait_sec")
    for index, cycle in enumerate(context.wash_cycles):
        _require_finite_nonnegative(
            cycle.rate_ul_min, f"wash_cycles[{index}].rate_ul_min"
        )
        _require_finite_nonnegative(
            cycle.duration_sec, f"wash_cycles[{index}].duration_sec"
        )

    if context.quality.good_batches < 0 or context.quality.max_bad_batches < 0:
        raise ValueError("quality batch limits must be non-negative")
    if context.quality.absorbance_shots < 1 or context.quality.fluorescence_shots < 1:
        raise ValueError("quality shot counts must be positive")
    if not np.isfinite(context.xray.exposure) or context.xray.exposure <= 0:
        raise ValueError("xray.exposure must be positive and finite")
    if not np.isfinite(context.xray.frame_acq_time) or context.xray.frame_acq_time <= 0:
        raise ValueError("xray.frame_acq_time must be positive and finite")


def _preflight(
    context: XrayUvvisPlanContext,
    suggestions: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    """Validate one complete suggestion before emitting any device message."""
    if len(suggestions) != 1 or not suggestions[0]:
        raise ValueError("xray_uvvis_acquire requires exactly one nonempty suggestion")

    suggestion = suggestions[0]
    configured_dofs = tuple(source.dof for source in context.sources)
    configured_set = set(configured_dofs)
    supplied_dofs = {name for name in suggestion if name.startswith("infusion_rate_")}
    if supplied_dofs != configured_set:
        raise ValueError(
            "suggestion infusion DOFs must match sources: "
            f"expected {sorted(configured_set)}, got {sorted(supplied_dofs)}"
        )

    unknown_fields = sorted(set(suggestion) - configured_set - {"_id"})
    if unknown_fields:
        raise ValueError(f"suggestion has unknown fields: {', '.join(unknown_fields)}")

    rates = tuple(float(suggestion[dof]) for dof in configured_dofs)
    if any(not np.isfinite(rate) or rate < 0 for rate in rates):
        raise ValueError("infusion rates must be finite and non-negative")
    if not any(rate > 0 for rate in rates):
        raise ValueError("at least one infusion rate must be greater than zero")
    return rates


def _unique_devices(devices: Sequence[Any]) -> list[Any]:
    seen: set[int] = set()
    unique: list[Any] = []
    for device in devices:
        identity = id(device)
        if identity not in seen:
            seen.add(identity)
            unique.append(device)
    return unique


def _configure_and_start(
    settings: Sequence[tuple[FlowSource | DilutionStage | WashCycle, float]],
    started: list[Any],
):
    """Configure every nonzero pump before starting each one in order."""
    for config, rate in settings:
        if rate == 0:
            continue
        yield from config.pump.set_infuse2(
            config.syringe_ml,
            syringe_material=config.material,
            set_target=config.set_target,
            target_vol=config.target_ml,
            target_unit="ml",
            infuse_rate=rate,
            infuse_unit="ul/min",
        )
    for config, rate in settings:
        if rate == 0:
            continue
        yield from config.pump.infuse_pump2()
        started.append(config.pump)


def _stop_running(pumps: Sequence[Any], started: list[Any]):
    for pump in _unique_devices(pumps):
        yield from pump.stop_pump2()
        started[:] = [running for running in started if running is not pump]


def _wait_for_equilibrium(
    pumps: Sequence[Any],
    mixer_lengths_cm: Sequence[float],
    *,
    ratio: float,
    tubing_id_mm: float = 1.016,
):
    """Wait for configured mixer residence time using live pump readbacks."""
    stages = (
        (
            (float(mixer_lengths_cm[0]), pumps[:2]),
            (float(mixer_lengths_cm[1]), pumps),
        )
        if len(mixer_lengths_cm) == 2
        else ((float(mixer_lengths_cm[0]), pumps),)
    )
    residence_seconds = 0.0
    for length_cm, stage_pumps in stages:
        total_rate = 0.0
        for pump in stage_pumps:
            rate = yield from bps.rd(pump.read_infuse_rate)
            unit = yield from bps.rd(pump.read_infuse_rate_unit)
            status = yield from bps.rd(pump.status)
            if status == "Infusing":
                if str(unit).lower() != "ul/min":
                    raise ValueError(
                        f"pump {pump.name!r} rate unit must be 'ul/min': {unit!r}"
                    )
                total_rate += float(rate)
        if total_rate <= 0:
            raise RuntimeError("active mixer flow must be greater than zero")
        mixer_volume_ul = pi * (tubing_id_mm / 2) ** 2 * length_cm * 10
        residence_seconds += 60 * mixer_volume_ul / total_rate
    yield from bps.sleep(residence_seconds * ratio)


def _measure_uvvis(
    context: XrayUvvisPlanContext,
    stream: Literal["fluorescence", "absorbance"],
    shots: int,
    *,
    settle_sec: float = 2,
):
    """Configure one optical mode and return its final event reading."""
    modes = {
        "fluorescence": ("High", "Low", "Dark", "Corrected Sample"),
        "absorbance": ("Low", "High", "Reference", "Absorbtion"),
    }
    led_value, shutter_value, correction, spectrum_type = modes[stream]
    state = (
        (yield from bps.rd(context.led)),
        (yield from bps.rd(context.uv_shutter)),
        (yield from bps.rd(context.qepro.correction)),
        (yield from bps.rd(context.qepro.spectrum_type)),
    )
    desired = (led_value, shutter_value, correction, spectrum_type)
    if state != desired:
        yield from bps.mv(
            context.qepro.correction,
            correction,
            context.qepro.spectrum_type,
            spectrum_type,
            context.led,
            led_value,
            context.uv_shutter,
            shutter_value,
        )
        yield from bps.sleep(settle_sec)

    reading: Mapping[str, Any] | None = None
    for _ in range(shots):
        reading = yield from bps.trigger_and_read([context.qepro], name=stream)
    return reading


def _emit_quality_event(signals: Mapping[str, Signal], result: Mapping[str, Any]):
    """Emit one fluorescence-quality event."""
    values = dict(result)
    peak_wavelength = float(values["peak_wavelength_nm"])
    values["peak_wavelength_nm"] = (
        peak_wavelength if np.isfinite(peak_wavelength) else -1.0
    )
    moves = [item for name, value in values.items() for item in (signals[name], value)]
    yield from bps.mv(*moves)
    yield from bps.create(name="fluorescence_quality")
    for signal in signals.values():
        yield from bps.read(cast(Any, signal))
    yield from bps.save()


def _measure_pl_with_quality_gate(
    context: XrayUvvisPlanContext,
    signals: Mapping[str, Signal],
):
    """Collect fluorescence batches until the configured quality limit."""
    good_count = 0
    bad_count = 0
    batch_index = 0
    while True:
        reading = yield from _measure_uvvis(
            context,
            "fluorescence",
            context.quality.fluorescence_shots,
        )
        if not context.quality.enabled:
            return

        x_field = context.qepro.x_axis.name
        y_field = context.qepro.output.name
        if reading is None or x_field not in reading or y_field not in reading:
            raise RuntimeError("fluorescence batch produced no matching QEPro event")
        wavelength = np.asarray(reading[x_field]["value"])
        intensity = np.asarray(reading[y_field]["value"])
        is_good, peak_wavelength = classify_pl(wavelength, intensity)
        if is_good:
            good_count += 1
        else:
            bad_count += 1
        yield from _emit_quality_event(
            signals,
            {
                "batch_index": batch_index,
                "verdict": "good" if is_good else "bad",
                "peak_wavelength_nm": peak_wavelength,
                "n_good_total": good_count,
                "n_bad_total": bad_count,
                "n_events_in_batch": context.quality.fluorescence_shots,
            },
        )
        batch_index += 1
        if (
            good_count >= context.quality.good_batches
            or bad_count >= context.quality.max_bad_batches
        ):
            return


def _prepare_xray_detector(context: XrayUvvisPlanContext):
    """Configure the area detector and return scan-plan metadata."""
    yield from bps.mv(
        context.xray_detector.cam.acquire_time,
        context.xray.frame_acq_time,
    )
    acquisition_time = float(
        (yield from bps.rd(context.xray_detector.cam.acquire_time))
    )
    if acquisition_time <= 0:
        raise ValueError("detector acquisition time must be positive")
    frame_count = max(1, int(ceil(context.xray.exposure / acquisition_time)))
    if hasattr(context.xray_detector, "images_per_set"):
        yield from bps.mv(context.xray_detector.images_per_set, frame_count)
    computed_exposure = frame_count * acquisition_time
    plan_metadata = {
        "time_per_frame": acquisition_time,
        "num_frames": frame_count,
        "requested_exposure": context.xray.exposure,
        "computed_exposure": computed_exposure,
        "type": "generator",
        "uid": str(uuid4()),
        "plan_name": "trigger",
    }
    return {
        "sp_time_per_frame": acquisition_time,
        "sp_num_frames": frame_count,
        "sp_requested_exposure": context.xray.exposure,
        "sp_computed_exposure": computed_exposure,
        "sp_type": "bps.trigger",
        "sp_uid": str(uuid4()),
        "sp_plan_name": "trigger",
        "sp_detector": context.xray_detector.name,
        "sp": plan_metadata,
    }


def _measure_scattering(context: XrayUvvisPlanContext):
    """Collect one scattering event and always close the fast shutter."""

    def acquire():
        yield from bps.mv(context.fast_shutter, -20)
        yield from bps.trigger_and_read([context.xray_detector], name="scattering")

    return (
        yield from bpp.finalize_wrapper(acquire(), bps.mv(context.fast_shutter, 20))
    )


def _sample_name(rates: Sequence[float], labels: Sequence[str]) -> str:
    return "_".join(
        component
        for label, rate in zip(labels, rates, strict=True)
        for component in (label, f"{int(rate):03d}")
    )


def _device_name(device: Any) -> str:
    return str(device.name)


def _config_metadata(config: Any) -> dict[str, Any]:
    """Serialize one frozen configuration object, replacing a pump by its name."""
    metadata = {field.name: getattr(config, field.name) for field in fields(config)}
    if "pump" in metadata:
        metadata["pump"] = _device_name(metadata["pump"])
    return metadata


def _serialized_config(context: XrayUvvisPlanContext) -> dict[str, Any]:
    return {
        "devices": {
            "qepro": _device_name(context.qepro),
            "led": _device_name(context.led),
            "uv_shutter": _device_name(context.uv_shutter),
            "fast_shutter": _device_name(context.fast_shutter),
            "xray_detector": _device_name(context.xray_detector),
        },
        "sources": [_config_metadata(source) for source in context.sources],
        "dilutions": [_config_metadata(stage) for stage in context.dilutions],
        "wash_cycles": [_config_metadata(cycle) for cycle in context.wash_cycles],
        "mixer_lengths_cm": list(context.mixer_lengths_cm),
        "residence_time_ratio": context.residence_time_ratio,
        "quality": _config_metadata(context.quality),
        "xray": _config_metadata(context.xray),
    }


def _build_run_metadata(
    context: XrayUvvisPlanContext,
    rates: tuple[float, ...],
    supplied: Mapping[str, Any] | None,
    detector_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge caller metadata before authoritative derived metadata."""
    metadata = deepcopy(dict(supplied or {}))
    for obsolete in (
        "flow_config",
        "xray_config",
        "wash_config",
        "quality_config",
        "quality_thresholds",
    ):
        metadata.pop(obsolete, None)
    metadata.update(detector_metadata)
    sample_name = _sample_name(
        rates, tuple(source.sample_label for source in context.sources)
    )
    metadata.update(
        {
            "sample_type": sample_name,
            "sample_name": sample_name,
            "infuse_rates": list(rates),
            "dof_names": [source.dof for source in context.sources],
            "precursors": [source.precursor for source in context.sources],
            "pumps": [_device_name(source.pump) for source in context.sources],
            "detectors": [
                _device_name(context.qepro),
                _device_name(context.xray_detector),
            ],
            "xray_uvvis_config": _serialized_config(context),
            "use_good_bad": context.quality.enabled,
        }
    )
    return metadata


def _cleanup_devices(context: XrayUvvisPlanContext, started: Sequence[Any]):
    """Attempt every pump and optical safe-state action before raising failures."""
    errors: list[Exception] = []
    for pump in reversed(_unique_devices(started)):
        try:
            yield from pump.stop_pump2()
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
    """Bind validated hardware once and return the Queue Server acquisition plan."""
    _validate_context(context)
    quality_signals = _new_quality_signals()

    def xray_uvvis_acquire(
        suggestions: Sequence[Mapping[str, Any]],
        actuators: Sequence[Any],
        sensors: Sequence[Any] | None = None,
        md: Mapping[str, Any] | None = None,
    ):
        """Acquire one correlated UV-Vis and X-ray optimization run."""
        del actuators, sensors
        rates = _preflight(context, suggestions)
        detector_metadata = yield from _prepare_xray_detector(context)
        run_metadata = _build_run_metadata(
            context,
            rates,
            md,
            detector_metadata,
        )
        started: list[Any] = []

        def cleanup():
            yield from _cleanup_devices(context, started)

        def acquisition():
            all_pumps = _unique_devices(
                [
                    *(source.pump for source in context.sources),
                    *(stage.pump for stage in context.dilutions),
                    *(cycle.pump for cycle in context.wash_cycles),
                ]
            )
            for pump in all_pumps:
                yield from pump.stop_pump2()

            yield from _configure_and_start(
                tuple(zip(context.sources, rates, strict=True)),
                started,
            )
            total_rate = sum(rates)
            dilution_settings = tuple(
                (stage, total_rate * stage.ratio) for stage in context.dilutions
            )
            before = tuple(
                item
                for item in dilution_settings
                if item[0].position == "before_equilibrium"
            )
            after = tuple(
                item
                for item in dilution_settings
                if item[0].position == "after_equilibrium"
            )
            yield from _configure_and_start(before, started)
            for stage, rate in before:
                if rate > 0 and stage.wait_sec:
                    yield from bps.sleep(stage.wait_sec)

            yield from _wait_for_equilibrium(
                tuple(source.pump for source in context.sources),
                context.mixer_lengths_cm,
                ratio=context.residence_time_ratio,
            )

            yield from _configure_and_start(after, started)
            for stage, rate in after:
                if rate > 0 and stage.wait_sec:
                    yield from bps.sleep(stage.wait_sec)

            yield from _measure_pl_with_quality_gate(context, quality_signals)
            yield from _measure_uvvis(
                context,
                "absorbance",
                context.quality.absorbance_shots,
            )
            yield from bps.mv(context.led, "Low", context.uv_shutter, "Low")

            yield from context.wrap_xray_run(
                _measure_scattering(context),
                context.xray.no_dark,
            )

            yield from _stop_running(tuple(started), started)
            for cycle in context.wash_cycles:
                yield from _configure_and_start(
                    ((cycle, cycle.rate_ul_min),),
                    started,
                )
                if cycle.rate_ul_min > 0 and cycle.duration_sec:
                    yield from bps.sleep(cycle.duration_sec)
                yield from _stop_running((cycle.pump,), started)

        plan = bpp.stage_wrapper(acquisition(), [context.qepro, context.xray_detector])
        plan = bpp.baseline_wrapper(
            plan, _unique_devices([source.pump for source in context.sources])
        )
        plan = _with_safe_cleanup(plan, cleanup)
        plan = bpp.run_wrapper(plan, md=run_metadata)
        plan = bpp.set_run_key_wrapper(plan, "xray_uvvis_acquire")
        return (yield from plan)

    return xray_uvvis_acquire
