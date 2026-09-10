from __future__ import annotations

import json
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from bluesky import plan_stubs as bps
from bluesky.run_engine import RunEngine
from ophyd import Component as Cpt
from ophyd import Device, Signal
from ophyd.status import DeviceStatus

from xpd_tools.optimization.plans import (
    DilutionStage,
    FlowSource,
    QualityPolicy,
    WashCycle,
    XraySettings,
    XrayUvvisPlanContext,
)


class FakeQEPro(Device):
    x_axis = Cpt(Signal, value=np.linspace(200.0, 950.0, 751))
    output = Cpt(Signal, value=np.zeros(751))
    spectrum_type = Cpt(Signal, value="Corrected Sample")
    correction = Cpt(Signal, value="Dark")

    def __init__(
        self,
        *args: Any,
        spectra: Sequence[np.ndarray] | None = None,
        fail_on_trigger: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spectra = list(spectra or [])
        self.fail_on_trigger = fail_on_trigger
        self.trigger_count = 0

    def trigger(self) -> DeviceStatus:
        self.trigger_count += 1
        status = DeviceStatus(self)
        if self.fail_on_trigger == self.trigger_count:
            status.set_exception(RuntimeError("QEPro trigger failed"))
            return status
        if self.spectra:
            spectrum = self.spectra.pop(0)
            self.output.put(np.asarray(spectrum, dtype=float))
        status.set_finished()
        return status


class FakePump(Device):
    read_infuse_rate = Cpt(Signal, value=0.0)
    read_infuse_rate_unit = Cpt(Signal, value="ul/min")
    status = Cpt(Signal, value="Stopped")

    def __init__(
        self,
        *args: Any,
        fail_start: bool = False,
        fail_stop_call: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fail_start = fail_start
        self.fail_stop_call = fail_stop_call
        self.configurations: list[dict[str, Any]] = []
        self.start_count = 0
        self.stop_count = 0

    def set_infuse2(
        self,
        input_size: float,
        *,
        syringe_material: str,
        set_target: bool,
        target_vol: float,
        target_unit: str,
        infuse_rate: float,
        infuse_unit: str,
    ):
        self.configurations.append(
            {
                "input_size": input_size,
                "syringe_material": syringe_material,
                "set_target": set_target,
                "target_vol": target_vol,
                "target_unit": target_unit,
                "infuse_rate": infuse_rate,
                "infuse_unit": infuse_unit,
            }
        )
        yield from bps.mv(
            self.read_infuse_rate,
            infuse_rate,
            self.read_infuse_rate_unit,
            infuse_unit,
        )

    def infuse_pump2(self):
        self.start_count += 1
        if self.fail_start:
            raise RuntimeError(f"failed to start {self.name}")
        yield from bps.mv(self.status, "Infusing")

    def stop_pump2(self):
        self.stop_count += 1
        if self.fail_stop_call == self.stop_count:
            raise RuntimeError(f"failed to stop {self.name}")
        yield from bps.mv(self.status, "Stopped")


class FakeAreaDetector(Device):
    class Cam(Device):
        acquire_time = Cpt(Signal, value=0.1)

    cam = Cpt(Cam, "")
    images_per_set = Cpt(Signal, value=1)
    image = Cpt(Signal, value=1.0)

    def __init__(self, *args: Any, fail_trigger: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fail_trigger = fail_trigger

    def trigger(self) -> DeviceStatus:
        status = DeviceStatus(self)
        if self.fail_trigger:
            status.set_exception(RuntimeError("X-ray trigger failed"))
        else:
            status.set_finished()
        return status


class _TiledStream:
    def __init__(self, data: Mapping[str, Any], *, failures: int = 0) -> None:
        self.data = {
            name: SimpleNamespace(values=np.asarray(value))
            for name, value in data.items()
        }
        self.failures = failures
        self.read_count = 0

    def read(self) -> dict[str, SimpleNamespace]:
        self.read_count += 1
        if self.read_count <= self.failures:
            raise OSError("stream is not ready")
        return self.data


class _TiledRun:
    def __init__(
        self,
        streams: Mapping[str, _TiledStream],
        *,
        metadata: Mapping[str, Any] | None = None,
        use_good_bad: bool = False,
    ) -> None:
        self.streams = dict(streams)
        start = dict(metadata or {})
        start.setdefault("use_good_bad", use_good_bad)
        self.metadata = {"start": start}

    def __getitem__(self, name: str) -> _TiledStream:
        return self.streams[name]


class _TiledCatalog:
    def __init__(
        self,
        runs: Mapping[str, _TiledRun],
        *,
        search_failures: int = 0,
    ) -> None:
        self.runs = dict(runs)
        self.search_failures = search_failures
        self.search_count = 0

    def __getitem__(self, uid: Hashable) -> _TiledRun:
        return self.runs[str(uid)]

    def search(self, query: Any) -> _TiledCatalog:
        self.search_count += 1
        if self.search_count <= self.search_failures:
            raise OSError("catalog is not ready")
        return self

    def keys(self) -> _TiledCatalog:
        return self

    def last(self) -> str:
        return next(reversed(self.runs))


def _run_from_documents(
    documents: Sequence[tuple[str, Mapping[str, Any]]], uid: str
) -> _TiledRun:
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
        stream_name: _TiledStream(
            {
                field: np.asarray([event[field] for event in stream_events])
                for field in stream_events[0]
            }
        )
        for stream_name, stream_events in events.items()
        if stream_events
    }
    return _TiledRun(streams, metadata=start)


@pytest.fixture
def tiled_fakes() -> SimpleNamespace:
    return SimpleNamespace(
        Stream=_TiledStream,
        Run=_TiledRun,
        Catalog=_TiledCatalog,
        run_from_documents=_run_from_documents,
    )


@pytest.fixture
def wavelength() -> np.ndarray:
    return np.arange(200.0, 951.0)


@pytest.fixture
def good_spectrum(wavelength: np.ndarray) -> np.ndarray:
    return 5000 * np.exp(-((wavelength - 660) ** 2) / (2 * 20**2))


@pytest.fixture
def bad_spectrum(wavelength: np.ndarray) -> np.ndarray:
    return 20 * np.exp(-((wavelength - 630) ** 2) / (2 * 20**2))


@pytest.fixture
def fake_qepro(
    wavelength: np.ndarray,
    good_spectrum: np.ndarray,
) -> FakeQEPro:
    device = FakeQEPro(name="QEPro", spectra=[good_spectrum] * 20)
    device.x_axis.put(wavelength)
    device.output.put(good_spectrum)
    return device


@pytest.fixture
def fake_pumps() -> dict[str, FakePump]:
    names = ("dds2_p1", "dds2_p2", "dds3_p1", "dds1_p1", "ultra2", "ultra1")
    return {name: FakePump(name=name) for name in names}


@pytest.fixture
def fake_area_detector() -> FakeAreaDetector:
    return FakeAreaDetector(name="xray_detector")


@pytest.fixture
def optical_signals() -> tuple[Signal, Signal, Signal]:
    return (
        Signal(name="led", value="Low"),
        Signal(name="uv_shutter", value="Low"),
        Signal(name="fast_shutter", value=20),
    )


@pytest.fixture
def documents(RE: RunEngine) -> list[tuple[str, dict[str, Any]]]:
    collected: list[tuple[str, dict[str, Any]]] = []
    RE.subscribe(lambda name, doc: collected.append((name, doc)))
    return collected


@pytest.fixture
def reference_config_factory(
    tmp_path: Path,
) -> Callable[..., Path]:

    def factory(
        phases: Sequence[tuple[str, bool]] = (("Target", False),),
        *,
        include_cif: bool = True,
    ) -> Path:
        radial = np.linspace(1.0, 25.0, 241)
        payload: list[dict[str, Any]] = []
        for index, (name, minimize) in enumerate(phases):
            gr_path = tmp_path / f"{name}.gr"
            np.savetxt(gr_path, np.column_stack((radial, np.sin(radial + index))))
            phase: dict[str, Any] = {
                "name": name,
                "gr_path": gr_path.name,
                "minimize": minimize,
            }
            if include_cif:
                cif_path = tmp_path / f"{name}.cif"
                cif_path.write_text(
                    "data_test\n"
                    "_symmetry_space_group_name_H-M 'P 1'\n"
                    "_cell_length_a 6\n_cell_length_b 6\n_cell_length_c 6\n"
                    "_cell_angle_alpha 90\n_cell_angle_beta 90\n"
                    "_cell_angle_gamma 90\n"
                    "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
                    "_atom_site_fract_x\n_atom_site_fract_y\n"
                    "_atom_site_fract_z\nCs1 Cs 0 0 0\n"
                )
                phase["cif_path"] = cif_path.name
            payload.append(phase)
        config_path = tmp_path / "references.json"
        config_path.write_text(json.dumps({"schema_version": 1, "phases": payload}))
        return config_path

    return factory


@pytest.fixture
def plan_context_factory(
    fake_qepro: FakeQEPro,
    fake_pumps: Mapping[str, FakePump],
    optical_signals: tuple[Signal, Signal, Signal],
    fake_area_detector: FakeAreaDetector,
) -> Callable[..., XrayUvvisPlanContext]:
    led, uv_shutter, fast_shutter = optical_signals
    default_quality = QualityPolicy(
        enabled=False, absorbance_shots=1, fluorescence_shots=1
    )

    def identity_wrapper(plan: Any, no_dark: bool):
        return plan

    def factory(
        *,
        sources: tuple[FlowSource, ...] | None = None,
        dilutions: tuple[DilutionStage, ...] = (),
        wash_cycles: tuple[WashCycle, ...] = (),
        quality: QualityPolicy | None = None,
        xray: XraySettings | None = None,
        mixer_lengths_cm: tuple[float, ...] = (0.0,),
        residence_time_ratio: float = 0.0,
        wrap_xray_run: Any = None,
    ) -> XrayUvvisPlanContext:
        configured_sources = sources
        if configured_sources is None:
            configured_sources = tuple(
                FlowSource(
                    dof=f"infusion_rate_{label}",
                    pump=fake_pumps[pump],
                    precursor=precursor,
                    sample_label=label,
                )
                for label, pump, precursor in (
                    ("CsPb", "dds2_p1", "CsPbOA"),
                    ("Br", "dds2_p2", "TOABr"),
                    ("I2", "dds3_p1", "ZnI2"),
                )
            )
        return XrayUvvisPlanContext(
            qepro=fake_qepro,
            led=led,
            uv_shutter=uv_shutter,
            fast_shutter=fast_shutter,
            xray_detector=fake_area_detector,
            wrap_xray_run=wrap_xray_run or identity_wrapper,
            sources=configured_sources,
            dilutions=dilutions,
            wash_cycles=wash_cycles,
            mixer_lengths_cm=mixer_lengths_cm,
            residence_time_ratio=residence_time_ratio,
            quality=quality or default_quality,
            xray=xray or XraySettings(),
        )

    return factory
