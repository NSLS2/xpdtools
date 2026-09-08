from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from bluesky import plan_stubs as bps
from bluesky.run_engine import RunEngine
from ophyd import Component as Cpt
from ophyd import Device, Signal
from ophyd.status import DeviceStatus


class FakeQEPro(Device):
    """Classic-Ophyd QEPro substitute with deterministic spectrum playback."""

    x_axis = Cpt(Signal, value=np.linspace(200.0, 950.0, 751))
    output = Cpt(Signal, value=np.zeros(751))
    sample = Cpt(Signal, value=np.zeros(751))
    dark = Cpt(Signal, value=np.zeros(751))
    reference = Cpt(Signal, value=np.ones(751))
    spectrum_type = Cpt(Signal, value="Corrected Sample")
    correction = Cpt(Signal, value="Dark")
    integration_time = Cpt(Signal, value=100)
    num_spectra = Cpt(Signal, value=1)
    buff_capacity = Cpt(Signal, value=1)

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
    """Classic-Ophyd pump substitute exposing the production plan methods."""

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
    """Minimal staged detector accepted by the X-ray plan path."""

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


@pytest.fixture
def wavelength() -> np.ndarray:
    """A one-nanometer UV-Vis wavelength grid."""
    return np.arange(200.0, 951.0)


@pytest.fixture
def good_spectrum(wavelength: np.ndarray) -> np.ndarray:
    """A deterministic 660 nm Gaussian PL spectrum."""
    return 5000 * np.exp(-((wavelength - 660) ** 2) / (2 * 20**2))


@pytest.fixture
def bad_spectrum(wavelength: np.ndarray) -> np.ndarray:
    """A deterministic spectrum below the acquisition quality threshold."""
    return 20 * np.exp(-((wavelength - 630) ** 2) / (2 * 20**2))


@pytest.fixture
def fake_qepro(
    wavelength: np.ndarray,
    good_spectrum: np.ndarray,
) -> FakeQEPro:
    """A QEPro initialized with a good deterministic spectrum."""
    device = FakeQEPro(name="QEPro", spectra=[good_spectrum] * 20)
    device.x_axis.put(wavelength)
    device.output.put(good_spectrum)
    return device


@pytest.fixture
def fake_pumps() -> dict[str, FakePump]:
    """Pumps used by standard flow, dilution, and wash configurations."""
    names = ("dds2_p1", "dds2_p2", "dds3_p1", "dds1_p1", "ultra2", "ultra1")
    return {name: FakePump(name=name) for name in names}


@pytest.fixture
def fake_area_detector() -> FakeAreaDetector:
    """A classic-Ophyd area detector for simulated X-ray acquisition."""
    return FakeAreaDetector(name="xray_detector")


@pytest.fixture
def optical_signals() -> tuple[Signal, Signal, Signal]:
    """LED, UV shutter, and fast-shutter signals."""
    return (
        Signal(name="led", value="Low"),
        Signal(name="uv_shutter", value="Low"),
        Signal(name="fast_shutter", value=20),
    )


@pytest.fixture
def documents(RE: RunEngine) -> list[tuple[str, dict[str, Any]]]:
    """Collect documents emitted by the shared RunEngine fixture."""
    collected: list[tuple[str, dict[str, Any]]] = []
    RE.subscribe(lambda name, doc: collected.append((name, doc)))
    return collected


@pytest.fixture
def reference_config_factory(
    tmp_path: Path,
) -> Callable[..., Path]:
    """Write external version-1 reference JSON and its referenced files."""

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
