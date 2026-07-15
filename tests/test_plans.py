"""Tests for the single_axis_flyscan plan."""

import asyncio
from pathlib import Path

import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    StaticPathProvider,
    UUIDFilenameProvider,
    callback_on_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.core._mock_signal_utils import (  # noqa: PLC2701
    _get_mock_signal_backend,
)
from ophyd_async.epics.adcore import ADWriterFactory, NDFileHDF5IO
from ophyd_async.fastcs.panda import DatasetTable, HDFPanda, PandaHdf5DatasetType

from xpdtools.detectors.pilatus4 import Pilatus4Detector
from xpdtools.motors import RotationMotor
from xpdtools.plans import single_axis_flyscan

NUM_IMAGES = 11
SCAN_START = 0.0
SCAN_STOP = 180.0


def _setup_flyscan_coordination(panda, motor, pilatus, num_images: int):
    """Set up cross-device mock coordination for flyscan testing.

    Wraps the motor setpoint callback to detect the fly move (2nd setpoint write)
    and coordinates PandA pcomp/pcap signals and detector num_captured.
    All async tasks are launched from within callbacks that fire in the RE's
    event loop, ensuring correct loop context.
    """
    # 1. pcap.arm -> pcap.active
    callback_on_mock_put(panda.pcap.arm, lambda v: set_mock_value(panda.pcap.active, v))

    # 2. Set PandA data prerequisites
    set_mock_value(panda.data.directory_exists, 1)
    set_mock_value(
        panda.data.datasets,
        DatasetTable(name=["enc1"], dtype=[PandaHdf5DatasetType.FLOAT_64]),
    )
    set_mock_value(panda.pcap.active, False)
    set_mock_value(panda.pcomp[1].active, False)
    set_mock_value(panda.data.num_captured, 0)

    # 3. Wrap motor setpoint callback to add fly coordination
    backend = _get_mock_signal_backend(motor.user_setpoint)
    original_motor_cb = backend._mock_put_callback
    move_count = [0]

    hdf = pilatus.get_plugin("hdf", NDFileHDF5IO)

    def _on_motor_setpoint(value):
        move_count[0] += 1
        if move_count[0] == 2:
            # Fly move started - set pcomp active and simulate captures
            set_mock_value(panda.pcomp[1].active, True)

            async def _simulate_captures():
                for i in range(num_images):
                    await asyncio.sleep(0.005)
                    set_mock_value(panda.data.num_captured, i + 1)
                    set_mock_value(hdf.num_captured, i + 1)
                set_mock_value(panda.pcomp[1].active, False)

            asyncio.ensure_future(_simulate_captures())

        # Call original motor mock callback (VelocityRespectingMotorMock)
        if original_motor_cb:
            return original_motor_cb(value)

    callback_on_mock_put(motor.user_setpoint, _on_motor_setpoint)


@pytest.fixture
async def devices(tmp_path: Path):
    path_provider = StaticPathProvider(
        UUIDFilenameProvider(), tmp_path, create_dir_depth=-2
    )
    async with init_devices(mock=True):
        panda = HDFPanda("PANDA:", path_provider)
        motor = RotationMotor("MOT:", encoder_pos_at_zero=0, name="rot_motor")
        pilatus1 = Pilatus4Detector(
            "DET1:",
            ADWriterFactory.hdf(path_provider),
            name="pilatus1",
        )
    # Set mock values for fast test execution
    set_mock_value(motor.encoder_resolution, 0.1)
    set_mock_value(motor.max_velocity, 1000.0)
    set_mock_value(motor.acceleration_time, 0.0)
    set_mock_value(pilatus1.driver.acquire_time, 0.001)
    set_mock_value(pilatus1.driver.acquire_period, 0.002)

    # Set up cross-device mock coordination
    _setup_flyscan_coordination(panda, motor, pilatus1, NUM_IMAGES)

    return pilatus1, panda, motor


@pytest.mark.parametrize(
    "time_based",
    [True, False],
)
def test_single_axis_flyscan(RE: RunEngine, devices, time_based: bool):
    """Test that single_axis_flyscan runs to completion and emits correct documents."""
    pilatus1, panda, motor = devices

    docs: dict[str, list] = {}

    def collect_doc(name, doc):
        docs.setdefault(name, []).append(doc)

    RE.subscribe(collect_doc)

    RE(
        single_axis_flyscan(
            detectors=[pilatus1],
            panda=panda,
            motor=motor,
            num_images=NUM_IMAGES,
            start=SCAN_START,
            stop=SCAN_STOP,
            time_based=time_based,
        )
    )

    # --- Verify core document structure ---
    assert "start" in docs
    assert "stop" in docs
    assert len(docs["start"]) == 1
    assert len(docs["stop"]) == 1

    start_doc = docs["start"][0]
    assert start_doc["plan_name"] == "single_axis_flyscan"
    assert start_doc["num_points"] == NUM_IMAGES
    assert start_doc["detectors"] == ["pilatus1"]

    stop_doc = docs["stop"][0]
    assert stop_doc["exit_status"] == "success"
    assert stop_doc["run_start"] == start_doc["uid"]

    # --- Verify descriptor ---
    assert "descriptor" in docs
    descriptors = docs["descriptor"]
    assert len(descriptors) >= 1
    # The "tomo" stream descriptor should contain data keys for pilatus1 and enc1
    tomo_descriptor = next(d for d in descriptors if d.get("name") == "tomo")
    assert "pilatus1" in tomo_descriptor["data_keys"]
    assert "enc1" in tomo_descriptor["data_keys"]

    # Pilatus data key should be external stream with correct shape
    pilatus_dk = tomo_descriptor["data_keys"]["pilatus1"]
    assert pilatus_dk["external"] == "STREAM:"
    assert pilatus_dk["shape"] == [1, 720, 1280]

    # PandA enc1 data key should be external stream (scalar)
    enc1_dk = tomo_descriptor["data_keys"]["enc1"]
    assert enc1_dk["external"] == "STREAM:"

    # --- Verify stream_resource documents ---
    assert "stream_resource" in docs
    stream_resources = docs["stream_resource"]

    # Should have at least 2: one for pilatus HDF, one for panda HDF
    pilatus_sr = next(sr for sr in stream_resources if sr["data_key"] == "pilatus1")
    panda_sr = next(sr for sr in stream_resources if sr["data_key"] == "enc1")

    # Pilatus stream resource checks
    assert pilatus_sr["mimetype"] == "application/x-hdf5"
    assert pilatus_sr["uri"].endswith(".h5")
    assert pilatus_sr["parameters"]["dataset"] == "/entry/data/data"
    assert "chunk_shape" in pilatus_sr["parameters"]

    # PandA stream resource checks
    assert panda_sr["mimetype"] == "application/x-hdf5"
    assert panda_sr["uri"].endswith(".h5")
    assert panda_sr["parameters"]["dataset"] == "/enc1"
    assert panda_sr["parameters"]["chunk_shape"] == (1024,)

    # --- Verify stream_datum documents ---
    assert "stream_datum" in docs
    stream_datums = docs["stream_datum"]

    # Filter datums by stream_resource uid
    pilatus_datums = [
        sd for sd in stream_datums if sd["stream_resource"] == pilatus_sr["uid"]
    ]
    panda_datums = [
        sd for sd in stream_datums if sd["stream_resource"] == panda_sr["uid"]
    ]

    assert len(pilatus_datums) >= 1
    assert len(panda_datums) >= 1

    # Verify indices cover all frames (start=0, stop=NUM_IMAGES)
    pilatus_last = pilatus_datums[-1]
    assert pilatus_last["indices"]["stop"] == NUM_IMAGES

    panda_last = panda_datums[-1]
    assert panda_last["indices"]["stop"] == NUM_IMAGES

    # Verify seq_nums are present and valid
    for datum in stream_datums:
        assert "seq_nums" in datum
        assert datum["seq_nums"]["start"] >= 0
        assert datum["seq_nums"]["stop"] > datum["seq_nums"]["start"]
