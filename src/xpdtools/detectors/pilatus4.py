"""Pilatus4 detector interface for XPD beamline at NSLS-II."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated as A

from ophyd_async.core import (
    DetectorTriggerLogic,
    DeviceMock,
    SignalDict,
    SignalR,
    SignalRW,
    StandardReadable,
    StrictEnum,
    SubsetEnum,
    callback_on_mock_put,
    default_mock_class,
    derived_signal_r,
    set_mock_put_proceeds,
    set_mock_value,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)
from ophyd_async.epics.adcore import (
    ADAcquireLogic,
    ADBaseDataType,
    ADBaseIO,
    ADWriterFactory,
    AreaDetector,
    NDFileHDF5IO,
    NDPluginBaseIO,
    prepare_exposures,
    trigger_info_from_num_images,
)
from ophyd_async.epics.core import (
    PvSuffix,
)


class Pilatus4TriggerMode(StrictEnum):
    """Trigger modes for the Pilatus4 detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#implementation-of-standard-driver-parameters
    """

    INTERNAL_SERIES = "Internal Series"
    INTERNAL_ENABLE = "Internal Enable"
    EXTERNAL_SERIES = "External Series"
    EXTERNAL_ENABLE = "External Enable"
    CONTINUOUS = "Continuous"
    EXTERNAL_GATE = "External Gate"


class Pilatus4ExtGateMode(StrictEnum):
    """External gate modes for the Pilatus4 detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#trigger-setup
    """

    PUMP_AND_PROBE = "Pump & Probe"
    HDR = "HDR"


class Pilatus4ROIMode(StrictEnum):
    """ROI modes for the Pilatus4 detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#readout-setup
    """

    DISABLED = "Disable"
    FOUR_M = "4M"


# TODO - add extra options in eiger2 and revert to StrictEnum
class Pilatus4CompressionAlgo(SubsetEnum):
    """Compression algorithms for the Pilatus4 detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#readout-setup
    """

    LZ4 = "LZ4"
    BSLZ4 = "BS LZ4"


class Pilatus4DataSource(StrictEnum):
    """Data sources for the Pilatus4 detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#readout-setup
    """

    NONE = "None"
    FILE_WRITER = "FileWriter"
    STREAM = "Stream"


class Pilatus4HDF5Format(StrictEnum):
    """HDF5 formats for the Pilatus4 detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#filewriter-interface
    """

    LEGACY = "Legacy"
    V2024_2 = "v2024.2"


class SimplonStreamVersion(StrictEnum):
    """Stream versions for the Pilatus4 detector.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#stream-interface
    """

    STREAM1 = "Stream"
    STREAM2 = "Stream2"


class Pilatus4DriverIO(StandardReadable, ADBaseIO):
    """Pilatus4 driver interface.

    See https://areadetector.github.io/areaDetector/ADEiger/eiger.html#implementation-of-standard-driver-parameters
    """

    # Standard Driver Parameters
    trigger_mode: A[SignalRW[Pilatus4TriggerMode], PvSuffix.rbv("TriggerMode")]
    num_images: A[SignalRW[int], PvSuffix.rbv("NumImages")]
    num_images_counter: A[SignalR[int], PvSuffix("NumImagesCounter_RBV")]
    num_exposures: A[SignalRW[int], PvSuffix.rbv("NumExposures")]
    acquire_time: A[SignalRW[float], PvSuffix.rbv("AcquireTime"), Format.CONFIG_SIGNAL]
    acquire_period: A[
        SignalRW[float], PvSuffix.rbv("AcquirePeriod"), Format.CONFIG_SIGNAL
    ]
    temperature_actual: A[SignalR[float], PvSuffix("TemperatureActual")]
    max_size_x: A[SignalR[int], PvSuffix("MaxSizeX_RBV")]
    max_size_y: A[SignalR[int], PvSuffix("MaxSizeY_RBV")]
    array_size_x: A[SignalR[int], PvSuffix("ArraySizeX_RBV")]
    array_size_y: A[SignalR[int], PvSuffix("ArraySizeY_RBV")]
    manufacturer: A[SignalR[str], PvSuffix("Manufacturer_RBV"), Format.CONFIG_SIGNAL]
    model: A[SignalR[str], PvSuffix("Model_RBV"), Format.CONFIG_SIGNAL]
    serial_number: A[SignalR[str], PvSuffix("SerialNumber_RBV"), Format.CONFIG_SIGNAL]
    firmware_version: A[
        SignalR[str], PvSuffix("FirmwareVersion_RBV"), Format.CONFIG_SIGNAL
    ]
    sdk_version: A[SignalR[str], PvSuffix("SDKVersion_RBV"), Format.CONFIG_SIGNAL]
    driver_version: A[SignalR[str], PvSuffix("DriverVersion_RBV"), Format.CONFIG_SIGNAL]

    # Detector Information
    description: A[SignalR[str], PvSuffix("Description_RBV"), Format.CONFIG_SIGNAL]
    x_pixel_size: A[SignalR[float], PvSuffix("XPixelSize_RBV"), Format.CONFIG_SIGNAL]
    y_pixel_size: A[SignalR[float], PvSuffix("YPixelSize_RBV"), Format.CONFIG_SIGNAL]
    sensor_material: A[
        SignalR[str], PvSuffix("SensorMaterial_RBV"), Format.CONFIG_SIGNAL
    ]
    sensor_thickness: A[
        SignalR[float], PvSuffix("SensorThickness_RBV"), Format.CONFIG_SIGNAL
    ]
    dead_time: A[SignalR[float], PvSuffix("DeadTime_RBV"), Format.CONFIG_SIGNAL]

    # Detector Status
    state: A[SignalR[str], PvSuffix("State_RBV")]
    error: A[SignalR[str], PvSuffix("Error_RBV")]
    temp0: A[SignalR[float], PvSuffix("Temp0_RBV")]
    humid0: A[SignalR[float], PvSuffix("Humid0_RBV")]

    # Acquisition Setup
    photon_energy: A[
        SignalRW[float], PvSuffix.rbv("PhotonEnergy"), Format.CONFIG_SIGNAL
    ]

    # Trigger Setup
    trigger_: A[SignalRW[float], PvSuffix("Trigger")]
    # trigger_exposure: A[SignalRW[float], PvSuffix.rbv("TriggerExposure")]
    num_triggers: A[SignalRW[int], PvSuffix.rbv("NumTriggers")]
    manual_trigger: A[SignalRW[bool], PvSuffix.rbv("ManualTrigger")]

    # Readout Setup
    roi_mode: A[SignalRW[Pilatus4ROIMode], PvSuffix.rbv("ROIMode")]
    flatfield_applied: A[SignalRW[bool], PvSuffix.rbv("FlatfieldApplied")]
    countrate_corr_applied: A[SignalRW[bool], PvSuffix.rbv("CountrateCorrApplied")]
    pixel_mask_applied: A[SignalRW[bool], PvSuffix.rbv("PixelMaskApplied")]
    auto_summation: A[SignalRW[bool], PvSuffix.rbv("AutoSummation")]
    compression_algo: A[
        SignalRW[Pilatus4CompressionAlgo], PvSuffix.rbv("CompressionAlgo")
    ]
    data_source: A[SignalRW[Pilatus4DataSource], PvSuffix.rbv("DataSource")]

    # Acquisition Status
    armed: A[SignalR[bool], PvSuffix("Armed")]
    bit_depth_image: A[SignalR[int], PvSuffix("BitDepthImage_RBV")]
    count_cutoff: A[SignalR[float], PvSuffix("CountCutoff_RBV")]

    # Stream Interface
    stream_enable: A[SignalRW[bool], PvSuffix.rbv("StreamEnable")]
    stream_state: A[SignalR[str], PvSuffix("StreamState_RBV")]
    stream_decompress: A[SignalRW[bool], PvSuffix.rbv("StreamDecompress")]
    stream_dropped: A[SignalR[int], PvSuffix("StreamDropped_RBV")]

    # Monitor Interface
    monitor_enable: A[SignalRW[bool], PvSuffix.rbv("MonitorEnable")]
    monitor_state: A[SignalR[str], PvSuffix("MonitorState_RBV")]
    monitor_timeout: A[SignalRW[float], PvSuffix.rbv("MonitorTimeout")]

    # Acquisition Metadata
    beam_x: A[SignalRW[float], PvSuffix.rbv("BeamX"), Format.CONFIG_SIGNAL]
    beam_y: A[SignalRW[float], PvSuffix.rbv("BeamY"), Format.CONFIG_SIGNAL]
    det_dist: A[SignalRW[float], PvSuffix.rbv("DetDist"), Format.CONFIG_SIGNAL]
    wavelength: A[SignalRW[float], PvSuffix.rbv("Wavelength"), Format.CONFIG_SIGNAL]

    # Detector Metadata
    chi_start: A[SignalRW[float], PvSuffix.rbv("ChiStart"), Format.CONFIG_SIGNAL]
    chi_incr: A[SignalRW[float], PvSuffix.rbv("ChiIncr"), Format.CONFIG_SIGNAL]
    kappa_start: A[SignalRW[float], PvSuffix.rbv("KappaStart"), Format.CONFIG_SIGNAL]
    kappa_incr: A[SignalRW[float], PvSuffix.rbv("KappaIncr"), Format.CONFIG_SIGNAL]
    omega_start: A[SignalRW[float], PvSuffix.rbv("OmegaStart"), Format.CONFIG_SIGNAL]
    omega_incr: A[SignalRW[float], PvSuffix.rbv("OmegaIncr"), Format.CONFIG_SIGNAL]
    phi_start: A[SignalRW[float], PvSuffix.rbv("PhiStart"), Format.CONFIG_SIGNAL]
    phi_incr: A[SignalRW[float], PvSuffix.rbv("PhiIncr"), Format.CONFIG_SIGNAL]
    two_theta_start: A[
        SignalRW[float], PvSuffix.rbv("TwoThetaStart"), Format.CONFIG_SIGNAL
    ]
    two_theta_incr: A[
        SignalRW[float], PvSuffix.rbv("TwoThetaIncr"), Format.CONFIG_SIGNAL
    ]

    # Minimum change allowed
    wavelength_eps: A[
        SignalRW[float], PvSuffix.rbv("WavelengthEps"), Format.CONFIG_SIGNAL
    ]
    energy_eps: A[SignalRW[float], PvSuffix.rbv("EnergyEps"), Format.CONFIG_SIGNAL]

    # FileWriter Interface
    # fw_enable: A[SignalRW[bool], PvSuffix.rbv("FWEnable")]
    # fw_state: A[SignalR[str], PvSuffix("FWState_RBV")]
    # fw_compression: A[SignalRW[bool], PvSuffix.rbv("FWCompression")]
    # fw_name_pattern: A[SignalRW[str], PvSuffix.rbv("FWNamePattern")]
    # sequence_id: A[SignalR[int], PvSuffix("SequenceId")]
    # save_files: A[SignalRW[bool], PvSuffix.rbv("SaveFiles")]
    # file_owner: A[SignalRW[str], PvSuffix.rbv("FileOwner")]
    # file_owner_grp: A[SignalRW[str], PvSuffix.rbv("FileOwnerGrp")]
    # file_perms: A[SignalRW[float], PvSuffix.rbv("FilePerms")]
    # fw_free: A[SignalR[float], PvSuffix("FWFree_RBV")]
    # fw_auto_remove: A[SignalRW[bool], PvSuffix.rbv("FWAutoRemove")]
    # fw_nimgs_per_file: A[SignalRW[int], PvSuffix.rbv("FWNImagesPerFile")]

    # Detector Status
    hv_reset_time: A[SignalRW[float], PvSuffix.rbv("HVResetTime")]
    hv_reset: A[SignalRW[bool], PvSuffix("HVReset", "HVReset")]
    hv_state: A[SignalR[str], PvSuffix("HVState_RBV")]

    # Acquisition Setup
    threshold: A[SignalRW[float], PvSuffix.rbv("ThresholdEnergy")]
    threshold1_enable: A[SignalRW[bool], PvSuffix.rbv("Threshold1Enable")]
    threshold2: A[SignalRW[float], PvSuffix.rbv("Threshold2Energy")]
    threshold2_enable: A[SignalRW[bool], PvSuffix.rbv("Threshold2Enable")]
    threshold_diff_enable: A[SignalRW[bool], PvSuffix.rbv("ThresholdDiffEnable")]
    counting_mode: A[SignalRW[str], PvSuffix.rbv("CountingMode"), Format.CONFIG_SIGNAL]

    # Trigger Setup
    ext_gate_mode: A[SignalRW[str], PvSuffix.rbv("ExtGateMode")]
    trigger_start_delay: A[SignalRW[float], PvSuffix.rbv("TriggerStartDelay")]

    # Readout Setup
    signed_data: A[SignalRW[bool], PvSuffix.rbv("SignedData"), Format.CONFIG_SIGNAL]

    # Stream Interface
    stream_version: A[
        SignalRW[SimplonStreamVersion],
        PvSuffix.rbv("StreamVersion"),
        Format.CONFIG_SIGNAL,
    ]

    # FileWriter Interface
    # fw_hdf5_format: A[SignalRW[EigerHDF5Format], PvSuffix.rbv("FWHDF5Format")]

    def __init__(self, prefix: str, name: str = ""):
        super().__init__(prefix, name=name)
        self.data_type = derived_signal_r(
            self.data_type_from_bit_depth_img,
            bit_depth=self.bit_depth_image,
            signed=self.signed_data,
        )

    def data_type_from_bit_depth_img(
        self, bit_depth: int, signed: bool
    ) -> ADBaseDataType:
        if signed:
            if bit_depth == 8:
                return ADBaseDataType.INT8
            elif bit_depth == 16:
                return ADBaseDataType.INT16
            elif bit_depth == 32:
                return ADBaseDataType.INT32
        else:
            if bit_depth == 8:
                return ADBaseDataType.UINT8
            elif bit_depth == 16:
                return ADBaseDataType.UINT16
            elif bit_depth == 32:
                return ADBaseDataType.UINT32
        raise ValueError(
            "Failed to infer data type! "
            f"Invalid bit depth and signedness {bit_depth}, {signed}"
        )


@dataclass
class Pilatus4TriggerLogic(DetectorTriggerLogic):
    """Trigger logic for Pilatus4 detectors."""

    driver: Pilatus4DriverIO

    def get_deadtime(self, config_values: SignalDict) -> float:
        return 0.001

    async def configure_stream2(self):
        """Configure the detector for stream2 acquisition mode."""
        coros = [
            self.driver.data_source.set(Pilatus4DataSource.STREAM),
            self.driver.stream_enable.set(True),
            self.driver.stream_version.set(SimplonStreamVersion.STREAM2),
            self.driver.compression_algo.set(Pilatus4CompressionAlgo.BSLZ4),
            self.driver.stream_decompress.set(False),
        ]
        await asyncio.gather(*coros)

    async def prepare_internal(self, num: int, livetime: float, deadtime: float):
        """Prepare the detector for internal series acquisition mode."""
        await asyncio.gather(
            self.configure_stream2(),
            self.driver.trigger_mode.set(Pilatus4TriggerMode.INTERNAL_SERIES),
            self.driver.num_triggers.set(1),
            prepare_exposures(self.driver, num, livetime, deadtime),
        )

    async def prepare_edge(self, num: int, livetime: float):
        """Prepare the detector for external series acquisition mode."""
        await asyncio.gather(
            self.configure_stream2(),
            self.driver.trigger_mode.set(Pilatus4TriggerMode.EXTERNAL_SERIES),
            self.driver.num_triggers.set(num),
            prepare_exposures(self.driver, 1, livetime, 0.001),
        )

    async def default_trigger_info(self):
        """Get default trigger info based on the number of images to acquire."""
        return await trigger_info_from_num_images(self.driver)


class Pilatus4DetectorMock(DeviceMock["Pilatus4Detector"]):
    """Mock behaviour that simulates Pilatus4 internal series acquisition."""

    async def connect(self, device: "Pilatus4Detector") -> None:
        """Mock signals to simulate Pilatus4 detector acquisition."""
        # Set default array sizes on driver
        set_mock_value(device.driver.array_size_x, 1280)
        set_mock_value(device.driver.array_size_y, 720)
        set_mock_value(device.driver.bit_depth_image, 32)
        set_mock_value(device.driver.signed_data, False)

        # Set default array sizes on HDF plugin if present
        try:
            hdf = device.get_plugin("hdf", NDFileHDF5IO)
            set_mock_value(hdf.array_size0, 720)
            set_mock_value(hdf.array_size1, 1280)
            set_mock_value(hdf.file_path_exists, True)
        except (AttributeError, TypeError):
            hdf = None

        # Set default signal values
        set_mock_value(device.driver.acquire_time, 1.0)
        set_mock_value(device.driver.acquire_period, 1.0001)
        set_mock_value(device.driver.num_images, 1)
        set_mock_value(device.driver.num_images_counter, 0)
        set_mock_value(device.driver.trigger_mode, Pilatus4TriggerMode.INTERNAL_SERIES)

        # Auto-adjust acquire_period when acquire_time exceeds it
        async def _on_acquire_time_write(value: float) -> None:
            current_period = await device.driver.acquire_period.get_value()
            if current_period < value:
                set_mock_value(device.driver.acquire_period, value + 0.0001)

        callback_on_mock_put(device.driver.acquire_time, _on_acquire_time_write)

        # Simulate acquisition on acquire=True
        async def _do_acquisition():
            trigger_mode = await device.driver.trigger_mode.get_value()
            if trigger_mode != Pilatus4TriggerMode.INTERNAL_SERIES:
                set_mock_put_proceeds(device.driver.acquire, True)
                return

            num_images = await device.driver.num_images.get_value()
            acquire_time = await device.driver.acquire_time.get_value()
            acquire_period = await device.driver.acquire_period.get_value()

            for i in range(num_images):
                await asyncio.sleep(acquire_time)
                set_mock_value(device.driver.num_images_counter, i + 1)
                if hdf is not None:
                    num_captured = await hdf.num_captured.get_value()
                    set_mock_value(hdf.num_captured, num_captured + 1)
                if i < num_images - 1:
                    await asyncio.sleep(acquire_period - acquire_time)

            set_mock_value(device.driver.acquire, False)
            set_mock_put_proceeds(device.driver.acquire, True)

        def _on_acquire_write(value: bool) -> None:
            if value:
                set_mock_put_proceeds(device.driver.acquire, False)
                asyncio.ensure_future(_do_acquisition())

        callback_on_mock_put(device.driver.acquire, _on_acquire_write)


@default_mock_class(Pilatus4DetectorMock)
class Pilatus4Detector(AreaDetector[Pilatus4DriverIO]):
    """Create an Pilatus4 AreaDetector instance.

    Parameters
    ----------
    prefix : str
        EPICS PV prefix for the detector
    writer_factories : ADWriterFactory
        Factories for file writer plugins and their data logics
    driver_suffix : str, optional
        Suffix for the driver PV, by default "cam1:"
    plugins : dict[str, NDPluginBaseIO], optional
        Additional areaDetector plugins to include, by default None
    config_sigs : Sequence[SignalR], optional
        Additional signals to include in configuration, by default ()
    name : str, optional
        Name for the detector device, by default ""
    """

    def __init__(
        self,
        prefix: str,
        *writer_factories: ADWriterFactory,
        driver_suffix="cam1:",
        plugins: dict[str, NDPluginBaseIO] | None = None,
        config_sigs: Sequence[SignalR] = (),
        name: str = "",
    ) -> None:
        driver = Pilatus4DriverIO(prefix + driver_suffix)
        super().__init__(
            driver,
            prefix,
            *writer_factories,
            acquire_logic=ADAcquireLogic(driver),
            trigger_logic=Pilatus4TriggerLogic(driver),
            plugins=plugins,
            config_sigs=config_sigs,
            name=name,
        )
