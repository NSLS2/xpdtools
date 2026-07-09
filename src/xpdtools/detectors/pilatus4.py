"""Pilatus4 detector interface for XPD beamline at NSLS-II."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated as A

from ophyd_async.core import (
    DetectorTriggerLogic,
    SignalDict,
    SignalR,
    SignalRW,
    StandardReadable,
    StrictEnum,
    SubsetEnum,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)
from ophyd_async.epics.adcore import (
    ADAcquireLogic,
    ADBaseIO,
    ADWriterFactory,
    AreaDetector,
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
    trigger: A[SignalRW[float], PvSuffix("Trigger")]
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


@dataclass
class Pilatus4TriggerLogic(DetectorTriggerLogic):
    """Trigger logic for Pilatus4 detectors."""

    driver: Pilatus4DriverIO

    def get_deadtime(self, config_values: SignalDict) -> float:
        return 0.001

    async def prepare_internal(self, num: int, livetime: float, deadtime: float):
        await self.driver.trigger_mode.set(Pilatus4TriggerMode.INTERNAL_SERIES)
        await prepare_exposures(self.driver, num, livetime, deadtime)

    async def prepare_edge(self, num: int, livetime: float):
        coros = [
            self.driver.trigger_mode.set(Pilatus4TriggerMode.EXTERNAL_SERIES),
            self.driver.num_triggers.set(num),
            self.driver.acquire_time.set(livetime),
            self.driver.num_images.set(1),
        ]
        await asyncio.gather(*coros)

    async def default_trigger_info(self):
        return await trigger_info_from_num_images(self.driver)


class Pilatus4Detector(AreaDetector[Pilatus4DriverIO]):
    """Create an Pilatus4 AreaDetector instance.

    :param prefix: EPICS PV prefix for the detector
    :param writer_factories: Factories for file writer plugins and their data logics
    :param driver_suffix: Suffix for the driver PV, defaults to "cam1:"
    :param plugins: Additional areaDetector plugins to include
    :param config_sigs: Additional signals to include in configuration
    :param name: Name for the detector device
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
