"""XPD-D Bluesky profile for the 28-ID-2 beamline at NSLS-II."""

import os

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.run_engine import RunEngine, call_in_bluesky_event_loop
from bluesky.utils import ProgressBarManager
from bluesky_tiled_plugins import TiledWriter
from IPython.core.getipython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from nslsii.ophyd_async.providers import NSLS2PathProvider
from ophyd_async.core import (
    init_devices,
)
from ophyd_async.epics.adcore import ADWriterFactory
from ophyd_async.fastcs.panda import HDFPanda
from tiled.client import from_uri

from xpdtools.detectors.pilatus4 import Pilatus4Detector
from xpdtools.motors import RotationMotor
from xpdtools.plans import single_axis_flyscan
from xpdtools.utils import ProposalIDPrompt, initialize_run_engine

XPDTOOLS_RUNNING_IN_CI = (
    os.environ.get("XPDTOOLS_RUNNING_IN_CI", "false").lower() == "true"
)

os.environ["REDIS_HOST"] = "xf28id2-xpdd-redis1.nsls2.bnl.gov"
os.environ["OPHYD_ASYNC_PRESERVE_DETECTOR_STATE"] = "YES"

# Create our RunEngine instance
RE: RunEngine = initialize_run_engine()

# Set some metadata that never changes.
RE.md["facility"] = "NSLS-II"
RE.md["group"] = "XPD-D"
RE.md["beamline_id"] = "28-ID-2"

# Setup progress bars
RE.waiting_hook = ProgressBarManager()  # type: ignore[assignment]


if not XPDTOOLS_RUNNING_IN_CI:
    tiled_writing_client = from_uri(
        "https://tiled.nsls2.bnl.gov",
        api_key=os.environ.get("TILED_BLUESKY_WRITING_API_KEY_XPDD", ""),
    )["xpdd"]["migration"]
else:
    from tiled.server.simple import SimpleTiledServer

    tiled_server = SimpleTiledServer(
        directory="/tmp/xpdtools",
        readable_storage="/tmp/tiled_server",
        api_key="xpdtools",
    )
    tiled_writing_client = from_uri(tiled_server.uri)
    tiled_reading_client = c = from_uri(tiled_server.uri)


# Setup writing to the database
tw = TiledWriter(tiled_writing_client)
RE.subscribe(tw)

# Setup BEC
bec = BestEffortCallback()
bec.disable_plots()
RE.subscribe(bec)

ipython = get_ipython()
if ipython is not None and isinstance(ipython, TerminalInteractiveShell):
    ipython.prompts = ProposalIDPrompt(RE, ipython)
    ipython.run_line_magic("autoawait", "call_in_bluesky_event_loop")
    if not XPDTOOLS_RUNNING_IN_CI:
        tiled_reading_client = c = from_uri("https://tiled.nsls2.bnl.gov")["xpdd"][
            "migration"
        ]


path_provider = NSLS2PathProvider(RE.md, beamline_tla="xpd", beamline_tla_suffix="-new")

with init_devices(mock=XPDTOOLS_RUNNING_IN_CI):
    panda1 = HDFPanda("XF:28ID2-ES{PANDA:1}:", path_provider)
    rot_motor = RotationMotor("XF:28IDD-ES:2{Twister}Mtr", encoder_pos_at_zero=-647)
    pilatus1 = Pilatus4Detector(
        "XF:28ID2-ES{Pilatus4-Det:1}", ADWriterFactory.hdf(path_provider)
    )
