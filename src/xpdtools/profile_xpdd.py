from tiled.client import from_uri
from bluesky_tiled_plugins import TiledWriter
from bluesky.run_engine import RunEngine, call_in_bluesky_event_loop
from bluesky.utils import ProgressBarManager
import bluesky.plan_stubs as bps
import bluesky.plans as bp

from pathlib import Path
from nslsii.ophyd_async.providers import NSLS2PathProvider
from nslsii.utils import open_redis_client
from redis_json_dict.redis_json_dict import RedisJSONDict
from IPython.core.getipython import get_ipython
import os
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from xpdtools.motors import RotationMotor
from xpdtools.utils import ProposalIDPrompt, show_docs
from xpdtools.detectors.pilatus4 import Pilatus4Detector
from ophyd_async.core import init_devices, StandardFlyer, YMDPathProvider, UUIDFilenameProvider
from ophyd_async.epics.adcore import ADWriterFactory
from ophyd_async.fastcs.panda import HDFPanda
from datetime import datetime
from tempfile import mkdtemp

XPDTOOLS_RUNNING_IN_CI = os.environ.get("XPDTOOLS_RUNNING_IN_CI", "false").lower() == "true"

if not XPDTOOLS_RUNNING_IN_CI:
    RE = RunEngine(RedisJSONDict(open_redis_client("xf28id2-xpdd-redis1.nsls2.bnl.gov", redis_ssl=True), ""))  # type: ignore (TODO: Loosen type of RE.md to Mapping from dict)
else:
    RE = RunEngine({"data_session": "pass-123456", "cycle": f"{datetime.today().year}-{datetime.today().month % 4 + 1}"})

# Set some metadata that never changes.
RE.md["facility"] = "NSLS-II"
RE.md["group"] = "XPD-D"
RE.md["beamline_id"] = "28-ID-2"

RE.waiting_hook = ProgressBarManager()  # type: ignore

if not XPDTOOLS_RUNNING_IN_CI:
    tiled_writing_client = from_uri("https://tiled.nsls2.bnl.gov", api_key=os.environ.get("TILED_BLUESKY_WRITING_API_KEY_XPD", ""))["xpdd"]["migration"]
else:
    from tiled.server.simple import SimpleTiledServer
    tiled_server = SimpleTiledServer(directory="/tmp/xpdtools", readable_storage="/tmp/tiled_server", api_key="xpdtools")
    tiled_writing_client = from_uri(tiled_server.uri)
    tiled_reading_client = c = from_uri(tiled_server.uri)


tw = TiledWriter(tiled_writing_client)
RE.subscribe(tw)


ipython = get_ipython()
if ipython is not None and isinstance(ipython, TerminalInteractiveShell):
    ipython.prompts = ProposalIDPrompt(RE, ipython)
    ipython.run_line_magic("autoawait", "call_in_bluesky_event_loop")
    if not XPDTOOLS_RUNNING_IN_CI:
        tiled_reading_client = c = from_uri("https://tiled.nsls2.bnl.gov")["xpdd"]["migration"]


path_provider = NSLS2PathProvider(RE.md, beamline_tla_suffix="-new")
if XPDTOOLS_RUNNING_IN_CI:
    path_provider._beamline_proposals_dir = Path(mkdtemp())


with init_devices(mock=XPDTOOLS_RUNNING_IN_CI):
    panda = HDFPanda("XF:28ID2-ES{PANDA:1}:", path_provider)
    rot_motor = RotationMotor("XF:28IDD-ES:2{Twister}Mtr", encoder_pos_at_zero=0)
    pilatus4 = Pilatus4Detector("XF:28ID2-ES{Pilatus4-Det:1}", ADWriterFactory.hdf(path_provider))