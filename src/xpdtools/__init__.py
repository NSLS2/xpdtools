"""
Copyright (c) 2026 Brookhaven National Laboratory. All rights reserved.

xpdtools: Tools for NSLS-II XPD beamline
"""

from __future__ import annotations

from ._version import version as __version__

__all__ = ["__version__"]

import IPython
import sys
from pathlib import Path
import argparse
from rich import print as rprint
from .utils import print_version_info


def start_profile():
    parser = argparse.ArgumentParser(description="Start a profile for xpdtools.")
    parser.add_argument("profile_name", nargs="?", default="", type=str, help="Name of the profile to start. If not provided, the default profile will be used.")
    args = parser.parse_args()
    rprint(f"[bold]Starting xpdtools profile: [green]{args.profile_name or 'default'}[/green][/bold]")
    print_version_info()
    profile_path = str(Path(__file__).parent / f"profile{'_' + args.profile_name if args.profile_name else ''}.py")
    sys.exit(IPython.start_ipython(argv=["-i", profile_path]))