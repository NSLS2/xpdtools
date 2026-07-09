"""General utility functions for xpdtools."""

from typing import Any

from bluesky.run_engine import RunEngine
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.terminal.prompts import Prompts
from pygments.token import Token
from rich import print as rprint


def print_version_info():
    """Print version information for bluesky, ophyd_async, tiled, and xpdtools."""
    from bluesky import __version__ as bluesky_version
    from ophyd_async import __version__ as ophyd_async_version
    from tiled import __version__ as tiled_version

    from xpdtools import __version__ as xpdtools_version

    rprint("\n[bold]Version Information[/bold]")
    rprint(f"  [bold]bluesky[/bold]: [blue]{bluesky_version}[/blue]")
    rprint(f"  [bold]ophyd_async[/bold]: [blue]{ophyd_async_version}[/blue]")
    rprint(f"  [bold]tiled[/bold]: [blue]{tiled_version}[/blue]")
    rprint(f"  [bold]xpdtools[/bold]: [blue]{xpdtools_version}[/blue]\n")


def show_docs(name: str, doc: dict[str, Any]):
    """Print out the bluesky documents in a readable format."""
    rprint(f"------- {name} ---------")
    rprint(doc)


class ProposalIDPrompt(Prompts):
    """Custom IPython prompt that shows the current proposal ID."""

    def __init__(self, RE: RunEngine, shell: TerminalInteractiveShell):
        super().__init__(shell)
        self._RE = RE

    def in_prompt_tokens(self, cli=None):
        return [
            (
                Token.Prompt,
                f"{self._RE.md.get('data_session', 'N/A')} [",
            ),
            (Token.PromptNum, str(self.shell.execution_count)),
            (Token.Prompt, "]: "),
        ]
