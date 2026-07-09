from rich import print as rprint
from typing import Any
from IPython.terminal.prompts import Prompts
from pygments.token import Token
from bluesky.run_engine import RunEngine
from IPython.terminal.interactiveshell import TerminalInteractiveShell


def print_version_info():
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
    rprint(f"------- {name} ---------")
    rprint(doc)


class ProposalIDPrompt(Prompts):

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