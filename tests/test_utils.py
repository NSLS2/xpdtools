"""Tests for xpdtools utility functions."""

import getpass

import pytest
from pygments.token import Token
from pytest_mock import MockerFixture

from xpdtools.utils import ProposalIDPrompt, start_beamtime


def test_start_beamtime(
    mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that start_beamtime suppresses prints.

    input/getpass prompts should still display.
    """

    mock_input = mocker.patch("builtins.input", return_value="testuser")
    mock_getpass = mocker.patch("getpass.getpass", return_value="testpass")

    sample_metadata = {
        "proposal": {
            "title": "Test Proposal",
            "type": "GU",
            "pi_name": "Test PI",
        }
    }

    def fake_sync_experiment(*args, **kwargs):
        # These prompts should still display (not suppressed by redirect_stdout)
        input("Username: ")
        getpass.getpass("Password: ")
        # Regular print output should be suppressed by redirect_stdout
        print("This should be suppressed")
        return sample_metadata

    mocker.patch(
        "xpdtools.utils.sync_experiment",
        side_effect=fake_sync_experiment,
    )

    start_beamtime(123456)

    mock_input.assert_called_once_with("Username: ")
    mock_getpass.assert_called_once_with("Password: ")

    captured = capsys.readouterr()
    assert "This should be suppressed" not in captured.out
    assert "123456" in captured.out
    assert "Test Proposal" in captured.out
    assert "GU" in captured.out
    assert "Test PI" in captured.out


def test_start_beamtime_no_proposal(
    mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test start_beamtime when no proposal metadata is returned."""

    mocker.patch("builtins.input", return_value="testuser")
    mocker.patch("getpass.getpass", return_value="testpass")

    def fake_sync_experiment(*args, **kwargs):
        input("Username: ")
        getpass.getpass("Password: ")
        return {}

    mocker.patch(
        "xpdtools.utils.sync_experiment",
        side_effect=fake_sync_experiment,
    )

    start_beamtime(999999)

    captured = capsys.readouterr()
    assert "999999" in captured.out


def test_proposal_id_prompt(mocker: MockerFixture) -> None:
    """Test ProposalIDPrompt formats prompt with proposal ID."""

    mock_shell = mocker.MagicMock()
    mock_shell.execution_count = 5

    mock_re = mocker.MagicMock()
    mock_re.md.get.return_value = "pass-123456"

    prompt = ProposalIDPrompt(mock_re, mock_shell)
    tokens = prompt.in_prompt_tokens()

    assert tokens == [
        (Token.Prompt, "pass-123456 ["),
        (Token.PromptNum, "5"),
        (Token.Prompt, "]: "),
    ]
    mock_re.md.get.assert_called_once_with("data_session", "N/A")


def test_proposal_id_prompt_no_session(mocker: MockerFixture) -> None:
    """Test that ProposalIDPrompt shows N/A when no data_session is set."""

    mock_shell = mocker.MagicMock()
    mock_shell.execution_count = 1

    mock_re = mocker.MagicMock()
    mock_re.md.get.return_value = "N/A"

    prompt = ProposalIDPrompt(mock_re, mock_shell)
    tokens = prompt.in_prompt_tokens()

    assert tokens == [
        (Token.Prompt, "N/A ["),
        (Token.PromptNum, "1"),
        (Token.Prompt, "]: "),
    ]
