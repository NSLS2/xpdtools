from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import xpd_tools.optimization.cli as cli


class _Context:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Client:
    def __init__(self) -> None:
        self.context = _Context()

    def __getitem__(self, key: str) -> _Client:
        assert key == cli.SANDBOX_CATALOG
        return self


class _Evaluator:
    instances: list[_Evaluator] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []
        self.instances.append(self)

    def __call__(
        self, uid: str, suggestions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.calls.append((uid, suggestions))
        return [{"Peak": 660.0, "_id": suggestions[0]["_id"]}]


def test_cli_requires_reference_config() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["uid"])
    assert exc.value.code == 2


def test_cli_rejects_empty_uid_input(
    reference_config_factory: Any,
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--pdf-references", str(reference_config_factory())])
    assert exc.value.code == 2


def test_evaluate_uids_assigns_ids_and_aborts_on_error() -> None:
    calls: list[str] = []

    def evaluator(
        uid: str, suggestions: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        calls.append(uid)
        if uid == "second":
            raise RuntimeError("stop")
        return [{"value": 1, "_id": suggestions[0]["_id"]}]

    with pytest.raises(RuntimeError, match="stop"):
        cli.evaluate_uids(["first", "second", "third"], evaluator)
    assert calls == ["first", "second"]


def test_cli_combines_uids_outputs_json_csv_and_closes_clients(
    tmp_path: Path,
    reference_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    uid_file = tmp_path / "uids.txt"
    uid_file.write_text("# comment\nsecond # trailing comment\n\n")
    output = tmp_path / "outcomes.csv"
    clients: list[tuple[str, _Client]] = []

    def from_uri(uri: str) -> _Client:
        client = _Client()
        clients.append((uri, client))
        return client

    def from_profile(profile: str) -> _Client:
        raise AssertionError(f"raw profile unexpectedly used: {profile}")

    _Evaluator.instances.clear()
    monkeypatch.setattr(cli, "from_uri", from_uri)
    monkeypatch.setattr(cli, "from_profile", from_profile)
    monkeypatch.setattr(cli, "XrayUvvisEvaluation", _Evaluator)

    result = cli.main(
        [
            "first",
            "--uids-file",
            str(uid_file),
            "--raw-profile",
            "ignored",
            "--raw-uri",
            "memory://raw",
            "--sandbox-uri",
            "memory://sandbox",
            "--pdf-references",
            str(reference_config_factory()),
            "--pdf-mode",
            "raw_only",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert [uri for uri, _ in clients] == ["memory://raw", "memory://sandbox"]
    assert all(client.context.closed for _, client in clients)
    assert _Evaluator.instances[0].calls == [
        ("first", [{"_id": 0}]),
        ("second", [{"_id": 1}]),
    ]
    assert json.loads(capsys.readouterr().out) == [
        {"Peak": 660.0, "_id": 0, "uid": "first"},
        {"Peak": 660.0, "_id": 1, "uid": "second"},
    ]
    frame = pd.read_csv(output)
    assert frame["uid"].tolist() == ["first", "second"]


def test_cli_uses_raw_profile_when_uri_is_absent(
    reference_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw = _Client()
    sandbox = _Client()
    profiles: list[str] = []
    monkeypatch.setattr(
        cli,
        "from_profile",
        lambda profile: profiles.append(profile) or raw,
    )
    monkeypatch.setattr(cli, "from_uri", lambda uri: sandbox)
    monkeypatch.setattr(cli, "XrayUvvisEvaluation", _Evaluator)

    assert (
        cli.main(
            [
                "uid",
                "--raw-profile",
                "local-profile",
                "--pdf-references",
                str(reference_config_factory()),
                "--pdf-mode",
                "raw_only",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert profiles == ["local-profile"]
    assert raw.context.closed and sandbox.context.closed
