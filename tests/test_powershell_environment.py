"""Verify the launcher imports dotenv values as data without exposing credentials."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_launcher_loads_env_without_overriding_process_or_printing_secrets(tmp_path):
    shell = shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is required for Windows launcher verification")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MAS_TEST_PRIVATE='dummy-secret;$(Write-Output injected)'\n"
        "MAS_TEST_EXISTING=from-file\n"
        "MAS_TEST_CACHE='C:/cache with spaces'\n",
        encoding="utf-8",
    )
    runner = tmp_path / "check.ps1"
    helper = Path(__file__).resolve().parents[1] / "scripts/load_environment.ps1"
    runner.write_text(
        "param($Helper, $Python, $EnvFile)\n"
        "$ErrorActionPreference = 'Stop'\n"
        "$env:MAS_TEST_EXISTING = 'from-process'\n"
        ". $Helper\n"
        "Import-ProjectEnvironment -Python $Python -EnvFile $EnvFile\n"
        "@{private=$env:MAS_TEST_PRIVATE;existing=$env:MAS_TEST_EXISTING;"
        "cache=$env:MAS_TEST_CACHE} | ConvertTo-Json -Compress\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(runner),
            str(helper),
            sys.executable,
            str(env_file),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    # A single JSON value means the importer did not print the dotenv contents.
    data = json.loads(result.stdout)
    assert data == {
        "private": "dummy-secret;$(Write-Output injected)",
        "existing": "from-process",
        "cache": "C:/cache with spaces",
    }
    assert not result.stderr
