"""Exercise disk-placement and ownership guards without creating a WSL distro."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows WSL host setup")

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/vllm_common.ps1"


def powershell_json(body):
    shell = shutil.which("powershell")
    if not shell:
        pytest.skip("Windows PowerShell is required")
    result = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
         "$ErrorActionPreference='Stop'\n. '" + str(HELPER) + "'\n" + body],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("bad_path", ["C:/model-cache", "D:/", "D:/cache/..", "//server/share/cache"])
def test_vllm_data_guard_rejects_other_drives_and_drive_root(bad_path):
    result = powershell_json(
        "$rejected=$false\n"
        f"try {{ Assert-VllmDataPath -Path '{bad_path}' | Out-Null }} catch {{ $rejected=$true }}\n"
        "@{rejected=$rejected} | ConvertTo-Json -Compress"
    )
    assert result == {"rejected": True}


def test_vllm_data_guard_preserves_paths_with_spaces():
    result = powershell_json(
        "@{path=(Assert-VllmDataPath -Path 'D:/HK261/vllm cache')} | ConvertTo-Json -Compress"
    )
    assert result == {"path": "D:\\HK261\\vllm cache"}


def test_vllm_space_guard_checks_physical_host_volume():
    result = powershell_json(
        "$rejected=$false\n"
        "try { Assert-VllmHostSpace -CFreeGiB 4.8 -DFreeGiB 9 -MinimumCFreeGiB 4 -MinimumDFreeGiB 10 } catch { $rejected=$true }\n"
        "Assert-VllmHostSpace -CFreeGiB 4.8 -DFreeGiB 46 -MinimumCFreeGiB 4 -MinimumDFreeGiB 35\n"
        "@{lowDRejected=$rejected} | ConvertTo-Json -Compress"
    )
    assert result == {"lowDRejected": True}


def test_vllm_setup_refuses_registered_distro_at_another_location():
    result = powershell_json(
        "$rejected=$false\n"
        "$records=@([pscustomobject]@{DistributionName='MAS-vLLM';BasePath='C:\\old-distro';Version=2})\n"
        "try { Assert-VllmDistroTarget -Distro 'MAS-vLLM' -InstallRoot 'D:/HK261/wsl/mas-vllm' -Records $records } catch { $rejected=$true }\n"
        "@{rejected=$rejected} | ConvertTo-Json -Compress"
    )
    assert result == {"rejected": True}


def test_vllm_setup_refuses_unowned_directory(tmp_path):
    assert tmp_path.drive.upper() == "D:", "Run Windows setup tests with TEMP/TMP on D"
    (tmp_path / "keep.txt").write_text("user data", encoding="utf-8")
    result = powershell_json(
        "$rejected=$false\n"
        f"try {{ Assert-VllmDistroTarget -Distro 'MAS-vLLM' -InstallRoot '{tmp_path}' -Records @() }} catch {{ $rejected=$_.Exception.Message -like '*unowned*' }}\n"
        "@{rejected=$rejected} | ConvertTo-Json -Compress"
    )
    assert result == {"rejected": True}
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_registered_but_incomplete_distro_still_needs_installation_space():
    result = powershell_json(
        "@{incomplete=(Get-VllmMinimumD -EnvironmentReady $false);"
        "complete=(Get-VllmMinimumD -EnvironmentReady $true)} | ConvertTo-Json -Compress"
    )
    assert result == {"incomplete": 35, "complete": 10}


def test_output_guard_checks_existing_child_junction(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    junction = tmp_path / "downloads"
    try:
        result = powershell_json(
            f"New-Item -ItemType Junction -Path '{junction}' -Target '{target}' | Out-Null\n"
            f"$safe=Get-VllmOutputPath -Root '{tmp_path}' -Child 'safe.json'\n"
            "$rejected=$false\n"
            f"try {{ Get-VllmOutputPath -Root '{tmp_path}' -Child 'downloads/rootfs.wsl' | Out-Null }} catch {{ $rejected=$true }}\n"
            "@{rejected=$rejected;safe=$safe} | ConvertTo-Json -Compress"
        )
        assert result == {"rejected": True, "safe": str(tmp_path / "safe.json")}
    finally:
        if junction.exists():
            os.rmdir(junction)


def test_failed_setup_replaces_stale_ready_status(tmp_path):
    output = tmp_path / "environment.json"
    previous = tmp_path / "environment.previous.json"
    output.write_text('{"environment_ready":true}', encoding="utf-8")
    result = powershell_json(
        f"Write-VllmSetupStatus -Output '{output}' -State running -PreviousOutput '{previous}'\n"
        f"Write-VllmSetupStatus -Output '{output}' -State failed -ErrorMessage 'disk low'\n"
        f"Write-VllmSetupStatus -Output '{output}' -State failed -ErrorMessage 'disk low' -PreviousOutput '{previous}'\n"
        f"@{{current=(Get-Content -Raw -LiteralPath '{output}' | ConvertFrom-Json);previous=(Get-Content -Raw -LiteralPath '{previous}' | ConvertFrom-Json)}} | ConvertTo-Json -Depth 5"
    )
    assert result["current"]["environment_ready"] is False
    assert result["current"]["state"] == "failed"
    assert result["current"]["error"] == "disk low"
    assert result["current"]["model_hardware_gate"] == "not_run"
    assert result["previous"]["environment_ready"] is True


def test_monitored_process_preserves_arguments_and_exit_code(tmp_path):
    script = tmp_path / "child script.ps1"
    script.write_text("param([string]$Value)\nWrite-Output $Value\nexit 7\n", encoding="utf-8")
    out = tmp_path / "stdout.log"
    err = tmp_path / "stderr.log"
    result = powershell_json(
        f"$code=Invoke-VllmSetupProcess -Command (Get-Command powershell.exe).Source -Arguments @('-NoProfile','-File','{script}','two words') -Output '{out}' -ErrorOutput '{err}' -Monitor {{ }}\n"
        "@{code=$code} | ConvertTo-Json -Compress"
    )
    assert result == {"code": 7}
    assert out.read_text(encoding="utf-8").strip() == "two words"
