"""Bounded readiness check for an OpenAI-compatible serving process."""
import time
import httpx


def wait_for_server(gateway, timeout=180, interval=1):
    deadline = time.monotonic() + timeout
    headers = {"Authorization": "Bearer " + gateway.key} if gateway.key else {}
    with httpx.Client(trust_env=False, headers=headers) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(gateway.base_url + "/models", timeout=min(5, max(0.001, deadline - time.monotonic())))
                if response.status_code in (401, 403):
                    raise RuntimeError("Local server rejected credentials; check LLM_API_KEY")
                if response.status_code == 200:
                    data = response.json()
                    if gateway.model not in {item["id"] for item in data["data"]}:
                        raise RuntimeError(f"Local endpoint does not serve model {gateway.model}")
                    return data
            except (httpx.TransportError, ValueError, KeyError, TypeError):
                pass
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
    raise TimeoutError("Local LLM server is not ready; start scripts/start_local_llm.ps1 and inspect its log")
