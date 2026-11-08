"""Optional ranged download from the original CIFAR host, verified by torchvision's MD5.

Useful on networks where a single connection to the dataset host is slow.
No alternate dataset or data transformation is involved.
"""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import httpx

URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
SIZE = 170498071
MD5 = "c58f30108f718f92721af3b95e74349a"


def main():
    root = Path("data")
    root.mkdir(exist_ok=True)
    target = root/"cifar-10-python.tar.gz"
    if target.exists() and hashlib.md5(target.read_bytes()).hexdigest() == MD5:
        print("CIFAR archive already verified")
        return
    cache = Path("tmp/cifar-download")
    cache.mkdir(parents=True, exist_ok=True)
    chunk = 4 * 1024 * 1024

    def fetch(index):
        first, last = index * chunk, min(SIZE, (index + 1)*chunk) - 1
        path = cache/f"part-{index:03}"
        if path.exists() and path.stat().st_size == last-first+1:
            return path
        response = httpx.get(URL + f"?part={index}", headers={"Range": f"bytes={first}-{last}"},
                             follow_redirects=True, timeout=120)
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != last-first+1:
            raise RuntimeError("Server did not honor the range request")
        path.write_bytes(response.content)
        print(f"part {index + 1}/{(SIZE + chunk - 1)//chunk}", flush=True)
        return path

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(fetch, range((SIZE + chunk - 1)//chunk)))
    tmp = target.with_suffix(".verified-part")
    with tmp.open("wb") as output:
        for path in parts:
            output.write(path.read_bytes())
    if hashlib.md5(tmp.read_bytes()).hexdigest() != MD5:
        raise RuntimeError("Archive MD5 mismatch; file was not installed")
    os.replace(tmp, target)
    print("CIFAR archive verified. CLI preprocessing will extract it.")


if __name__ == "__main__":
    main()
