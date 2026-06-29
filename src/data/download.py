from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any

import requests
from tqdm import tqdm

logger = logging.getLogger("ai-ml-template")


def download_with_retry(
    url: str,
    dest: str,
    expected_sha256: str | None = None,
    max_retries: int = 3,
) -> None:
    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            temp_dest = dest + ".part"
            hasher = hashlib.sha256()

            with open(temp_dest, "wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=os.path.basename(dest),
                leave=False,
            ) as pb:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        hasher.update(chunk)
                        pb.update(len(chunk))

            actual_hash = hasher.hexdigest()

            if expected_sha256 is not None and actual_hash != expected_sha256:
                raise ValueError(
                    f"SHA256 mismatch for {url}. "
                    f"Expected: {expected_sha256}, Got: {actual_hash}"
                )

            os.rename(temp_dest, dest)
            logger.info("Downloaded %s -> %s (%d bytes)", url, dest, total_size)
            return

        except Exception as e:
            if os.path.exists(dest + ".part"):
                os.remove(dest + ".part")

            if attempt < max_retries:
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "Download attempt %d/%d failed: %s. Retrying in %ds...",
                    attempt,
                    max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to download {url} after {max_retries} attempts."
                ) from e
