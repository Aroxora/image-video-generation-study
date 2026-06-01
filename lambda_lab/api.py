"""
A tiny, dependency-free client for the Lambda Cloud public API (v1).

Auth: Lambda uses HTTP Basic with your API key as the username and an empty
password (`curl -u "$LAMBDA_API_KEY:" ...`). Generate a key at
https://cloud.lambda.ai/api-keys and export it as LAMBDA_API_KEY.

Only the standard library is used, so this runs on a bare Python with no `pip
install`. Endpoints are read straight from the official docs at
https://docs.lambda.ai/public-cloud/cloud-api/ — see API_BASE / OPS below; if
Lambda renames anything, those two constants are the only edit needed.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

# Lambda rebranded lambdalabs.com -> lambda.ai; the cloud API host followed.
# Both hosts have historically resolved, but this is the current canonical one.
API_BASE = os.environ.get("LAMBDA_API_BASE", "https://cloud.lambda.ai/api/v1")

# Credentials are looked up in this order; the file form keeps the secret OUT of
# the repo (these paths are git-ignored / outside the tree). Never commit a key.
CRED_PATHS = (
    Path.home() / ".lambda_lab" / "credentials",
    Path(".lambda_lab") / "credentials",
)


def _key_from_file() -> str:
    """Read LAMBDA_API_KEY=... (or a bare key) from the first credentials file found."""
    for p in CRED_PATHS:
        try:
            if not p.is_file():
                continue
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("LAMBDA_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
                if line.startswith("secret_"):
                    return line
        except OSError:
            continue
    return ""


class LambdaCloudError(RuntimeError):
    """Raised on any non-2xx response; carries status + parsed body for triage."""

    def __init__(self, status: int, message: str, body: object = None):
        super().__init__(f"[{status}] {message}")
        self.status = status
        self.body = body


class LambdaCloud:
    def __init__(self, api_key: str | None = None, *, base: str = API_BASE, timeout: float = 60.0):
        self.api_key = api_key or os.environ.get("LAMBDA_API_KEY", "") or _key_from_file()
        if not self.api_key:
            raise LambdaCloudError(
                0, "no API key: set LAMBDA_API_KEY, pass api_key=, or write ~/.lambda_lab/credentials"
            )
        self.base = base.rstrip("/")
        self.timeout = timeout
        # Basic auth, key as username, empty password.
        token = base64.b64encode(f"{self.api_key}:".encode()).decode()
        self._auth = f"Basic {token}"

    # ---- low-level request ------------------------------------------------
    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{self.base}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", self._auth)
        req.add_header("Accept", "application/json")
        # The API sits behind Cloudflare, which 1010-blocks the default
        # "Python-urllib/x" agent. A curl-style UA sails through (override via env).
        req.add_header("User-Agent", os.environ.get("LAMBDA_LAB_UA", "curl/8.7.1"))
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode()
                parsed = json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raw = e.read().decode() if e.fp else ""
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = {"raw": raw}
            msg = ""
            if isinstance(parsed, dict):
                err = parsed.get("error") or {}
                msg = err.get("message") or parsed.get("message") or raw
            raise LambdaCloudError(e.code, msg or e.reason, parsed) from None
        except urllib.error.URLError as e:
            raise LambdaCloudError(0, f"network error: {e.reason}") from None
        # The API wraps payloads in {"data": ...}.
        return parsed.get("data", parsed) if isinstance(parsed, dict) else parsed

    # ---- instance types ---------------------------------------------------
    def instance_types(self) -> dict:
        """GET /instance-types — {name: {instance_type, regions_with_capacity_available}}."""
        return self._request("GET", "/instance-types")

    def regions_with_capacity(self, instance_type: str) -> list[str]:
        info = self.instance_types().get(instance_type) or {}
        regions = info.get("regions_with_capacity_available", [])
        return [r.get("name", r) if isinstance(r, dict) else r for r in regions]

    # ---- instances --------------------------------------------------------
    def list_instances(self) -> list[dict]:
        """GET /instances."""
        return self._request("GET", "/instances")

    def get_instance(self, instance_id: str) -> dict:
        """GET /instances/{id}."""
        return self._request("GET", f"/instances/{instance_id}")

    def find_instance_by_name(self, name: str) -> dict | None:
        """Idempotency helper: return a live instance tagged with `name`, if any."""
        for inst in self.list_instances():
            if inst.get("name") == name and inst.get("status") != "terminated":
                return inst
        return None

    def launch(
        self,
        instance_type: str,
        region: str,
        ssh_key_names: list[str],
        *,
        name: str | None = None,
        file_system_names: list[str] | None = None,
        quantity: int = 1,
    ) -> list[str]:
        """POST /instance-operations/launch -> list of new instance ids."""
        body = {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": ssh_key_names,
            "quantity": quantity,
        }
        if name:
            body["name"] = name
        if file_system_names:
            body["file_system_names"] = file_system_names
        out = self._request("POST", "/instance-operations/launch", body)
        return out.get("instance_ids", [])

    def terminate(self, instance_ids: list[str]) -> dict:
        """POST /instance-operations/terminate."""
        return self._request("POST", "/instance-operations/terminate", {"instance_ids": instance_ids})

    def restart(self, instance_ids: list[str]) -> dict:
        return self._request("POST", "/instance-operations/restart", {"instance_ids": instance_ids})

    # ---- ssh keys ---------------------------------------------------------
    def list_ssh_keys(self) -> list[dict]:
        return self._request("GET", "/ssh-keys")

    def add_ssh_key(self, name: str, public_key: str | None = None) -> dict:
        """POST /ssh-keys. With no public_key, Lambda generates one and returns the
        private key once (save it!). Pass a public_key to register an existing key."""
        body = {"name": name}
        if public_key:
            body["public_key"] = public_key
        return self._request("POST", "/ssh-keys", body)

    def ensure_ssh_key(self, name: str, public_key: str) -> str:
        """Register `name` if absent; return the key name to use at launch."""
        existing = {k.get("name") for k in self.list_ssh_keys()}
        if name not in existing:
            self.add_ssh_key(name, public_key)
        return name

    # ---- filesystems (persistent storage that survives termination) -------
    def list_filesystems(self) -> list[dict]:
        return self._request("GET", "/file-systems")

    def find_filesystem(self, name: str) -> dict | None:
        for fs in self.list_filesystems():
            if fs.get("name") == name:
                return fs
        return None

    # ---- polling helpers --------------------------------------------------
    def wait_until_active(self, instance_id: str, *, timeout_s: float = 900, poll_s: float = 12) -> dict:
        """Block until the instance reports status 'active' and has an IP."""
        deadline = time.time() + timeout_s
        last = {}
        while time.time() < deadline:
            last = self.get_instance(instance_id)
            status = last.get("status")
            if status == "active" and last.get("ip"):
                return last
            if status in {"terminated", "error", "unhealthy"}:
                raise LambdaCloudError(0, f"instance {instance_id} entered status={status}", last)
            time.sleep(poll_s)
        raise LambdaCloudError(0, f"timed out waiting for {instance_id} to go active", last)
