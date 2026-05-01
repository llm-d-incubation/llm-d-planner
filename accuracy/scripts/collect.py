"""
Pull vLLM memory validation results from the cluster PVC to a local directory.

Creates a temporary busybox reader pod that mounts the PVC, uses kubectl cp
to copy logs/ and runs/ for the given vLLM version, then deletes the pod.

Usage:
    python accuracy/scripts/collect.py \
      [--vllm-version v0.19.0]          # falls back to sweep.yaml vllm_image tag
      [--sweep accuracy/scripts/sweep.yaml]
      [--namespace llmdplanner]
      [--pvc vllm-mem-data]
      [--out data/benchmarks/memory/]
      [--dry-run]

Results land in: <out>/<version>/logs/ and <out>/<version>/runs/
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

READER_POD_NAME = "vllm-mem-reader"
POLL_INTERVAL_S = 2
POD_READY_TIMEOUT_S = 60


def _resolve_version(vllm_version_arg: str | None, sweep_path: str | None) -> str:
    if vllm_version_arg:
        return vllm_version_arg
    if sweep_path:
        config = yaml.safe_load(Path(sweep_path).read_text())
        image = config.get("defaults", {}).get("vllm_image", "")
        if ":" in image:
            return image.split(":")[-1]
    print(
        "Error: cannot resolve vLLM version. Provide --vllm-version or "
        "--sweep pointing to a sweep.yaml with defaults.vllm_image set.",
        file=sys.stderr,
    )
    sys.exit(1)


def _reader_pod_manifest(namespace: str, pvc: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": READER_POD_NAME, "namespace": namespace},
        "spec": {
            "restartPolicy": "Never",
            "volumes": [{"name": "data", "persistentVolumeClaim": {"claimName": pvc}}],
            "containers": [{
                "name": "reader",
                "image": "busybox",
                "command": ["sh", "-c", "sleep 3600"],
                "volumeMounts": [{"name": "data", "mountPath": "/data"}],
            }],
        },
    }


def _kubectl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", *args], capture_output=True, text=True, check=check)


def _wait_for_pod_running(namespace: str, timeout: int = POD_READY_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = _kubectl("get", "pod", READER_POD_NAME, "-n", namespace,
                          "-o", "jsonpath={.status.phase}", check=False)
        if result.stdout.strip() == "Running":
            return
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(
        f"Reader pod did not reach Running within {timeout}s. "
        f"Check: kubectl describe pod {READER_POD_NAME} -n {namespace}"
    )


def collect(
    vllm_version: str,
    namespace: str,
    pvc: str,
    out: str,
    dry_run: bool = False,
) -> None:
    src_base = f"/data/results/{vllm_version}"
    dst_base = Path(out) / vllm_version

    for subdir in ("logs", "runs"):
        src = f"{READER_POD_NAME}:{src_base}/{subdir}/"
        dst = dst_base / subdir
        print(f"  {'[dry-run] ' if dry_run else ''}kubectl cp {src} → {dst}")

    if dry_run:
        return

    manifest_json = json.dumps(_reader_pod_manifest(namespace, pvc))
    subprocess.run(
        ["kubectl", "apply", "-f", "-", "-n", namespace],
        input=manifest_json, text=True, check=True,
    )

    try:
        print("Waiting for reader pod to be Running...", flush=True)
        _wait_for_pod_running(namespace)

        for subdir in ("logs", "runs"):
            src = f"{READER_POD_NAME}:{src_base}/{subdir}/"
            dst = dst_base / subdir
            dst.mkdir(parents=True, exist_ok=True)
            print(f"Copying {subdir}...", flush=True)
            _kubectl("cp", "-n", namespace, src, str(dst))

    finally:
        print("Deleting reader pod...", flush=True)
        _kubectl("delete", "pod", READER_POD_NAME, "-n", namespace,
                 "--ignore-not-found", check=False)

    runs = list((dst_base / "runs").glob("*.json"))
    logs = list((dst_base / "logs").glob("*.log"))
    print(f"\nCollected {len(runs)} JSON results and {len(logs)} logs → {dst_base}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vllm-version", default=None)
    ap.add_argument("--sweep", default="accuracy/scripts/sweep.yaml")
    ap.add_argument("--namespace", default="llmdplanner")
    ap.add_argument("--pvc", default="vllm-mem-data")
    ap.add_argument("--out", default="data/benchmarks/memory/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    version = _resolve_version(args.vllm_version, args.sweep)
    print(f"vLLM version: {version}", flush=True)

    collect(
        vllm_version=version,
        namespace=args.namespace,
        pvc=args.pvc,
        out=args.out,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
