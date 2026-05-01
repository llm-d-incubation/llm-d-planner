"""
In-cluster orchestrator: reads sweep.yaml, submits sub-jobs sequentially,
fetches startup logs when each pod is ready, and parses results to JSON.

Designed to run inside a Kubernetes Job (in-cluster config). Requires the
vllm-mem-orchestrator ServiceAccount with Job+Pod+Pod/log RBAC.

Usage (run by the orchestrator Job — scripts mounted at /scripts/ via ConfigMap):
    python sweep_runner.py --config /sweep/sweep.yaml --results /data/results/
"""
import argparse
import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

NAMESPACE = "llmdplanner"


# ── Matrix expansion ──────────────────────────────────────────────────────────

def _expandable_fields() -> list[str]:
    return ["tp", "pp", "dp", "max_model_len", "dtype", "kv_cache_dtype"]


def expand_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = config.get("defaults", {})
    expanded = []
    for entry in config["runs"]:
        base = {**defaults, **entry}
        for field in _expandable_fields():
            if isinstance(base.get(field), list):
                for val in base[field]:
                    run = copy.deepcopy(base)
                    run[field] = val
                    expanded.append(run)
                break
        else:
            expanded.append(base)
    return expanded


def make_run_id(run: dict[str, Any]) -> str:
    model_slug = re.sub(r"[^a-z0-9]+", "-", (run.get("_label") or run["model"]).lower()).strip("-")
    gpu_slug = re.sub(r"[^a-z0-9]+", "-", run["gpu"].lower()).strip("-")
    params = f"tp{run['tp']}pp{run['pp']}dp{run['dp']}"

    # Append short discriminators for non-default sensitivity dimensions so
    # that dtype/kv_cache_dtype/quantization sweeps get unique result files.
    suffix = ""
    dtype = run.get("dtype") or "auto"
    if dtype != "auto":
        suffix += "-dt" + dtype.replace("float", "f").replace("bfloat", "bf")  # dtf16 / dtbf16
    kv = run.get("kv_cache_dtype") or "auto"
    if kv != "auto":
        suffix += "-kv" + kv.replace("float", "f").replace("bfloat", "bf")  # kvf16 / kvfp8
    quant = run.get("quantization")
    if quant:
        suffix += f"-q{quant}"  # qfp8 / qawq

    tail = f"--{gpu_slug}--{params}--{run['max_model_len']}{suffix}"
    rid = f"{model_slug}{tail}"
    if len(rid) > 52:
        model_slug = model_slug[: len(model_slug) - (len(rid) - 52)]
        rid = f"{model_slug}{tail}"
    return rid.strip("-")


# ── K8s orchestration ─────────────────────────────────────────────────────────

def _load_k8s() -> tuple[Any, Any]:
    """Load in-cluster K8s config and return (BatchV1Api, CoreV1Api)."""
    from kubernetes import client, config
    config.load_incluster_config()
    return client.BatchV1Api(), client.CoreV1Api()


def _build_job_manifest(run_id: str, run: dict[str, Any]) -> dict[str, Any]:
    """Build a Job manifest dict for a single vLLM run."""
    num_gpus = run["tp"] * run["pp"]
    node_selector = run.get("node_selector", {})
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"vllm-mem-{run_id}",
            "namespace": run["namespace"],
            "labels": {"app": "vllm-mem-validation", "run-id": run_id},
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 3600,
            "template": {
                "metadata": {"labels": {"app": "vllm-mem-validation", "run-id": run_id}},
                "spec": {
                    "restartPolicy": "Never",
                    "volumes": [{"name": "data", "persistentVolumeClaim":
                                 {"claimName": run["results_pvc"]}}],
                    "containers": [{
                        "name": "vllm",
                        "image": run["vllm_image"],
                        "command": ["vllm", "serve", run["model"]],
                        "args": [
                            f"--tensor-parallel-size={run['tp']}",
                            f"--pipeline-parallel-size={run['pp']}",
                            f"--data-parallel-size={run['dp']}",
                            f"--gpu-memory-utilization={run['gpu_memory_utilization']}",
                            f"--max-model-len={run['max_model_len']}",
                            "--no-enable-prefix-caching",
                            # Argument-sensitivity fields — only emitted when non-default
                            *([f"--dtype={run['dtype']}"] if run.get("dtype") and run["dtype"] != "auto" else []),
                            *([f"--quantization={run['quantization']}"] if run.get("quantization") else []),
                            *([f"--kv-cache-dtype={run['kv_cache_dtype']}"] if run.get("kv_cache_dtype") and run["kv_cache_dtype"] != "auto" else []),
                            *(["--trust-remote-code"] if run.get("trust_remote_code") else []),
                        ],
                        "env": [
                            {"name": "HF_TOKEN", "valueFrom":
                             {"secretKeyRef": {"name": run.get("hf_token_secret", "hf-token"), "key": "token"}}},
                            {"name": "HF_HOME", "value": "/data/models"},
                            {"name": "HOME", "value": "/data"},
                            {"name": "XDG_CACHE_HOME", "value": "/data/.cache"},
                            {"name": "FLASHINFER_WORKSPACE_DIR", "value": "/data/.cache/flashinfer"},
                            {"name": "VLLM_ATTENTION_BACKEND", "value": "FLASH_ATTN"},
                            {"name": "VLLM_LOGGING_LEVEL", "value": "DEBUG"},
                        ],
                        "resources": {
                            "limits": {"nvidia.com/gpu": num_gpus},
                            "requests": {"nvidia.com/gpu": num_gpus},
                        },
                        "volumeMounts": [{"name": "data", "mountPath": "/data"}],
                        "startupProbe": {
                            "httpGet": {"path": "/health", "port": 8000},
                            "initialDelaySeconds": 60,
                            "periodSeconds": 10,
                            "failureThreshold": 180,  # 30 minutes
                            "successThreshold": 1,
                        },
                    }],
                    "nodeSelector": node_selector,
                    "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists",
                                     "effect": "NoSchedule"}],
                },
            },
        },
    }


def _submit_sub_job(batch_api: Any, run_id: str, run: dict[str, Any]) -> None:
    from kubernetes import client
    manifest = _build_job_manifest(run_id, run)
    batch_api.create_namespaced_job(namespace=run["namespace"], body=manifest)
    print(f"  Submitted: vllm-mem-{run_id}", flush=True)


class PodFailedError(RuntimeError):
    """Raised when a vLLM pod terminates with a non-zero exit or phase=Failed."""
    def __init__(self, pod_name: str, message: str) -> None:
        super().__init__(message)
        self.pod_name = pod_name


def _wait_for_pod_ready(core_api: Any, run_id: str, namespace: str,
                        timeout: int = 2400) -> str:
    """Block until the pod's startupProbe passes. Returns the pod name."""
    from kubernetes import watch
    w = watch.Watch()
    label_sel = f"run-id={run_id}"
    print("  Waiting for pod ready (startupProbe)...", flush=True)
    for event in w.stream(core_api.list_namespaced_pod,
                          namespace=namespace,
                          label_selector=label_sel,
                          timeout_seconds=timeout):
        pod = event["object"]
        pod_name = pod.metadata.name
        # Detect terminal failure immediately rather than waiting for timeout
        if pod.status.phase == "Failed":
            w.stop()
            raise PodFailedError(pod_name, f"Pod {pod_name} failed (phase=Failed)")
        if pod.status.container_statuses:
            for cs in pod.status.container_statuses:
                if cs.ready:
                    w.stop()
                    print(f"  Pod ready: {pod_name}", flush=True)
                    return pod_name
                # Terminated with non-zero exit = OOM or crash
                if cs.state and cs.state.terminated and cs.state.terminated.exit_code != 0:
                    w.stop()
                    reason = cs.state.terminated.reason or "unknown"
                    raise PodFailedError(
                        pod_name,
                        f"Pod {pod_name} terminated: {reason} "
                        f"(exit {cs.state.terminated.exit_code})"
                    )
    raise TimeoutError(f"Pod for run-id={run_id} did not become ready within {timeout}s")


def _fetch_pod_log(core_api: Any, pod_name: str, namespace: str,
                   previous: bool = False) -> str:
    return core_api.read_namespaced_pod_log(
        name=pod_name, namespace=namespace, previous=previous
    )


def _cleanup_stale_job(batch_api: Any, run_id: str, namespace: str) -> None:
    """Delete a leftover job from a previous crashed sweep before re-submitting."""
    from kubernetes import client
    try:
        batch_api.delete_namespaced_job(
            name=f"vllm-mem-{run_id}", namespace=namespace,
            body=client.V1DeleteOptions(propagation_policy="Background"),
        )
        print("  Cleaned up stale job before submit.", flush=True)
    except Exception:
        pass  # Job doesn't exist — expected on first run


def _delete_job(batch_api: Any, run_id: str, namespace: str) -> None:
    from kubernetes import client
    job_name = f"vllm-mem-{run_id}"
    batch_api.delete_namespaced_job(
        name=job_name, namespace=namespace,
        body=client.V1DeleteOptions(propagation_policy="Foreground"),
    )
    print(f"  Deleted Job: {job_name}", flush=True)


def run_sweep(runs: list[dict[str, Any]], results_dir: Path) -> None:
    # In-cluster: parse_log is mounted at /scripts/ via ConfigMap.
    # Locally: fall back to the sibling parse_log.py for testing.
    try:
        import parse_log as pl
    except ModuleNotFoundError:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("parse_log", Path(__file__).parent / "parse_log.py")
        pl = _ilu.module_from_spec(_spec)  # type: ignore[assignment]
        _spec.loader.exec_module(pl)

    batch_api, core_api = _load_k8s()
    logs_dir = results_dir / "logs"
    runs_dir = results_dir / "runs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures = 0

    for i, run in enumerate(runs, 1):
        run_id = make_run_id(run)
        json_path = runs_dir / f"{run_id}.json"
        log_path = logs_dir / f"{run_id}.log"
        namespace = run.get("namespace", NAMESPACE)

        print(f"\n[{i}/{len(runs)}] {run_id}", flush=True)

        if json_path.exists():
            try:
                existing = json.loads(json_path.read_text())
                if existing.get("skipped"):
                    print("  Re-running — previous result was a skipped placeholder.", flush=True)
                else:
                    print("  Skipping — result already exists.", flush=True)
                    continue
            except Exception:
                print("  Skipping — result already exists.", flush=True)
                continue

        try:
            _cleanup_stale_job(batch_api, run_id, namespace)
            _submit_sub_job(batch_api, run_id, run)
            pod_name = _wait_for_pod_ready(core_api, run_id, namespace)
            log_text = _fetch_pod_log(core_api, pod_name, namespace)
            log_path.write_text(log_text)
            print(f"  Log saved: {log_path}", flush=True)
        except Exception as e:
            print(f"  Run failed: {e}", flush=True)
            # Save failure log before the job is deleted in the finally block.
            # PodFailedError carries the pod name directly; other failures
            # (e.g. TimeoutError) require a pod list to find it.
            try:
                if isinstance(e, PodFailedError):
                    fail_pod = e.pod_name
                else:
                    pods = core_api.list_namespaced_pod(
                        namespace=namespace, label_selector=f"run-id={run_id}"
                    )
                    fail_pod = pods.items[0].metadata.name if pods.items else None
                if fail_pod:
                    # For terminated containers use previous=True to get the
                    # crashed container's output rather than an empty current log.
                    terminated = isinstance(e, PodFailedError)
                    try:
                        fail_log = _fetch_pod_log(core_api, fail_pod, namespace,
                                                  previous=terminated)
                    except Exception:
                        fail_log = _fetch_pod_log(core_api, fail_pod, namespace)
                    fail_path = logs_dir / f"{run_id}.FAILED.log"
                    fail_path.write_text(fail_log)
                    print(f"  Failure log saved: {fail_path}", flush=True)
                else:
                    print("  No pod found — failure log unavailable.", flush=True)
            except Exception as log_err:
                print(f"  Could not save failure log: {log_err}", flush=True)
            failures += 1
            continue
        finally:
            try:
                _delete_job(batch_api, run_id, namespace)
            except Exception as e:
                print(f"  Warning: could not delete Job: {e}", flush=True)

        try:
            parsed = pl.parse(log_path)
        except Exception as e:
            print(f"  Parse failed: {e}", flush=True)
            failures += 1
            continue

        try:
            record = {
                "model": run["model"],
                "gpu": run["gpu"],
                "vllm_args": {
                    "tensor_parallel_size": run["tp"],
                    "pipeline_parallel_size": run["pp"],
                    "data_parallel_size": run["dp"],
                    "max_model_len": run["max_model_len"],
                    "gpu_memory_utilization": float(run["gpu_memory_utilization"]),
                    "dtype": run.get("dtype", "auto"),
                    "quantization": run.get("quantization"),
                    "kv_cache_dtype": run.get("kv_cache_dtype", "auto"),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "log_path": str(log_path),
                **parsed,
                # planner_predicted and error_pct are added in a separate calibration step
            }
            if "_sweep_dim" in run:
                record["_sweep_dim"] = run["_sweep_dim"]

            json_path.write_text(json.dumps(record, indent=2))
            print(f"  JSON saved: {json_path}", flush=True)
            successes += 1
        except Exception as e:
            print(f"  Failed to save result: {e}", flush=True)
            failures += 1

    print(f"\nSweep complete. {successes} succeeded, {failures} failed. Results in {results_dir}", flush=True)


# ── Version extraction ────────────────────────────────────────────────────────

def _extract_version(vllm_image: str) -> str:
    """Parse the version tag from a vLLM image string.

    Examples:
        vllm/vllm-openai:v0.19.0  ->  v0.19.0
        myregistry/vllm:latest    ->  latest
        vllm-openai               ->  unknown
    """
    if ":" in vllm_image:
        return vllm_image.split(":")[-1]
    print(f"  Warning: no tag found in vllm_image '{vllm_image}', using 'unknown'", flush=True)
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/sweep/sweep.yaml")
    ap.add_argument("--results", default="/data/results/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    version = _extract_version(config.get("defaults", {}).get("vllm_image", ""))
    results_dir = Path(args.results) / version
    runs = expand_matrix(config)
    print(f"vLLM version: {version}", flush=True)
    print(f"Results dir:  {results_dir}", flush=True)
    print(f"Matrix expanded: {len(runs)} runs", flush=True)
    for r in runs:
        print(f"  {make_run_id(r)}", flush=True)

    if args.dry_run:
        print("Dry run complete — no Jobs submitted.")
        return

    run_sweep(runs, results_dir)


if __name__ == "__main__":
    main()
