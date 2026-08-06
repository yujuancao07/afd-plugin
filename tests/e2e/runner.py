#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run DeepSeekV2-Lite DeepEP and AFD GSM8K E2E scenarios."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tests.e2e.helpers_gsm8k import (
    _extract_gsm8k_accuracy,
    _extract_gsm8k_sample_count,
    _run_lm_eval,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ASYNC_AFD_CONNECTOR = "CAMAsyncAFDConnector"
PROCESS_TERMINATION_TIMEOUT_S = 20
PROCESS_POLL_INTERVAL_S = 0.2
PROCESS_REAP_TIMEOUT_S = 5
LOG_THREAD_JOIN_TIMEOUT_S = 2
GSM8K_SAMPLE_LIMIT = 7
GSM8K_THRESHOLD_ENV = "AFD_GSM8K_THRESHOLD"
GSM8K_TOLERANCE_ENV = "AFD_GSM8K_TOLERANCE"
DEFAULT_GSM8K_THRESHOLD = 0.20
DEFAULT_GSM8K_TOLERANCE = 0.05


def main() -> int:
    args = parse_args()
    configure_scenario(args)
    attention_devices = parse_csv(args.attention_devices)
    ffn_devices = parse_csv(args.ffn_devices)
    validate_topology(args, attention_devices, ffn_devices)

    processes: list[subprocess.Popen[str]] = []
    log_threads: list[threading.Thread] = []
    handled_signals = (signal.SIGTERM, signal.SIGINT)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in handled_signals
    }
    received_signal: int | None = None

    def exit_after_cleanup(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        if received_signal is not None:
            return
        received_signal = signum
        raise SystemExit(128 + signum)

    for signum in handled_signals:
        signal.signal(signum, exit_after_cleanup)

    try:
        if args.baseline:
            role_devices = {"baseline": attention_devices}
            launch_order = (("baseline", "BASELINE"),)
        else:
            role_devices = {
                "attention": attention_devices,
                "ffn": ffn_devices,
            }
            launch_order = (
                ("attention", "ATTN"),
                ("ffn", "FFN"),
            )
            if not uses_async_connector(args):
                launch_order = tuple(reversed(launch_order))

        for role, label in launch_order:
            command = (
                build_baseline_command(args)
                if role == "baseline"
                else build_vllm_command(args, role=role)
            )
            visible_devices = ",".join(role_devices[role])
            process_env = build_env(visible_devices, args, role=role)
            print_command(
                label,
                command,
                args.device_backend,
                visible_devices,
            )
            process = start_process(
                role,
                command,
                process_env,
            )
            processes.append(process)
            log_threads.append(stream_output(role, process))
            ensure_alive(process, f"{label} process exited during startup")

        wait_for_openai_api(args, processes)
        ensure_processes_alive(processes)

        run_gsm8k_evaluation(args)

        ensure_processes_alive(processes)
        return 0
    finally:
        try:
            terminate_processes(processes)
        finally:
            try:
                for thread in log_threads:
                    thread.join(timeout=LOG_THREAD_JOIN_TIMEOUT_S)
            finally:
                for signum, previous_handler in previous_handlers.items():
                    signal.signal(signum, previous_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a manual DeepSeekV2 AFD E2E smoke test.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--scenario",
        choices=["baseline-graph", "afd-eager", "afd-graph", "afd-graph-dbo"],
        required=True,
        help="Fixed E2E scenario to run.",
    )
    parser.add_argument(
        "--gsm8k-output-path",
        required=True,
        help="Directory or file path where lm-eval writes GSM8K results.",
    )
    parser.add_argument(
        "--vllm-bin",
        default="vllm",
        help="vLLM executable to run. Defaults to 'vllm'.",
    )
    parser.add_argument(
        "--num-attention-ranks",
        dest="num_attention_ranks",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num-ffn-ranks",
        dest="num_ffn_ranks",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--attention-devices",
        default="0",
        help=(
            "Comma-separated device IDs for the Attention serve process. "
            "The number of devices must match Attention DP size."
        ),
    )
    parser.add_argument(
        "--ffn-devices",
        default="1",
        help=(
            "Comma-separated device IDs for the FFN serve process. "
            "The number of devices must match FFN DP size."
        ),
    )
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port-base", type=int, default=8000)
    parser.add_argument("--afd-host", default="127.0.0.1")
    parser.add_argument("--afd-port", type=int, default=1239)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--prompt", default="San Francisco is a")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help=(
            "Number of completion requests to send. Defaults to the number of "
            "Attention ranks."
        ),
    )
    parser.add_argument(
        "--request-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent completion requests. Defaults to --num-requests.",
    )
    parser.add_argument(
        "--served-model-name-prefix",
        default="deepseek-v2-lite-afd",
        help="Prefix used for role-specific served model names.",
    )
    parser.add_argument(
        "--cuda-graph-full-decode-only",
        action="store_true",
        help="Run without --enforce-eager and set cudagraph_mode=FULL_DECODE_ONLY.",
    )
    parser.add_argument(
        "--cudagraph-capture-size",
        type=int,
        default=64,
        help=(
            "Capture size used for max-num-seqs, max-num-batched-tokens, "
            "max-cudagraph-capture-size, and cudagraph-capture-sizes."
        ),
    )
    parser.add_argument(
        "--enable-dbo",
        action="store_true",
        help="Enable vLLM DBO/ubatching for both AFD roles.",
    )
    parser.add_argument(
        "--dbo-decode-token-threshold",
        type=int,
        default=1,
        help="Value passed to --dbo-decode-token-threshold when DBO is enabled.",
    )
    parser.add_argument(
        "--dbo-prefill-token-threshold",
        type=int,
        default=None,
        help=(
            "Value passed to --dbo-prefill-token-threshold when DBO is enabled. "
            "Defaults to --cudagraph-capture-size."
        ),
    )
    parser.add_argument(
        "--use-decode-bench-connector",
        action="store_true",
        help="Pass an AFDDecodeBenchConnector kv-transfer-config to Attention.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size. Defaults to 1.",
    )
    parser.add_argument(
        "--attention-tp-size",
        type=int,
        default=None,
        help="Attention tensor parallelism size. Defaults to --tp-size.",
    )
    parser.add_argument(
        "--ffn-tp-size",
        type=int,
        default=None,
        help="FFN tensor parallelism size. Defaults to --tp-size.",
    )
    parser.add_argument(
        "--afd-connector",
        default=None,
        help=(
            "AFD connector name. Defaults to P2pNcclAFDConnector for GPU and "
            "CAMP2pAFDConnector for NPU."
        ),
    )
    parser.add_argument(
        "--afd-async",
        action="store_true",
        help="Set additional_config['afd']['async']=true.",
    )
    parser.add_argument(
        "--compute-gate-on-attention",
        action="store_true",
        help="Set additional_config['afd']['compute_gate_on_attention']=true.",
    )
    parser.add_argument(
        "--afd-connector-extra-config",
        action="append",
        default=[],
        help=(
            "JSON object merged into "
            "additional_config['afd']['connector_extra_config']."
        ),
    )
    parser.add_argument(
        "--expect-text",
        default=None,
        help="If set, assert that every completion response contains this text.",
    )
    parser.add_argument(
        "--device-backend",
        choices=["gpu", "npu"],
        default="gpu",
        help="Device backend. 'gpu' uses CUDA workers, 'npu' uses Ascend workers.",
    )
    parser.add_argument(
        "--common-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added to all processes.",
    )
    parser.add_argument(
        "--attention-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added only to Attention processes.",
    )
    parser.add_argument(
        "--ffn-vllm-arg",
        action="append",
        default=[],
        help="Extra single-token vLLM arg added only to FFN processes.",
    )
    return parser.parse_args()


def configure_scenario(args: argparse.Namespace) -> None:
    """Overwrite topology and feature flags for the selected fixed scenario."""
    scenario_settings = {
        "baseline-graph": (True, True, False, 1, 0),
        "afd-eager": (False, False, False, 2, 1),
        "afd-graph": (False, True, False, 2, 1),
        "afd-graph-dbo": (False, True, True, 2, 1),
    }
    baseline, use_graph, enable_dbo, attention_ranks, ffn_ranks = (
        scenario_settings[args.scenario]
    )
    args.baseline = baseline
    args.cuda_graph_full_decode_only = use_graph
    args.enable_dbo = enable_dbo
    args.num_attention_ranks = attention_ranks
    args.num_ffn_ranks = ffn_ranks
    args.tp_size = 1
    args.attention_tp_size = 1
    args.ffn_tp_size = 1
    if use_graph:
        args.cudagraph_capture_size = 8
    if enable_dbo:
        args.dbo_decode_token_threshold = 1
        args.dbo_prefill_token_threshold = 8


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_topology(
    args: argparse.Namespace,
    attention_devices: list[str],
    ffn_devices: list[str],
) -> None:
    if len(attention_devices) != len(set(attention_devices)):
        raise ValueError("Attention devices must be unique")
    if not args.baseline and len(ffn_devices) != len(set(ffn_devices)):
        raise ValueError("FFN devices must be unique")
    if not args.baseline and set(attention_devices) & set(ffn_devices):
        raise ValueError("Attention and FFN devices must not overlap")
    if args.num_attention_ranks != len(attention_devices):
        raise ValueError(
            "--num-attention-ranks must match the number of --attention-devices",
        )
    if not args.baseline and args.num_ffn_ranks != len(ffn_devices):
        raise ValueError("--num-ffn-ranks must match the number of --ffn-devices")
    if args.baseline:
        if args.num_attention_ranks != 1 or args.num_ffn_ranks != 0:
            raise ValueError(
                "baseline E2E requires one Attention rank and no FFN ranks",
            )
        if role_tp_size(args, "attention") != 1:
            raise ValueError("baseline E2E requires Attention TP=1")
        return
    if args.num_attention_ranks != 2 or args.num_ffn_ranks != 1:
        raise ValueError("AFD E2E requires two Attention ranks and one FFN rank")
    for role, rank_count in (
        ("attention", args.num_attention_ranks),
        ("ffn", args.num_ffn_ranks),
    ):
        tp_size = role_tp_size(args, role)
        if tp_size < 1:
            raise ValueError(f"{role} TP size must be positive")
        if rank_count % tp_size != 0:
            raise ValueError(
                f"{role} rank count must be divisible by TP size "
                f"(ranks={rank_count}, tp={tp_size})",
            )


def build_baseline_command(args: argparse.Namespace) -> list[str]:
    """Build the native single-process baseline command without AFD config."""
    if any(
        arg == "--additional-config" or arg.startswith("--additional-config=")
        for arg in args.common_vllm_arg
    ):
        raise ValueError(
            "baseline --common-vllm-arg cannot inject --additional-config",
        )
    cmd = [
        args.vllm_bin,
        "serve",
        args.model,
        "--served-model-name",
        served_model_name(args, "baseline"),
        "--data-parallel-size",
        "1",
        "--tensor-parallel-size",
        "1",
        "--enable-expert-parallel",
    ]
    if args.cuda_graph_full_decode_only:
        capture_size = str(args.cudagraph_capture_size)
        cmd.extend(
            [
                "--max-num-seqs",
                capture_size,
                "--max-num-batched-tokens",
                capture_size,
                "--max-cudagraph-capture-size",
                capture_size,
                "--cudagraph-capture-sizes",
                capture_size,
                "--compilation-config",
                json.dumps(
                    {"cudagraph_mode": "FULL_DECODE_ONLY"},
                    separators=(",", ":"),
                ),
            ],
        )
    else:
        cmd.append("--enforce-eager")
    cmd.extend(["--host", args.api_host, "--port", str(attention_api_port(args))])
    cmd.extend(args.common_vllm_arg)
    return cmd


def build_vllm_command(
    args: argparse.Namespace,
    *,
    role: str,
) -> list[str]:
    tp_size = role_tp_size(args, role)
    role_total_ranks = (
        args.num_attention_ranks if role == "attention" else args.num_ffn_ranks
    )
    role_dp_size = max(1, role_total_ranks // tp_size)
    is_npu = args.device_backend == "npu"
    connector = args.afd_connector or (
        "CAMP2pAFDConnector" if is_npu else "P2pNcclAFDConnector"
    )

    afd_config = {
        "afd": {
            "role": role,
            "connector": connector,
            "host": args.afd_host,
            "port": args.afd_port,
            "num_attention_ranks": args.num_attention_ranks,
            "num_ffn_ranks": args.num_ffn_ranks,
        },
    }
    if args.afd_async:
        afd_config["afd"]["async"] = True
    if args.compute_gate_on_attention:
        afd_config["afd"]["compute_gate_on_attention"] = True
    connector_extra_config = parse_afd_connector_extra_config(
        args.afd_connector_extra_config,
    )
    if connector_extra_config:
        afd_config["afd"]["connector_extra_config"] = connector_extra_config
    cmd = [
        args.vllm_bin,
        "serve",
        args.model,
        "--served-model-name",
        served_model_name(args, role),
        "--data-parallel-size",
        str(role_dp_size),
        "--tensor-parallel-size",
        str(tp_size),
        "--enable-expert-parallel",
        "--additional-config",
        json.dumps(afd_config, separators=(",", ":")),
    ]
    if args.cuda_graph_full_decode_only:
        capture_size = str(args.cudagraph_capture_size)
        cmd.extend(
            [
                "--max-num-seqs",
                capture_size,
                "--max-num-batched-tokens",
                capture_size,
                "--max-cudagraph-capture-size",
                capture_size,
                "--cudagraph-capture-sizes",
                capture_size,
                "--compilation-config",
                json.dumps(
                    {"cudagraph_mode": "FULL_DECODE_ONLY"},
                    separators=(",", ":"),
                ),
            ],
        )
    else:
        cmd.append("--enforce-eager")

    if args.enable_dbo:
        prefill_threshold = (
            args.dbo_prefill_token_threshold
            if args.dbo_prefill_token_threshold is not None
            else args.cudagraph_capture_size
        )
        cmd.extend(
            [
                "--enable-dbo",
                "--dbo-decode-token-threshold",
                str(args.dbo_decode_token_threshold),
                "--dbo-prefill-token-threshold",
                str(prefill_threshold),
            ],
        )

    if role == "attention":
        cmd.extend(
            ["--host", args.api_host, "--port", str(attention_api_port(args))],
        )
        if args.use_decode_bench_connector:
            cmd.extend(["--kv-transfer-config", decode_bench_connector_config()])
        cmd.extend(args.attention_vllm_arg)
    else:
        cmd.extend(
            ["--host", args.api_host, "--port", str(ffn_api_port(args))],
        )
        cmd.extend(args.ffn_vllm_arg)
    cmd.extend(args.common_vllm_arg)
    return cmd


def role_tp_size(args: argparse.Namespace, role: str) -> int:
    if role == "attention":
        return args.attention_tp_size or args.tp_size
    if role == "ffn":
        return args.ffn_tp_size or args.tp_size
    raise ValueError(f"unknown AFD role {role!r}")


def parse_afd_connector_extra_config(values: list[str]) -> dict[str, Any]:
    connector_extra_config: dict[str, Any] = {}
    for raw_value in values:
        value = json.loads(raw_value)
        if not isinstance(value, dict):
            raise ValueError("--afd-connector-extra-config must be a JSON object")
        connector_extra_config.update(value)
    return connector_extra_config


def uses_async_connector(args: argparse.Namespace) -> bool:
    return args.afd_connector == ASYNC_AFD_CONNECTOR


def decode_bench_connector_config() -> str:
    return json.dumps(
        {
            "kv_connector": "AFDDecodeBenchConnector",
            "kv_connector_module_path": "tools.benchmarks.decode_bench",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "fill_mean": 0.015,
                "fill_std": 0.0,
            },
        },
        separators=(",", ":"),
    )


def served_model_name(args: argparse.Namespace, role: str) -> str:
    return f"{args.served_model_name_prefix}-{role}"


def attention_api_port(args: argparse.Namespace) -> int:
    return args.api_port_base


def ffn_api_port(args: argparse.Namespace) -> int:
    return args.api_port_base + 1


def run_gsm8k_evaluation(args: argparse.Namespace) -> None:
    """Run the fixed GSM8K workload against the scenario's public API."""
    role = "baseline" if args.baseline else "attention"
    results = _run_lm_eval(
        f"http://{args.api_host}:{attention_api_port(args)}",
        served_model_name(args, role),
        output_path=args.gsm8k_output_path,
        tokenizer=args.model,
        limit=GSM8K_SAMPLE_LIMIT,
    )
    sample_count = _extract_gsm8k_sample_count(results)
    assert sample_count == GSM8K_SAMPLE_LIMIT, (
        f"GSM8K evaluated {sample_count} samples; expected {GSM8K_SAMPLE_LIMIT}"
    )
    minimum_accuracy = float(
        os.environ.get(GSM8K_THRESHOLD_ENV, str(DEFAULT_GSM8K_THRESHOLD)),
    ) - float(os.environ.get(GSM8K_TOLERANCE_ENV, str(DEFAULT_GSM8K_TOLERANCE)))
    accuracy = _extract_gsm8k_accuracy(results)
    assert accuracy >= minimum_accuracy, (
        f"GSM8K accuracy {accuracy:.4f} is below the required "
        f"threshold {minimum_accuracy:.4f}"
    )


def build_env(
    visible_devices: str,
    args: argparse.Namespace,
    *,
    role: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "18000")
    env[visible_devices_env_name(args.device_backend)] = visible_devices
    if args.device_backend != "npu":
        env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    if args.baseline:
        env["VLLM_PLUGINS"] = "ascend" if args.device_backend == "npu" else ""
    else:
        env["VLLM_PLUGINS"] = (
            "ascend,afd" if args.device_backend == "npu" else "afd"
        )
    env["PYTHONUNBUFFERED"] = "1"
    if (
        args.device_backend == "npu"
        and role in ("attention", "ffn")
        and role_tp_size(args, role) <= 1
    ):
        env.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)
    env.pop("AFD_PLUGIN_EARLY_ENGINE_PATCH", None)
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not current_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{current_pythonpath}"
    )
    return env


def start_process(
    name: str,
    command: list[str],
    env: dict[str, str],
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )


def stream_output(
    name: str,
    process: subprocess.Popen[str],
) -> threading.Thread:
    def worker() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{name}] {line}", end="")

    thread = threading.Thread(target=worker, name=f"{name}-log-stream", daemon=True)
    thread.start()
    return thread


def wait_for_openai_api(
    args: argparse.Namespace,
    processes: list[subprocess.Popen[str]],
) -> None:
    deadline = time.monotonic() + args.startup_timeout
    url = f"http://{args.api_host}:{attention_api_port(args)}/v1/models"
    last_error: BaseException | None = None

    while time.monotonic() < deadline:
        for process in processes:
            returncode = process.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"vLLM process exited before Attention API was ready "
                    f"(returncode={returncode}, command={process.args!r})",
                )
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print(f"\nAttention API is ready at {url}")
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(2)

    raise TimeoutError(
        f"Timed out waiting for Attention API at {url}; last error={last_error!r}",
    )


def request_completion(args: argparse.Namespace) -> dict[str, Any]:
    url = f"http://{args.api_host}:{attention_api_port(args)}/v1/completions"
    payload = {
        "model": served_model_name(args, "attention"),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = read_http_error_body(exc)
        raise RuntimeError(
            f"Completion request to {url} failed with HTTP {exc.code} "
            f"{exc.reason}: {body}",
        ) from exc
    return json.loads(body)


def read_http_error_body(error: urllib.error.HTTPError) -> str:
    body = error.read().decode("utf-8", errors="replace")
    return body if body else "<empty response body>"


def request_completions(args: argparse.Namespace) -> list[dict[str, Any]]:
    request_count = (
        int(args.num_requests)
        if args.num_requests is not None
        else max(int(args.num_attention_ranks), 1)
    )
    if request_count == 1:
        return [request_completion(args)]

    responses: list[dict[str, Any] | None] = [None] * request_count
    concurrency = (
        int(args.request_concurrency)
        if args.request_concurrency is not None
        else request_count
    )
    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as executor:
        futures = {
            executor.submit(request_completion, args): request_idx
            for request_idx in range(request_count)
        }
        failures: list[tuple[int, BaseException]] = []
        for future in as_completed(futures):
            request_idx = futures[future]
            try:
                responses[request_idx] = future.result()
            except Exception as exc:
                failures.append((request_idx, exc))

    if failures:
        failure_messages = "; ".join(
            f"request {request_idx}: {failure}" for request_idx, failure in failures
        )
        raise RuntimeError(f"Completion requests failed: {failure_messages}")

    return [response for response in responses if response is not None]


def ensure_alive(process: subprocess.Popen[str], message: str) -> None:
    returncode = process.poll()
    if returncode is not None:
        raise RuntimeError(f"{message} (returncode={returncode})")


def ensure_processes_alive(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"vLLM process exited unexpectedly (returncode={returncode})",
            )


def terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    failures: list[str] = []
    live_pgids: list[int] = []

    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError as exc:
            failures.append(
                f"SIGTERM failed for process group {process.pid}: {exc}",
            )
        live_pgids.append(process.pid)

    deadline = time.monotonic() + PROCESS_TERMINATION_TIMEOUT_S
    leaders_to_poll = list(processes)
    while live_pgids:
        unreaped_leaders: list[subprocess.Popen[str]] = []
        for process in leaders_to_poll:
            try:
                returncode = process.poll()
            except Exception as exc:
                failures.append(f"poll failed for process {process.pid}: {exc}")
                continue
            if returncode is None:
                unreaped_leaders.append(process)
        leaders_to_poll = unreaped_leaders

        surviving_pgids: list[int] = []
        for pgid in live_pgids:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                continue
            except OSError as exc:
                failures.append(
                    f"liveness check failed for process group {pgid}: {exc}",
                )
            surviving_pgids.append(pgid)
        live_pgids = surviving_pgids
        if not live_pgids or time.monotonic() >= deadline:
            break
        time.sleep(PROCESS_POLL_INTERVAL_S)

    for pgid in live_pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except OSError as exc:
            failures.append(
                f"SIGKILL failed for process group {pgid}: {exc}",
            )

    for process in processes:
        try:
            process.wait(timeout=PROCESS_REAP_TIMEOUT_S)
        except Exception as exc:
            failures.append(f"wait failed for process {process.pid}: {exc}")
            continue
        if process.poll() is None:
            failures.append(f"process {process.pid} could not be reaped")

    if failures:
        raise RuntimeError("; ".join(failures))


def visible_devices_env_name(device_backend: str) -> str:
    return (
        "ASCEND_RT_VISIBLE_DEVICES"
        if device_backend == "npu"
        else "CUDA_VISIBLE_DEVICES"
    )


def print_command(
    name: str,
    command: list[str],
    device_backend: str,
    visible_devices: str,
) -> None:
    printable = " ".join(shell_quote(token) for token in command)
    env_name = visible_devices_env_name(device_backend)
    print(f"\n=== Starting {name} ({env_name}={visible_devices}) ===")
    print(printable)


def shell_quote(value: str) -> str:
    if value and all(char.isalnum() or char in "@%_+=:,./-" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    sys.exit(main())
