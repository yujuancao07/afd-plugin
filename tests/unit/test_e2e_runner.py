from __future__ import annotations

import argparse
import importlib
import io
import json
import signal
import subprocess
import sys
import threading
import time
import urllib.error

import pytest

from tests.e2e import helpers_gsm8k, runner

E2E_ENTRYPOINT_ENV_VARS = (
    "AFD_E2E_BACKEND",
    "AFD_E2E_DEVICES",
    "AFD_GPU_E2E_MODEL",
    "AFD_GPU_E2E_VLLM_BIN",
    "AFD_NPU_E2E_MODEL",
    "AFD_NPU_E2E_VLLM_BIN",
)


def _e2e_entrypoint():
    return importlib.import_module("tests.e2e.test_deepseek_v2_lite")


def _set_e2e_entrypoint_env(monkeypatch, backend: str) -> None:
    for name in E2E_ENTRYPOINT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AFD_E2E_BACKEND", backend)
    if backend == "gpu":
        monkeypatch.setenv("AFD_E2E_DEVICES", "2, 4, 6")
        monkeypatch.setenv("AFD_GPU_E2E_MODEL", "/models/deepseek-v2-lite-gpu")
        monkeypatch.setenv("AFD_GPU_E2E_VLLM_BIN", "/opt/gpu/bin/vllm")
    elif backend == "npu":
        monkeypatch.setenv("AFD_E2E_DEVICES", "1, 3, 5")
        monkeypatch.setenv("AFD_NPU_E2E_MODEL", "/models/deepseek-v2-lite-npu")
        monkeypatch.setenv("AFD_NPU_E2E_VLLM_BIN", "/opt/npu/bin/vllm")


def test_e2e_entrypoint_builds_gpu_runner_command(monkeypatch, tmp_path):
    _set_e2e_entrypoint_env(monkeypatch, "gpu")
    output_path = tmp_path / "afd-graph"

    command = _e2e_entrypoint().build_runner_command("afd-graph", output_path)

    assert command == [
        sys.executable,
        "-m",
        "tests.e2e.runner",
        "--model",
        "/models/deepseek-v2-lite-gpu",
        "--vllm-bin",
        "/opt/gpu/bin/vllm",
        "--device-backend",
        "gpu",
        "--attention-devices",
        "2,4",
        "--ffn-devices",
        "6",
        "--scenario",
        "afd-graph",
        "--gsm8k-output-path",
        str(output_path),
    ]


def test_e2e_entrypoint_builds_npu_runner_command(monkeypatch, tmp_path):
    _set_e2e_entrypoint_env(monkeypatch, "npu")
    output_path = tmp_path / "afd-graph-dbo"

    command = _e2e_entrypoint().build_runner_command(
        "afd-graph-dbo",
        output_path,
    )

    assert command == [
        sys.executable,
        "-m",
        "tests.e2e.runner",
        "--model",
        "/models/deepseek-v2-lite-npu",
        "--vllm-bin",
        "/opt/npu/bin/vllm",
        "--device-backend",
        "npu",
        "--attention-devices",
        "1,3",
        "--ffn-devices",
        "5",
        "--scenario",
        "afd-graph-dbo",
        "--gsm8k-output-path",
        str(output_path),
    ]


@pytest.mark.parametrize(("backend", "expected_device"), [("gpu", "2"), ("npu", "1")])
def test_e2e_entrypoint_baseline_passes_one_attention_device(
    monkeypatch,
    tmp_path,
    backend,
    expected_device,
):
    _set_e2e_entrypoint_env(monkeypatch, backend)

    command = _e2e_entrypoint().build_runner_command(
        "baseline-graph",
        tmp_path / "baseline-graph",
    )

    assert command[command.index("--attention-devices") + 1] == expected_device


@pytest.mark.parametrize(
    ("backend", "missing_name"),
    [
        (None, "AFD_E2E_BACKEND"),
        ("gpu", "AFD_E2E_DEVICES"),
        ("gpu", "AFD_GPU_E2E_MODEL"),
        ("npu", "AFD_E2E_DEVICES"),
        ("npu", "AFD_NPU_E2E_MODEL"),
    ],
)
def test_e2e_entrypoint_rejects_missing_required_configuration(
    monkeypatch,
    tmp_path,
    backend,
    missing_name,
):
    _set_e2e_entrypoint_env(monkeypatch, backend or "")
    monkeypatch.delenv(missing_name, raising=False)

    with pytest.raises(RuntimeError, match=missing_name):
        _e2e_entrypoint().build_runner_command("afd-eager", tmp_path / "results")


def test_e2e_entrypoint_rejects_unknown_backend(monkeypatch, tmp_path):
    _set_e2e_entrypoint_env(monkeypatch, "tpu")

    with pytest.raises(RuntimeError, match="AFD_E2E_BACKEND"):
        _e2e_entrypoint().build_runner_command("afd-eager", tmp_path / "results")


@pytest.mark.parametrize(
    "backend",
    ["gpu", "npu"],
)
def test_e2e_entrypoint_rejects_wrong_device_count(
    monkeypatch,
    tmp_path,
    backend,
):
    _set_e2e_entrypoint_env(monkeypatch, backend)
    monkeypatch.setenv("AFD_E2E_DEVICES", "0,1")

    with pytest.raises(RuntimeError, match="AFD_E2E_DEVICES"):
        _e2e_entrypoint().build_runner_command("afd-eager", tmp_path / "results")


@pytest.mark.parametrize(
    "backend",
    ["gpu", "npu"],
)
def test_e2e_entrypoint_rejects_reused_devices(
    monkeypatch,
    tmp_path,
    backend,
):
    _set_e2e_entrypoint_env(monkeypatch, backend)
    monkeypatch.setenv("AFD_E2E_DEVICES", "0,0,2")

    with pytest.raises(RuntimeError, match="unique"):
        _e2e_entrypoint().build_runner_command("afd-eager", tmp_path / "results")


def test_e2e_entrypoint_forwards_cancellation_and_reaps_runner(monkeypatch):
    entrypoint = _e2e_entrypoint()
    monkeypatch.setattr(entrypoint, "os", runner.os, raising=False)
    monkeypatch.setattr(entrypoint, "signal", signal, raising=False)
    command = [sys.executable, "-m", "tests.e2e.runner"]
    previous_handlers = {
        signal.SIGTERM: object(),
        signal.SIGINT: object(),
    }
    installed_handlers = {}
    signal_calls = []
    kill_calls = []
    popen_calls = []

    class FakeProcess:
        pid = 321
        returncode = None
        wait_calls = []

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if len(self.wait_calls) == 1:
                installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
                pytest.fail("the first cancellation signal must unwind the call")
            if len(self.wait_calls) == 2:
                installed_handlers[signal.SIGINT](signal.SIGINT, None)
                raise subprocess.TimeoutExpired(command, timeout)
            self.returncode = -9
            return self.returncode

    process = FakeProcess()

    def fake_popen(actual_command, **kwargs):
        popen_calls.append((actual_command, kwargs))
        return process

    def fake_signal(signum, handler):
        signal_calls.append((signum, handler))
        installed_handlers[signum] = handler

    monkeypatch.setattr(entrypoint.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        entrypoint.signal,
        "getsignal",
        lambda signum: previous_handlers[signum],
    )
    monkeypatch.setattr(entrypoint.signal, "signal", fake_signal)
    monkeypatch.setattr(
        entrypoint.os,
        "killpg",
        lambda pid, sig: kill_calls.append((pid, sig)),
        raising=False,
    )
    monkeypatch.setattr(entrypoint.signal, "SIGKILL", 9, raising=False)

    with pytest.raises(SystemExit) as error:
        entrypoint._run_runner(command)

    assert error.value.code == 128 + signal.SIGTERM
    assert popen_calls == [
        (
            command,
            {"cwd": entrypoint.REPO_ROOT, "start_new_session": True},
        ),
    ]
    assert kill_calls == [(321, signal.SIGTERM), (321, 9)]
    assert process.wait_calls == [None, entrypoint.RUNNER_CLEANUP_TIMEOUT_S, None]
    assert signal_calls[-2:] == list(previous_handlers.items())


def test_e2e_entrypoint_fails_for_a_nonzero_runner_exit(monkeypatch):
    entrypoint = _e2e_entrypoint()
    monkeypatch.setattr(entrypoint, "signal", signal, raising=False)
    command = [sys.executable, "-m", "tests.e2e.runner"]

    class FailedProcess:
        pid = 321
        returncode = 17

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(
        entrypoint.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FailedProcess(),
    )
    monkeypatch.setattr(entrypoint.signal, "signal", lambda *_args: None)

    with pytest.raises(subprocess.CalledProcessError) as error:
        entrypoint._run_runner(command)

    assert error.value.returncode == 17


def test_e2e_entrypoint_reaps_runner_when_kill_signal_fails(monkeypatch):
    entrypoint = _e2e_entrypoint()
    monkeypatch.setattr(entrypoint, "os", runner.os, raising=False)
    monkeypatch.setattr(entrypoint, "signal", signal, raising=False)
    command = [sys.executable, "-m", "tests.e2e.runner"]
    installed_handlers = {}

    class FakeProcess:
        pid = 321
        returncode = None
        wait_calls = []

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if len(self.wait_calls) == 1:
                installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
            if len(self.wait_calls) == 2:
                raise subprocess.TimeoutExpired(command, timeout)
            self.returncode = -9
            return self.returncode

    process = FakeProcess()

    def fake_killpg(_pid, sig):
        if sig == signal.SIGKILL:
            raise OSError("kill failed")

    monkeypatch.setattr(
        entrypoint.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(entrypoint.signal, "getsignal", lambda _signum: None)
    monkeypatch.setattr(
        entrypoint.signal,
        "signal",
        lambda signum, handler: installed_handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(entrypoint.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(entrypoint.signal, "SIGKILL", 9, raising=False)

    with pytest.raises(OSError, match="kill failed"):
        entrypoint._run_runner(command)

    assert process.wait_calls == [None, entrypoint.RUNNER_CLEANUP_TIMEOUT_S, None]


def test_e2e_entrypoint_queues_a_signal_during_runner_spawn(monkeypatch):
    entrypoint = _e2e_entrypoint()
    command = [sys.executable, "-m", "tests.e2e.runner"]
    previous_handlers = {
        signal.SIGTERM: object(),
        signal.SIGINT: object(),
    }
    installed_handlers = {}
    signal_calls = []
    kill_calls = []

    class FakeProcess:
        pid = 321
        returncode = None
        wait_calls = []

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            self.returncode = 128 + signal.SIGTERM
            return self.returncode

    process = FakeProcess()

    def signal_during_popen(*_args, **_kwargs):
        installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
        return process

    def fake_signal(signum, handler):
        signal_calls.append((signum, handler))
        installed_handlers[signum] = handler

    monkeypatch.setattr(entrypoint.subprocess, "Popen", signal_during_popen)
    monkeypatch.setattr(
        entrypoint.signal,
        "getsignal",
        lambda signum: previous_handlers[signum],
    )
    monkeypatch.setattr(entrypoint.signal, "signal", fake_signal)
    monkeypatch.setattr(
        entrypoint.os,
        "killpg",
        lambda pid, sig: kill_calls.append((pid, sig)),
        raising=False,
    )

    with pytest.raises(SystemExit) as error:
        entrypoint._run_runner(command)

    assert error.value.code == 128 + signal.SIGTERM
    assert kill_calls == [(321, signal.SIGTERM)]
    assert process.wait_calls == [entrypoint.RUNNER_CLEANUP_TIMEOUT_S]
    assert signal_calls[-2:] == list(previous_handlers.items())


def test_e2e_entrypoint_keeps_the_first_signal_during_spawn_drain(monkeypatch):
    entrypoint = _e2e_entrypoint()
    command = [sys.executable, "-m", "tests.e2e.runner"]
    installed_handlers = {}
    kill_calls = []
    injected_second_signal = False

    class FakeProcess:
        pid = 321
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 128 + signal.SIGTERM
            return self.returncode

    process = FakeProcess()

    def signal_during_popen(*_args, **_kwargs):
        installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
        return process

    def trace_spawn_drain(frame, event, _arg):
        nonlocal injected_second_signal
        if (
            event != "line"
            or frame.f_code.co_name != "_run_runner"
            or injected_second_signal
        ):
            return trace_spawn_drain
        state = frame.f_locals
        process_is_registered = state.get("process") is not None
        round2_drain_window = (
            state.get("pending_signal", object()) is None
            and state.get("received_signal") is None
        )
        round3_drain_window = (
            state.get("received_signal") == signal.SIGTERM
            and state.get("forwarded") is False
        )
        if process_is_registered and (round2_drain_window or round3_drain_window):
            injected_second_signal = True
            installed_handlers[signal.SIGINT](signal.SIGINT, None)
        return trace_spawn_drain

    monkeypatch.setattr(entrypoint.subprocess, "Popen", signal_during_popen)
    monkeypatch.setattr(entrypoint.signal, "getsignal", lambda _signum: None)
    monkeypatch.setattr(
        entrypoint.signal,
        "signal",
        lambda signum, handler: installed_handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(
        entrypoint.os,
        "killpg",
        lambda pid, sig: kill_calls.append((pid, sig)),
        raising=False,
    )

    previous_trace = sys.gettrace()
    sys.settrace(trace_spawn_drain)
    try:
        with pytest.raises(SystemExit) as error:
            entrypoint._run_runner(command)
    finally:
        sys.settrace(previous_trace)

    assert injected_second_signal is True
    assert error.value.code == 128 + signal.SIGTERM
    assert kill_calls == [(321, signal.SIGTERM)]


def test_e2e_entrypoint_waits_for_the_full_nested_cleanup_budget(monkeypatch):
    entrypoint = _e2e_entrypoint()
    command = [sys.executable, "-m", "tests.e2e.runner"]
    nested_cleanup_bound_s = 80
    installed_handlers = {}
    kill_calls = []

    class SlowCleanupProcess:
        pid = 321
        returncode = None
        wait_calls = []

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            if len(self.wait_calls) == 1:
                installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
            if timeout is not None and timeout <= nested_cleanup_bound_s:
                raise subprocess.TimeoutExpired(command, timeout)
            self.returncode = 128 + signal.SIGTERM
            return self.returncode

    process = SlowCleanupProcess()

    monkeypatch.setattr(
        entrypoint.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(entrypoint.signal, "getsignal", lambda _signum: None)
    monkeypatch.setattr(
        entrypoint.signal,
        "signal",
        lambda signum, handler: installed_handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(
        entrypoint.os,
        "killpg",
        lambda pid, sig: kill_calls.append((pid, sig)),
        raising=False,
    )
    monkeypatch.setattr(entrypoint.signal, "SIGKILL", 9, raising=False)

    with pytest.raises(SystemExit) as error:
        entrypoint._run_runner(command)

    assert error.value.code == 128 + signal.SIGTERM
    assert kill_calls == [(321, signal.SIGTERM)]
    assert process.wait_calls == [None, entrypoint.RUNNER_CLEANUP_TIMEOUT_S]


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        model="deepseek-ai/DeepSeek-V2-Lite",
        vllm_bin="vllm",
        api_host="127.0.0.1",
        api_port_base=18100,
        afd_host="127.0.0.1",
        afd_port=1239,
        served_model_name_prefix="deepseek-v2-lite-afd",
        prompt="San Francisco is a",
        max_tokens=16,
        temperature=0.0,
        num_requests=None,
        request_concurrency=None,
        num_attention_ranks=2,
        num_ffn_ranks=1,
        attention_devices="0,1",
        ffn_devices="2",
        device_backend="gpu",
        tp_size=1,
        attention_tp_size=None,
        ffn_tp_size=None,
        scenario="afd-eager",
        baseline=False,
        cuda_graph_full_decode_only=False,
        cudagraph_capture_size=64,
        enable_dbo=False,
        dbo_decode_token_threshold=1,
        dbo_prefill_token_threshold=None,
        afd_connector=None,
        afd_async=False,
        compute_gate_on_attention=False,
        afd_connector_extra_config=[],
        use_decode_bench_connector=False,
        common_vllm_arg=[],
        attention_vllm_arg=[],
        ffn_vllm_arg=[],
        gsm8k_output_path="/tmp/gsm8k-results",
    )


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("baseline-graph", (True, True, False, 1, 0, 1)),
        ("afd-eager", (False, False, False, 2, 1, 1)),
        ("afd-graph", (False, True, False, 2, 1, 1)),
        ("afd-graph-dbo", (False, True, True, 2, 1, 1)),
    ],
)
def test_configure_scenario_overwrites_fixed_topology_and_features(
    scenario,
    expected,
):
    args = _args()
    args.scenario = scenario
    args.num_attention_ranks = 99
    args.num_ffn_ranks = 99
    args.tp_size = 99
    args.cuda_graph_full_decode_only = False
    args.enable_dbo = False

    runner.configure_scenario(args)

    assert (
        args.baseline,
        args.cuda_graph_full_decode_only,
        args.enable_dbo,
        args.num_attention_ranks,
        args.num_ffn_ranks,
        args.tp_size,
    ) == expected
    assert args.attention_tp_size == 1
    assert args.ffn_tp_size == 1
    if args.cuda_graph_full_decode_only:
        assert args.cudagraph_capture_size == 8
    if args.enable_dbo:
        assert args.dbo_decode_token_threshold == 1
        assert args.dbo_prefill_token_threshold == 8


def test_build_baseline_command_uses_native_single_process_graph_server():
    args = _args()
    args.scenario = "baseline-graph"
    runner.configure_scenario(args)

    command = runner.build_baseline_command(args)

    assert "--additional-config" not in command
    assert command[command.index("--data-parallel-size") + 1] == "1"
    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    assert json.loads(command[command.index("--compilation-config") + 1]) == {
        "cudagraph_mode": "FULL_DECODE_ONLY",
    }


@pytest.mark.parametrize(
    "passthrough_arg",
    ["--additional-config", '--additional-config={"afd":{}}'],
)
def test_build_baseline_command_rejects_additional_config_passthrough(
    passthrough_arg,
):
    args = _args()
    args.scenario = "baseline-graph"
    args.common_vllm_arg = [passthrough_arg]
    runner.configure_scenario(args)

    with pytest.raises(ValueError, match="--additional-config"):
        runner.build_baseline_command(args)


def test_validate_topology_accepts_baseline_without_ffn_ranks():
    args = _args()
    args.scenario = "baseline-graph"
    runner.configure_scenario(args)

    runner.validate_topology(args, ["0"], [])


@pytest.mark.parametrize(
    ("attention_devices", "ffn_devices", "error_message"),
    [
        (["0", "0"], ["2"], "Attention devices must be unique"),
        (["0", "1"], ["2", "2"], "FFN devices must be unique"),
        (["0", "1"], ["1"], "Attention and FFN devices must not overlap"),
    ],
)
def test_validate_topology_rejects_reused_devices(
    attention_devices,
    ffn_devices,
    error_message,
):
    args = _args()

    with pytest.raises(ValueError, match=error_message):
        runner.validate_topology(args, attention_devices, ffn_devices)


@pytest.mark.parametrize(
    ("scenario", "backend", "expected_plugins"),
    [
        ("baseline-graph", "gpu", ""),
        ("baseline-graph", "npu", "ascend"),
        ("afd-eager", "gpu", "afd"),
        ("afd-eager", "npu", "ascend,afd"),
    ],
)
def test_build_env_uses_the_scenario_plugin_allowlist(
    scenario,
    backend,
    expected_plugins,
):
    args = _args()
    args.scenario = scenario
    args.device_backend = backend
    runner.configure_scenario(args)

    env = runner.build_env("0", args, role="baseline" if args.baseline else "attention")

    assert env["VLLM_PLUGINS"] == expected_plugins


@pytest.mark.parametrize(
    ("backend", "visible_devices_env"),
    [
        ("gpu", "CUDA_VISIBLE_DEVICES"),
        ("npu", "ASCEND_RT_VISIBLE_DEVICES"),
    ],
)
def test_print_command_reports_the_actual_visible_devices_environment(
    backend,
    visible_devices_env,
    capsys,
):
    runner.print_command("ATTN", ["vllm", "serve"], backend, "0,1")

    assert f"({visible_devices_env}=0,1)" in capsys.readouterr().out


def test_extract_gsm8k_sample_count_returns_effective_count():
    assert helpers_gsm8k._extract_gsm8k_sample_count(
        {"n-samples": {"gsm8k": {"effective": 7}}},
    ) == 7


@pytest.mark.parametrize(
    "results",
    [
        {},
        {"n-samples": {"gsm8k": {"effective": "not-a-number"}}},
        {"n-samples": {"gsm8k": {"effective": 7.5}}},
    ],
)
def test_extract_gsm8k_sample_count_rejects_missing_or_malformed_results(results):
    with pytest.raises((KeyError, ValueError), match="GSM8K sample count"):
        helpers_gsm8k._extract_gsm8k_sample_count(results)


def test_run_lm_eval_uses_builtin_gsm8k_and_inherits_hf_home(
    monkeypatch,
    tmp_path,
):
    popen_calls = []
    hf_home = str(tmp_path / "huggingface")
    monkeypatch.setenv("HF_HOME", hf_home)

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        raise RuntimeError("stop after inspecting lm-eval invocation")

    monkeypatch.setattr(helpers_gsm8k.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="stop after inspecting"):
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000",
            "model",
            output_path=str(tmp_path / "results"),
        )

    command, kwargs = popen_calls[0]
    assert command[command.index("--tasks") + 1] == "gsm8k"
    assert "--include_path" not in command
    assert kwargs["env"]["HF_HOME"] == hf_home


def test_run_lm_eval_timeout_cleans_its_group_reaps_and_joins(monkeypatch, tmp_path):
    popen_calls = []
    group_alive = True
    group_signal_calls = []
    reader_join_calls = []

    class FakeProcess:
        pid = 404
        stdout = io.StringIO("")
        returncode = None
        direct_kill_calls = 0
        wait_calls = []

        def kill(self):
            self.direct_kill_calls += 1

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            self.returncode = -9
            return self.returncode

    class FakeReader:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout=None):
            reader_join_calls.append(timeout)

        def is_alive(self):
            return False

    process = FakeProcess()

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return process

    def fake_killpg(pid, sig):
        nonlocal group_alive
        group_signal_calls.append((pid, sig))
        if sig == 0 and not group_alive:
            raise ProcessLookupError
        if sig == signal.SIGKILL:
            group_alive = False

    monotonic_values = iter([100.0, 7301.0, 8000.0, 8021.0])
    monkeypatch.setattr(helpers_gsm8k.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(helpers_gsm8k.threading, "Thread", FakeReader)
    monkeypatch.setattr(
        helpers_gsm8k.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(helpers_gsm8k, "signal", signal, raising=False)
    monkeypatch.setattr(helpers_gsm8k.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(helpers_gsm8k.signal, "SIGKILL", 9, raising=False)

    with pytest.raises(TimeoutError, match="lm-eval exceeded"):
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000",
            "model",
            output_path=str(tmp_path / "results"),
        )

    assert popen_calls[0][1]["start_new_session"] is True
    assert process.direct_kill_calls == 0
    assert group_signal_calls == [
        (404, signal.SIGTERM),
        (404, 0),
        (404, 9),
    ]
    assert process.wait_calls == [helpers_gsm8k.LM_EVAL_REAP_TIMEOUT_S]
    assert reader_join_calls == [helpers_gsm8k.LM_EVAL_READER_JOIN_TIMEOUT_S]


def test_run_lm_eval_rejects_a_reader_that_does_not_join(monkeypatch, tmp_path):
    reader_join_calls = []

    class CompletedProcess:
        pid = 404
        stdout = io.StringIO(
            '{"results":{"gsm8k":{"exact_match":1.0}}}\n',
        )
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    class StuckReader:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout=None):
            reader_join_calls.append(timeout)

        def is_alive(self):
            return True

    monkeypatch.setattr(
        helpers_gsm8k.subprocess,
        "Popen",
        lambda *_args, **_kwargs: CompletedProcess(),
    )
    monkeypatch.setattr(helpers_gsm8k.threading, "Thread", StuckReader)
    monkeypatch.setattr(helpers_gsm8k.time, "monotonic", lambda: 100.0)

    with pytest.raises(RuntimeError, match="reader thread"):
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000",
            "model",
            output_path=str(tmp_path / "results"),
        )

    assert reader_join_calls == [helpers_gsm8k.LM_EVAL_READER_JOIN_TIMEOUT_S]


def test_run_lm_eval_proxies_a_constructor_signal_through_cleanup(
    monkeypatch,
    tmp_path,
):
    events = []
    mask_calls = []
    installed_handlers = {}
    group_alive = True
    popen_kwargs = {}

    def previous_term_handler(signum, _frame):
        events.append(("delegate", signum))
        raise SystemExit(128 + signum)

    def previous_int_handler(signum, _frame):
        events.append(("delegate", signum))
        raise SystemExit(128 + signum)

    previous_handlers = {
        signal.SIGTERM: previous_term_handler,
        signal.SIGINT: previous_int_handler,
    }

    class FakeProcess:
        pid = 404
        stdout = io.StringIO("")
        returncode = None

        def wait(self, timeout=None):
            events.append("wait")
            self.returncode = -signal.SIGTERM
            return self.returncode

    process = FakeProcess()

    def fake_popen(*_args, **kwargs):
        events.append(("spawn", None))
        popen_kwargs.update(kwargs)
        installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
        return process

    def fake_signal(signum, handler):
        installed_handlers[signum] = handler
        action = "restore" if handler is previous_handlers[signum] else "install"
        events.append((action, signum))

    def fake_killpg(_pid, sig):
        nonlocal group_alive
        if sig == signal.SIGTERM:
            events.append(("cleanup-term", sig))
            events.append(("second-signal", signal.SIGINT))
            installed_handlers[signal.SIGINT](signal.SIGINT, None)
            group_alive = False
        elif sig == 0:
            events.append(("probe", sig))
            if not group_alive:
                raise ProcessLookupError

    monkeypatch.setattr(
        helpers_gsm8k,
        "POSIX_SIGNAL_MASK_SUPPORTED",
        True,
        raising=False,
    )
    monkeypatch.setattr(
        helpers_gsm8k.signal,
        "pthread_sigmask",
        lambda *args: mask_calls.append(args),
        raising=False,
    )
    monkeypatch.setattr(
        helpers_gsm8k.signal,
        "getsignal",
        lambda signum: previous_handlers[signum],
    )
    monkeypatch.setattr(helpers_gsm8k.signal, "signal", fake_signal)
    monkeypatch.setattr(helpers_gsm8k.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(helpers_gsm8k.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(helpers_gsm8k.time, "monotonic", lambda: 100.0)

    with pytest.raises(SystemExit) as error:
        helpers_gsm8k._run_lm_eval(
            "http://127.0.0.1:8000",
            "model",
            output_path=str(tmp_path / "results"),
    )

    assert error.value.code == 128 + signal.SIGTERM
    assert events == [
        ("install", signal.SIGTERM),
        ("install", signal.SIGINT),
        ("spawn", None),
        ("delegate", signal.SIGTERM),
        ("cleanup-term", signal.SIGTERM),
        ("second-signal", signal.SIGINT),
        ("probe", 0),
        "wait",
        ("restore", signal.SIGTERM),
        ("restore", signal.SIGINT),
    ]
    assert mask_calls == []
    assert popen_kwargs["start_new_session"] is True


@pytest.mark.parametrize("backend", ["gpu", "npu"])
def test_run_gsm8k_evaluation_uses_the_builtin_task_and_fixed_limit(
    monkeypatch,
    backend,
):
    args = _args()
    args.device_backend = backend
    calls = []

    def fake_run_lm_eval(base_url, model_name, **kwargs):
        calls.append((base_url, model_name, kwargs))
        return {
            "n-samples": {"gsm8k": {"effective": 7}},
            "results": {"gsm8k": {"exact_match": 0.20}},
        }

    monkeypatch.setattr(runner, "_run_lm_eval", fake_run_lm_eval)

    runner.run_gsm8k_evaluation(args)

    assert calls == [
        (
            "http://127.0.0.1:18100",
            "deepseek-v2-lite-afd-attention",
            {
                "output_path": "/tmp/gsm8k-results",
                "tokenizer": "deepseek-ai/DeepSeek-V2-Lite",
                "limit": 7,
            },
        ),
    ]


def test_run_gsm8k_evaluation_rejects_a_non_seven_sample_result(
    monkeypatch,
):
    args = _args()
    monkeypatch.setattr(
        runner,
        "_run_lm_eval",
        lambda *_args, **_kwargs: {
            "n-samples": {"gsm8k": {"effective": 6}},
            "results": {"gsm8k": {"exact_match": 1.0}},
        },
    )

    with pytest.raises(AssertionError, match="expected 7"):
        runner.run_gsm8k_evaluation(args)


def test_request_completions_raises_after_a_concurrent_request_fails(monkeypatch):
    """A successful completion must not hide a sibling request failure."""
    args = _args()
    call_count = 0
    call_lock = threading.Lock()

    def fake_request_completion(_args):
        nonlocal call_count
        with call_lock:
            call_count += 1
            request_number = call_count
        if request_number == 2:
            raise RuntimeError("broken")
        return {"id": request_number}

    monkeypatch.setattr(runner, "request_completion", fake_request_completion)

    with pytest.raises(RuntimeError, match="broken"):
        runner.request_completions(args)

    assert call_count == 2


def test_request_completions_preserves_request_order(monkeypatch):
    args = _args()
    args.num_requests = 3
    args.request_concurrency = 3
    call_count = 0
    call_lock = threading.Lock()

    def fake_request_completion(_args):
        nonlocal call_count
        with call_lock:
            request_number = call_count
            call_count += 1
        if request_number == 0:
            time.sleep(0.02)
        return {"id": request_number}

    monkeypatch.setattr(runner, "request_completion", fake_request_completion)

    assert runner.request_completions(args) == [{"id": 0}, {"id": 1}, {"id": 2}]


def test_ensure_processes_alive_reports_exited_process_returncode():
    process = argparse.Namespace(poll=lambda: 17)

    with pytest.raises(RuntimeError, match="returncode=17"):
        runner.ensure_processes_alive([process])


def test_terminate_processes_uses_a_shared_deadline_then_reaps_every_process(
    monkeypatch,
):
    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid
            self.returncode = None
            self.wait_calls = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls += 1
            self.returncode = 0

    first_process = FakeProcess(101)
    second_process = FakeProcess(102)
    kill_calls = []

    def fake_killpg(pid, sig):
        kill_calls.append((pid, sig))
        process = first_process if pid == first_process.pid else second_process
        if sig == 0 and process.returncode is not None:
            raise ProcessLookupError
        if pid == first_process.pid and sig == signal.SIGTERM:
            first_process.returncode = 0
        if pid == second_process.pid and sig == signal.SIGKILL:
            second_process.returncode = 0

    monotonic_values = iter([100.0, 121.0])
    monkeypatch.setattr(runner.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(runner.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic_values))

    runner.terminate_processes([first_process, second_process])

    assert kill_calls == [
        (101, signal.SIGTERM),
        (102, signal.SIGTERM),
        (101, 0),
        (102, 0),
        (102, signal.SIGKILL),
    ]
    assert first_process.wait_calls == 1
    assert second_process.wait_calls == 1


def test_terminate_processes_cleans_a_live_group_after_its_leader_exits(
    monkeypatch,
):
    class ExitedLeader:
        pid = 101
        returncode = 0
        wait_calls = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls += 1
            return self.returncode

    process = ExitedLeader()
    group_alive = True
    kill_calls = []

    def fake_killpg(pid, sig):
        nonlocal group_alive
        kill_calls.append((pid, sig))
        if sig == 0 and not group_alive:
            raise ProcessLookupError
        if sig == signal.SIGKILL:
            group_alive = False

    monotonic_values = iter([100.0, 121.0])
    monkeypatch.setattr(runner.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(runner.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic_values))

    runner.terminate_processes([process])

    assert kill_calls == [
        (101, signal.SIGTERM),
        (101, 0),
        (101, 9),
    ]
    assert process.wait_calls == 1


def test_terminate_processes_treats_a_missing_group_as_already_clean(
    monkeypatch,
):
    class ExitedLeader:
        pid = 101
        returncode = 0
        wait_calls = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls += 1
            return self.returncode

    process = ExitedLeader()
    kill_calls = []

    def fake_killpg(pid, sig):
        kill_calls.append((pid, sig))
        raise ProcessLookupError

    monkeypatch.setattr(runner.os, "killpg", fake_killpg, raising=False)

    runner.terminate_processes([process])

    assert kill_calls == [(101, signal.SIGTERM)]
    assert process.wait_calls == 1


def test_terminate_processes_reaps_a_zombie_leader_without_waiting_the_deadline(
    monkeypatch,
):
    group_alive = True
    kill_calls = []
    monotonic_calls = 0

    class ZombieLeader:
        pid = 101
        poll_calls = 0
        wait_calls = 0

        def poll(self):
            nonlocal group_alive
            self.poll_calls += 1
            group_alive = False
            return 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            return 0

    process = ZombieLeader()

    def fake_killpg(pid, sig):
        kill_calls.append((pid, sig))
        if sig == 0 and not group_alive:
            raise ProcessLookupError

    def fake_monotonic():
        nonlocal monotonic_calls
        monotonic_calls += 1
        if monotonic_calls > 1:
            raise AssertionError("zombie-only group consumed the cleanup deadline")
        return 100.0

    monkeypatch.setattr(runner.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(runner.time, "monotonic", fake_monotonic)

    runner.terminate_processes([process])

    assert kill_calls == [(101, signal.SIGTERM), (101, 0)]
    assert process.poll_calls >= 1
    assert process.wait_calls == 1


def test_terminate_processes_reports_signal_and_reap_failures(monkeypatch):
    class FakeProcess:
        pid = 101

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise subprocess.SubprocessError("reap failed")

    def fake_killpg(_pid, sig):
        if sig == 0:
            return
        if sig == signal.SIGTERM:
            raise OSError("term failed")
        raise OSError("kill failed")

    monotonic_values = iter([100.0, 121.0])
    monkeypatch.setattr(runner.os, "killpg", fake_killpg, raising=False)
    monkeypatch.setattr(runner.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(monotonic_values))

    with pytest.raises(RuntimeError) as error:
        runner.terminate_processes([FakeProcess()])

    assert "term failed" in str(error.value)
    assert "kill failed" in str(error.value)
    assert "reap failed" in str(error.value)


def test_main_checks_processes_and_restores_signal_handlers_when_cleanup_fails(
    monkeypatch,
):
    args = _args()
    args.attention_devices = "0"
    args.ffn_devices = "1"
    args.afd_connector = None
    process = argparse.Namespace(poll=lambda: None)
    checked_processes = []
    installed_handlers = []
    previous_handlers = {
        signal.SIGTERM: "previous-term-handler",
        signal.SIGINT: "previous-int-handler",
    }

    class FakeLogThread:
        joined = False

        def join(self, timeout):
            self.joined = True

    log_thread = FakeLogThread()

    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "validate_topology", lambda *_args: None)
    monkeypatch.setattr(runner, "build_vllm_command", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "build_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "start_process", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(runner, "stream_output", lambda *_args: log_thread)
    monkeypatch.setattr(runner, "wait_for_openai_api", lambda *_args: None)
    monkeypatch.setattr(runner, "run_gsm8k_evaluation", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "ensure_processes_alive",
        lambda processes: checked_processes.append(list(processes)),
    )
    monkeypatch.setattr(
        runner,
        "terminate_processes",
        lambda _processes: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )
    monkeypatch.setattr(
        runner.signal,
        "getsignal",
        lambda signum: previous_handlers[signum],
    )
    monkeypatch.setattr(
        runner.signal,
        "signal",
        lambda signum, handler: installed_handlers.append((signum, handler)),
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        runner.main()

    assert checked_processes == [[process, process]] * 2
    assert log_thread.joined is True
    assert installed_handlers[2:] == list(previous_handlers.items())
    with pytest.raises(SystemExit) as exit_error:
        installed_handlers[0][1](signal.SIGTERM, None)
    assert exit_error.value.code == 128 + signal.SIGTERM


def test_main_ignores_a_second_signal_while_cleanup_finishes(monkeypatch):
    args = _args()
    args.attention_devices = "0"
    args.ffn_devices = "1"
    args.afd_connector = None
    process = argparse.Namespace(poll=lambda: None)
    installed_handlers = {}
    signal_calls = []
    cleanup_events = []
    previous_handlers = {
        signal.SIGTERM: "previous-term-handler",
        signal.SIGINT: "previous-int-handler",
    }

    class FakeLogThread:
        joined = False

        def join(self, timeout):
            self.joined = True

    log_thread = FakeLogThread()

    def fake_signal(signum, handler):
        signal_calls.append((signum, handler))
        installed_handlers[signum] = handler

    def trigger_first_signal(*_args):
        installed_handlers[signal.SIGTERM](signal.SIGTERM, None)

    def cleanup_with_second_signal(_processes):
        cleanup_events.append("started")
        installed_handlers[signal.SIGTERM](signal.SIGTERM, None)
        cleanup_events.append("completed")

    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "validate_topology", lambda *_args: None)
    monkeypatch.setattr(runner, "build_vllm_command", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "build_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "start_process", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(runner, "stream_output", lambda *_args: log_thread)
    monkeypatch.setattr(runner, "wait_for_openai_api", trigger_first_signal)
    monkeypatch.setattr(runner, "terminate_processes", cleanup_with_second_signal)
    monkeypatch.setattr(
        runner.signal,
        "getsignal",
        lambda signum: previous_handlers[signum],
    )
    monkeypatch.setattr(runner.signal, "signal", fake_signal)

    with pytest.raises(SystemExit) as error:
        runner.main()

    assert error.value.code == 128 + signal.SIGTERM
    assert cleanup_events == ["started", "completed"]
    assert log_thread.joined is True
    assert signal_calls[-2:] == list(previous_handlers.items())


def test_runner_drops_flashcomm_for_npu_role_without_tp(monkeypatch):
    args = _args()
    args.device_backend = "npu"
    args.ffn_tp_size = 1
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")

    env = runner.build_env("2,3", args, role="ffn")

    assert "VLLM_ASCEND_ENABLE_FLASHCOMM1" not in env


def test_runner_forces_gpu_v1_model_runner(monkeypatch):
    args = _args()
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

    env = runner.build_env("0,1", args, role="attention")

    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_request_completion_includes_http_error_body(monkeypatch):
    args = _args()
    error = urllib.error.HTTPError(
        url="http://127.0.0.1:18100/v1/completions",
        code=500,
        msg="Internal Server Error",
        hdrs=None,
        fp=io.BytesIO(b'{"error":"CUDA out of memory"}'),
    )

    def fake_urlopen(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        runner.request_completion(args)
