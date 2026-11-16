from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks import device_monitor


def test_unavailable_host_counters_do_not_bias_cross_rank_summary(monkeypatch):
    monkeypatch.setattr(device_monitor.wandb, "run", None)
    table, summary = device_monitor.log_prof_data(
        [
            {"cpu_tree_pss_gb": -1, "cpu_pinned_allocated_gb": -1, "cpu_mem_gb": 20},
            {"cpu_tree_pss_gb": 12, "cpu_pinned_allocated_gb": -1, "cpu_mem_gb": 30},
        ],
        iteration=200,
    )
    assert table["cpu_tree_pss_gb"].tolist() == [-1, 12]
    assert summary.loc["cpu_tree_pss_gb"].tolist() == [12, 12, 12]
    assert summary.loc["cpu_pinned_allocated_gb"].tolist() == [-1, -1, -1]
    assert summary.loc["cpu_mem_gb"].tolist() == [25, 30, 20]


@pytest.mark.parametrize("supports_pinned_stats", [False, True])
def test_memory_attribution_reaches_logged_metrics(monkeypatch, supports_pinned_stats, tmp_path):
    monitor = device_monitor.DeviceMonitor(log_memory_detail=False)
    monitor.rank = 0
    monitor.world_size = 1
    monitor.handle = object()
    monitor.local_dir = str(tmp_path)
    cpu = {"cpu_mem_gb": 20.0, "cpu_tree_pss_gb": 12.0, "cpu_trainer_rss_gb": 10.0}
    monkeypatch.setattr(device_monitor, "process_tree_memory", lambda process: cpu.copy())
    for name in (
        "max_memory_allocated",
        "max_memory_reserved",
        "temperature",
        "power_draw",
        "utilization",
        "clock_rate",
    ):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    if supports_pinned_stats:
        monkeypatch.setattr(
            torch.cuda.memory,
            "host_memory_stats",
            lambda: {"allocated_bytes.current": 2 * 1024**3, "reserved_bytes.current": 5 * 1024**3},
        )
    else:
        monkeypatch.delattr(torch.cuda.memory, "host_memory_stats", raising=False)
    monkeypatch.setattr(
        device_monitor.pynvml, "nvmlDeviceGetMemoryInfo", lambda handle: SimpleNamespace(used=0, free=0)
    )
    logged = []

    def log_data(rows, iteration):
        logged.extend(rows)
        return device_monitor.pd.DataFrame(rows), device_monitor.pd.DataFrame({"cpu_mem_gb": [20]})

    monkeypatch.setattr(device_monitor, "log_prof_data", log_data)
    monitor.every_n_impl(None, None, {}, {}, None, iteration=200)
    assert logged[0]["cpu_mem_gb"] == 20
    assert logged[0]["cpu_tree_pss_gb"] == 12
    assert logged[0]["cpu_pinned_allocated_gb"] == (2 if supports_pinned_stats else -1)
    assert logged[0]["cpu_pinned_reserved_gb"] == (5 if supports_pinned_stats else -1)
    assert (tmp_path / "prof_data_000000200.csv").is_file()
    assert (tmp_path / "summary_000000200.csv").is_file()


def test_wandb_auth_failure_does_not_abort_monitoring_or_scalar_logging(monkeypatch):
    monkeypatch.setattr(device_monitor.wandb, "run", object())
    monkeypatch.setattr(device_monitor.wandb, "Table", lambda **kwargs: object())
    calls = []

    def upload(data, step):
        calls.append((data, step))
        if "DeviceMonitor/prof_data" in data:
            raise device_monitor.wandb.errors.AuthenticationError("test authentication failure")

    monkeypatch.setattr(device_monitor.wandb, "log", upload)
    table, summary = device_monitor.log_prof_data([{"cpu_mem_gb": 20}], iteration=1400)
    assert table["cpu_mem_gb"].tolist() == [20]
    assert summary.loc["cpu_mem_gb", "Avg"] == 20
    assert len(calls) == 2
    assert calls[1] == (
        {"DeviceMonitor/min_cpu_mem_gb": 20, "DeviceMonitor/max_cpu_mem_gb": 20, "DeviceMonitor/avg_cpu_mem_gb": 20},
        1400,
    )


def test_wandb_monitor_upload_failure_does_not_expose_error_details(monkeypatch):
    monkeypatch.setattr(
        device_monitor.wandb,
        "log",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            device_monitor.wandb.errors.AuthenticationError("secret-proxy-credential")
        ),
    )
    warnings = []
    monkeypatch.setattr(device_monitor.log, "warning", warnings.append)
    device_monitor._log_wandb_monitor_data({"mem/allocated": 1}, 1400)
    assert len(warnings) == 1
    assert "AuthenticationError" in warnings[0]
    assert "secret-proxy-credential" not in warnings[0]
