from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from cosmos_framework.utils.host_memory import cgroup_memory, process_tree_memory

GIB = 1024**3


class Process:
    def __init__(self, rss, pss, uss, children=(), error=None):
        self.rss = rss
        self.pss = pss
        self.uss = uss
        self.descendants = children
        self.error = error

    def children(self, recursive):
        assert recursive
        return self.descendants

    def memory_info(self):
        if isinstance(self.error, psutil.NoSuchProcess):
            raise self.error
        return SimpleNamespace(rss=self.rss * GIB)

    def memory_full_info(self):
        if self.error:
            raise self.error
        return SimpleNamespace(pss=self.pss * GIB, uss=self.uss * GIB)


@pytest.mark.parametrize("exit_child", [False, True])
def test_shared_rss_is_not_reported_as_physical_memory(monkeypatch, exit_child):
    child = Process(10, 6, 2)
    children = [child]
    if exit_child:
        children.append(Process(100, 100, 100, error=psutil.NoSuchProcess(123)))
    parent = Process(10, 6, 2, children=children)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=100 * GIB, available=70 * GIB))
    monkeypatch.setattr("cosmos_framework.utils.host_memory.cgroup_memory", lambda: {})
    result = process_tree_memory(parent)
    assert result["cpu_mem_gb"] == 20
    assert result["cpu_trainer_rss_gb"] == result["cpu_children_rss_gb"] == 10
    assert result["cpu_tree_pss_gb"] == 12
    assert result["cpu_tree_uss_gb"] == 4
    assert result["cpu_tree_process_count"] == 2
    assert result["cpu_host_available_gb"] == 70
    assert result["cpu_host_used_gb"] == 30


def test_denied_pss_is_unavailable_not_zero(monkeypatch):
    parent = Process(10, 6, 2, children=[Process(10, 6, 2, error=psutil.AccessDenied(123))])
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=100 * GIB, available=70 * GIB))
    monkeypatch.setattr("cosmos_framework.utils.host_memory.cgroup_memory", lambda: {})
    result = process_tree_memory(parent)
    assert result["cpu_mem_gb"] == 20
    assert result["cpu_tree_pss_gb"] == result["cpu_tree_uss_gb"] == -1


@pytest.mark.parametrize("membership,mount_root", [("/", "/"), ("/pod/container", "/pod")])
def test_cgroup_mount_namespace_resolution(tmp_path, membership, mount_root):
    proc = tmp_path / "proc/self"
    proc.mkdir(parents=True)
    mount = tmp_path / "cgroup"
    root = mount / Path(membership).relative_to(mount_root)
    root.mkdir(parents=True)
    (proc / "cgroup").write_text(f"0::{membership}\n")
    (proc / "mountinfo").write_text(f"1 0 0:1 {mount_root} {mount} rw - cgroup2 cgroup rw\n")
    (root / "memory.current").write_text(str(10 * GIB))
    (root / "memory.max").write_text(str(20 * GIB))
    (root / "memory.stat").write_text(f"anon {4 * GIB}\nfile {6 * GIB}\nshmem {2 * GIB}\n")
    assert cgroup_memory(tmp_path / "proc") == {
        "cpu_cgroup_current_gb": 10,
        "cpu_cgroup_limit_gb": 20,
        "cpu_cgroup_anon_gb": 4,
        "cpu_cgroup_file_gb": 6,
        "cpu_cgroup_shmem_gb": 2,
    }


def test_missing_cgroup_counters_are_unavailable(tmp_path):
    assert set(cgroup_memory(tmp_path).values()) == {-1}
