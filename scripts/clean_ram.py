#!/usr/bin/env python3
"""
Clean system RAM:
  1. Calls malloc_trim(0) in each validator/miner/NLP-server Python process
     via gdb to return fragmented heap back to the OS.
  2. Drops the Linux page/slab cache (/proc/sys/vm/drop_caches).
     Requires root (run with sudo).

Usage:
    sudo python scripts/clean_ram.py
"""
import subprocess
import sys


KEYWORDS = ["neurons.miner", "neurons.validator", "nlp_server.server"]


def total_ram_used_gb() -> float:
    try:
        out = subprocess.check_output(["free", "-m"], text=True)
        for line in out.splitlines():
            if line.startswith("Mem:"):
                return int(line.split()[2]) / 1024
    except Exception:
        pass
    return 0.0


def rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def find_python_pids() -> list[tuple[str, int]]:
    found = []
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
        for line in out.splitlines():
            if "python" not in line.lower():
                continue
            for kw in KEYWORDS:
                if kw in line:
                    pid = int(line.split()[1])
                    found.append((kw, pid))
                    break
    except Exception:
        pass
    return found


def malloc_trim_via_gdb(pid: int) -> bool:
    try:
        subprocess.run(
            ["gdb", "-p", str(pid), "-batch",
             "-ex", "call malloc_trim(0)",
             "-ex", "detach"],
            capture_output=True, timeout=15,
        )
        return True
    except Exception:
        return False


def drop_page_cache() -> bool:
    try:
        subprocess.run(["sync"], check=True)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except PermissionError:
        print("  ⚠  Need root to drop page cache — run with sudo.")
        return False
    except Exception as e:
        print(f"  ✗  drop_caches failed: {e}")
        return False


def main() -> None:
    before = total_ram_used_gb()
    print(f"RAM used before: {before:.2f} GB\n")

    procs = find_python_pids()
    if not procs:
        print("No matching Python processes found.")
    else:
        print("Calling malloc_trim in each process via gdb …")
        for name, pid in procs:
            rss_before = rss_mb(pid)
            ok = malloc_trim_via_gdb(pid)
            rss_after = rss_mb(pid)
            freed = max(0.0, rss_before - rss_after)
            status = "✓" if ok else "✗"
            print(f"  {status}  pid {pid:>7}  {name}")
            print(f"           RSS {rss_before:.0f} MB → {rss_after:.0f} MB  (freed {freed:.0f} MB)")

    print("\nDropping OS page/slab cache …")
    ok = drop_page_cache()
    print(f"  {'✓' if ok else '✗'}  drop_caches")

    after = total_ram_used_gb()
    print(f"\nRAM used after:  {after:.2f} GB  (freed {max(0.0, before - after):.2f} GB)")


if __name__ == "__main__":
    main()
