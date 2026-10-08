"""Stage timings and process/GPU/RAM peaks, including CPU staging and save."""
import os
import resource
import subprocess
import threading
import time
import socket
from contextlib import contextmanager
import torch

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class Resources:
    def __init__(self):
        self.started = time.monotonic(); self.stages = {}; self.process_gpu_bytes = 0
        self.stop = threading.Event(); self.thread = threading.Thread(target=self.poll, daemon=True)
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.thread.start()
        return self
    def poll(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
                for line in out.stdout.splitlines():
                    pid, memory = line.split(",")
                    if int(pid) == os.getpid():
                        self.process_gpu_bytes = max(self.process_gpu_bytes, int(memory)*1024**2)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self.stop.wait(1)
    @contextmanager
    def stage(self, name):
        synchronize(); start = time.monotonic()
        try:
            yield
        finally:
            synchronize(); self.stages[name] = self.stages.get(name, 0)+time.monotonic()-start
    def result(self):
        synchronize()
        seconds = time.monotonic()-self.started
        return dict(seconds=seconds, gpu_hours=seconds/3600, stage_seconds=self.stages,
                    max_cuda_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    max_cuda_reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                    process_gpu_memory_peak=self.process_gpu_bytes,
                    host_rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*
                        (1 if __import__("sys").platform=="darwin" else 1024),
                    cpu_threads=torch.get_num_threads(), host=socket.gethostname(),
                    cpu_cores=os.cpu_count(), load_average=os.getloadavg(),
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    cuda_version=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    measurement="reserved GPU wall-time, includes host work; not CUDA kernel busy time")
    def __exit__(self, *args):
        self.stop.set(); self.thread.join(timeout=4)
