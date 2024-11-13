import time
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
import psutil
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    NVMLError,
)


class EnhancedPerformanceLoggingWrapper(Wrapper):
    def __init__(self, env, log_interval=1000):
        super().__init__(env)
        self.log_interval = log_interval
        self.step_times = deque(maxlen=log_interval)
        self.render_times = deque(maxlen=log_interval)
        self.reset_times = deque(maxlen=100)
        self.step_count = 0
        self.process = psutil.Process()
        self.initial_ram = self.process.memory_info().rss / (1024 * 1024)  # Initial RAM usage in MB
        try:
            nvmlInit()
            self.gpu_handle = nvmlDeviceGetHandleByIndex(0)  # Assuming we're using the first GPU
        except NVMLError:
            self.gpu_handle = None

    def step(self, action):
        start_time = time.perf_counter()
        observation, reward, terminated, truncated, info = self.env.step(action)
        end_time = time.perf_counter()
        self.step_times.append(end_time - start_time)

        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            self.log_performance()

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        start_time = time.perf_counter()
        result = self.env.reset(**kwargs)
        end_time = time.perf_counter()
        self.reset_times.append(end_time - start_time)

        print(f"Reset time: {self.reset_times[-1]:.6f} seconds")

        return result

    def render(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = self.env.render(*args, **kwargs)
        end_time = time.perf_counter()
        self.render_times.append(end_time - start_time)
        return result

    def log_performance(self):
        stats = self.get_performance_stats()
        print(f"Performance stats after {self.step_count} steps:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    def get_performance_stats(self):
        current_ram = self.process.memory_info().rss / (1024 * 1024)  # Current RAM usage in MB
        cpu_percent = self.process.cpu_percent()

        stats = {
            "avg_step_time": f"{np.mean(self.step_times):.6f} seconds",
            "avg_render_time": f"{np.mean(self.render_times):.6f} seconds",
            "avg_reset_time": f"{np.mean(self.reset_times):.6f} seconds",
            "ram_usage": f"{current_ram:.2f} MB (change: {current_ram - self.initial_ram:.2f} MB)",
            "cpu_usage": f"{cpu_percent:.2f}%",
        }

        if self.gpu_handle:
            try:
                memory_info = nvmlDeviceGetMemoryInfo(self.gpu_handle)
                utilization = nvmlDeviceGetUtilizationRates(self.gpu_handle)
                stats["gpu_memory"] = f"{memory_info.used / (1024**2):.2f} MB / {memory_info.total / (1024**2):.2f} MB"
                stats["gpu_utilization"] = f"{utilization.gpu}%"
            except NVMLError:
                pass

        return stats
