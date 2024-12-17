from prometheus_client import start_http_server, Gauge
from pynvml import *
import time

# Initialize NVML
try:
    nvmlInit()
except NVMLError as error:
    print(f"Error initializing NVML: {error}")
    exit(1)

# Metrics
gpu_memory_total = Gauge('gpu_memory_total_bytes', 'Total GPU Memory in Bytes', ['gpu_id'])
gpu_memory_used = Gauge('gpu_memory_used_bytes', 'Used GPU Memory in Bytes', ['gpu_id'])
gpu_utilization = Gauge('gpu_utilization_percentage', 'GPU Utilization Percentage', ['gpu_id'])
gpu_power_usage = Gauge('gpu_power_usage_watts', 'Current GPU Power Consumption in Watts', ['gpu_id'])

def get_gpu_info():
    try:
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            utilization = nvmlDeviceGetUtilizationRates(handle)
            power_usage = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert milliwatts to watts

            # Set metrics for each GPU
            gpu_memory_total.labels(gpu_id=f"gpu_{i}").set(memory_info.total)
            gpu_memory_used.labels(gpu_id=f"gpu_{i}").set(memory_info.used)
            gpu_utilization.labels(gpu_id=f"gpu_{i}").set(utilization.gpu)
            gpu_power_usage.labels(gpu_id=f"gpu_{i}").set(power_usage)
    except NVMLError as error:
        print(f"Error reading GPU info: {error}")

def main():
    # Start Prometheus HTTP server on port 8000
    start_http_server(8001)
    print("Prometheus metrics server started on port 8000")

    while True:
        get_gpu_info()
        time.sleep(1)  # Update every 1 seconds

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        nvmlShutdown()
