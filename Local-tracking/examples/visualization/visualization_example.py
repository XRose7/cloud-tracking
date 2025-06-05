from prometheus_client import start_http_server, Gauge
import random
import time
import psutil

RANDOM_VALUE = Gauge('random_value', 'Simulated random value (0-100)')
CPU_USAGE = Gauge('system_cpu_percent', 'Real CPU usage percentage')

def generate_metrics():
    psutil.cpu_percent(interval=None)

    while True:
        random_num = random.uniform(0, 100)
        RANDOM_VALUE.set(random_num)
        print(random_num)
        cpu_percent = psutil.cpu_percent(interval=None)
        CPU_USAGE.set(cpu_percent)
        
        time.sleep(1)

if __name__ == '__main__':
    start_http_server(8000)
    generate_metrics()

