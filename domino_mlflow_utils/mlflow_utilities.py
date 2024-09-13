
import pandas as pd
import random
import base64
import pandas as pd
from PIL import Image
from io import BytesIO
import subprocess
import time
import mlflow
from mlflow import MlflowClient
import time
import threading
import psutil
import nvidia_smi 
import pandas as pd

class DominoMLflowUtilities:
    def __init__(self):
        mlflow.end_run()
        initialized = True
        self.t1 = None
        
    def init(self,experiment_name,config=None):    
        mlflow.set_experiment(experiment_name)
        run = mlflow.active_run()
        print(f"Checking for active runs, Active Run= {run}")
        if run:
            print("Ending active prior to starting new run. Ending run_id: {}".format(run.info.run_id))     
            mlflow.end_run()
        self.run = mlflow.start_run()
        print("Started new run with run_id: {}".format(self.run.info.run_id))     
        if config:
            mlflow.log_params(vars(config))
        self.t1 = threading.Thread(target=publish_utilization_metrics, args=(self.run.info.run_id,1))
        self.t1.start()
        
    
    def finish(self):
        mlflow.end_run()
        if self.t1:
            self.t1.join()
        mlflow.end_run()
        print("run_id: {}; status: {}".format(self.run.info.run_id, self.run.info.status))
        print("--")
        # Check for any active runs
        r = mlflow.active_run()
        if r:
            print("After Finish - Active run: {}".format(r.info.run_id))
            
def get_utilization_metrics():
    metrics = {}
    cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
    i = 0
    for c in cpu_usage:
        key = f"cpu{i}"
        metrics[key] = cpu_usage[i]
        i = i + 1

    gpu_found = False
    try:
        nvidia_smi.nvmlInit()
        gpu_found = True
    except:
        #print('GPU not found')
        do_not_track_gpu = True
    if gpu_found:
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            dir(mem)
            key = f"gpu{i}_global_mem_utilization_mb"
            metrics[key] = ((mem.total - mem.free) / 1024 ** 2) / (mem.total / 1024 ** 2)

            key = f"gpu{i}_gpu_utilization"
            metrics[key] = util.gpu / 100.0

            key = f"gpu{i}_gpu_memory_utilization"
            metrics[key] = util.memory / 100.0
    return metrics


def publish_utilization_metrics(run_id, sample_every_n_seconds=5):
    client = MlflowClient()
    status = client.get_run(run_id).to_dictionary()['info']['status']
    x = 0
    while not status == 'FINISHED':
        metrics = get_utilization_metrics()
        mlflow.log_metrics(metrics, step=x)
        x = x + 1
        time.sleep(sample_every_n_seconds)
        status = client.get_run(run_id).to_dictionary()['info']['status']