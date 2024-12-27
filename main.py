import os
import sys
import time
import psutil
import importlib
import argparse
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import warnings
warnings.simplefilter("ignore")
sys.stderr = open(os.devnull, 'w')

def run_worker(rank, world_size, run_master):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=600)

    p = psutil.Process()

    if rank == 0:
        p.cpu_affinity([0])
        print(f"Child #{rank}: affinity now {p.cpu_affinity()}", flush=True)

        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master()
    else:
        p.cpu_affinity([rank-1])
        print(f"Child #{rank}: affinity now {p.cpu_affinity()}", flush=True)

        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    rpc.shutdown()

if __name__ == "__main__":
    # Parse the command line argument to get the filename (module name)
    parser = argparse.ArgumentParser(description='Run the distributed RPC model')
    parser.add_argument('module', type=str, help='Specify the module name to import run_master from')
    args = parser.parse_args()

    # Dynamically import the module using the module name passed as argument
    module_name = args.module
    try:
        # Import the module dynamically
        imported_module = importlib.import_module(module_name)
        # Get the run_master function from the imported module
        run_master = getattr(imported_module, 'run_master')
    except ImportError as e:
        print(f"Failed to import module {module_name}: {e}")
        exit(1)
    except AttributeError:
        print(f"Module {module_name} does not have a function called 'run_master'")
        exit(1)

    # World size (number of processes for master and workers)
    world_size = 5  # 1 master + 4 workers
    
    # Measure execution time
    tik = time.time()
    
    # Start multi-processing
    # Create different instances of the run_worker processes(workers) for each CPU -> each one their own data INDEPENDENTLY
    mp.spawn(run_worker, args=(world_size, run_master), nprocs=world_size, join=True)
    
    tok = time.time()
    print(f"execution time = {tok - tik}")
