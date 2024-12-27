# student_template.py

import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd
import torch.optim as optim

from models import VGG, Partition, num_classes, image_w, image_h, num_batches, batch_size

class DistVggNet(nn.Module):
    """
    Assemble partitions as an nn.Module and define pipelining logic.
    The model is split across different workers.
    """
    def __init__(self, workers, vgg_model, partition_indices):
        super(DistVggNet, self).__init__()
        # Get the layers from the VGG model
        layers = list(vgg_model.features.children()) + list(vgg_model.classifier.children())

        self.partitions = []
        for i, worker in enumerate(workers):
            # Get the modules for this partition
            modules = layers[partition_indices[i]:partition_indices[i+1]]
            # Send the partition to the worker
            p_rref = rpc.remote(
                worker,
                Partition,
                args=(modules,),
                timeout=0
            )
            self.partitions.append(p_rref)

    # Forward function: SPECIFIES HOW THE INPUT DATA IS TRANDFORMED AS IT PASSES THROUGH THE LAYERS OF THE NETWORK

    def forward(self, x):
        # Split input into micro-batches
        micro_batches = torch.chunk(x, chunks=64)  

        micro_batch_results = []

        # Process each micro-batch through the pipeline
        for micro_batch in micro_batches:
            x_rref = RRef(micro_batch)

            for p_rref in self.partitions[:-1]:
                # the x_reference of each mini-batch is being updated EVERY LOOP,
                # and passed to the next processor/CPU
                x_rref = p_rref.remote().forward(x_rref)

            out_fut = self.partitions[-1].rpc_async().forward(x_rref)
            micro_batch_results.append(out_fut)

        results = [fut.wait() for fut in micro_batch_results]
        return torch.cat(results, dim=0)

    def parameter_rrefs(self):
        remote_params = []
        for p_rref in self.partitions:
            remote_params.extend(p_rref.remote().parameter_rrefs().to_here())
        return remote_params

def run_master():
    # Create the VGG model
    vgg_model = VGG()

    # Define how layers are divided between workers: indices split model into different blocks
    partition_indices = [0, 12, 23, 34, 44]

    # Initialize the distributed model
    workers = ["worker1", "worker2", "worker3", "worker4"]

    model = DistVggNet(workers, vgg_model, partition_indices)
    loss_fn = nn.MSELoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    for i in range(num_batches):
        print(f"Processing batch {i}", flush=True)
        # Generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
                      .scatter_(1, one_hot_indices, 1)

        # The distributed autograd context is the dedicated scope for the
        # distributed backward pass to store gradients, which can later be
        # retrieved using the context_id by the distributed optimizer.
        with dist_autograd.context() as context_id:
            outputs = model(inputs)
            print("Start of backpropagation and optimization", flush=True)
            dist_autograd.backward(context_id, [loss_fn(outputs, labels)])
            opt.step(context_id)
            print("End of backpropagation and optimization", flush=True)