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

    def forward_partition_fut(self, x, partition_id):
        """
        Forward pass in a distributed manner. Send input to the
        designated partition and return a future object for asynchronous operation.

        :param x: Input tensor
        :param partition_id: ID of the partition to which the input should be sent
        :return: Future object representing the partition's output
        """
        x_rref = RRef(x)
        out_fut = self.partitions[partition_id].rpc_async().forward(x_rref)
        return out_fut # This is a future object, wait() must be called to get the actual output


    def parameter_rrefs(self):
        remote_params = []
        for p_rref in self.partitions:
            remote_params.extend(p_rref.remote().parameter_rrefs().to_here())
        return remote_params

def run_master():
    # Create the VGG model
    vgg_model = VGG()

    # Initialize the distributed model
    workers = ["worker1", "worker2", "worker3", "worker4","worker1", "worker2", "worker3", "worker4"]
    
    # Define how layers are divided between workers: indices split model into different blocks
    partition_indices = [0, 5, 11, 16, 22, 27, 33, 38, 44]

    # Number of stages in the pipeline
    num_stages = len(partition_indices) - 1

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
                      
        # Split inputs and labels into microbatches
        microbatch_size = 4     # batch_size  = microbatch_size * num_microbatches
        assert inputs.size(0) % microbatch_size == 0
        num_microbatches = inputs.size(0) // microbatch_size
        inputs_microbatches = torch.split(inputs, microbatch_size)

        # The distributed autograd context is the dedicated scope for the
        # distributed backward pass to store gradients, which can later be
        # retrieved using the context_id by the distributed optimizer.
        with dist_autograd.context() as context_id:

            output_futs = [None for _ in range(num_microbatches)]
            total_steps = num_microbatches + num_stages - 1

            # Loop over all steps 
            for step in range(total_steps):
                # For each steps, we will handle all possible microbatches. 
                #   - Which can be calculated by mb_id = step - stage_id
                for stage_id in range(num_stages):
                    mb_id = step - stage_id

                    # If its smaller than 0 || larger than total number of microbatches
                    # - then it is not a valid batch and will be skipped
                    if (0 <= mb_id < num_microbatches):
                        if (stage_id == 0):
                            out_fut = model.forward_partition_fut(inputs_microbatches[mb_id], stage_id)
                        else:
                            out_fut = model.forward_partition_fut(output_futs[mb_id].wait(), stage_id)
                        output_futs[mb_id] = out_fut

            outputs = torch.cat(torch.futures.wait_all(output_futs))
            print("Start of backpropagation and optimization", flush=True)
            dist_autograd.backward(context_id, [loss_fn(outputs, labels)])
            opt.step(context_id)
            print("End of backpropagation and optimization", flush=True)