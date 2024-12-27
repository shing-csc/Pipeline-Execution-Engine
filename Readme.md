## Pipeline Execution Engine

### Hardware Requirements
A computer with at least 4 CPU cores, running Ubuntu Linux or Windows Subsystem for Linux 2 (WSL2).



### Environment Requirements
PyTorch Version >= 1.7.0
Python Version >= 3.6.8

We suggest using a completely new Python environment with Anaconda. 
Below is the setup:

conda create -n pipeline python=3.10
conda activate pipeline
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia



### Testing of Pipelines
mkdir -p logs
python main.py pipelineFileName > logs/pipeline_num.log



## Comparisons between different pipelines
Pipeline 1 - Four-stage CPU pipeline with PyTorch's RPC framework
Approach: 
Dividing the input data to optimal number of microbatches, with optimal model partition.

Pipeline 2 - Four-stage CPU pipeline with staggered pipeline scheduling mechanism
Approach: 
A custom scheduling mechanism handles the staggered execution of microbatches across the four pipeline stages. Introduces a step-based execution strategy to ensure efficient resource utilization and minimize idle time for each stage.

Pipeline 3 - Eight-stage CPU pipeline with staggered pipeline scheduling mechanism

Approach: Extends the staggered scheduling mechanism to an eight-stage pipeline, enabling finer granularity in model partitioning. Introduces a higher scalability and throughput compared to the four-stage pipelines.

sequential.py - A reference of a sequential order of a pipeline without microbatches, partition, stage optimization.



### Execution time for different pipelines

Execution time for Pipeline1:  187ms

Execution time for Pipeline2:  155ms

Execution time for Pipeline3:  130ms

Execution time for sequential: 385ms