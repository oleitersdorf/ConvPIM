# ConvPIM: Evaluating Digital Processing-in-Memory through Convolutional Neural Network Acceleration
## Overview
This is the evaluation environment for the following paper, 

`O. Leitersdorf, R. Ronen, and S. Kvatinsky, “ConvPIM: Evaluating Digital Processing-in-Memory through Convolutional Neural Network Acceleration,” 2023.` 

The repository includes the `PyTorch` scripts that were profiled to generate the results in the paper (using `NVIDIA NVML` and `NVIDIA Nsight Systems`), and the results for the different experiments. 

## User Information
### Dependencies
The evaluation environment is implemented using `PyTorch` with bindings to `NVIDIA NVML` through `pynvml` for the power measurements. The scripts are further executed through the `NVIDIA Nsight Systems` profiler for additional profiler measurements.

### Organization
The repository is organized as follows:
- `README.md`: Provides general information on the evaluation methodology.
- `experiments`: Includes the `PyTorch` scripts that represent the different experiments in the paper.
- `results`: Contains the raw output from the PyTorch scripts as well as a summary of the overall results.