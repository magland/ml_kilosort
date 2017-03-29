## KiloSort processor plugin to MountainLab

This project wraps the spike sorting algorithm, [KiloSort](https://github.com/cortex-lab/KiloSort), as a processor plugin of [MountainLab](https://github.com/magland/mountainlab).

Right now the processor executes in CPU-only mode. But, as KiloSort is designed to run using GPU, this capabilities will be incremented soon.

This plugin is at a very early stage and is under prelim development. More info coming soon.

## Instructions

You must have Matlab with the signal processing toolbox installed, invokable by the command "matlab".

First, install MountainLab. Then clone this repository into the packages directory:

```bash
cd mountainlab/packages
git clone https://github.com/magland/ml_kilosort.git
```

If you would like to run KiloSort in GPU mode (recommended) you must first make sure MATLAB has cuda capabilities. That means that cuda must be installed, you have the matlab parallel computing toolbox, and mexcuda matlab command works properly. For Ubuntu 16.04 you should install cuda 8.0 and make sure you have the most recent version of matlab (at least 2017a). When you are ready do the following to compile the KiloSort cuda files. From matlab console:

```
cd mountainlab/packages/ml_kilosort/KiloSort/CUDA
mexGPUall
```

If successful you will get a new mexMPmuFEAT.mexa64 file (at least on Linux).

If the processing daemon is running, restart it:

```bash
mp-daemon-restart
```

Now you should have a new processor registered in mountainprocess. Try:

```bash
mp-spec ml_kilosort
```

To run a prelim comparison with mountainsort on synthetic data follow the instructions in ml_kilosort/test_compare.


