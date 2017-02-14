## KiloSort processor plugin to MountainLab

This project wraps the spike sorting algorithm, [KiloSort](https://github.com/cortex-lab/KiloSort), as a processor plugin of [MountainLab](https://github.com/magland/mountainlab).

Right now the processor executes in CPU-only mode. But, as KiloSort is designed to run using GPU, this capabilities will be incremented soon.

This plugin is at a very early stage and is under prelim development. More info coming soon.

## Instructions

First, install MountainLab.

```bash
cd mountainlab/packages
git clone https://github.com/magland/ml_kilosort.git
```

If it's running, restart the processing daemon:

```bash
mp-daemon-restart
```

Now you should have a new processor registered in mountainprocess. Try:

```bash
mp-spec ml_kilosort
```

To run a prelim comparison with mountainsort follow the instructions in ml_kilosort/test_compare.


