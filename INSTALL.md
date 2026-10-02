# Installation and prerequisites

## Supported setup

The existing artifact documentation reports a Lenovo SR650 V3 with Intel Xeon
Gold 5420+ (Sapphire Rapids), RHEL 9.4 or Ubuntu 22.04. AMX experiments require an
Intel CPU and operating system with AMX support. A generic x86 virtual machine
is not an equivalent test environment.

Install Git, GNU Make, a C/C++ compiler with the AMX intrinsic support required
by the component Makefiles, and Python with pip/venv. Python 3.10 is a reasonable
starting point for the root requirements on Ubuntu 22.04; the authors' exact
Python/compiler versions are not recorded here.

```sh
git clone https://github.com/jkalya/gatebleed.git
cd gatebleed
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

The root requirements cover analysis packages, not every model-specific
experiment. Read the README in the chosen component for its extra dependencies,
inputs and hardware configuration. There is no single root-level build target.

## Local performance-stage example

The existing `01_performance_stages/README.md` documents this sequence:

```sh
cd 01_performance_stages
make
python3 get_rdtscp_data.py
```

Then run `python3 plot.py PATH_TO_GENERATED_CSV`, replacing the argument with
the CSV produced by the preceding command. Compare the plot with
`plot_original.png`; the original documentation notes that OS versions can
change the observed latency stages.

## Other components

Use each component's own README for its setup and execution steps:

- [Performance stages](01_performance_stages/README.md)
- [Remote covert channel](03_remote_gatebleed_covert_channel/README.md)
- [Remote Spectre experiment](04_remote_gatebleed_spectre/README.md)
- [Transformer timing](05_gatebleed_transformer_model/README.md)
- [MoE membership inference](06_gatebleed_moe_transformer_mia/README.md)
- [CNN membership inference](07_gatebleed_cnn_model_mia/README.md)

## Reproducibility record

Record `git rev-parse HEAD`, `uname -r`, CPU model, compiler version and
`python -m pip freeze` with results. The instructions above were assembled from
the checked-in artifact files; they are not a new hardware validation. A
maintainer should confirm a clean installation on the supported system before
publishing a tested release.
