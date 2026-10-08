# HTCondor Batch Submission for SND@LHC MCTruth Pipeline

This directory contains the HTCondor configuration and execution scripts to run the 400 partition Monte Carlo neutrino categorization pipeline concurrently across CERN's HTCondor batch cluster.

---

## Directory Overview

- **`mctruth_neutrinos.sub`**: HTCondor job submit file configured for AlmaLinux9, `group_u_SNDLHC.users`, and `longlunch` (up to 2h per partition).
- **`run_mctruth_partition.sh`**: Job worker wrapper executed on each HTCondor compute node. Sets up the `sndsw` environment, loads custom neutrino C++ libraries, creates the partition directory on EOS, and runs `mctruth_neutrinos.py` with `--skip-existing`.
- **`sndswEnv.sh`**: Frozen static environment variables (`LD_LIBRARY_PATH`, `PYTHONPATH`, `ROOTSYS`, `ROOT_INCLUDE_PATH`, etc.) exported from the verified working setup.
- **`args_partitions.txt`**: The exact list of 400 valid partitions (numeric directories 0 through 400 found in the GENIE MC dataset).
- **`submit.sh`**: Helper wrapper to submit all 400 jobs from `lxplus`.
- **`status.sh`**: Helper script to monitor job queue state and live progress of output `*_truth.root` files generated on EOS.
- **`out/`**, **`err/`**, **`log/`**: Standard log directories for HTCondor job standard output, error, and cluster execution logs.

---

## Quick Start (from `lxplus`)

### 1. Submit All 400 Partitions

From the repository root on `lxplus`:
```bash
./htcondor/submit.sh
```
Or directly using `condor_submit`:
```bash
cd htcondor
condor_submit mctruth_neutrinos.sub
```

### 2. Check Job & Output Status

At any time, run:
```bash
./htcondor/status.sh
```
This shows:
- The total number and percentage of categorized `*_truth.root` files already written to `/eos/user/i/idioniso/snd-numu-charm/data/`.
- The live HTCondor queue summary (`condor_q`).

### 3. Custom Options

- **Change Job Flavour** (e.g. `workday` for 8 hours instead of default `longlunch`):
  ```bash
  condor_submit mctruth_neutrinos.sub flavour=workday
  ```
- **Submit a Subset of Partitions**:
  Create a file (e.g. `htcondor/args_rerun.txt`) containing the partition numbers to run, and submit:
  ```bash
  condor_submit mctruth_neutrinos.sub args_file=args_rerun.txt
  ```

---

## Features & Safety

- **Automatic Skip of Existing Files**: Worker jobs run with `--skip-existing`. Any partition that has already finished (including partitions already processed interactively) will be recognized immediately and skipped safely without re-doing work.
- **Sandboxed Execution**: Worker nodes transfer and source `sndswEnv.sh` to ensure reproducible ROOT `v6-36-04` and `sndsw` execution without relying on unstable interactive subshell state.
- **Auto Retries**: HTCondor is configured with `max_retries = 2` to automatically recover from transient network blips or EOS timeout hiccups.
