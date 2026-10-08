# HTCondor Batch Submission for Chained Tracking + MCTruth Pipeline

This directory contains the HTCondor configuration and execution scripts to run the two-stage chained pipeline per partition concurrently across CERN HTCondor:

```
[Raw Genie-TGeant4 MC]
         │
         ▼ (Step 1: run_dimuon_reco.py, ~1-2 min)
[sndLHC.Genie-TGeant4_digCPP_2MuTrks.root]
         │
         ▼ (Step 2: mctruth_neutrinos.py, ~10-30 sec)
[sndLHC.Genie-TGeant4_digCPP_2MuTrks_truth.root]
```

---

## Directory Overview

- **`pipeline.sub`**: HTCondor job submit file configured for AlmaLinux9, `group_u_SNDLHC.users`, and `longlunch` (up to 2h per partition).
- **`run_pipeline_partition.sh`**: Job worker wrapper executed on each HTCondor compute node. Executes Step 1 (dimuon tracking), verifies the tracking output ROOT file, and then immediately runs Step 2 (MCTruth extraction) on that file with `--skip-existing`.
- **`sndswEnv.sh`**: Frozen static environment variables (`LD_LIBRARY_PATH`, `PYTHONPATH`, `ROOTSYS`, `ROOT_INCLUDE_PATH`, etc.) exported from the verified working setup.
- **`args_partitions.txt`**: The exact list of 400 valid partitions (numeric directories 0 through 400 found in the GENIE MC dataset).
- **`submit.sh`**: Helper wrapper to submit all 400 jobs from `lxplus`.
- **`status.sh`**: Helper script to monitor job queue state and live counts of both `*_2MuTrks.root` and `*_2MuTrks_truth.root` files on EOS.
- **`out/`**, **`err/`**, **`log/`**: Standard log directories for HTCondor job output, error, and cluster execution logs.

---

## Quick Start (from `lxplus`)

### 1. Submit All 400 Partitions

From the repository root on `lxplus`:
```bash
./htcondor/tracking_and_truth/submit.sh
```
Or directly using `condor_submit`:
```bash
cd htcondor/tracking_and_truth
condor_submit pipeline.sub
```

### 2. Check Live Status

At any time, run:
```bash
./htcondor/tracking_and_truth/status.sh
```
This shows:
- The total count and percentage of completed Step 1 tracked files (`_2MuTrks.root`).
- The total count and percentage of completed Step 2 final truth files (`_2MuTrks_truth.root`).
- The live HTCondor queue summary (`condor_q`).

### 3. Custom Options & Overrides

- **Custom Input/Output Patterns**:
  ```bash
  condor_submit htcondor/tracking_and_truth/pipeline.sub \
    input_pattern="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP.root" \
    track_pattern="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_2MuTrks.root" \
    truth_pattern="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_2MuTrks_truth.root"
  ```
- **Change Job Flavour**:
  ```bash
  condor_submit htcondor/tracking_and_truth/pipeline.sub flavour=workday
  ```
- **Submit a Subset of Partitions**:
  ```bash
  condor_submit htcondor/tracking_and_truth/pipeline.sub args_file=args_custom.txt
  ```

---

## Key Benefits

- **No Double-Queuing**: The batch slot executes both steps back-to-back without sending the second step to wait in the queue.
- **Fast Execution**: Step 2 runs only on the events surviving the 2-track requirement, completing in seconds per partition.
- **Idempotence**: Both steps run with `--skip-existing`. If interrupted or re-submitted, finished stages are skipped automatically.
