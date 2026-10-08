# HTCondor Batch Submission for SND@LHC Downstream Dimuon Tracking

This directory contains the HTCondor configuration and execution scripts to run downstream dimuon tracking (`run_TrackSelections.py -t dimuon_DS --nTracks 2 -ht`) across all Monte Carlo partitions concurrently using [`scripts/run_dimuon_reco.py`](../../scripts/run_dimuon_reco.py).

---

## Directory Overview

- **`tracking.sub`**: HTCondor job submit file configured for AlmaLinux9, `group_u_SNDLHC.users`, and `microcentury` (up to 1 hour per partition, while reconstruction takes ~1–3 minutes).
- **`run_tracking_partition.sh`**: Job worker wrapper executed on each HTCondor compute node. Sets up the `sndsw` environment, resolves input/output paths, ensures output directory exists, and executes `run_dimuon_reco.py` with `--skip-existing`.
- **`sndswEnv.sh`**: Frozen static environment variables (`LD_LIBRARY_PATH`, `PYTHONPATH`, `ROOTSYS`, `ROOT_INCLUDE_PATH`, etc.) exported from the verified working setup.
- **`args_partitions.txt`**: The exact list of 400 valid partitions (numeric directories 0 through 400 found in the GENIE MC dataset).
- **`submit.sh`**: Helper wrapper to submit all 400 jobs from `lxplus`.
- **`status.sh`**: Helper script to monitor job queue state and live count of output `*_dimuonReco.root` files on EOS.
- **`out/`**, **`err/`**, **`log/`**: Standard log directories for HTCondor job output, error, and cluster execution logs.

---

## Quick Start (from `lxplus`)

### 1. Submit All 400 Partitions

From the repository root on `lxplus`:
```bash
./htcondor/tracking/submit.sh
```
Or directly using `condor_submit`:
```bash
cd htcondor/tracking
condor_submit tracking.sub
```

### 2. Check Job & Output Status

At any time, run:
```bash
./htcondor/tracking/status.sh
```
This shows:
- The total count and percentage of completed `*_dimuonReco.root` files on EOS.
- The live HTCondor queue summary (`condor_q`).

### 3. Custom Options & Overrides

- **Custom Input or Output Patterns**:
  ```bash
  condor_submit tracking.sub \
    input_pattern="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP.root" \
    output_pattern="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_dimuonReco.root"
  ```
- **Change Job Flavour** (e.g. `longlunch` for 2 hours instead of default `microcentury`):
  ```bash
  condor_submit tracking.sub flavour=longlunch
  ```
- **Submit a Subset of Partitions**:
  ```bash
  condor_submit tracking.sub args_file=args_custom.txt
  ```

---

## Default Paths & Reconstruction Settings

- **Input Pattern**: `/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000/%s/sndLHC.Genie-TGeant4_digCPP.root`
- **Output Pattern**: `/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_dimuonReco.root`
- **Track Type**: `dimuon_DS`
- **Minimum Tracks**: `2`
- **Hough Tracking**: Enabled (`-ht`)
