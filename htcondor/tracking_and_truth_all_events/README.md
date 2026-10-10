# HTCondor Batch Submission: Chained Pipeline (All Events)
## Tracking (All Events) -> MCTruth Extraction -> 2D Event Displays

This directory runs the complete end-to-end 3-step chained pipeline per partition on CERN HTCondor, keeping **all events** (even those with less than two reconstructed DS tracks), extracting truth branches (including `mctruth_weight = 1.0`), and generating 2D event displays with `--mc-truth` and `--recoMuons` enabled.

```
[Raw Genie-TGeant4 MC Partition]
               │
               ▼ Step 1: run_dimuon_reco.py --all-events (-t dimuon_DS --nTracks 0 -ht)
[sndLHC.Genie-TGeant4_dig_2MuTrks.root]  (Keeps all events)
               │
               ▼ Step 2: mctruth_neutrinos.py
[sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root] (Extracts truth tree + mctruth_weight)
               │
               ▼ Step 3: generate_2DEventDisplays.py (--mc-truth --recoMuons)
[event_displays/partitions/displays_part<p>.root]
               │
               ▼ Merge: merge_displays_nu14_2022.sh / merge_displays_old_100fb_2022.sh
[event_displays/..._displays.root]  <-- Master ROOT File
```

---

## Presets Supported

### 1. 2022 $\nu_\mu$ ($15000\text{ fb}^{-1}$) Production (`nu14_2022`, 1000 Partitions)
- Dataset: `sndlhc_15000fb-1_2022_down/nu14/volume_volTarget` (partitions 1..1000)
- **Submit**:
  ```bash
  ./htcondor/tracking_and_truth_all_events/submit_nu14_2022.sh
  ```
- **Status**:
  ```bash
  ./htcondor/tracking_and_truth_all_events/status_nu14_2022.sh
  ```
- **Merge Displays**:
  ```bash
  ./htcondor/tracking_and_truth_all_events/merge_displays_nu14_2022.sh
  ```

### 2. Older $100\text{ fb}^{-1}$ Production (`old_100fb_2022`, 401 Partitions)
- Dataset: `sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000` (partitions 0..400)
- **Submit**:
  ```bash
  ./htcondor/tracking_and_truth_all_events/submit_old_100fb_2022.sh
  ```
- **Status**:
  ```bash
  ./htcondor/tracking_and_truth_all_events/status_old_100fb_2022.sh
  ```
- **Merge Displays**:
  ```bash
  ./htcondor/tracking_and_truth_all_events/merge_displays_old_100fb_2022.sh
  ```

---

## Running Locally in Parallel (Alternative to HTCondor)

To run all events locally across worker threads on an interactive node:

- **2022 $\nu_\mu$ ($15000\text{ fb}^{-1}$)**:
  ```bash
  ./scripts/run_pipeline_parallel.py --preset nu14_2022 -j 10 --force --all-events
  ```

- **Older $100\text{ fb}^{-1}$ Production**:
  ```bash
  ./scripts/run_pipeline_parallel.py --preset old_100fb_2022 -j 10 --force --all-events
  ```

---

## Directory Overview

- **`pipeline.sub`**: HTCondor job submit file configured for `AlmaLinux9`, `group_u_SNDLHC.users`, and `longlunch` (up to 2h per job) with `FORCE=1 ALL_EVENTS=1`.
- **`run_pipeline_partition.sh`**: Chained worker script that executes all three steps sequentially within the worker slot:
  1. Tracking with all events stored (`run_dimuon_reco.py -t dimuon_DS --all-events`)
  2. Truth extraction (`mctruth_neutrinos.py`)
  3. 2D Event Display generation (`generate_2DEventDisplays.py --mc-truth --recoMuons`)
- **`args_nu14_volTarget_1000.txt`**: Partitions 1..1000 for the $15000\text{ fb}^{-1}$ production.
- **`args_old_100fb_400.txt`**: Partitions 0..400 for the older $100\text{ fb}^{-1}$ production.
- **`sndswEnv.sh`**: Frozen static environment variables (`LD_LIBRARY_PATH`, `PYTHONPATH`, `ROOTSYS`, etc.).
- **`out/`**, **`err/`**, **`log/`**: HTCondor log directories.
