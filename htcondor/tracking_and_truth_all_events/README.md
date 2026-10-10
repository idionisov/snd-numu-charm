# HTCondor Batch Submission: Chained Pipeline (All Events)
## Tracking (All Events) -> MCTruth Extraction -> 2D Event Displays

This directory runs the complete end-to-end 3-step chained pipeline per partition on CERN HTCondor, keeping **all events** (even those with less than two reconstructed DS tracks), extracting truth branches (including `mctruth_weight = 1.0`), and generating 2D event displays with `--mc-truth` and `--recoMuons` enabled.

```
[Raw Genie-TGeant4 MC Partition]
               │
               ▼ Step 1: run_dimuon_reco.py --all-events (-t dimuon_DS --nTracks 0 -ht)
[sndLHC.Genie-TGeant4_dig_2MuTrks.root]  (Contains all events)
               │
               ▼ Step 2: mctruth_neutrinos.py
[sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root] (Contains truth tree + mctruth_weight)
               │
               ▼ Step 3: generate_2DEventDisplays.py (--mc-truth --recoMuons)
[event_displays/partitions/displays_part<p>.root]
               │
               ▼ Merge: merge_displays_nu14_2022.sh (scripts/merge_event_displays.py)
[event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_displays.root]  <-- Master ROOT File
```

---

## Directory Overview

- **`pipeline.sub`**: HTCondor job submit file configured for `AlmaLinux9`, `group_u_SNDLHC.users`, and `longlunch` (up to 2h per job) with `FORCE=1 ALL_EVENTS=1`.
- **`run_pipeline_partition.sh`**: Chained worker script that executes all three steps sequentially within the worker slot:
  1. Tracking with all events stored (`run_dimuon_reco.py -t dimuon_DS --all-events`)
  2. Truth extraction (`mctruth_neutrinos.py`)
  3. 2D Event Display generation (`generate_2DEventDisplays.py --mc-truth --recoMuons`)
- **`args_nu14_volTarget_1000.txt`**: Partitions 1..1000 for the $15000\text{ fb}^{-1}$ production.
- **`sndswEnv.sh`**: Frozen static environment variables (`LD_LIBRARY_PATH`, `PYTHONPATH`, `ROOTSYS`, etc.).
- **`submit_nu14_2022.sh`** (or `submit.sh`): Submit all 1000 jobs to HTCondor from an `lxplus` node.
- **`status_nu14_2022.sh`** (or `status.sh`): Live monitoring reporting progress percentages for all stages.
- **`merge_displays_nu14_2022.sh`** (or `merge_displays.sh`): Merge partition display files into the master display file.
- **`out/`**, **`err/`**, **`log/`**: HTCondor log directories.

---

## Quick Start (from `lxplus`)

### 1. Submit All 1000 Jobs
```bash
./htcondor/tracking_and_truth_all_events/submit.sh
```
*(Or directly: `condor_submit htcondor/tracking_and_truth_all_events/pipeline.sub`)*

### 2. Monitor Progress
```bash
./htcondor/tracking_and_truth_all_events/status.sh
```

### 3. Merge Event Displays into Master File
```bash
./htcondor/tracking_and_truth_all_events/merge_displays.sh
```

Outputs:
- Tracked files: `/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/<partition>/sndLHC.Genie-TGeant4_dig_2MuTrks.root`
- Truth files: `/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/<partition>/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root`
- Partition Displays: `/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/partitions/displays_part<p>.root`
- Master Displays: `/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_displays.root`
