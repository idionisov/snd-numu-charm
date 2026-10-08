# HTCondor Batch Submission: Chained Tracking -> MCTruth -> Event Displays

This directory runs the complete end-to-end 3-step chained pipeline per partition on CERN HTCondor:

```
[Raw Genie-TGeant4 MC Partition]
               │
               ▼ Step 1: run_dimuon_reco.py (~1–2 min)
[sndLHC.Genie-TGeant4_digCPP_2MuTrks.root]
               │
               ▼ Step 2: mctruth_neutrinos.py (~10–30 sec)
[sndLHC.Genie-TGeant4_digCPP_2MuTrks_truth.root]
               │
               ▼ Step 3: generate_2DEventDisplays.py (--mc-truth --muonReco) (~5–15 sec)
[event_displays/partitions/displays_part<p>.root]
               │
               ▼ Merge: merge_displays.sh (or scripts/merge_event_displays.py)
[event_displays/numu_dimuon_signal_event_displays.root]  <-- Single Master ROOT File
```

---

## Directory Overview

- **`pipeline.sub`**: HTCondor job submit file configured for `AlmaLinux9`, `group_u_SNDLHC.users`, and `longlunch` (up to 2h per job).
- **`run_pipeline_partition.sh`**: Chained worker script that executes all three steps sequentially within the same worker slot:
  1. Dimuon tracking (`run_dimuon_reco.py -t dimuon_DS --nTracks 2 -ht`)
  2. Truth extraction (`mctruth_neutrinos.py`)
  3. 2D Event Display generation (`generate_2DEventDisplays.py --mc-truth --muonReco`)
- **`args_partitions.txt`**: All 400 valid MC partition numbers.
- **`sndswEnv.sh`**: Frozen static environment variables (`LD_LIBRARY_PATH`, `PYTHONPATH`, `ROOTSYS`, etc.).
- **`submit.sh`**: Helper submission script to submit all 400 jobs from `lxplus`.
- **`status.sh`**: Monitor reporting completed files for all three stages and `condor_q`.
- **`merge_displays.sh`**: Helper script to merge all per-partition event display ROOT files into the single master ROOT file.
- **`out/`**, **`err/`**, **`log/`**: HTCondor log directories.

---

## Quick Start (from `lxplus`)

### 1. Submit All 400 Jobs

```bash
./htcondor/tracking_and_truth/submit.sh
```
*(Or directly: `condor_submit htcondor/tracking_and_truth/pipeline.sub`)*

### 2. Check Live Status

```bash
./htcondor/tracking_and_truth/status.sh
```
This shows the progress percentage for:
- Step 1: `_2MuTrks.root`
- Step 2: `_2MuTrks_truth.root`
- Step 3: `displays_part*.root`
- Master single ROOT file status

### 3. Merge All Event Displays into One Single ROOT File

Once the batch jobs finish (or at any intermediate point):
```bash
./htcondor/tracking_and_truth/merge_displays.sh
```

---

## 2022 $\nu_\mu$ Production (`sndlhc_15000fb-1_2022_down/nu14/volume_volTarget`, 1000 Partitions)

To run the full pipeline on the 1000-partition $15000\text{ fb}^{-1}$ production:

### 1. Submit all 1000 partitions:
```bash
./htcondor/tracking_and_truth/submit_nu14_2022.sh
```

### 2. Monitor status:
```bash
./htcondor/tracking_and_truth/status_nu14_2022.sh
```

### 3. Merge displays into single file:
```bash
./htcondor/tracking_and_truth/merge_displays_nu14_2022.sh
```
Outputs:
- Tracked files: `/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/<partition>/sndLHC.Genie-TGeant4_dig_2MuTrks.root`
- Truth files: `/eos/user/i/idioniso/snd-numu-charm/data/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget/<partition>/sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root`
- Master Displays: `/eos/user/i/idioniso/snd-numu-charm/event_displays/sndlhc_15000fb-1_2022_down_nu14_volume_volTarget_displays.root`

---

## Custom Parameter Overrides

You can override any path pattern directly when submitting:
```bash
condor_submit htcondor/tracking_and_truth/pipeline.sub \
  input_pattern="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP.root" \
  track_pattern="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_2MuTrks.root" \
  truth_pattern="/eos/user/i/idioniso/snd-numu-charm/data/%s/sndLHC.Genie-TGeant4_digCPP_2MuTrks_truth.root" \
  disp_pattern="/eos/user/i/idioniso/snd-numu-charm/event_displays/partitions/displays_part%s.root"
```
