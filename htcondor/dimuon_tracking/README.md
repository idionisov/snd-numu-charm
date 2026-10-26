# Stage 5: Dimuon Track Reconstruction & Skimming

Modular HTCondor batch submission setup for Stage 5 of the SND@LHC neutrino simulation pipeline.

## Pipeline Stage
Performs Scifi track reconstruction, charm dimuon skimming, and 2D event display generation:
- **Input**:
  - `sndLHC.Genie-TGeant4_MCEB_dig.root` in `/eos/user/i/idioniso/snd-numu-charm/data/newP8/<partition>/`
- **Output**:
  - Tracked file: `sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root` (contains `Reco_MuonTracks`, `cbmsim;1`, `cbmsim;2`, `mctruth;1`)
  - Direct charm skim: `sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonDirect.root`
  - Indirect charm skim: `sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth_charmDimuonIndirect.root`
  - 2D Event Displays: `/eos/user/i/idioniso/snd-numu-charm/event_displays/newP8_MCEB/partitions/{direct,indirect}/displays_part<p>.root`

## Files
- `dimuon_tracking.sub`: HTCondor submit description file.
- `run_dimuon_tracking_partition.sh`: Worker script executing `run_dimuon_reco.py`, `filter_charm_dimuon_categories.py`, and `generate_2DEventDisplays.py`.
- `submit_dimuon_tracking.sh`: Submission wrapper script supporting partition subsets, `--two-tracks-only`, `--skip-displays`, `--force`, and `--dry-run`.
- `status_dimuon_tracking.sh`: Monitor displaying completed tracking files, direct/indirect skims, and display files on EOS.
- `verify_dimuon_tracking_outputs.py`: Deep auditor verifying `Reco_MuonTracks`, `mctruth` integrity, and skim counts.
- `sndswEnv.sh`: Exact unmodified environment configuration for lxplus worker nodes.
- `args_nu14_volTarget_1000.txt`: Partition IDs 1 to 1000.

## Usage Examples

### 1. Test Single Partition Locally
```bash
./run_dimuon_tracking_partition.sh 17
```

### 2. Submit Subset of Partitions to HTCondor
```bash
# Partitions 1 to 10:
./submit_dimuon_tracking.sh -p 1-10

# Specific partitions:
./submit_dimuon_tracking.sh -p 1,5,10,17
```

### 3. Submit All 1000 Partitions
```bash
./submit_dimuon_tracking.sh
```

### 4. Monitor & Audit
```bash
./status_dimuon_tracking.sh
python3 verify_dimuon_tracking_outputs.py -p 1-1000
```
