# Stage 4: Detector Digitization

Modular HTCondor batch submission setup for Stage 4 of the SND@LHC neutrino simulation pipeline.

## Pipeline Stage
Simulates detector response (Scifi and MuFilter SiPM channels, time-walk, saturation, electronic thresholds, and cluster creation):
- **Input**:
  - `sndLHC.Genie-TGeant4_MCEB.root` (Stage 2/3 MCEB file with unpruned truth and `mctruth`)
  - `geofile_full.Genie-TGeant4.root` (Detector geometry file)
- **Output**:
  - `sndLHC.Genie-TGeant4_MCEB_dig.root` in `/eos/user/i/idioniso/snd-numu-charm/data/newP8/<partition>/`
  - (and `sndLHC.Genie-TGeant4_dig.root` if raw sim digitization was missing)

## Files
- `digi.sub`: HTCondor submit description file.
- `run_digi_partition.sh`: Worker script executing `run_digiSND.py` in local scratch, verifying `Digi_ScifiHits` and `Digi_MuFilterHits` branches, and transferring output to EOS.
- `submit_digi.sh`: Submission wrapper script with support for partition subsets, `--force`, and `--dry-run`.
- `status_digi.sh`: Monitor displaying available MCEB files and completed digitized partitions on EOS.
- `verify_digi_outputs.py`: Deep auditor verifying digitized branch presence and entry counts.
- `sndswEnv.sh`: Exact unmodified environment configuration for lxplus worker nodes.
- `args_nu14_volTarget_1000.txt`: Partition IDs 1 to 1000.

## Usage Examples

### 1. Test Single Partition Locally
```bash
./run_digi_partition.sh 17
```

### 2. Submit Subset of Partitions to HTCondor
```bash
# Partitions 1 to 10:
./submit_digi.sh -p 1-10

# Specific partitions:
./submit_digi.sh -p 1,5,10,17
```

### 3. Submit All 1000 Partitions
```bash
./submit_digi.sh
```

### 4. Monitor & Audit
```bash
./status_digi.sh
python3 verify_digi_outputs.py -p 1-1000
```
