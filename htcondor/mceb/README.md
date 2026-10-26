# Stage 2: Monte Carlo Event Builder (MCEB)

Modular HTCondor batch submission setup for running the Monte Carlo Event Builder step (`run_MCEventBuilder.py`) of SND@LHC neutrino interactions.

## Pipeline Stage
- **Input**:
  - `sndLHC.Genie-TGeant4.root` (Geant4 simulation from Stage 1)
  - `geofile_full.Genie-TGeant4.root` (Detector geometry file)
- **Output**:
  - `sndLHC.Genie-TGeant4_MCEB.root` in `/eos/user/i/idioniso/snd-numu-charm/data/newP8/<partition>/`

## Files
- `mceb.sub`: HTCondor submit description file.
- `run_mceb_partition.sh`: Worker script staging input files into local scratch, executing `run_MCEventBuilder.py`, validating ROOT tree integrity, and transferring output to EOS.
- `submit_mceb.sh`: Submission wrapper script with support for partition lists/ranges, `--force`, `--dry-run`, and flavour settings.
- `status_mceb.sh`: Quick status monitor showing available input simulation files and completed MCEB partitions.
- `verify_mceb_outputs.py`: Deep integrity auditor checking `cbmsim` entry counts and non-zombie status.
- `sndswEnv.sh`: Exact unmodified environment configuration for lxplus worker nodes.
- `args_nu14_volTarget_1000.txt`: Partition IDs 1 to 1000.

## Usage Examples

### 1. Test Single Partition Locally
```bash
./run_mceb_partition.sh 17
```

### 2. Submit Subset of Partitions to HTCondor
```bash
# Partitions 1 to 10:
./submit_mceb.sh -p 1-10

# Specific partitions:
./submit_mceb.sh -p 1,5,10,17
```

### 3. Submit All 1000 Partitions
```bash
./submit_mceb.sh
```

### 4. Monitor & Audit
```bash
./status_mceb.sh
python3 verify_mceb_outputs.py -p 1-1000
```
