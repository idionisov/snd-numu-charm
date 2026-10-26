# Stage 3: MC Truth Extraction & Transplantation

Modular HTCondor batch submission setup for Stage 3 of the SND@LHC neutrino simulation pipeline.

## Pipeline Stage
Enriches Monte Carlo Event Builder (MCEB) files with full unpruned truth from original Geant4 simulation files:
- **Truth Source**: `sndLHC.Genie-TGeant4.root` (contains unpruned `MCTrack`, `ScifiPoint`, `MuFilterPoint`)
- **Input MCEB**: `sndLHC.Genie-TGeant4_MCEB.root`
- **Output Target**: `sndLHC.Genie-TGeant4_MCEB.root` (in-place update or distinct target)
  - `cbmsim;1`: Original MCEB tree preserved intact
  - `cbmsim;2`: Cloned tree with unpruned `MCTrack`, `ScifiPoint`, and `MuFilterPoint` transplanted via 1-to-1 event number matching
  - `mctruth;1`: Comprehensive categorized truth observables tree populated on the fly via `NeutrinoTruthProcessor`

## Files
- `mctruth.sub`: HTCondor submit description file.
- `run_mctruth_partition.sh`: Worker script executing `map_mctruth_to_mceb.py`, verifying all three tree cycles, and reporting results.
- `submit_mctruth.sh`: Submission wrapper script with support for partition subsets, custom output patterns, `--force`, and `--dry-run`.
- `status_mctruth.sh`: Monitor displaying available MCEB files and completed truth-enriched partitions on EOS.
- `verify_mctruth_outputs.py`: Deep auditor verifying `cbmsim;1`, `cbmsim;2`, `mctruth`, and unpruned track sizes.
- `sndswEnv.sh`: Exact unmodified environment configuration for lxplus worker nodes.
- `args_nu14_volTarget_1000.txt`: Partition IDs 1 to 1000.

## Usage Examples

### 1. Test Single Partition Locally
```bash
./run_mctruth_partition.sh 17
```

### 2. Submit Subset of Partitions to HTCondor
```bash
# Partitions 1 to 10:
./submit_mctruth.sh -p 1-10

# Specific partitions:
./submit_mctruth.sh -p 1,5,10,17
```

### 3. Submit All 1000 Partitions
```bash
./submit_mctruth.sh
```

### 4. Monitor & Audit
```bash
./status_mctruth.sh
python3 verify_mctruth_outputs.py -p 1-1000
```
