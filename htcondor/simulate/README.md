# Stage 1: Geant4 Neutrino Simulation (Pythia8 Decayer)

Modular HTCondor batch submission setup for running the Geant4 simulation step of SND@LHC neutrino interactions using Pythia8 (`DecayConfigPy8.C`).

## Pipeline Stage
- **Input**: GENIE files (`*gst.root`, `*ghep.root`) from `/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2mmRangeCut/sndlhc_15000fb-1_2022_down/nu14/volume_volTarget/<partition>`
- **Output**: Simulated detector events and geometry in `/eos/user/i/idioniso/snd-numu-charm/data/newP8/<partition>/`:
  - `sndLHC.Genie-TGeant4.root` (Geant4 simulation truth hits & tracks)
  - `geofile_full.Genie-TGeant4.root` (Detector geometry file)
  - `*.gst.root` / `*.ghep.root` (Staged input GENIE records)

## Files
- `simulate.sub`: HTCondor submit description file.
- `run_simulate_partition.sh`: Worker script executing `run_simSND.py` inside local scratch, verifying file integrity, and moving outputs to EOS.
- `submit_simulate.sh`: Helper CLI script for submitting jobs (supports partition subsets, force mode, dry run).
- `status_simulate.sh`: Status monitor checking completed partitions on EOS and HTCondor queue.
- `verify_simulate_outputs.py`: Detailed ROOT file integrity auditor.
- `sndswEnv.sh`: Exact static environment configuration for lxplus worker nodes.
- `args_nu14_volTarget_1000.txt`: List of target partition IDs (1..1000).

## Usage Examples

### 1. Test Single Partition Locally
```bash
./run_simulate_partition.sh 17
```

### 2. Submit Subset of Partitions to HTCondor
```bash
# Partitions 1 to 10:
./submit_simulate.sh -p 1-10

# Specific partitions:
./submit_simulate.sh -p 1,5,10,17
```

### 3. Submit All 1000 Partitions
```bash
./submit_simulate.sh
```

### 4. Monitor & Audit
```bash
./status_simulate.sh
python3 verify_simulate_outputs.py -p 1-1000
```
