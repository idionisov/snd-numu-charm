# HTCondor Simulation, MCEB, and Digitization (Pythia8) Pipeline

This directory provides the HTCondor batch submission setup to regenerate all 1000 partitions of neutrino Monte Carlo using the updated Pythia8 external decayer macro (`gconfig/DecayConfigPy8.C`).

---

## Pipeline Workflow

For each partition (1..1000):
1. **Geant4 Simulation (`shipLHC/run_simSND.py`)**:
   - Takes the input GENIE neutrino interactions:
     `/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/2mmRangeCut/sndlhc_15000fb-1_2022_down/nu14/volume_volTarget/<partition>/sndlhc_+volTarget_SND_LHC_SNDG18_02a_01_000.0.gst.root`
   - Configures Pythia8 as external decayer via local `gconfig/DecayConfigPy8.C`.
   - Propagates particles through the 2022 SND@LHC geometry (`volTarget`).
   - Generates `sndLHC.Genie-TGeant4.root` and `geofile_full.Genie-TGeant4.root`.
2. **MC Event Builder (`shipLHC/run_MCEventBuilder.py`)**:
   - Assembles hits into `sndLHC.Genie-TGeant4_MCEB.root`.
3. **Digitization (`shipLHC/run_digiSND.py`)**:
   - Runs on raw simulation: generates `sndLHC.Genie-TGeant4_dig.root`.
   - Runs on MCEB simulation: generates `sndLHC.Genie-TGeant4_MCEB_dig.root`.
4. **Final Export**:
   - All files are staged and saved to `/eos/user/i/idioniso/snd-numu-charm/data/newP8/<partition>/`.

---

## Directory Layout

- [`sim_pipeline.sub`](./sim_pipeline.sub): HTCondor submission configuration.
- [`run_sim_partition.sh`](./run_sim_partition.sh): Worker script running the full 4-step pipeline per partition on local scratch.
- [`submit_simP8.sh`](./submit_simP8.sh): Helper script to submit all 1000 partitions (or a subset) to HTCondor.
- [`status_simP8.sh`](./status_simP8.sh): Fast multi-threaded EOS progress monitor.
- [`verify_simP8_outputs.py`](./verify_simP8_outputs.py): Integrity check verifying ROOT trees and entry consistency.
- [`sndswEnv.sh`](./sndswEnv.sh): Environment setup script ensuring local branch paths and decayer are loaded.
- [`args_nu14_volTarget_1000.txt`](./args_nu14_volTarget_1000.txt): Partition IDs (1 to 1000).

---

## Usage

### 1. Submit Jobs to HTCondor (from lxplus)

```bash
# Submit all 1000 partitions (JobFlavour: workday, up to 8 hours)
./submit_simP8.sh

# Or submit a test partition or subset:
condor_submit sim_pipeline.sub -queue "partition in 1 2 3 4 5"
```

### 2. Monitor Progress

```bash
./status_simP8.sh
```

### 3. Verify Outputs

```bash
# Check first 5 partitions
python3 verify_simP8_outputs.py -p 1-5

# Check all 1000 partitions with 16 parallel threads
python3 verify_simP8_outputs.py -p 1-1000 -j 16
```
