# Stage 6: Event Prefiltering (Essentials)

Runs the universal event prefiltering pipeline (`scripts/prefilter.py`) on reconstructed and skimmed Monte Carlo files:
- **Main Stream**: `sndLHC.Genie-TGeant4_MCEB_dig_2MuTrks_truth.root` -> `prefiltered/newP8/essentials/<p>/...`
- **Direct Signal**: `..._charmDimuonDirect.root` -> `prefiltered/newP8/essentials-direct/<p>/...`
- **Indirect Signal**: `..._charmDimuonIndirect.root` -> `prefiltered/newP8/essentials-indirect/<p>/...`

## Job Characteristics
- **Flavour**: `microcentury` (max runtime 1h, typical execution time ~30-60 seconds per partition)
- **CPUs**: 1 CPU per job (`-w 1 -j 1`)
- **Memory**: 2 GB
- **Disk**: 2 GB

## Standalone Execution

```bash
cd /afs/cern.ch/work/i/idioniso/snd-numu-charm/htcondor/prefilter

# Submit all 1000 partitions
./submit_prefilter.sh -p 1-1000

# Monitor progress
./status_prefilter.sh

# Verify outputs
python3 verify_prefilter_outputs.py -p 1-1000

# Merge cutflow and CSV
python3 merge_prefilter_cutflow.py
```

## DAGMan Integration

Prefilter is integrated as the 6th stage in the DAGMan workflow:
```
MCEB -> TRUTH -> DIGI -> TRACK -> PREFILTER
```

Submit with:
```bash
cd /afs/cern.ch/work/i/idioniso/snd-numu-charm/htcondor/dagman
./submit_dag.sh -p 1-1000 -s mceb,mctruth,digi,track,prefilter --force
```
