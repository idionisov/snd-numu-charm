# HTCondor Batch Submission Pipelines for SND@LHC

This directory houses the HTCondor batch submission setups for running Monte Carlo pipelines concurrently across the CERN HTCondor batch cluster.

---

## Pipelines

1. **[`tracking_and_truth/`](./tracking_and_truth)**:
   - **Recommended Chained Pipeline**: Runs dimuon tracking (`_2MuTrks.root`) followed immediately by truth extraction (`_2MuTrks_truth.root`) within a single worker slot per partition.
   - Job Flavour: `longlunch` (up to 2h per job).
   - Submit helper: `./htcondor/tracking_and_truth/submit.sh`
   - Status helper: `./htcondor/tracking_and_truth/status.sh`

2. **[`tracking/`](./tracking)**:
   - **Standalone Dimuon Tracking**: Runs downstream dimuon tracking (`run_TrackSelections.py -t dimuon_DS --nTracks 2 -ht`) via [`scripts/run_dimuon_reco.py`](../scripts/run_dimuon_reco.py).
   - Generates: `sndLHC.Genie-TGeant4_digCPP_dimuonReco.root` across all 400 partitions on EOS.
   - Job Flavour: `microcentury` (up to 1h per job).
   - Submit helper: `./htcondor/tracking/submit.sh`
   - Status helper: `./htcondor/tracking/status.sh`

3. **[`mctruth_neutrinos/`](./mctruth_neutrinos)**:
   - **Standalone MCTruth**: Neutrino event categorization and truth branch extraction on raw or pre-filtered MC files via [`scripts/mctruth_neutrinos.py`](../scripts/mctruth_neutrinos.py).
   - Generates: `sndLHC.Genie-TGeant4_digCPP_truth.root` across all 400 partitions on EOS.
   - Job Flavour: `longlunch` (up to 2h per job).
   - Submit helper: `./htcondor/mctruth_neutrinos/submit.sh`
   - Status helper: `./htcondor/mctruth_neutrinos/status.sh`
