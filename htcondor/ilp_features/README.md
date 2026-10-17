# HTCondor Batch Submission: ILP Track Reconstruction & Feature Extraction

This directory provides HTCondor job submission to run Integer Linear Programming (ILP) Downstream (DS) MuFilter track reconstruction and feature extraction across all 1000 partitions of the $15000\text{ fb}^{-1}$ 2022 $\nu_\mu$ production with job flavour **`workday`** (up to 8 hours per job).

Each job processes one partition file in-place:
```
[sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root]
               │
               ▼ extract_ilp_features.py (ILP optimization via scipy.optimize.milp)
[sndLHC.Genie-TGeant4_dig_2MuTrks_truth.root]  <-- Injects new "ilp" TTree alongside "cbmsim"
```

The injected `ilp` `TTree` contains purely data-compatible reconstruction variables:
- 1-track and 2-track optimal costs $J^*(1)$ and $J^*(2)$
- Dimuon gain metric $\Delta J = J^*(1) - J^*(2)$
- Reconstructed 3D track parameters (slopes, intercepts, $\chi^2$, NDF, planes, timing, QDC)
- Analytical 3D distance of closest approach ($\text{DCA}_{3\text{D}}$) and reconstructed vertex
- Combined dimuon quality $\chi^2_{2\mu}$ and composite discriminant score $\mathcal{D}_{2\mu}$

---

## Quick Usage

From any CERN `lxplus` node:

### 1. Submit All 1000 Jobs (flavour `workday`)
```bash
./htcondor/ilp_features/submit_nu14_2022.sh
```
or:
```bash
condor_submit htcondor/ilp_features/ilp.sub
```

### 2. Check Job Status & Partition Progress
```bash
./htcondor/ilp_features/status_nu14_2022.sh
```

### 3. Custom Partition Range (Optional)
To test on a subset first (e.g. partition 1 and 100):
```bash
echo -e "1\n100" > /tmp/test_parts.txt
./htcondor/ilp_features/submit_nu14_2022.sh args_file=/tmp/test_parts.txt
```

---

## Directory Files

- **`ilp.sub`**: HTCondor job description file configured for AlmaLinux9, flavour `workday` (8h), and 4GB memory.
- **`run_ilp_partition.sh`**: Worker script executed in the Condor slot for each partition.
- **`submit_nu14_2022.sh`**: Submits 1000 partition jobs.
- **`status_nu14_2022.sh`**: Monitors queue and partition processing progress.
- **`args_nu14_volTarget_1000.txt`**: Partitions 1..1000.
- **`sndswEnv.sh`**: Pre-configured environment variables for worker nodes.
- **`out/`**, **`err/`**, **`log/`**: Job logs and standard outputs.
