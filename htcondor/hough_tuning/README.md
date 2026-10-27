# HTCondor Hough Tracking Parameter Tuning Grid Search

Distributed optimization of SND@LHC Downstream Dimuon Tracking Hough parameters across HTCondor batch slots.

Evaluates configurations simultaneously across 4 benchmark datasets:
1. Genie Direct Signal (TPR)
2. Particle Gun Signal (TPR)
3. Genie Background (FPR)
4. Passing Muons `pmu2022` Background (FPR)

Objective metric: Asimov Discovery Significance $Z = \sqrt{2 \cdot [(s + b)\ln(1 + s/b) - s]}$ and Score $(1 - \text{FPR}) \cdot \text{TPR}^2$.

---

## Directory Structure

```
htcondor/hough_tuning/
├── hough_tuning.sub           # HTCondor job submission description
├── run_hough_tuning_job.sh    # Worker execution wrapper for each job slice
├── submit_hough_tuning.sh     # Interactive submission script
├── status_hough_tuning.sh     # Batch monitoring script
├── merge_hough_tuning_csv.py  # Gathers and merges per-job CSVs into master CSV
├── sndswEnv.sh                # Complete CVMFS / ROOT / Python environment
├── out/                       # Standard output logs
├── err/                       # Standard error logs
└── log/                       # HTCondor cluster logs
```

---

## Quick Start

### 1. Submit Grid Search (e.g. 50 parallel job slices)

```bash
# Dry-run verification:
./htcondor/hough_tuning/submit_hough_tuning.sh -j 50 --dry-run

# Submit 50 jobs with microcentury flavour (~20 min max):
./htcondor/hough_tuning/submit_hough_tuning.sh -j 50 -f microcentury
```

### 2. Monitor Job Progress

```bash
./htcondor/hough_tuning/status_hough_tuning.sh -j 50
```

### 3. Merge Results and View Scoreboard

Once all jobs have finished, merge the generated per-job CSV files into the consolidated results CSV:

```bash
python3 htcondor/hough_tuning/merge_hough_tuning_csv.py \
    -i /eos/user/i/idioniso/snd-numu-charm/plots/hough_tuning/jobs \
    -o /eos/user/i/idioniso/snd-numu-charm/plots/hough_tuning/hough_fom_pairwise_results.csv \
    --sort-by asimov_z \
    --plot
```

The output CSV will match the structure and column format of [`hough_fom_pairwise_results.csv`](file:///eos/user/i/idioniso/snd-numu-charm/plots/hough_tuning/hough_fom_pairwise_results.csv).
