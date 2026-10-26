#!/usr/bin/env python3
"""
htcondor/dagman/generate_dag.py
-------------------------------
Generates a complete per-partition HTCondor DAGMan workflow file for the
SND@LHC neutrino simulation, event building, truth extraction, digitization,
and track reconstruction pipeline.

Architecture per partition:
  SIM_<p> -> MCEB_<p> -> TRUTH_<p> -> DIGI_<p> -> TRACK_<p>

Features:
- Fast partitions proceed immediately without waiting for other partitions.
- Configurable stage subsets (--stages sim,mceb,mctruth,digi,track).
- Automatic retries per node for network/FUSE resilience.
- DAGMan configuration integration (concurrency limits & log scanning).

Usage:
  python3 generate_dag.py -p 1-10 -o test.dag
  python3 generate_dag.py -p 1-1000 -o pipeline.dag
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Optional


STAGE_CONFIG = {
    "sim": {
        "name": "SIM",
        "sub": "sub/simulate.sub",
    },
    "mceb": {
        "name": "MCEB",
        "sub": "sub/mceb.sub",
    },
    "mctruth": {
        "name": "TRUTH",
        "sub": "sub/mctruth.sub",
    },
    "digi": {
        "name": "DIGI",
        "sub": "sub/digi.sub",
    },
    "track": {
        "name": "TRACK",
        "sub": "sub/dimuon_tracking.sub",
    },
}

ALL_STAGES_ORDER = ["sim", "mceb", "mctruth", "digi", "track"]


def parse_partitions(part_arg: Optional[str]) -> List[int]:
    """Parse partition range or comma-separated list."""
    if not part_arg or part_arg.strip().lower() == "all":
        return list(range(1, 1001))

    partitions = set()
    for token in part_arg.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start, end = int(parts[0]), int(parts[1])
                for p in range(min(start, end), max(start, end) + 1):
                    partitions.add(p)
        elif token.isdigit():
            partitions.add(int(token))

    return sorted(list(partitions))


def main():
    parser = argparse.ArgumentParser(description="Generate HTCondor DAGMan file for SND@LHC pipeline")
    parser.add_argument("-p", "--partitions", type=str, default="1-1000", help="Partition range (e.g. 1-1000, 1-10, 1,2,5)")
    parser.add_argument("-o", "--output", type=str, default="pipeline.dag", help="Output DAG filename")
    parser.add_argument("-s", "--stages", type=str, default="all", help="Comma-separated stages: sim,mceb,mctruth,digi,track (default: all)")
    parser.add_argument("-r", "--retries", type=int, default=2, help="Number of automatic retries per node (default: 2)")
    parser.add_argument("--force", action="store_true", help="Set force=1 to overwrite existing outputs")
    parser.add_argument("--skip-displays", action="store_true", help="Set skip_displays=1 in tracking stage")
    parser.add_argument("--config", type=str, default="dagman.config", help="DAGMan config file (default: dagman.config)")
    args = parser.parse_args()

    partitions = parse_partitions(args.partitions)
    if not partitions:
        print("ERROR: No valid partitions specified.", file=sys.stderr)
        sys.exit(1)

    if args.stages.strip().lower() == "all":
        active_stages = list(ALL_STAGES_ORDER)
    else:
        req = [s.strip().lower() for s in args.stages.split(",") if s.strip()]
        active_stages = [s for s in ALL_STAGES_ORDER if s in req]

    if not active_stages:
        print(f"ERROR: No valid stages found in '{args.stages}'. Valid: {ALL_STAGES_ORDER}", file=sys.stderr)
        sys.exit(1)

    force_val = "1" if args.force else "0"
    skip_disp_val = "1" if args.skip_displays else "0"

    print("==========================================================")
    print(" Generating HTCondor DAGMan Workflow File")
    print(f" Partitions   : {len(partitions)} ({partitions[0]}..{partitions[-1]})")
    print(f" Stages       : {' -> '.join([s.upper() for s in active_stages])}")
    print(f" Retries      : {args.retries} per node")
    print(f" Force Mode   : {force_val}")
    print(f" Output File  : {args.output}")
    print("==========================================================")

    lines = []
    lines.append("# ==============================================================================")
    lines.append("# HTCondor DAGMan Workflow: SND@LHC Simulation & Reconstruction Pipeline")
    lines.append(f"# Generated partitions: {len(partitions)} | Stages: {' -> '.join(active_stages)}")
    lines.append("# ==============================================================================\n")

    if args.config and os.path.exists(os.path.join(os.path.dirname(args.output) or ".", args.config)):
        lines.append(f"CONFIG {args.config}\n")
    elif args.config:
        lines.append(f"CONFIG {args.config}\n")

    for p in partitions:
        lines.append(f"# --- Partition {p} ---")
        prev_node = None
        for stage in active_stages:
            node_name = f"{STAGE_CONFIG[stage]['name']}_{p}"
            sub_file = STAGE_CONFIG[stage]["sub"]
            lines.append(f"JOB {node_name} {sub_file}")

            # Vars
            vars_list = [f'partition="{p}"', f'force="{force_val}"']
            if stage == "track":
                vars_list.append(f'skip_displays="{skip_disp_val}"')
            lines.append(f"VARS {node_name} {' '.join(vars_list)}")

            if args.retries > 0:
                lines.append(f"RETRY {node_name} {args.retries}")

            if prev_node is not None:
                lines.append(f"PARENT {prev_node} CHILD {node_name}")

            prev_node = node_name

        lines.append("")

    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")

    total_nodes = len(partitions) * len(active_stages)
    print(f" Successfully generated '{args.output}' with {total_nodes} total job nodes!")
    print(f" Submit with: condor_submit_dag {args.output}")


if __name__ == "__main__":
    main()
