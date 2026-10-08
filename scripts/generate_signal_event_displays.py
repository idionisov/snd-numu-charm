#!/usr/bin/env python3
"""
scripts/generate_signal_event_displays.py
-----------------------------------------
Backward-compatible wrapper forwarder.
Directs to `scripts/generate_2DEventDisplays.py`.
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_SCRIPT = os.path.join(SCRIPT_DIR, "generate_2DEventDisplays.py")

if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, NEW_SCRIPT] + sys.argv[1:])
