#!/usr/bin/env python3
"""
scripts/filter_neutrinos.py (Forwarder)
---------------------------------------
Forwarder to scripts/mctruth_neutrinos.py.
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_new_script = os.path.join(_script_dir, "mctruth_neutrinos.py")

os.execv(sys.executable, [sys.executable, _new_script] + sys.argv[1:])
