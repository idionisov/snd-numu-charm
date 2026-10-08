#!/usr/bin/env python3
"""
scripts/select_neutrino_category.py (Forwarder)
-----------------------------------------------
Forwarder to scripts/mctruth_neutrino_category.py.
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_new_script = os.path.join(_script_dir, "mctruth_neutrino_category.py")

os.execv(sys.executable, [sys.executable, _new_script] + sys.argv[1:])
