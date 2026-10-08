#!/usr/bin/env python3
"""
scripts/filter_signal_numu_charm.py (Forwarder)
-----------------------------------------------
DEPRECATED: This script has been superseded by scripts/filter_neutrinos.py.

Signal event skimming has been replaced with universal neutrino interaction categorization.
All events are categorized by flavor, current, primary outgoing particles (channel ID lookup),
and charm direct decay channel so custom signatures can be filtered downstream.

This forwarder redirects execution directly to scripts/filter_neutrinos.py.
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_new_script = os.path.join(_script_dir, "filter_neutrinos.py")

print("[Notice] scripts/filter_signal_numu_charm.py has been superseded by scripts/filter_neutrinos.py.", file=sys.stderr)
print("[Notice] Running scripts/filter_neutrinos.py with your arguments...\n", file=sys.stderr)

os.execv(sys.executable, [sys.executable, _new_script] + sys.argv[1:])
