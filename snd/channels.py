"""
snd.channels
------------
Interaction channel categorization and persistent lookup table management.
Assigns canonical unique IDs to combinations of outgoing immediate interaction products
and charm hadron decay channels.
"""

from __future__ import annotations

import os
import json
import time
import threading
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter
from contextlib import contextmanager

try:
    from filelock import FileLock
except ImportError:
    FileLock = None


# Particle PDG to standard notation mapping
PDG_TO_NAME: Dict[int, str] = {
    # Leptons
    11: "e-",
    -11: "e+",
    12: "nu_e",
    -12: "anti_nu_e",
    13: "mu-",
    -13: "mu+",
    14: "nu_mu",
    -14: "anti_nu_mu",
    15: "tau-",
    -15: "tau+",
    16: "nu_tau",
    -16: "anti_nu_tau",
    # Gauge bosons
    22: "gamma",
    23: "Z0",
    24: "W+",
    -24: "W-",
    # Light mesons
    111: "pi0",
    211: "pi+",
    -211: "pi-",
    130: "K_L0",
    310: "K_S0",
    311: "K0",
    -311: "anti_K0",
    321: "K+",
    -321: "K-",
    221: "eta",
    331: "eta'",
    113: "rho0",
    213: "rho+",
    -213: "rho-",
    223: "omega",
    333: "phi",
    # Charmed mesons
    411: "D+",
    -411: "D-",
    421: "D0",
    -421: "anti_D0",
    431: "Ds+",
    -431: "Ds-",
    413: "D*+",
    -413: "D*-",
    423: "D*0",
    -423: "anti_D*0",
    433: "Ds*+",
    -433: "Ds*-",
    443: "J/psi",
    # Baryons
    2112: "n",
    -2112: "anti_n",
    2212: "p",
    -2212: "anti_p",
    3122: "Lambda0",
    -3122: "anti_Lambda0",
    3222: "Sigma+",
    -3222: "anti_Sigma-",
    3212: "Sigma0",
    -3212: "anti_Sigma0",
    3112: "Sigma-",
    -3112: "anti_Sigma+",
    3322: "Xi0",
    -3322: "anti_Xi0",
    3312: "Xi-",
    -3312: "anti_Xi+",
    3334: "Omega-",
    -3334: "anti_Omega+",
    # Charmed baryons
    4122: "Lambda_c+",
    -4122: "anti_Lambda_c-",
    4222: "Sigma_c++",
    -4222: "anti_Sigma_c--",
    4212: "Sigma_c+",
    -4212: "anti_Sigma_c-",
    4112: "Sigma_c0",
    -4112: "anti_Sigma_c0",
    4232: "Xi_c+",
    -4232: "anti_Xi_c-",
    4132: "Xi_c0",
    -4132: "anti_Xi_c0",
    4332: "Omega_c0",
    -4332: "anti_Omega_c0",
}


def pdg_to_name(pdg: int) -> str:
    """Return standard particle symbol for a PDG code, falling back to string representation."""
    if pdg in PDG_TO_NAME:
        return PDG_TO_NAME[pdg]
    # Check nuclei
    if pdg > 1000000000:
        z = (pdg % 10000000) // 10000
        a = (pdg % 10000) // 10
        return f"Nucleus(Z={z},A={a})"
    return f"PDG({pdg})"


def is_charmed_hadron(pdg: int) -> bool:
    """Return True if particle code corresponds to a charmed hadron."""
    code = abs(pdg)
    if code < 400:
        return False
    if code in [411, 421, 431, 4122, 4222, 4212, 4112, 4232, 4132, 4332, 413, 423, 433]:
        return True
    # General quark inspection
    mod10 = code // 10
    q1 = mod10 % 10
    q2 = (mod10 // 10) % 10
    q3 = (mod10 // 100) % 10
    q4 = (mod10 // 1000) % 10
    return (q1 == 4 or q2 == 4 or q3 == 4 or q4 == 4)


def format_channel_formula(pdgs: List[int]) -> str:
    """
    Format a sorted list of particle PDG codes into a standard physics equation string:
    e.g. [13, 211, 211, 2112] -> 'mu- + 2pi+ + n'
    """
    if not pdgs:
        return "empty"

    counts = Counter(pdgs)
    # Sort order: leptons first, then charmed hadrons, then light mesons, then baryons, then others
    def sort_key(item):
        pdg, _ = item
        abs_p = abs(pdg)
        if abs_p in [11, 12, 13, 14, 15, 16]:
            priority = 0
        elif is_charmed_hadron(pdg):
            priority = 1
        elif abs_p in [211, 111, 321, 311, 310, 130, 221, 331]:
            priority = 2
        elif abs_p in [2212, 2112, 3122, 3222, 3212, 3112, 3322, 3312]:
            priority = 3
        else:
            priority = 4
        return (priority, -abs_p)

    sorted_items = sorted(counts.items(), key=sort_key)
    parts = []
    for pdg, count in sorted_items:
        name = pdg_to_name(pdg)
        prefix = f"{count}" if count > 1 else ""
        parts.append(f"{prefix}{name}")

    return " + ".join(parts)


class ChannelLookupManager:
    """
    Persistent lookup table manager for neutrino interaction channels and charm decay channels.
    Maintains a JSON lookup table in config/interaction_channels.json.
    Assigns consistent unique integer IDs to every unique combination of products.
    Thread-safe and process-safe against concurrent reads and writes across worker processes.
    """

    _instance: Optional[ChannelLookupManager] = None
    _lock = threading.Lock()

    def __init__(self, table_path: Optional[str] = None):
        if table_path is None:
            # Default to repo root config directory
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            table_path = os.path.join(repo_root, "config", "interaction_channels.json")

        self.table_path = table_path
        self.lock_path = table_path + ".lock"
        self._lock = threading.Lock()
        self.primary_channels: Dict[str, dict] = {} # key "pdg1,pdg2,..." -> channel info
        self.primary_by_id: Dict[int, dict] = {}
        self.charm_decay_channels: Dict[str, dict] = {} # key "parent:d1,d2,..." -> decay info
        self.charm_by_id: Dict[int, dict] = {}
        self._dirty = False

        self.load()

    @classmethod
    def get_instance(cls, table_path: Optional[str] = None) -> ChannelLookupManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(table_path)
            return cls._instance

    @contextmanager
    def _file_lock(self, timeout: float = 60.0):
        """Cross-process file lock context manager."""
        if FileLock is not None:
            lock = FileLock(self.lock_path, timeout=timeout)
            with lock:
                yield
        else:
            import fcntl
            os.makedirs(os.path.dirname(os.path.abspath(self.lock_path)), exist_ok=True)
            fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
            start_t = time.time()
            locked = False
            try:
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        locked = True
                        break
                    except (BlockingIOError, OSError):
                        if time.time() - start_t > timeout:
                            raise TimeoutError(f"Timeout waiting for channel table lock {self.lock_path}")
                        time.sleep(0.05)
                yield
            finally:
                if locked:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
                os.close(fd)

    def _load_unlocked(self):
        """Read latest data from disk without locking (assumes caller holds locks)."""
        if os.path.exists(self.table_path):
            try:
                with open(self.table_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.primary_channels = data.get("primary_channels", {})
                self.charm_decay_channels = data.get("charm_decay_channels", {})
                self.primary_by_id = {
                    v["channel_id"]: v
                    for v in self.primary_channels.values()
                    if isinstance(v, dict) and "channel_id" in v
                }
                self.charm_by_id = {
                    v["decay_id"]: v
                    for v in self.charm_decay_channels.values()
                    if isinstance(v, dict) and "decay_id" in v
                }
                return
            except Exception as e:
                print(f"[Warning] Could not load channel lookup table from {self.table_path}: {e}")

        self.primary_channels = {}
        self.charm_decay_channels = {}
        self.primary_by_id = {}
        self.charm_by_id = {}

    def _save_unlocked(self):
        """Write current data atomically to disk with a process-unique temp file (assumes caller holds locks)."""
        os.makedirs(os.path.dirname(os.path.abspath(self.table_path)), exist_ok=True)
        data = {
            "metadata": {
                "description": "Lookup table of unique SND@LHC neutrino interaction channels and charm decays",
                "total_primary_channels": len(self.primary_channels),
                "total_charm_decay_channels": len(self.charm_decay_channels),
            },
            "primary_channels": self.primary_channels,
            "charm_decay_channels": self.charm_decay_channels,
        }
        tmp_path = f"{self.table_path}.tmp.{os.getpid()}_{time.time_ns()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, self.table_path)
            self._dirty = False
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def load(self):
        """Thread-safe and process-safe load from disk."""
        with self._lock:
            with self._file_lock():
                self._load_unlocked()

    def save(self):
        """Thread-safe and process-safe save to disk, merging any memory updates."""
        with self._lock:
            if not self._dirty:
                return
            with self._file_lock():
                # Read latest from disk to avoid overwriting channels registered by other workers
                current_local_primary = dict(self.primary_channels)
                current_local_charm = dict(self.charm_decay_channels)
                self._load_unlocked()

                # Merge any missing primary channels
                for k, v in current_local_primary.items():
                    if k not in self.primary_channels:
                        existing_ids = [
                            entry["channel_id"]
                            for entry in self.primary_channels.values()
                            if isinstance(entry, dict) and "channel_id" in entry
                        ]
                        next_id = max(existing_ids, default=0) + 1
                        v["channel_id"] = next_id
                        self.primary_channels[k] = v
                        self.primary_by_id[next_id] = v

                # Merge any missing charm decay channels
                for k, v in current_local_charm.items():
                    if k not in self.charm_decay_channels:
                        existing_ids = [
                            entry["decay_id"]
                            for entry in self.charm_decay_channels.values()
                            if isinstance(entry, dict) and "decay_id" in entry
                        ]
                        next_id = max(existing_ids, default=0) + 1
                        v["decay_id"] = next_id
                        self.charm_decay_channels[k] = v
                        self.charm_by_id[next_id] = v

                self._save_unlocked()

    def get_or_register_channel(
        self,
        primary_pdgs: List[int],
        nu_pdg: int = 0
    ) -> Tuple[int, str, dict]:
        """
        Lookup or register a unique combination of immediate interaction products.
        Returns:
            (channel_id, channel_name, channel_dict)
        """
        sorted_pdgs = sorted([int(p) for p in primary_pdgs])
        key = ",".join(str(p) for p in sorted_pdgs)

        # Fast path: check in-memory cache
        with self._lock:
            if key in self.primary_channels:
                entry = self.primary_channels[key]
                return entry["channel_id"], entry["formula"], entry

        # Slow path: new channel needs registration under cross-process lock
        with self._lock:
            with self._file_lock():
                # Re-check disk in case another worker just registered this channel
                self._load_unlocked()
                if key in self.primary_channels:
                    entry = self.primary_channels[key]
                    return entry["channel_id"], entry["formula"], entry

                existing_ids = [
                    v["channel_id"]
                    for v in self.primary_channels.values()
                    if isinstance(v, dict) and "channel_id" in v
                ]
                next_id = max(existing_ids, default=0) + 1
                formula = format_channel_formula(sorted_pdgs)

                has_charm = any(is_charmed_hadron(p) for p in sorted_pdgs)
                charm_pdgs = [p for p in sorted_pdgs if is_charmed_hadron(p)]
                leptons = [p for p in sorted_pdgs if abs(p) in [11, 12, 13, 14, 15, 16]]
                is_cc = any(abs(p) in [11, 13, 15] for p in sorted_pdgs)
                is_nc = not is_cc

                entry = {
                    "channel_id": next_id,
                    "key": key,
                    "formula": formula,
                    "primary_pdgs": sorted_pdgs,
                    "multiplicity": len(sorted_pdgs),
                    "is_cc": is_cc,
                    "is_nc": is_nc,
                    "has_charm": has_charm,
                    "charm_pdgs": charm_pdgs,
                    "primary_leptons": leptons,
                    "incoming_nu_pdg": nu_pdg,
                }

                self.primary_channels[key] = entry
                self.primary_by_id[next_id] = entry
                self._save_unlocked()
                return next_id, formula, entry

    def get_or_register_charm_decay(
        self,
        parent_pdg: int,
        daughter_pdgs: List[int]
    ) -> Tuple[int, str, str, dict]:
        """
        Lookup or register a unique charm hadron direct decay channel.
        Returns:
            (decay_id, decay_formula, decay_mode, decay_dict)
            decay_mode is one of: "to_muon", "to_electron", "hadronic", "other"
        """
        parent_p = int(parent_pdg)
        sorted_daughters = sorted([int(p) for p in daughter_pdgs])
        key = f"{parent_p}:" + ",".join(str(p) for p in sorted_daughters)

        # Fast path: check in-memory cache
        with self._lock:
            if key in self.charm_decay_channels:
                entry = self.charm_decay_channels[key]
                return entry["decay_id"], entry["formula"], entry["mode"], entry

        # Slow path: new charm decay needs registration under cross-process lock
        with self._lock:
            with self._file_lock():
                self._load_unlocked()
                if key in self.charm_decay_channels:
                    entry = self.charm_decay_channels[key]
                    return entry["decay_id"], entry["formula"], entry["mode"], entry

                existing_ids = [
                    v["decay_id"]
                    for v in self.charm_decay_channels.values()
                    if isinstance(v, dict) and "decay_id" in v
                ]
                next_id = max(existing_ids, default=0) + 1
                parent_name = pdg_to_name(parent_p)
                daughter_formula = format_channel_formula(sorted_daughters)
                full_formula = f"{parent_name} -> {daughter_formula}"

                has_muon = any(abs(p) == 13 for p in sorted_daughters)
                has_electron = any(abs(p) == 11 for p in sorted_daughters)
                has_pion = any(abs(p) in [211, 111] for p in sorted_daughters)
                has_kaon = any(abs(p) in [321, 311, 310, 130] for p in sorted_daughters)

                if has_muon:
                    mode = "to_muon"
                elif has_electron:
                    mode = "to_electron"
                elif all(abs(p) > 100 for p in sorted_daughters if p != 22):
                    mode = "hadronic"
                else:
                    mode = "other"

                entry = {
                    "decay_id": next_id,
                    "key": key,
                    "parent_pdg": parent_p,
                    "parent_name": parent_name,
                    "daughter_pdgs": sorted_daughters,
                    "formula": full_formula,
                    "daughter_formula": daughter_formula,
                    "mode": mode,
                    "has_direct_muon": has_muon,
                    "has_direct_electron": has_electron,
                    "has_direct_pion": has_pion,
                    "has_direct_kaon": has_kaon,
                    "n_daughters": len(sorted_daughters),
                }

                self.charm_decay_channels[key] = entry
                self.charm_by_id[next_id] = entry
                self._save_unlocked()
                return next_id, full_formula, mode, entry
