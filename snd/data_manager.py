import os
import sys
from typing import List, Union, Optional, Set, Sequence, Dict, Any, Tuple
import ROOT


_LIBRARIES_LOADED = False


def load_trident_libraries(repo_root: Optional[str] = None):
    global _LIBRARIES_LOADED
    if _LIBRARIES_LOADED:
        return
    _LIBRARIES_LOADED = True

    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    build_lib = os.path.join(repo_root, "build", "lib")
    analysis_inc = os.path.join(repo_root, "analysis")
    cuts_inc = os.path.join(repo_root, "cuts")

    ROOT.gInterpreter.AddIncludePath(analysis_inc)
    ROOT.gInterpreter.AddIncludePath(cuts_inc)

    sndsw_root = os.environ.get("SNDSW_ROOT", "")
    if sndsw_root:
        ROOT.gInterpreter.AddIncludePath(os.path.join(sndsw_root, "include"))
        ROOT.gInterpreter.AddIncludePath(os.path.join(sndsw_root, "analysis/tools"))
        for h_tool in ["sndGeometryGetter.h", "sndTchainGetter.h", "sndConfiguration.h"]:
            for d in [os.path.join(sndsw_root, "include"), os.path.join(sndsw_root, "analysis/tools")]:
                ht_path = os.path.join(d, h_tool)
                if os.path.exists(ht_path):
                    ROOT.gInterpreter.ProcessLine(f'#include "{ht_path}"')
                    break

    for h_name in ["DataManager.h", "NeutrinoTruthProcessor.h", "MuonNeutrinoTruthProcessor.h", "PreselectionProcessor.h"]:
        header_file = os.path.join(analysis_inc, h_name)
        if os.path.exists(header_file):
            ROOT.gInterpreter.ProcessLine(f'#include "{header_file}"')

    ROOT.gSystem.AddDynamicPath(build_lib)

    for lib_name in ["libtrident_analysis.so", "libtrident_cuts.so", "libsnd_analysis_tools.so"]:
        lib_path = os.path.join(build_lib, lib_name)
        if os.path.exists(lib_path):
            ROOT.gSystem.Load(lib_name)


class DataManager:
    def __init__(
        self,
        source: Optional[Union[str, Sequence[str], int, ROOT.TChain]] = None,
        tree_name: Optional[str] = None,
        num_threads: Optional[int] = None,
        run: Optional[int] = None,
        n_files: int = -1,
        csv_data_path: Optional[str] = None,
        geo_path: Optional[str] = None,
        csv_geo_path: Optional[str] = None,
        init_geo: bool = False,
        load_libraries: bool = True,
    ):
        if load_libraries:
            load_trident_libraries()

        tree_str = tree_name if tree_name is not None else ""
        threads_int = num_threads if (num_threads is not None and num_threads > 0) else 0

        # Determine run number if provided
        run_num: Optional[int] = None
        if run is not None:
            run_num = int(run)
        elif isinstance(source, int):
            run_num = int(source)

        if run_num is not None:
            csv_str = csv_data_path if csv_data_path is not None else ""
            self._cpp = ROOT.snd.DataManager(run_num, n_files, csv_str, tree_str, threads_int)
        elif isinstance(source, ROOT.TChain):
            self._cpp = ROOT.snd.DataManager(source, False, threads_int)
        elif isinstance(source, (list, tuple)):
            files = list(source)
            if n_files > 0:
                files = files[:n_files]
            vec = ROOT.std.vector('string')()
            for s in files:
                vec.push_back(str(s))
            self._cpp = ROOT.snd.DataManager(vec, tree_str, threads_int)
        elif source is not None:
            if n_files > 0:
                resolved = list(ROOT.snd.DataManager.ResolveFiles(str(source)))
                vec = ROOT.std.vector('string')()
                for s in resolved[:n_files]:
                    vec.push_back(str(s))
                self._cpp = ROOT.snd.DataManager(vec, tree_str, threads_int)
            else:
                self._cpp = ROOT.snd.DataManager(str(source), tree_str, threads_int)
        else:
            self._cpp = ROOT.snd.DataManager()

        if geo_path:
            self._cpp.SetGeoPath(str(geo_path))

        if init_geo:
            self.init_geometry(geo_path=geo_path, csv_file_path=csv_geo_path)

        self._branch_names: Optional[Set[str]] = None

    @property
    def tree_name(self) -> str:
        return str(self._cpp.GetTreeName())

    @property
    def files(self) -> List[str]:
        return [str(f) for f in self._cpp.GetFiles()]

    @property
    def num_files(self) -> int:
        return int(self._cpp.GetNumFiles())

    @property
    def branch_names(self) -> Set[str]:
        if self._branch_names is None:
            self._branch_names = set([str(b) for b in self._cpp.GetBranchNamesList()])
        return self._branch_names

    def has_branch(self, name: str) -> bool:
        return bool(self._cpp.HasBranch(name))

    def has_event_header(self) -> bool:
        return bool(self._cpp.HasEventHeader())

    @property
    def run_number(self) -> Optional[int]:
        rn = int(self._cpp.GetRunNumber())
        return rn if rn > 0 else None

    @property
    def fill_number(self) -> Optional[int]:
        fn = int(self._cpp.GetFillNumber())
        return fn if fn > 0 else None

    def get_fill_number(self) -> Optional[int]:
        return self.fill_number

    @property
    def lumi_path(self) -> Optional[str]:
        p = str(self._cpp.GetLumiPath())
        return p if p else None

    def get_lumi_path(self, lumi_dir: str = "/eos/experiment/sndlhc/atlas_lumi") -> Optional[str]:
        p = str(self._cpp.GetLumiPath(lumi_dir))
        return p if p else None

    @property
    def data_base_path(self) -> Optional[str]:
        dbp = str(self._cpp.GetDataBasePath())
        return dbp if dbp else None

    @property
    def geo_path(self) -> Optional[str]:
        gp = str(self._cpp.GetGeoPath())
        return gp if gp else None

    @geo_path.setter
    def geo_path(self, path: str) -> None:
        self._cpp.SetGeoPath(str(path))

    def get_geo_path(self, csv_file_path: Optional[str] = None) -> Optional[str]:
        gp = str(self._cpp.GetGeoPath(csv_file_path or ""))
        return gp if gp else None

    def init_geometry(self, geo_path: Optional[str] = None, csv_file_path: Optional[str] = None) -> Tuple[Any, Any]:
        geom = self._cpp.InitGeometry(geo_path or "", csv_file_path or "")
        return (geom.first, geom.second)

    @property
    def has_geometry(self) -> bool:
        return bool(self._cpp.HasGeometry())

    @property
    def scifi(self) -> Any:
        return self._cpp.GetScifi()

    @property
    def mufilter(self) -> Any:
        return self._cpp.GetMuFilter()

    @property
    def geometry(self) -> Tuple[Any, Any]:
        g = self._cpp.GetGeometry()
        return (g.first, g.second)

    @property
    def configuration(self) -> Optional[Any]:
        cfg = self._cpp.GetConfiguration()
        return cfg if cfg else None

    def isMC(self) -> bool:
        return bool(self._cpp.IsMC())

    @property
    def is_mc(self) -> bool:
        return self.isMC()

    @property
    def entries(self) -> int:
        return int(self._cpp.GetEntries())

    def __len__(self) -> int:
        return self.entries

    def get_chain(self) -> ROOT.TChain:
        return self._cpp.GetChain()

    def rdf(
        self,
        range_limit: Optional[int] = None,
        progress: bool = False,
        every_seconds: float = 30.0,
        every_events: int = 0,
        progress_label: Optional[str] = None,
    ) -> ROOT.RDataFrame:
        """
        Returns a high-performance ROOT RDataFrame connected to the dataset.
        """
        dataframe = self._cpp.GetDataFrame()
        if range_limit is not None and range_limit > 0:
            dataframe = dataframe.Range(range_limit)
        if progress:
            from .progress import attach_progress_printer
            total = range_limit if (range_limit is not None and range_limit > 0) else self.entries
            label = progress_label if progress_label is not None else "DataManager"
            dataframe = attach_progress_printer(
                dataframe,
                total_events=total,
                every_seconds=every_seconds,
                every_events=every_events,
                label=label,
            )
        return dataframe

    def df(
        self,
        columns: Optional[Sequence[str]] = None,
        range_limit: Optional[int] = None,
        query: Optional[str] = None,
        rdf: Optional[ROOT.RDataFrame] = None,
        progress: bool = False,
        every_seconds: float = 30.0,
        progress_label: Optional[str] = None,
    ) -> Any:
        """
        Converts dataset observables into a flattened Python pandas DataFrame.

        Parameters:
            columns: Optional list of branch or column names to extract.
                     If None, extracts all known scalar and 1D vector columns.
            range_limit: Optional entry limit (e.g. 5000).
            query: Optional selection filter string (e.g. "reco_chi2 < 10").
            rdf: Optional pre-configured ROOT RDataFrame node to extract from.
                 If None, creates one via self.rdf(range_limit=range_limit).
            progress: Enable 30-second progress logging during conversion.
            every_seconds: Status update interval in seconds.
            progress_label: Label prefix for progress reporter.

        Returns:
            pandas.DataFrame containing flat arrays of the requested columns.
        """
        import pandas as pd

        active_rdf = rdf if rdf is not None else self.rdf(
            range_limit=range_limit,
            progress=progress,
            every_seconds=every_seconds,
            progress_label=progress_label or "DataManager.df",
        )

        if query is not None and len(query.strip()) > 0:
            active_rdf = active_rdf.Filter(query)

        if range_limit is not None and range_limit > 0 and rdf is not None:
            active_rdf = active_rdf.Range(range_limit)

        if columns is None:
            known_jagged = {
                "MCTrack", "ScifiPoint", "MuFilterPoint", "EmulsionDetPoint",
                "Digi_ScifiHits", "Digi_ScifiHits2MCPoints",
                "Digi_MuFilterHits", "Digi_MuFilterHits2MCPoints",
                "EventHeader.", "truth", "fittedTracks", "Reco_MuonTracks",
                "Cluster_Scifi", "Cluster_Mufi",
            }
            selected_cols = []
            from .truth_branches import TRIDENT_ANALYSIS_COLUMNS
            for c in TRIDENT_ANALYSIS_COLUMNS:
                if self.has_branch(c):
                    selected_cols.append(c)

            for b in self.branch_names:
                if b not in selected_cols and b not in known_jagged and "." not in b:
                    selected_cols.append(b)
        else:
            selected_cols = list(columns)

        if not selected_cols:
            return pd.DataFrame()

        np_dict = active_rdf.AsNumpy(selected_cols)

        clean_dict = {}
        for k, v in np_dict.items():
            if isinstance(v, (list, tuple)) or hasattr(v, "__len__"):
                if len(v) > 0 and hasattr(v[0], "__iter__") and not isinstance(v[0], (str, bytes)):
                    clean_dict[k] = ["".join(str(ch) for ch in item) for item in v]
                elif hasattr(v, "dtype") and v.dtype.kind in ['S', 'U', 'O', 'V']:
                    clean_dict[k] = [
                        x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else str(x)
                        for x in v
                    ]
                else:
                    clean_dict[k] = v
            else:
                clean_dict[k] = v

        return pd.DataFrame(clean_dict)

    def get_rdataframe(self, range_limit: Optional[int] = None) -> ROOT.RDataFrame:
        return self.rdf(range_limit=range_limit)

    @property
    def rdataframe(self) -> ROOT.RDataFrame:
        return self.rdf()

    def get_time_range(self, cached: bool = True) -> Optional[Tuple[int, int]]:
        tr = self._cpp.GetTimeRange(cached)
        if tr.first > 0 and tr.second > 0:
            return (int(tr.first), int(tr.second))
        return None

    def get_duration(self, cached: bool = True) -> Optional[float]:
        dur = self._cpp.GetDuration(cached)
        return float(dur) if dur >= 0.0 else None

    @property
    def duration(self) -> Optional[float]:
        return self.get_duration()

    @property
    def time_range(self) -> Optional[Tuple[int, int]]:
        return self.get_time_range()

    def get_duration_str(self, cached: bool = True) -> Optional[str]:
        dur_str = str(self._cpp.GetDurationStr(cached))
        return dur_str if dur_str and dur_str != "N/A" else None

    def get_time_range_fast(self, cached: bool = True) -> Optional[Tuple[int, int]]:
        tr = self._cpp.GetTimeRangeFast(cached)
        if tr.first > 0 and tr.second > 0:
            return (int(tr.first), int(tr.second))
        return None

    def get_duration_fast(self, cached: bool = True) -> Optional[float]:
        dur = self._cpp.GetDurationFast(cached)
        return float(dur) if dur >= 0.0 else None

    @property
    def duration_fast(self) -> Optional[float]:
        return self.get_duration_fast()

    @property
    def time_range_fast(self) -> Optional[Tuple[int, int]]:
        return self.get_time_range_fast()

    def get_duration_fast_str(self, cached: bool = True) -> Optional[str]:
        dur_str = str(self._cpp.GetDurationFastStr(cached))
        return dur_str if dur_str and dur_str != "N/A" else None

    def get_lumi(
        self,
        start_ts: Optional[Union[int, float]] = None,
        end_ts: Optional[Union[int, float]] = None,
        lumi_dir: str = "/eos/experiment/sndlhc/atlas_lumi",
        cached: bool = True,
    ) -> float:
        s_ts = int(start_ts) if start_ts is not None else -1
        e_ts = int(end_ts) if end_ts is not None else -1
        return float(self._cpp.GetLumi(s_ts, e_ts, lumi_dir, cached))

    @property
    def lumi(self) -> float:
        return self.get_lumi()

    def get_lumi_fast(
        self,
        lumi_dir: str = "/eos/experiment/sndlhc/atlas_lumi",
        cached: bool = True,
    ) -> float:
        return float(self._cpp.GetLumiFast(lumi_dir, cached))

    @property
    def lumi_fast(self) -> float:
        return self.get_lumi_fast()

    def get_lumi_exact(
        self,
        lumi_dir: str = "/eos/experiment/sndlhc/atlas_lumi",
        cached: bool = True,
    ) -> float:
        return float(self._cpp.GetLumiExact(lumi_dir, cached))

    @property
    def lumi_exact(self) -> float:
        return self.get_lumi_exact()

    def get_lumi_str(
        self,
        start_ts: Optional[Union[int, float]] = None,
        end_ts: Optional[Union[int, float]] = None,
        lumi_dir: str = "/eos/experiment/sndlhc/atlas_lumi",
        cached: bool = True,
    ) -> str:
        s_ts = int(start_ts) if start_ts is not None else -1
        e_ts = int(end_ts) if end_ts is not None else -1
        return str(self._cpp.GetLumiStr(s_ts, e_ts, lumi_dir, cached))

    def get_lumi_fast_str(
        self,
        lumi_dir: str = "/eos/experiment/sndlhc/atlas_lumi",
        cached: bool = True,
    ) -> str:
        return str(self._cpp.GetLumiFastStr(lumi_dir, cached))

    def summary(self) -> str:
        return str(self._cpp.Summary())

    def print(self) -> None:
        self._cpp.Print()

    def __repr__(self) -> str:
        mc_str = "True" if self.isMC() else "False"
        run_part = f", run={self.run_number}" if self.run_number else ""
        fill_part = f", fill={self.fill_number}" if self.fill_number else ""
        dur_str = self.get_duration_fast_str()
        dur_part = f", duration='{dur_str}'" if dur_str else ""
        geo_part = ", geo=OK" if self.has_geometry else ""
        return f"DataManager(tree='{self.tree_name}'{run_part}{fill_part}, is_mc={mc_str}, files={self.num_files}, entries={self.entries:,}{dur_part}{geo_part})"

    def to_pandas(
        self,
        columns: Optional[Union[List[str], Sequence[str]]] = None,
        query: Optional[str] = None,
        exclude_jagged: bool = True,
        categorical_columns: bool = True,
        max_events: Optional[int] = None,
    ):
        import pandas as pd

        known_jagged = {
            "MCTrack", "ScifiPoint", "MuFilterPoint", "EmulsionDetPoint",
            "Digi_ScifiHits", "Digi_ScifiHits2MCPoints",
            "Digi_MuFilterHits", "Digi_MuFilterHits2MCPoints",
            "EventHeader.", "truth"
        }

        df = self.rdf()

        if query is not None and len(query.strip()) > 0:
            df = df.Filter(query)

        if max_events is not None and max_events > 0:
            df = df.Filter(f"rdfentry_ < {max_events}")

        if columns is None:
            from .truth_branches import TRIDENT_ANALYSIS_COLUMNS
            selected_cols = [c for c in TRIDENT_ANALYSIS_COLUMNS if self.has_branch(c)]
            for b in self.branch_names:
                if b not in selected_cols and b not in known_jagged and "." not in b:
                    selected_cols.append(b)
        else:
            selected_cols = list(columns)

        np_dict = df.AsNumpy(selected_cols)

        clean_dict = {}
        for k, v in np_dict.items():
            if hasattr(v, "dtype") and v.dtype.kind in ['S', 'U', 'O', 'V']:
                clean_dict[k] = [
                    x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else str(x)
                    for x in v
                ]
            else:
                clean_dict[k] = v

        pdf = pd.DataFrame(clean_dict)

        if categorical_columns:
            for cat_col in ["proc_name", "region_name", "emission_proc_name"]:
                if cat_col in pdf.columns:
                    pdf[cat_col] = pdf[cat_col].astype("category")

        return pdf

    @classmethod
    def load_libraries(cls, repo_root: Optional[str] = None):
        load_trident_libraries(repo_root=repo_root)

    @classmethod
    def from_run(
        cls,
        run: int,
        n_files: int = -1,
        csv_data_path: Optional[str] = None,
        tree_name: Optional[str] = None,
        num_threads: Optional[int] = None,
        init_geo: bool = False,
        geo_path: Optional[str] = None,
        csv_geo_path: Optional[str] = None,
    ) -> "DataManager":
        return cls(
            run=run,
            n_files=n_files,
            csv_data_path=csv_data_path,
            tree_name=tree_name,
            num_threads=num_threads,
            init_geo=init_geo,
            geo_path=geo_path,
            csv_geo_path=csv_geo_path,
        )

    @staticmethod
    def get_data_base_path(run: int, csv_file_path: Optional[str] = None) -> str:
        load_trident_libraries()
        return str(ROOT.snd.DataManager.FetchDataBasePath(int(run), csv_file_path or ""))

    @staticmethod
    def get_geo_path_for_run(run: int, csv_file_path: Optional[str] = None) -> str:
        load_trident_libraries()
        return str(ROOT.snd.DataManager.FetchGeoPath(int(run), csv_file_path or ""))

    @staticmethod
    def fetch_tchain(
        run_or_file: Union[int, str],
        n_files: int = -1,
        csv_file_path: Optional[str] = None,
    ) -> ROOT.TChain:
        load_trident_libraries()
        if isinstance(run_or_file, int):
            return ROOT.snd.DataManager.FetchTChain(run_or_file, n_files, csv_file_path or "")
        return ROOT.snd.DataManager.FetchTChain(str(run_or_file))

    @staticmethod
    def fetch_geometry(
        geo_path_or_run: Union[int, str],
        csv_file_path: Optional[str] = None,
    ) -> Tuple[Any, Any]:
        load_trident_libraries()
        if isinstance(geo_path_or_run, int):
            geom = ROOT.snd.DataManager.FetchGeometry(geo_path_or_run, csv_file_path or "")
        else:
            geom = ROOT.snd.DataManager.FetchGeometry(str(geo_path_or_run))
        return (geom.first, geom.second)

    @staticmethod
    def copy_metadata(source_file: str, target_file: str, exclude_trees: Optional[List[str]] = None) -> None:
        if exclude_trees is None:
            exclude_trees = ["cbmsim", "rawConv"]
        vec = ROOT.std.vector('string')()
        for e in exclude_trees:
            vec.push_back(str(e))
        ROOT.snd.DataManager.CopyMetadata(source_file, target_file, vec)

    @staticmethod
    def fetch_lumi(
        fill: int,
        start_ts: Optional[Union[int, float]] = None,
        end_ts: Optional[Union[int, float]] = None,
        lumi_dir: str = "/eos/experiment/sndlhc/atlas_lumi",
    ) -> float:
        load_trident_libraries()
        s_ts = int(start_ts) if start_ts is not None else -1
        e_ts = int(end_ts) if end_ts is not None else -1
        return float(ROOT.snd.DataManager.FetchLumi(int(fill), s_ts, e_ts, lumi_dir))
