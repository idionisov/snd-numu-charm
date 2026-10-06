"""
snd.cuts
--------
Physics and detector analysis cuts for SND@LHC datasets.
Includes RDataFrame-compatible functors and cut classes.
"""

from __future__ import annotations
import ROOT
from .data_manager import load_trident_libraries


def get_avg_scifi_fiducial_cut(
    vertical_min: float = 200.0,
    vertical_max: float = 1200.0,
    horizontal_min: float = 300.0,
    horizontal_max: float = 1336.0,
    reversed: bool = False,
) -> ROOT.snd.AvgScifiFiducialCut:
    """
    Instantiate and return the C++ snd::AvgScifiFiducialCut object for RDataFrame.

    Parameters:
        vertical_min: Minimum average vertical SciFi channel number (default: 200.0).
        vertical_max: Maximum average vertical SciFi channel number (default: 1200.0).
        horizontal_min: Minimum average horizontal SciFi channel number (default: 300.0).
        horizontal_max: Maximum average horizontal SciFi channel number (default: 1336.0, i.e. 128*12 - 200).
        reversed: If True, select events outside the fiducial boundary (default: False).

    Returns:
        ROOT.snd.AvgScifiFiducialCut instance.
    """
    load_trident_libraries()
    return ROOT.snd.AvgScifiFiducialCut(
        float(vertical_min),
        float(vertical_max),
        float(horizontal_min),
        float(horizontal_max),
        bool(reversed),
    )


class AvgScifiFiducialCut:
    """
    Python wrapper for snd::AvgScifiFiducialCut.
    Usable directly as a callable in Python or passed to RDataFrame.Filter().
    """

    def __new__(
        cls,
        vertical_min: float = 200.0,
        vertical_max: float = 1200.0,
        horizontal_min: float = 300.0,
        horizontal_max: float = 1336.0,
        reversed: bool = False,
    ):
        return get_avg_scifi_fiducial_cut(
            vertical_min=vertical_min,
            vertical_max=vertical_max,
            horizontal_min=horizontal_min,
            horizontal_max=horizontal_max,
            reversed=reversed,
        )
