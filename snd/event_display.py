#!/usr/bin/env python3
"""
snd.event_display
-----------------
Self-contained 2D Event Display for SND@LHC signal events (XZ and YZ projections).
Faithfully reproduces the official SND@LHC event display look-and-feel:
  - Exact detector geometry outlines and colors (Fe blocks in kGreen-6, Scifi/MuFilter in kBlue+1/kRed)
  - Color-coding for SciFi hit densities (hits/cm) and MuFilter QDC units using the official Viridis palette
  - Official CERN & SND@LHC logo banner and Run/Event info subpads
  - Original hit density & QDC gradient legend matching 2dEventDisplay.py
  - MC truth tracks (Primary muon mu1 in kBlue, Charm decay muon mu2 in kRed, charm flight in kGreen+2)
  - Physics truth kinematics summary box & track legend
"""

import os
import sys
import math
from array import array
import numpy as np
import ROOT

# Ensure ROOT operates in batch mode
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetPalette(ROOT.kViridis)

DEFAULT_GEOFILE = "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root"

LOGO_PATHS = [
    os.path.expandvars("$SNDSW_ROOT/shipLHC/Large__SND_Logo_black_cut.png"),
    "/afs/cern.ch/user/i/idioniso/snd_master/sndsw/shipLHC/Large__SND_Logo_black_cut.png",
    "/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/sw/slc9_x86-64/sndsw/master-local1/shipLHC/Large__SND_Logo_black_cut.png",
]

PDG_NAMES = {
    14: "#nu_{#mu}",
    -14: "#bar{#nu}_{#mu}",
    13: "#mu^{-}",
    -13: "#mu^{+}",
    411: "D^{+}",
    -411: "D^{-}",
    421: "D^{0}",
    -421: "#bar{D}^{0}",
    431: "D_{s}^{+}",
    -431: "D_{s}^{-}",
    4122: "#Lambda_{c}^{+}",
    -4122: "#bar{#Lambda}_{c}^{-}",
}


def get_scifi_hit_density(points, x_range: float = 0.5):
    """
    Takes list of (z, coord) and returns array with number of hits within x_range cm.
    Matches getSciFiHitDensity in official 2dEventDisplay.py.
    """
    n = len(points)
    ret = []
    r2 = x_range * x_range
    for i in range(n):
        z_i, c_i = points[i]
        density = 0
        for j in range(n):
            z_j, c_j = points[j]
            if ((z_i - z_j) ** 2 + (c_i - c_j) ** 2) <= r2:
                density += 1
        ret.append(density)
    return ret


class Snd2DEventDisplay:
    """
    Self-contained 2D Event Display mimicking the official sndsw EventDisplay_Task look-and-feel.
    Includes hit density coloring for SciFi and QDC coloring for MuFilter.
    """

    def __init__(
        self,
        geo_file: str = DEFAULT_GEOFILE,
        z_start: float = 250.0,
        z_length: float = 350.0,
        x_min: float = -100.0,
        x_max: float = 10.0,
        y_min: float = -30.0,
        y_max: float = 80.0,
        color_by_qdc_and_density: bool = True,
        max_density: int = 40,
        max_qdc: float = 3200.0,
        draw_logo: bool = True,
        draw_text: bool = True,
    ):
        self.geo_file = geo_file
        self.z_min = z_start
        self.z_max = z_start + z_length
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.color_by_qdc_and_density = color_by_qdc_and_density
        self.max_density = max_density
        self.max_qdc = max_qdc
        self.draw_logo = draw_logo
        self.draw_text = draw_text

        self.geo = None
        self.scifi_mod = None
        self.mufilter_mod = None

        self._vec_a = ROOT.TVector3()
        self._vec_b = ROOT.TVector3()

        # Locate logo file
        self.logo_path = None
        for p in LOGO_PATHS:
            if os.path.exists(p):
                self.logo_path = p
                break

        # Setup geometry and baseline outlines
        self._setup_detector_node_defs()
        self._init_geometry()

    def _setup_detector_node_defs(self):
        """Original color mappings matching sndsw EventDisplay_Task."""
        self.nodes = {"volMuFilter_1/volFeBlockEnd_1": ROOT.kGreen - 6}
        for i in range(2):
            self.nodes[f"volVeto_1/volVetoPlane_{i}_{i}"] = ROOT.kRed
            for j in range(7):
                self.nodes[f"volVeto_1/volVetoPlane_{i}_{i}/volVetoBar_1{i}{j:0>3d}"] = ROOT.kRed
            self.nodes[f"volVeto_1/subVetoBox_{i}"] = ROOT.kGray + 1

        for i in range(4):
            self.nodes[f"volMuFilter_1/volMuDownstreamDet_{i}_{i+7}"] = ROOT.kBlue + 1
            for j in range(60):
                self.nodes[f"volMuFilter_1/volMuDownstreamDet_{i}_{i+7}/volMuDownstreamBar_ver_3{i}{j+60:0>3d}"] = ROOT.kBlue + 1
                if i < 3:
                    self.nodes[f"volMuFilter_1/volMuDownstreamDet_{i}_{i+7}/volMuDownstreamBar_hor_3{i}{j:0>3d}"] = ROOT.kBlue + 1
            self.nodes[f"volMuFilter_1/subDSBox_{i+7}"] = ROOT.kGray + 1

        for i in range(5):
            self.nodes[f"volTarget_1/ScifiVolume{i+1}_{i+1}000000"] = ROOT.kBlue + 1
            self.nodes[f"volTarget_1/volWallborder_{i}"] = ROOT.kGray
            self.nodes[f"volMuFilter_1/subUSBox_{i+2}"] = ROOT.kGray + 1
            self.nodes[f"volMuFilter_1/volMuUpstreamDet_{i}_{i+2}"] = ROOT.kBlue + 1
            for j in range(10):
                self.nodes[f"volMuFilter_1/volMuUpstreamDet_{i}_{i+2}/volMuUpstreamBar_2{i}00{j}"] = ROOT.kBlue + 1
            self.nodes[f"volMuFilter_1/volFeBlock_{i}"] = ROOT.kGreen - 6

        for i in range(7, 10):
            self.nodes[f"volMuFilter_1/volFeBlock_{i}"] = ROOT.kGreen - 6

        self.pass_nodes = {"Block", "Wall"}
        self.x_nodes = {"UpstreamBar", "VetoBar", "hor"}
        self.proj_idx = {"X": 0, "Y": 1}

    def _init_geometry(self):
        """Loads geometry and builds baseline polylines for detector outlines."""
        import SndlhcGeo

        if not os.path.exists(self.geo_file):
            raise FileNotFoundError(f"Geofile not found at {self.geo_file}")

        self.geo = SndlhcGeo.GeoInterface(self.geo_file)
        self.scifi_mod = self.geo.modules.get("Scifi")
        self.mufilter_mod = self.geo.modules.get("MuFilter")

        nav = ROOT.gGeoManager.GetCurrentNavigator()
        self.detector_polylines = {"X": {}, "Y": {}}

        for node_suffix, color in self.nodes.items():
            node_path = "/cave_1/Detector_0/" + node_suffix
            for p in ["X", "Y"]:
                if not nav.cd(node_path):
                    continue
                node_obj = nav.GetCurrentNode()
                shape = node_obj.GetVolume().GetShape()
                dx, dy, dz = shape.GetDX(), shape.GetDY(), shape.GetDZ()
                ox, oy, oz = shape.GetOrigin()[0], shape.GetOrigin()[1], shape.GetOrigin()[2]

                p_pts = {}
                if p == "X" and not any(xn in node_path for xn in self.x_nodes):
                    p_pts["LB"] = array("d", [-dx + ox, oy, -dz + oz])
                    p_pts["LT"] = array("d", [dx + ox, oy, -dz + oz])
                    p_pts["RB"] = array("d", [-dx + ox, oy, dz + oz])
                    p_pts["RT"] = array("d", [dx + ox, oy, dz + oz])
                elif p == "Y" and "ver" not in node_path:
                    p_pts["LB"] = array("d", [ox, -dy + oy, -dz + oz])
                    p_pts["LT"] = array("d", [ox, dy + oy, -dz + oz])
                    p_pts["RB"] = array("d", [ox, -dy + oy, dz + oz])
                    p_pts["RT"] = array("d", [ox, dy + oy, dz + oz])
                else:
                    continue

                m_pts = {}
                for key, pt in p_pts.items():
                    m_pts[key] = array("d", [0.0, 0.0, 0.0])
                    nav.LocalToMaster(pt, m_pts[key])

                poly = ROOT.TPolyLine()
                c_idx = self.proj_idx[p]
                poly.SetPoint(0, m_pts["LB"][2], m_pts["LB"][c_idx])
                poly.SetPoint(1, m_pts["LT"][2], m_pts["LT"][c_idx])
                poly.SetPoint(2, m_pts["RT"][2], m_pts["RT"][c_idx])
                poly.SetPoint(3, m_pts["RB"][2], m_pts["RB"][c_idx])
                poly.SetPoint(4, m_pts["LB"][2], m_pts["LB"][c_idx])
                poly.SetLineColor(color)
                poly.SetLineWidth(1)
                # Fill iron blocks and walls with alpha 0.5 matching original display
                if any(pn in node_path for pn in self.pass_nodes):
                    poly.SetFillColorAlpha(color, 0.5)

                self.detector_polylines[p][node_path] = poly

    def _draw_detectors(self, pad_x, pad_y):
        """Draws baseline detector outlines into XZ and YZ pads."""
        pad_x.cd()
        for poly in self.detector_polylines["X"].values():
            if poly.GetFillColor() != 0:
                poly.DrawClone("f same")
            poly.DrawClone("same")

        pad_y.cd()
        for poly in self.detector_polylines["Y"].values():
            if poly.GetFillColor() != 0:
                poly.DrawClone("f same")
            poly.DrawClone("same")

    def _draw_logo_and_info(self, pad, pad_num: int, run_number: int, event_number: int):
        """Draws the official SND@LHC logo and Run/Event text matching EventDisplay_Task."""
        pad.cd()
        drawn_objs = []

        if self.draw_logo and self.logo_path:
            pad_logo = ROOT.TPad(f"logo_{pad_num}", f"logo_{pad_num}", 0.10, 0.10, 0.19, 0.26)
            pad_logo.SetBorderSize(0)
            pad_logo.SetFillStyle(4000)
            pad_logo.SetFillColorAlpha(0, 0)
            pad_logo.Draw()
            pad_logo.cd()
            logo_img = ROOT.TImage.Open(self.logo_path)
            if logo_img:
                logo_img.SetConstRatio(True)
                logo_img.Draw()
                drawn_objs.extend([pad_logo, logo_img])
            pad.cd()

        if self.draw_text:
            pad_text = ROOT.TPad(f"info_{pad_num}", f"info_{pad_num}", 0.17, 0.08, 0.47, 0.28)
            pad_text.SetBorderSize(0)
            pad_text.SetFillStyle(4000)
            pad_text.SetFillColorAlpha(0, 0)
            pad_text.Draw()
            pad_text.cd()
            text_info = ROOT.TLatex()
            text_info.SetTextAlign(11)
            text_info.SetTextFont(42)
            text_info.SetTextSize(0.15)
            text_info.DrawLatex(0.0, 0.60, "SND@LHC Experiment, CERN")
            text_info.DrawLatex(0.0, 0.30, f"Run / Event: {run_number} / {event_number}")
            drawn_objs.extend([pad_text, text_info])
            pad.cd()

        return drawn_objs

    def _draw_density_and_qdc_legend(self, pad):
        """
        Draws hit density (SciFi) and QDC (MuFilter) color scales in the lower panel.
        Matches drawLegend in 2dEventDisplay.py:264-297.
        """
        pad.cd()
        drawn_objs = []
        n_legend_points = 5

        pad_leg = ROOT.TPad(f"density_qdc_legend_{pad.GetName()}", "density_qdc_legend", 0.48, 0.08, 0.86, 0.28)
        pad_leg.SetBorderSize(0)
        pad_leg.SetFillStyle(4000)
        pad_leg.Draw()
        pad_leg.cd()

        text_leg = ROOT.TLatex()
        text_leg.SetTextAlign(11)
        text_leg.SetTextFont(42)
        text_leg.SetTextSize(0.14)

        palette = ROOT.TColor.GetPalette()
        n_pal = len(palette)

        for i in range(n_legend_points):
            dens_val = int(i * self.max_density / (n_legend_points - 1))
            qdc_val = int(i * self.max_qdc / (n_legend_points - 1))
            x_pos = (i + 0.3) * (1.0 / (n_legend_points + 2.0))
            marker_x = (i + 0.15) * (1.0 / (n_legend_points + 2.0))

            if i < (n_legend_points - 1):
                text_leg.DrawLatex(x_pos, 0.58, f"{dens_val}")
                text_leg.DrawLatex(x_pos, 0.22, f"{qdc_val}")
            else:
                text_leg.DrawLatex(x_pos, 0.58, f"{dens_val} SciFi hits/cm")
                text_leg.DrawLatex(x_pos, 0.22, f"{qdc_val} QDC units")

            # SciFi Density Ellipse
            scifi_color_idx = int(float(i) / (n_legend_points - 1) * (n_pal - 1))
            el = ROOT.TEllipse(marker_x, 0.64, 0.05 / 4.0, 0.07)
            el.SetFillStyle(1001)
            el.SetFillColor(palette[scifi_color_idx])
            el.SetLineColor(ROOT.kBlack)
            el.SetLineWidth(1)
            el.Draw("same")
            drawn_objs.append(el)

            # MuFilter QDC Box
            qdc_color_idx = int(float(i) / (n_legend_points - 1) * (n_pal - 1))
            box = ROOT.TBox(marker_x - 0.05 / 4.0, 0.28 - 0.06, marker_x + 0.05 / 4.0, 0.28 + 0.06)
            box.SetFillStyle(1001)
            box.SetFillColor(palette[qdc_color_idx])
            box.SetLineColor(ROOT.kBlack)
            box.SetLineWidth(1)
            box.Draw("same")
            drawn_objs.append(box)

        drawn_objs.extend([pad_leg, text_leg])
        pad.cd()
        return drawn_objs

    def _extract_truth(self, tree):
        """Extracts truth information from tree event."""
        truth = {
            "nu_e": 0.0,
            "vtx": (0.0, 0.0, -9999.0),
            "charm_pdg": 0,
            "decay_length_3d": 0.0,
            "mu1_p": 0.0,
            "mu2_p": 0.0,
            "dimuon_mass": 0.0,
            "dimuon_opening_angle_mrad": 0.0,
            "has_candidate": False,
        }

        if hasattr(tree, "nu_e"):
            truth["nu_e"] = float(tree.nu_e)
        if hasattr(tree, "vtx_x"):
            truth["vtx"] = (float(tree.vtx_x), float(tree.vtx_y), float(tree.vtx_z))
        if hasattr(tree, "charm_pdg"):
            truth["charm_pdg"] = int(tree.charm_pdg)
        if hasattr(tree, "decay_length_3d"):
            truth["decay_length_3d"] = float(tree.decay_length_3d)
        if hasattr(tree, "mu1_p"):
            truth["mu1_p"] = float(tree.mu1_p)
        if hasattr(tree, "mu2_p"):
            truth["mu2_p"] = float(tree.mu2_p)
        if hasattr(tree, "dimuon_mass"):
            truth["dimuon_mass"] = float(tree.dimuon_mass)
        if hasattr(tree, "dimuon_opening_angle_mrad"):
            truth["dimuon_opening_angle_mrad"] = float(tree.dimuon_opening_angle_mrad)
        if hasattr(tree, "has_candidate"):
            truth["has_candidate"] = bool(tree.has_candidate)

        return truth

    def _extrapolate_track(self, p0, trk, max_len: float = 60.0):
        """
        Extrapolates track along its physical 3D momentum direction.
        Avoids dividing by Pz (singularities when Pz -> 0 or transverse).
        """
        p_tot = trk.GetP()
        if p_tot < 0.005:
            return None
        flight_len = min(max_len, max(5.0, 40.0 * p_tot))
        dx = flight_len * trk.GetPx() / p_tot
        dy = flight_len * trk.GetPy() / p_tot
        dz = flight_len * trk.GetPz() / p_tot
        x1 = min(self.x_max, max(self.x_min, p0[0] + dx))
        y1 = min(self.y_max, max(self.y_min, p0[1] + dy))
        z1 = min(self.z_max, max(self.z_min, p0[2] + dz))
        return (x1, y1, z1)

    def _find_truth_tracks(self, tree, truth_info):
        """Finds trajectory coordinates for mu1, charm, and mu2."""
        tracks = {
            "mu1": None,
            "charm": None,
            "mu2": None,
            "vtx": truth_info["vtx"],
            "charm_decay": None,
        }

        vtx = truth_info["vtx"]
        if vtx[2] < -9000.0 and hasattr(tree, "MCTrack") and tree.MCTrack.GetEntries() > 0:
            vtx = (tree.MCTrack[0].GetStartX(), tree.MCTrack[0].GetStartY(), tree.MCTrack[0].GetStartZ())
            tracks["vtx"] = vtx

        mu1_trk = None
        charm_trk = None
        mu2_trk = None

        if hasattr(tree, "MCTrack"):
            for i, trk in enumerate(tree.MCTrack):
                pdg = trk.GetPdgCode()
                mother = trk.GetMotherId()
                abs_pdg = abs(pdg)

                # Primary muon (CC)
                if abs_pdg == 13 and mother == 0 and mu1_trk is None:
                    mu1_trk = (i, trk)
                # Primary charm hadron
                elif abs_pdg in [411, 421, 431, 4122, 4232, 4132, 4332] and mother == 0 and charm_trk is None:
                    charm_trk = (i, trk)

            # Secondary muon from charm decay
            if charm_trk is not None:
                charm_id = charm_trk[0]
                for i, trk in enumerate(tree.MCTrack):
                    if abs(trk.GetPdgCode()) == 13 and i != (mu1_trk[0] if mu1_trk else -1):
                        curr = trk
                        curr_mid = curr.GetMotherId()
                        is_from_charm = False
                        while curr_mid >= 0 and curr_mid < tree.MCTrack.GetEntries():
                            if curr_mid == charm_id:
                                is_from_charm = True
                                break
                            curr = tree.MCTrack[curr_mid]
                            curr_mid = curr.GetMotherId()
                        if is_from_charm:
                            mu2_trk = (i, trk)
                            break

        # Trajectory for Primary Muon
        if mu1_trk is not None:
            trk = mu1_trk[1]
            p0 = (trk.GetStartX(), trk.GetStartY(), trk.GetStartZ())
            pts = [p0]
            if hasattr(tree, "ScifiPoint"):
                for p in tree.ScifiPoint:
                    if p.GetTrackID() == mu1_trk[0]:
                        pts.append((p.GetX(), p.GetY(), p.GetZ()))
            if hasattr(tree, "MuFilterPoint"):
                for p in tree.MuFilterPoint:
                    if p.GetTrackID() == mu1_trk[0]:
                        pts.append((p.GetX(), p.GetY(), p.GetZ()))
            if len(pts) == 1:
                p1 = self._extrapolate_track(p0, trk, max_len=200.0 if trk.GetP() > 5.0 else 60.0)
                if p1:
                    pts.append(p1)
            pts.sort(key=lambda p: p[2])
            tracks["mu1"] = pts

        # Trajectory for Charm
        if charm_trk is not None:
            c_start = (charm_trk[1].GetStartX(), charm_trk[1].GetStartY(), charm_trk[1].GetStartZ())
            c_decay = None
            if mu2_trk is not None:
                c_decay = (mu2_trk[1].GetStartX(), mu2_trk[1].GetStartY(), mu2_trk[1].GetStartZ())
            elif truth_info.get("decay_length_3d", 0) > 0 and charm_trk[1].GetP() > 0:
                L = truth_info["decay_length_3d"]
                p = charm_trk[1].GetP()
                c_decay = (
                    c_start[0] + L * charm_trk[1].GetPx() / p,
                    c_start[1] + L * charm_trk[1].GetPy() / p,
                    c_start[2] + L * charm_trk[1].GetPz() / p,
                )
            if c_decay is not None:
                tracks["charm"] = [c_start, c_decay]
                tracks["charm_decay"] = c_decay

        # Trajectory for Mu2
        if mu2_trk is not None:
            trk = mu2_trk[1]
            p0 = (trk.GetStartX(), trk.GetStartY(), trk.GetStartZ())
            pts = [p0]
            if hasattr(tree, "ScifiPoint"):
                for p in tree.ScifiPoint:
                    if p.GetTrackID() == mu2_trk[0]:
                        pts.append((p.GetX(), p.GetY(), p.GetZ()))
            if hasattr(tree, "MuFilterPoint"):
                for p in tree.MuFilterPoint:
                    if p.GetTrackID() == mu2_trk[0]:
                        pts.append((p.GetX(), p.GetY(), p.GetZ()))
            if len(pts) == 1:
                p1 = self._extrapolate_track(p0, trk, max_len=50.0)
                if p1:
                    pts.append(p1)
            pts.sort(key=lambda p: p[2])
            tracks["mu2"] = pts

        return tracks

    def draw_event(
        self,
        tree,
        event_idx: int = 0,
        canvas_name: str = "simpleDisplay",
        canvas_title: str = "2d event display",
    ) -> ROOT.TCanvas:
        """
        Builds the 2D event display TCanvas mimicking EventDisplay_Task simpleDisplay.
        """
        tree.GetEntry(event_idx)
        truth = self._extract_truth(tree)
        tracks = self._find_truth_tracks(tree, truth)

        run_id = 0
        if hasattr(tree, "EventHeader"):
            try:
                run_id = tree.EventHeader.GetRunId()
            except Exception:
                pass

        # Canvas matching original nx=1200, ny=1600
        canvas = ROOT.TCanvas(canvas_name, canvas_title, 1200, 1600)
        canvas.Divide(1, 2)

        palette = ROOT.TColor.GetPalette()
        n_pal = len(palette)

        # Pad 1: XZ Projection
        pad_xz = canvas.cd(1)
        pad_xz.SetMargin(0.08, 0.03, 0.08, 0.05)
        hist_xz = ROOT.TH2F(
            f"xz_{event_idx}",
            "; z [cm]; x [cm]",
            500, self.z_min, self.z_max, 100, self.x_min, self.x_max,
        )
        hist_xz.SetStats(0)
        hist_xz.GetXaxis().SetTitleSize(0.04)
        hist_xz.GetYaxis().SetTitleSize(0.04)
        hist_xz.Draw("axis")

        # Pad 2: YZ Projection
        pad_yz = canvas.cd(2)
        pad_yz.SetMargin(0.08, 0.03, 0.08, 0.05)
        hist_yz = ROOT.TH2F(
            f"yz_{event_idx}",
            "; z [cm]; y [cm]",
            500, self.z_min, self.z_max, 100, self.y_min, self.y_max,
        )
        hist_yz.SetStats(0)
        hist_yz.GetXaxis().SetTitleSize(0.04)
        hist_yz.GetYaxis().SetTitleSize(0.04)
        hist_yz.Draw("axis")

        # 1. Draw baseline detectors (original outlines & colors)
        self._draw_detectors(pad_xz, pad_yz)

        # 2. Extract SciFi hits and compute hit densities
        scifi_pts_x = []
        scifi_pts_y = []
        if hasattr(tree, "Digi_ScifiHits"):
            for hit in tree.Digi_ScifiHits:
                det_id = hit.GetDetectorID()
                self.scifi_mod.GetSiPMPosition(det_id, self._vec_a, self._vec_b)
                if hit.isVertical():
                    scifi_pts_x.append((self._vec_a.Z(), self._vec_a.X()))
                else:
                    scifi_pts_y.append((self._vec_a.Z(), self._vec_a.Y()))

        # Draw SciFi Hits with hit density color coding
        scifi_markers = []
        for pts, target_pad in [(scifi_pts_x, pad_xz), (scifi_pts_y, pad_yz)]:
            if not pts:
                continue
            target_pad.cd()
            if self.color_by_qdc_and_density:
                densities = np.clip(get_scifi_hit_density(pts), 0, self.max_density)
                # Sort indices so highest density points are drawn last (on top)
                sorted_indices = np.argsort(densities)
                for idx in sorted_indices:
                    z, coord = pts[idx]
                    dens = densities[idx]
                    c_idx = int(float(dens) / self.max_density * (n_pal - 1))
                    col = palette[c_idx]
                    el = ROOT.TEllipse(z, coord, 1.2, 1.2)
                    el.SetFillStyle(1001)
                    el.SetFillColor(col)
                    el.SetLineWidth(0)
                    el.Draw("same")
                    scifi_markers.append(el)
            else:
                g = ROOT.TGraph()
                g.SetMarkerStyle(20)
                g.SetMarkerSize(1.5)
                g.SetMarkerColor(ROOT.kBlue + 2)
                for i, (z, coord) in enumerate(pts):
                    g.SetPoint(i, z, coord)
                g.Draw("P same")
                scifi_markers.append(g)

        # 3. MuFilter Fired Bars: QDC color coding or solid fill
        nav = ROOT.gGeoManager.GetCurrentNavigator()
        filled_bars = []
        if hasattr(tree, "Digi_MuFilterHits"):
            for hit in tree.Digi_MuFilterHits:
                det_id = hit.GetDetectorID()
                sys_id = hit.GetSystem()
                self.mufilter_mod.GetPosition(det_id, self._vec_a, self._vec_b)
                cur_path = nav.GetPath()

                # Calculate total QDC for this hit
                this_qdc = 0.0
                ns = max(1, hit.GetnSides())
                for side in range(ns):
                    for m in range(hit.GetnSiPMs()):
                        q = hit.GetSignal(m + side * hit.GetnSiPMs())
                        if q > 0:
                            this_qdc += q

                if self.color_by_qdc_and_density:
                    qdc_capped = min(this_qdc, self.max_qdc)
                    qdc_col_idx = int(qdc_capped / self.max_qdc * (n_pal - 1))
                    bar_color = palette[qdc_col_idx]
                else:
                    bar_color = ROOT.kRed + 1 if sys_id == 1 else ROOT.kBlack

                bar_thick = 5 if sys_id == 3 else 3

                # Color ONLY the specific fired bar in its measured projection
                for p, target_pad in [("X", pad_xz), ("Y", pad_yz)]:
                    if cur_path in self.detector_polylines[p]:
                        orig_poly = self.detector_polylines[p][cur_path]
                        fpoly = ROOT.TPolyLine(orig_poly)
                        fpoly.SetFillColor(bar_color)
                        fpoly.SetLineColor(bar_color)
                        fpoly.SetLineWidth(bar_thick)
                        target_pad.cd()
                        fpoly.Draw("f same")
                        fpoly.Draw("same")
                        filled_bars.append(fpoly)


        # 4. Draw MC Truth Tracks (Matching original simpleTracking color scheme)
        track_objs = []
        # Primary Muon 1: Solid Blue (kBlue), line width 2
        if tracks["mu1"] is not None and len(tracks["mu1"]) > 1:
            mu1_line_xz = ROOT.TPolyLine()
            mu1_line_yz = ROOT.TPolyLine()
            for idx, pt in enumerate(tracks["mu1"]):
                mu1_line_xz.SetPoint(idx, pt[2], pt[0])
                mu1_line_yz.SetPoint(idx, pt[2], pt[1])

            for line in [mu1_line_xz, mu1_line_yz]:
                line.SetLineColor(ROOT.kBlue)
                line.SetLineWidth(2)
                line.SetLineStyle(1)

            pad_xz.cd()
            mu1_line_xz.Draw("same")
            pad_yz.cd()
            mu1_line_yz.Draw("same")
            track_objs.extend([mu1_line_xz, mu1_line_yz])

        # Charm Hadron Flight: Green dashed line (kGreen+2), line width 3
        if tracks["charm"] is not None and len(tracks["charm"]) > 1:
            charm_line_xz = ROOT.TPolyLine()
            charm_line_yz = ROOT.TPolyLine()
            for idx, pt in enumerate(tracks["charm"]):
                charm_line_xz.SetPoint(idx, pt[2], pt[0])
                charm_line_yz.SetPoint(idx, pt[2], pt[1])

            for line in [charm_line_xz, charm_line_yz]:
                line.SetLineColor(ROOT.kGreen + 2)
                line.SetLineWidth(3)
                line.SetLineStyle(2)

            pad_xz.cd()
            charm_line_xz.Draw("same")
            pad_yz.cd()
            charm_line_yz.Draw("same")
            track_objs.extend([charm_line_xz, charm_line_yz])

        # Secondary Decay Muon 2: Solid Red (kRed), line width 2
        if tracks["mu2"] is not None and len(tracks["mu2"]) > 1:
            mu2_line_xz = ROOT.TPolyLine()
            mu2_line_yz = ROOT.TPolyLine()
            for idx, pt in enumerate(tracks["mu2"]):
                mu2_line_xz.SetPoint(idx, pt[2], pt[0])
                mu2_line_yz.SetPoint(idx, pt[2], pt[1])

            for line in [mu2_line_xz, mu2_line_yz]:
                line.SetLineColor(ROOT.kRed)
                line.SetLineWidth(2)
                line.SetLineStyle(1)

            pad_xz.cd()
            mu2_line_xz.Draw("same")
            pad_yz.cd()
            mu2_line_yz.Draw("same")
            track_objs.extend([mu2_line_xz, mu2_line_yz])

        # Interaction Vertex & Charm Decay Vertex Markers
        vtx = tracks["vtx"]
        if vtx[2] > -9000.0:
            m_vtx_xz = ROOT.TMarker(vtx[2], vtx[0], 29)
            m_vtx_yz = ROOT.TMarker(vtx[2], vtx[1], 29)
            for m in [m_vtx_xz, m_vtx_yz]:
                m.SetMarkerColor(ROOT.kMagenta + 2)
                m.SetMarkerSize(2.2)
            pad_xz.cd()
            m_vtx_xz.Draw("same")
            pad_yz.cd()
            m_vtx_yz.Draw("same")
            track_objs.extend([m_vtx_xz, m_vtx_yz])

        if tracks["charm_decay"] is not None:
            cdec = tracks["charm_decay"]
            m_dec_xz = ROOT.TMarker(cdec[2], cdec[0], 34)
            m_dec_yz = ROOT.TMarker(cdec[2], cdec[1], 34)
            for m in [m_dec_xz, m_dec_yz]:
                m.SetMarkerColor(ROOT.kOrange + 2)
                m.SetMarkerSize(2.0)
            pad_xz.cd()
            m_dec_xz.Draw("same")
            pad_yz.cd()
            m_dec_yz.Draw("same")
            track_objs.extend([m_dec_xz, m_dec_yz])

        # Track endpoint markers
        if tracks["mu1"] is not None and len(tracks["mu1"]) > 1:
            end_pt = tracks["mu1"][-1]
            m_end_xz = ROOT.TMarker(end_pt[2], end_pt[0], 20)
            m_end_yz = ROOT.TMarker(end_pt[2], end_pt[1], 20)
            for m in [m_end_xz, m_end_yz]:
                m.SetMarkerColor(ROOT.kBlue)
                m.SetMarkerSize(1.0)
            pad_xz.cd()
            m_end_xz.Draw("same")
            pad_yz.cd()
            m_end_yz.Draw("same")
            track_objs.extend([m_end_xz, m_end_yz])

        if tracks["mu2"] is not None and len(tracks["mu2"]) > 1:
            end_pt = tracks["mu2"][-1]
            m_end_xz = ROOT.TMarker(end_pt[2], end_pt[0], 20)
            m_end_yz = ROOT.TMarker(end_pt[2], end_pt[1], 20)
            for m in [m_end_xz, m_end_yz]:
                m.SetMarkerColor(ROOT.kRed)
                m.SetMarkerSize(1.0)
            pad_xz.cd()
            m_end_xz.Draw("same")
            pad_yz.cd()
            m_end_yz.Draw("same")
            track_objs.extend([m_end_xz, m_end_yz])

        # 5. Draw Official SND@LHC Logo and Run/Event info subpads
        logo_objs_1 = self._draw_logo_and_info(pad_xz, 1, run_id, event_idx)
        logo_objs_2 = self._draw_logo_and_info(pad_yz, 2, run_id, event_idx)

        # 6. Draw Density and QDC Color Scale Legend (Bottom of XZ pad matching 2dEventDisplay.py)
        legend_objs_1 = []
        legend_objs_2 = []
        if self.color_by_qdc_and_density:
            legend_objs_1 = self._draw_density_and_qdc_legend(pad_xz)

        # 7. Physics Truth Kinematic Summary Table (XZ Upper Left Area)
        pad_xz.cd()
        pave_info = ROOT.TPaveText(0.10, 0.70, 0.52, 0.92, "NDC")
        pave_info.SetBorderSize(1)
        pave_info.SetFillColor(ROOT.kWhite)
        pave_info.SetTextAlign(12)
        pave_info.SetTextFont(42)
        pave_info.SetTextSize(0.028)

        # Dynamic topology classification
        has_cand = truth.get("has_candidate", False) or (truth["charm_pdg"] != 0 and truth["mu2_p"] > 0)
        has_charm = truth.get("has_charm", False) or truth["charm_pdg"] != 0
        is_numu = truth.get("is_numu_cc", False) or (truth.get("is_cc", False) and truth["mu1_p"] > 0)

        if has_cand:
            topo_tag = "#nu_{#mu} CC Dimuon + Charm"
        elif has_charm:
            topo_tag = "#nu_{#mu} CC + Charm Hadron"
        elif is_numu:
            topo_tag = "#nu_{#mu} CC Interaction"
        elif truth.get("is_cc", False):
            topo_tag = "#nu CC Interaction"
        else:
            topo_tag = "Neutral Current (NC) Interaction"

        pave_info.AddText(f"#bf{{SND@LHC Simulation}}  {topo_tag}")
        pave_info.AddText(f"E_{{#nu}} = {truth['nu_e']:.1f} GeV  |  Vertex: ({vtx[0]:.1f}, {vtx[1]:.1f}, {vtx[2]:.1f}) cm")

        if truth["mu1_p"] > 0:
            pave_info.AddText(f"#mu_{{1}} (Primary Lepton): p = {truth['mu1_p']:.1f} GeV/c")
        else:
            pave_info.AddText("Outgoing Lepton: None (NC / Hadron Shower)")

        if truth["charm_pdg"] != 0:
            charm_name = PDG_NAMES.get(truth["charm_pdg"], f"PDG {truth['charm_pdg']}")
            pave_info.AddText(f"Charm Hadron: #bf{{{charm_name}}}  |  Flight L_{{3D}} = {truth['decay_length_3d']:.2f} cm")

        if truth["mu2_p"] > 0:
            pave_info.AddText(f"#mu_{{2}} (Charm Decay): p = {truth['mu2_p']:.1f} GeV/c  |  M_{{#mu#mu}} = {truth['dimuon_mass']:.2f} GeV/c^{{2}}")

        pave_info.Draw("same")

        # Track & Topology Legend (XZ Upper Right Area)
        leg = ROOT.TLegend(0.68, 0.66, 0.96, 0.92)
        leg.SetBorderSize(1)
        leg.SetFillColor(ROOT.kWhite)
        leg.SetTextFont(42)
        leg.SetTextSize(0.027)

        d_mu1 = ROOT.TLine()
        d_mu1.SetLineColor(ROOT.kBlue)
        d_mu1.SetLineWidth(2)

        d_charm = ROOT.TLine()
        d_charm.SetLineColor(ROOT.kGreen + 2)
        d_charm.SetLineWidth(3)
        d_charm.SetLineStyle(2)

        d_mu2 = ROOT.TLine()
        d_mu2.SetLineColor(ROOT.kRed)
        d_mu2.SetLineWidth(2)

        d_vtx = ROOT.TMarker(0, 0, 29)
        d_vtx.SetMarkerColor(ROOT.kMagenta + 2)
        d_vtx.SetMarkerSize(1.6)

        d_dec = ROOT.TMarker(0, 0, 34)
        d_dec.SetMarkerColor(ROOT.kOrange + 2)
        d_dec.SetMarkerSize(1.4)

        if tracks.get("mu1") is not None and len(tracks["mu1"]) > 1:
            leg.AddEntry(d_mu1, "#mu_{1} (Prompt Muon)", "l")
        if tracks.get("charm") is not None and len(tracks["charm"]) > 1:
            leg.AddEntry(d_charm, "Charm Hadron Flight", "l")
        if tracks.get("mu2") is not None and len(tracks["mu2"]) > 1:
            leg.AddEntry(d_mu2, "#mu_{2} (Charm Decay Muon)", "l")
        if vtx[2] > -9000.0:
            leg.AddEntry(d_vtx, "Primary Interaction Vertex", "p")
        if tracks.get("charm_decay") is not None:
            leg.AddEntry(d_dec, "Charm Decay Vertex", "p")
        leg.Draw("same")

        # Keep Python object references alive on canvas to prevent GC
        canvas._keep_alive = [
            hist_xz, hist_yz,
            pave_info, leg,
            d_mu1, d_charm, d_mu2, d_vtx, d_dec,
        ]
        canvas._keep_alive.extend(scifi_markers)
        canvas._keep_alive.extend(filled_bars)
        canvas._keep_alive.extend(track_objs)
        canvas._keep_alive.extend(logo_objs_1)
        canvas._keep_alive.extend(logo_objs_2)
        canvas._keep_alive.extend(legend_objs_1)
        canvas._keep_alive.extend(legend_objs_2)

        canvas.Update()
        return canvas

    def save_to_root(
        self,
        tfile: ROOT.TFile,
        tree,
        event_idx: int = 0,
        name: str = None,
        subdir: str = "eventdisplay",
    ):
        """
        Generates and saves the TCanvas into the given ROOT TFile under subdir.
        """
        if name is None:
            name = f"event_{event_idx}"

        cur_dir = tfile.GetDirectory(subdir)
        if not cur_dir:
            cur_dir = tfile.mkdir(subdir)
        cur_dir.cd()

        canvas = self.draw_event(tree, event_idx, canvas_name=name, canvas_title=f"Event {event_idx}")
        canvas.Write(name)
        return canvas

    def save_to_image(
        self,
        tree,
        event_idx: int = 0,
        output_path: str = "event_display.png",
    ):
        """
        Renders the event display and saves it to an image (PNG, PDF, SVG).
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        canvas = self.draw_event(tree, event_idx)
        canvas.Print(output_path)
        return output_path
