import uproot
import numpy as np
from collections import defaultdict
from particle import PDGID, Particle

import hist
import matplotlib.pyplot as plt
import mplhep as mh

# Hard-coded path to Pythia8 installation
import sys
sys.path.append("/Users/cvilela/SND/CharmProduction/pythia8318/lib/")
import pythia8

# Hard-coded path to location of GENIE gst files
data = uproot.concatenate([f"sndlhc_+volTarget_SND_LHC_SNDG18_02a_01_000.0.gst.{i}.root:gst" for i in range(1, 2)])

n_ev = 0
n_charm = 0
charm_pdg = defaultdict(int)

charm_momenta = []
prim_muon_momentum = []

for event in data:
    if not event.cc:
        continue
    
    n_ev += 1
    if event.charm == 1:
        n_charm += 1

        prim_muon_pdg = 13 if event.neu == 14 else -13
        prim_muon_momentum.append([prim_muon_pdg, event.pxl, event.pyl, event.pzl])

        charm_list = []
        
        for i_f in range(event.nf):
            pdg = PDGID(event.pdgf[i_f])
            if pdg.has_charm:
                charm_pdg[pdg] += 1
                charm_list.append([pdg, event.pxf[i_f], event.pyf[i_f], event.pzf[i_f]])
        charm_momenta.append(charm_list)

print(f"Number of CC events {n_ev}, with charm {n_charm}. Ratio = {n_charm/n_ev:.2f}")
for charm_had in charm_pdg.keys():
    print(f"Fraction of {charm_had}: {charm_pdg[charm_had]/n_ev:.2f}")

# Generate charm hadron decays with pythia8
pythia8_commands = ["processLevel:all = off", # insert decaying hadron by hand
                    "Next:numberCount = 0", # silence counter
                    "Random:setSeed = on", # manually set seed
                    "ParticleDecays:limitTau0 = off"] # remove lifetime limit
# Turn on decays for charm hadrons
for charm_had in charm_pdg.keys():
    pythia8_commands.append(f"{charm_had}:mayDecay = on")

pythia = pythia8.Pythia("", False)
for cmd in pythia8_commands:
    pythia.readString(cmd)
pythia.init()

muon_momenta = []

for i_event in range(len(charm_momenta)):
    this_muon = []

    for j_charm in range(len(charm_momenta[i_event])):
    
        pythia.event.reset()
        
        pdg = charm_momenta[i_event][j_charm][0]
        m = Particle.from_pdgid(pdg).mass/1000
        px = charm_momenta[i_event][j_charm][1]
        py = charm_momenta[i_event][j_charm][2]
        pz = charm_momenta[i_event][j_charm][3]
        E = (m**2 + px**2 + py**2 + pz**2)**0.5
        
        # append(id, status, mother1, mother2, px, py, pz, e, m): mu- at rest
        pythia.event.append(pdg, # pdg
                            1, # status
                            0, # mother1
                            0, # mother2
                            px, py, pz, E, m)
        
        if not pythia.next():
            continue
        
        for i in range(pythia.event.size()):
            p = pythia.event[i]
            if p.isFinal() and abs(p.id()) == 13: #muon
                this_muon.append([p.id(), p.px(), p.py(), p.pz()])
    muon_momenta.append(this_muon)

muon_multiplicity = defaultdict(int)
for m in muon_momenta:
    muon_multiplicity[len(m)] += 1

for mult in muon_multiplicity.keys():
    print(f"Fraction of charm events with {mult} muons: {muon_multiplicity[mult]/len(charm_momenta):.2f} ({muon_multiplicity[mult]})")

h_cos_angle = hist.new.Reg(10, 0.9, 1., name = "opening", label = r"cos$\theta_{\mu\mu}$").Weight()
h_cos_2d = hist.new.Reg(10, 0.9, 1, name = "primary", label = r"Primary muon cos$\theta$").Reg(10, 0.9, 1, name = "charm", label = r"Charm-daughter muon cos$\theta$").Weight()

h_angle_opening = hist.new.Reg(20, 0, 0.2, name = "opening", label = r"$\theta_{\mu\mu}$").Weight()
h_angle_2d = hist.new.Reg(20, 0, 0.2, name = "primary", label = r"Primary muon $\theta$").Reg(20, 0, 0.2, name = "charm", label = r"Charm-daughter muon $\theta$").Weight()

h_mom_2d = hist.new.Reg(10, 0, 500, name = "primary", label = "Primary muon momentum [GeV]").Reg(10, 0, 500, name = "charm", label = "Charm-daughter muon mom. [GeV]").Weight()

for i_event in range(len(muon_momenta)):
    if len(muon_momenta[i_event]) == 1:
        abs_p_prim_mu = np.sqrt(np.sum(np.square(prim_muon_momentum[i_event][1:4])))
        abs_p_charm_mu = np.sqrt(np.sum(np.square(muon_momenta[i_event][0][1:4])))
        
        this_cos = np.sum([muon_momenta[i_event][0][j]*prim_muon_momentum[i_event][j] for j in range(1, 4)])/(abs_p_prim_mu*abs_p_charm_mu)

        h_cos_angle.fill(this_cos)
        h_cos_2d.fill(prim_muon_momentum[i_event][3]/abs_p_prim_mu, muon_momenta[i_event][0][3]/abs_p_charm_mu)

        h_angle_opening.fill(np.arccos(this_cos))
        h_angle_2d.fill(np.arccos(prim_muon_momentum[i_event][3]/abs_p_prim_mu), np.arccos(muon_momenta[i_event][0][3]/abs_p_charm_mu))

        h_mom_2d.fill(abs_p_prim_mu, abs_p_charm_mu)

fig1, ax1 = plt.subplots(2, 2)
mh.histplot(h_cos_angle, ax = ax1[0][0])
mh.hist2dplot(h_cos_2d, ax = ax1[1][1])
mh.histplot(h_cos_2d.integrate("primary"), ax = ax1[1][0])
mh.histplot(h_cos_2d.integrate("charm"), ax = ax1[0][1])
plt.tight_layout()

fig3, ax3 = plt.subplots(2, 2)
mh.histplot(h_angle_opening, ax = ax3[0][0])
mh.hist2dplot(h_angle_2d, ax = ax3[1][1])
mh.histplot(h_angle_2d.integrate("primary"), ax = ax3[1][0])
mh.histplot(h_angle_2d.integrate("charm"), ax = ax3[0][1])
plt.tight_layout()

fig2, ax2, = plt.subplots(2, 2)
ax2[0,0].axis('off')
mh.hist2dplot(h_mom_2d, ax = ax2[1][1])
mh.histplot(h_mom_2d.integrate("primary"), ax = ax2[1][0])
mh.histplot(h_mom_2d.integrate("charm"), ax = ax2[0][1])
plt.tight_layout()

print(f"Fraction of di-muon events with more than 50 mrad opening angle: {np.sum(h_angle_opening[0.05j:].values())/np.sum(h_angle_opening.values()):.2f}")

print(h_angle_opening)

plt.show()
