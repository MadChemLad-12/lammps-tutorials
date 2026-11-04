import subprocess
import numpy as np
from ase.build import graphene, make_supercell, molecule
from ase.cluster import Octahedron
from ase.io import read, write
from ase.visualize import view

tol = 2.0    # packing tolerance in Angstroms
vac = 15.0   # real vacuum above the liquid (or a deliberate film + wall)

# Nafion parameters
NAFION_XYZ = "/home/user/Documents/Structures/Nafion/Simple_nafion.xyz"
N_NAFION = 2
HEAD_MODE = "graphene"   # "graphene" | "pt" | "free"
HEAD_ZMAX = 7.0          # head atoms below Z_G + this (Å), graphene mode
HEAD_SHELL = 5.0         # head atoms within r_max + this of NP centroid (Å), pt mode

# Find nafion head atoms (S + bonded O)
def head_indices(naf):
    """1-based Packmol indices of each sulfonate head: S plus the O atoms bonded to it."""
    sym = naf.get_chemical_symbols()
    idx = set()
    for iS in (k for k, s in enumerate(sym) if s == "S"):
        d = naf.get_distances(iS, range(len(naf)))
        idx |= {iS} | {k for k, s in enumerate(sym) if s == "O" and d[k] < 1.9}
    return sorted(i + 1 for i in idx)

def head_block(idx, pt):
    ids = " ".join(map(str, idx))
    if HEAD_MODE == "graphene":
        c = f"below plane 0. 0. 1. {Z_G + HEAD_ZMAX:.2f}"
    elif HEAD_MODE == "pt":
        cen = pt.positions.mean(axis=0)
        r = np.linalg.norm(pt.positions - cen, axis=1).max() + HEAD_SHELL
        c = f"inside sphere {cen[0]:.2f} {cen[1]:.2f} {cen[2]:.2f} {r:.2f}"
    else:
        return ""
    return f"  atoms {ids}\n    {c}\n  end atoms\n"

# --- Orthogonal graphene sheet (a x a*sqrt(3) cell), periodic in x and y ---
gr = graphene(size=(1, 1, 1))
gr = make_supercell(gr, [[1, 0, 0], [1, 2, 0], [0, 0, 1]])  # rectangular cell
gr = gr.repeat((10, 6, 1))            # ~24.6 x 25.6 A; size it so the particle has room
gr.center(vacuum=0, axis=2)
gr.positions[:, 2] = 0.0
Lx, Ly = gr.cell[0, 0], gr.cell[1, 1]

# --- Pt nanoparticle: truncated octahedron, (111) facet pointing down ---
np_ = Octahedron("Pt", length=5, cutoff=2)
np_.rotate((1, 1, 1), (0, 0, -1))     # rotates the [111] direction onto -z
np_.center()
np_.positions[:, 0] += Lx / 2 - np_.positions[:, 0].mean()
np_.positions[:, 1] += Ly / 2 - np_.positions[:, 1].mean()
np_.positions[:, 2] += 3.3 - np_.positions[:, 2].min()   # ~PBE-D3-like separation, relax later

solid = gr + np_
z_top = solid.positions[:, 2].max()
write("solid.xyz", solid)
write("water.xyz", molecule("H2O"))

# --- Liquid region above the graphene, number of waters from volume ---
z0, z1 = 2.5, z_top + 25.0
V = Lx * Ly * (z1 - z0)                              # A^3 (a slight overcount: includes NP volume)
n_water = int(0.9 * 0.0334 * V)                      # ~0.0334 molecules/A^3 at 1 g/cc; 0.9 = slack for NP

inp = f"""tolerance {tol}
filetype xyz
output liquid.xyz
structure solid.xyz
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure
structure water.xyz
  number {n_water}
  inside box {tol/2} {tol/2} {z0} {Lx - tol/2} {Ly - tol/2} {z1}
end structure
"""
open("pack.inp", "w").write(inp)
subprocess.run("packmol < pack.inp", shell=True, check=True)

system = read("liquid.xyz")
system.set_cell([gr.cell[0], gr.cell[1], [0, 0, z1 + vac]])
system.set_pbc(True)
write("data.lammps", system, format="lammps-data", atom_style="atomic",
      specorder=["Pt", "C", "O", "H", "F", "S"], masses=True)
view(system)  # optional: visualize the system

from ase.neighborlist import neighbor_list
# --- better check: ignore intramolecular pairs ---
n_solid = len(solid)
i, j, d = neighbor_list("ijd", system, 1.5)
mol = lambda k: np.where(k >= n_solid, (k - n_solid) // 3, -1 - k)  # unique id per solid atom
bad = ~((i >= n_solid) & (j >= n_solid) & (mol(i) == mol(j)))
print("non-bonded pairs < 1.5 A:", bad.sum(), "min:", d[bad].min() if bad.any() else None)