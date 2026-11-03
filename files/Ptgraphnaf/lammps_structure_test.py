import subprocess
import numpy as np
from ase.build import graphene, make_supercell, molecule
from ase.cluster import Octahedron
from ase.io import read, write
from ase.visualize import view

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

inp = f"""tolerance 2.0
filetype xyz
output liquid.xyz
structure solid.xyz
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure
structure water.xyz
  number {n_water}
  inside box 0. 0. {z0} {Lx} {Ly} {z1}
end structure
"""
open("pack.inp", "w").write(inp)
subprocess.run("packmol < pack.inp", shell=True, check=True)

system = read("liquid.xyz")
system.set_cell([gr.cell[0], gr.cell[1], [0, 0, z1 + 2.0]])
system.set_pbc(True)
write("data.lammps", system, format="lammps-data", atom_style="atomic",
      specorder=["Pt", "C", "O", "H", "F", "S"], masses=True)
#view(system)  # optional: visualize the system

# CHECK
from ase.neighborlist import neighbor_list
check = read("data.lammps", format="lammps-data", style="atomic", units="metal")
print(len(check), check.cell.cellpar(), check.pbc)
print(check.get_chemical_formula())
i, j, d = neighbor_list("ijd", system, 1.2)
print("pairs closer than 1.2 A:", len(d), "min:", d.min() if len(d) else None)