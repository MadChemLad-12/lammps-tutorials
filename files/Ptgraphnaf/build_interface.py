#!/usr/bin/env python
"""
Pt nanoparticle / graphene / Nafion / water builder for LAMMPS (MACE) simulations.

Every adjustable variable lives in Config (edit the defaults, or override in main()).
All functions receive cfg, so nothing is hard-coded further down.
Atom order in data.lammps: [graphene C + Pt] [Nafion copies] [waters (O,H,H)].
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from ase import Atoms
from ase.build import graphene, make_supercell, molecule
from ase.cluster import Decahedron, Icosahedron, Octahedron, wulff_construction
from ase.io import read, write
from ase.neighborlist import neighbor_list


# =============================================================================
# CONFIG - everything you normally change is here
# =============================================================================
@dataclass
class Config:
    # ---- Pt nanoparticle ----------------------------------------------------
    np_shape: str = "octahedron"      # octahedron | decahedron | icosahedron | wulff
    np_sizes_nm: tuple = (0.8, 1.0, 1.15, 1.3, 1.5)   # volume-equivalent diameters, nm
    oct_family: str = "any"           # octahedron only: any | truncated | cuboctahedron
    facet_down: str = "111"           # 111 | 100 | largest  (facet facing the graphene)
    np_rotate_z: float = 0.0          # degrees about z (align NP with graphene lattice)
    wulff_surfaces: tuple = ((1, 1, 1), (1, 0, 0), (1, 1, 0))
    wulff_energies: tuple = (1.49, 1.73, 1.76)        # relative surface energies -> shape only
    a_pt: float = 3.92                # Pt lattice constant, A

    # ---- graphene support ---------------------------------------------------
    layers: int = 4                   # AB-stacked layers
    layer_spacing: float = 3.35       # A
    graphene_cells: tuple = None      # (nx, ny) orthogonal cells (2.46 x 4.26 A); None = auto from NP
    margin: float = 7.0               # graphene beyond the NP edge, A (NP-image gap = 2*margin)
    z_base: float = 5.0               # z of the bottom graphene layer, A (keeps it off the cell edge)

    # ---- NP / support contact ----------------------------------------------
    embedded: bool = True             # True: NP sunk into the top layer; False: sits on top
    np_gap: float = 3.3               # not embedded: NP bottom this far ABOVE the top layer, A
    embed_depth: float = 1.2          # embedded: NP bottom this far BELOW the top layer, A
    embed_cutoff: float = 2.0         # embedded: remove top-layer C closer than this to any Pt, A

    # ---- Nafion -------------------------------------------------------------
    nafion_xyz: str = "/home/user/Documents/Structures/Nafion/Simple_nafion.xyz"
    n_nafion: int = 2                 # 0 = no Nafion
    head_mode: str = "graphene"       # graphene | pt | free
    head_zmax: float = 7.0            # graphene mode: head atoms below z_top_layer + this, A
    head_shell: float = 5.0           # pt mode: head atoms within r_max + this of NP centroid, A
    nafion_density: float = 1.9       # g/cc, only used to estimate excluded volume

    # ---- water / packing ----------------------------------------------------
    tol: float = 2.0                  # Packmol tolerance, A
    water_above: float = 12.0         # water above NP top, A
    water_floor: float = 2.5          # water starts this far above the top graphene layer, A
    water_fill: float = 0.90          # fraction of 1 g/cc to pack (slack for NP / Nafion)
    vac: float = 10.0                 # vacuum above the liquid, A
    seed: int = 421
    run_packmol: bool = True

    # ---- output -------------------------------------------------------------
    outdir: str = "structures"
    specorder: tuple = ("Pt", "C", "O", "H", "F", "S")
    view: bool = True                # ase.visualize.view of each finished system


RHO_W = 0.0334          # water molecules / A^3 at 1 g/cc
A_CC = 1.42             # graphene C-C bond, A


# =============================================================================
# Nanoparticle
# =============================================================================
def vol_eq_nm(n_atoms, a_pt):
    """Diameter (nm) of a sphere with the same volume as n_atoms fcc Pt atoms."""
    return (3 * n_atoms * a_pt**3 / (2 * np.pi)) ** (1 / 3) / 10.0


def extent(atoms, a_pt):
    """Max atom-atom distance + one nearest-neighbour spacing (A): a 'physical' diameter."""
    p = atoms.positions
    return max(np.linalg.norm(p - q, axis=1).max() for q in p) + a_pt / np.sqrt(2)


def aspect(atoms):
    """sqrt(largest/smallest eigenvalue) of the position covariance: 1 = round."""
    w = np.linalg.eigvalsh(np.cov((atoms.positions - atoms.positions.mean(0)).T))
    return float(np.sqrt(w[-1] / max(w[0], 1e-9)))


def _candidates(cfg, n_target):
    """Yield (label, Atoms) for every particle of the requested shape family."""
    shape = cfg.np_shape.lower()
    if shape == "octahedron":
        for L in range(2, 16):
            for c in range((L - 1) // 2 + 1):
                if cfg.oct_family == "truncated" and not (c >= 1 and L == 3 * c + 1):
                    continue
                if cfg.oct_family == "cuboctahedron" and not (c >= 1 and L == 2 * c + 1):
                    continue
                yield f"Octahedron(L={L},c={c})", Octahedron("Pt", length=L, cutoff=c)
    elif shape == "decahedron":
        for p in range(2, 7):
            for q in range(1, 6):
                for r in range(0, 3):
                    yield f"Decahedron(p={p},q={q},r={r})", Decahedron("Pt", p=p, q=q, r=r)
    elif shape == "icosahedron":
        for n in range(2, 7):
            yield f"Icosahedron(shells={n})", Icosahedron("Pt", noshells=n)
    elif shape in ("wulff", "cluster"):
        # wulff_construction only roughly honours the requested size, so scan around it
        for n_req in range(max(4, int(0.8 * n_target)), int(1.25 * n_target) + 2, 2):
            cl = wulff_construction("Pt", list(cfg.wulff_surfaces), list(cfg.wulff_energies),
                                    n_req, "fcc", rounding="closest",
                                    latticeconstant=cfg.a_pt)
            yield f"Wulff(N_req={n_req})", cl
    else:
        raise ValueError(f"unknown np_shape {cfg.np_shape!r}")


def select_np(cfg, d_nm):
    """Pick the particle of the chosen shape whose volume-equivalent diameter is closest."""
    n_target = max(4, round(np.pi / 6 * (10 * d_nm) ** 3 / (cfg.a_pt**3 / 4)))
    best = None
    for label, cl in _candidates(cfg, n_target):
        a = Atoms(cl)
        a.set_pbc(False)
        if not 4 <= len(a) <= 600:
            continue
        # size error dominates; asphericity breaks ties between equally sized variants
        score = abs(vol_eq_nm(len(a), cfg.a_pt) - d_nm) + 0.02 * (aspect(a) - 1.0)
        if best is None or score < best[0]:
            best = (score, label, a)
    if best is None:
        raise RuntimeError("no candidate particles found")
    return best[1], best[2]


def facet_groups(pos):
    """Group convex-hull triangles into planar facets -> list of (outward normal, area)."""
    hull = ConvexHull(pos)
    groups = []
    for eq, simp in zip(hull.equations, hull.simplices):
        n = eq[:3] / np.linalg.norm(eq[:3])
        t = pos[simp]
        area = 0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0]))
        for g in groups:
            if np.dot(g[0], n) > 0.999:
                g[1] += area
                break
        else:
            groups.append([n, area])
    return [(n, a) for n, a in groups]


def orient_facet_down(cfg, atoms):
    """Rotate so the chosen facet is horizontal and faces -z (toward the support)."""
    shape, want = cfg.np_shape.lower(), cfg.facet_down
    cubic = shape in ("octahedron", "wulff", "cluster")     # cubic axes along x,y,z
    pos = atoms.positions - atoms.positions.mean(0)
    groups = facet_groups(pos)

    def match(n):
        if want == "largest":
            return True
        a = np.sort(np.abs(n))
        if want == "111":
            if cubic:
                return np.allclose(a, 1 / np.sqrt(3), atol=0.05)
            return abs(n[2]) > 0.1 if shape == "decahedron" else True   # skip vertical (100) sides
        if want == "100":
            if cubic:
                return np.allclose(a, [0, 0, 1], atol=0.05)
            return shape == "decahedron" and abs(n[2]) < 0.1            # decahedron (100) sides
        raise ValueError(f"facet_down must be 111, 100 or largest, got {want!r}")

    cand = [(n, a) for n, a in groups if match(n)]
    if not cand:
        raise ValueError(f"{cfg.np_shape} has no {want} facet (try facet_down='largest')")
    n, _ = max(cand, key=lambda t: t[1])
    atoms.rotate(n, (0, 0, -1))
    atoms.rotate(cfg.np_rotate_z, "z")
    atoms.center()


def build_np(cfg, d_nm):
    label, atoms = select_np(cfg, d_nm)
    orient_facet_down(cfg, atoms)
    atoms.set_cell(None)
    atoms.set_pbc(False)
    return label, atoms


# =============================================================================
# Graphene support + NP placement / embedding
# =============================================================================
def build_graphene(cfg, np_width_xy):
    unit = graphene(size=(1, 1, 1))
    unit = make_supercell(unit, [[1, 0, 0], [1, 2, 0], [0, 0, 1]])   # 2.46 x 4.26 A
    ax, ay = unit.cell[0, 0], unit.cell[1, 1]
    if cfg.graphene_cells is None:
        L = max(np_width_xy) + 2 * cfg.margin                        # ~square sheet
        nx, ny = int(np.ceil(L / ax)), int(np.ceil(L / ay))
    else:
        nx, ny = cfg.graphene_cells
    sheet = unit.repeat((nx, ny, 1))
    sheet.set_pbc((True, True, False))
    gr = Atoms()
    for k in range(cfg.layers):
        layer = sheet.copy()
        if k % 2 == 1:                       # AB (Bernal) stacking: shift one C-C bond along y
            layer.positions[:, 1] += A_CC
            layer.wrap()
        layer.positions[:, 2] = cfg.z_base + k * cfg.layer_spacing
        layer.set_tags(k)
        gr += layer
    Lx, Ly = nx * ax, ny * ay
    gr.set_cell(np.diag([Lx, Ly, 0.0]))
    return gr, Lx, Ly


def assemble_solid(cfg, np_, gr, Lx, Ly):
    """Place the NP on / in the top layer; drop overlapping C when embedded."""
    z_top_layer = cfg.z_base + (cfg.layers - 1) * cfg.layer_spacing
    p = np_.positions.copy()
    p[:, 0] += Lx / 2 - 0.5 * (p[:, 0].min() + p[:, 0].max())
    p[:, 1] += Ly / 2 - 0.5 * (p[:, 1].min() + p[:, 1].max())
    z_bottom = z_top_layer - cfg.embed_depth if cfg.embedded else z_top_layer + cfg.np_gap
    p[:, 2] += z_bottom - p[:, 2].min()
    np_ = Atoms("Pt" * len(np_), positions=p)

    n_removed = 0
    if cfg.embedded:
        top = np.where(gr.get_tags() == cfg.layers - 1)[0]
        dmin = np.linalg.norm(gr.positions[top][:, None, :] - p[None, :, :], axis=2).min(axis=1)
        drop = top[dmin < cfg.embed_cutoff]
        n_removed = len(drop)
        keep = np.setdiff1d(np.arange(len(gr)), drop)
        gr = gr[keep]
        if cfg.layers > 1:
            low = np.where(gr.get_tags() < cfg.layers - 1)[0]
            dl = np.linalg.norm(gr.positions[low][:, None, :] - p[None, :, :], axis=2).min()
            if dl < 2.0:
                print(f"   WARNING: NP is {dl:.2f} A from a lower graphene layer - reduce embed_depth")

    solid = gr + np_
    if "tags" in solid.arrays:               # keep Packmol's xyz reader happy
        del solid.arrays["tags"]
    dpc = np.linalg.norm(gr.positions[:, None, :] - p[None, :, :], axis=2).min()
    return solid, np_, z_top_layer, n_removed, dpc


# =============================================================================
# Nafion + Packmol
# =============================================================================
def head_indices(naf):
    """1-based Packmol indices of each sulfonate head: S plus the O atoms bonded to it."""
    sym = naf.get_chemical_symbols()
    idx = set()
    for iS in (k for k, s in enumerate(sym) if s == "S"):
        d = naf.get_distances(iS, range(len(naf)))
        idx |= {iS} | {k for k, s in enumerate(sym) if s == "O" and d[k] < 1.9}
    if not idx:
        raise ValueError("no sulfur found in the Nafion file")
    return sorted(i + 1 for i in idx)


def head_block(cfg, idx, pt, z_top_layer):
    ids = " ".join(map(str, idx))
    if cfg.head_mode == "graphene":
        c = f"below plane 0. 0. 1. {z_top_layer + cfg.head_zmax:.2f}"
    elif cfg.head_mode == "pt":
        cen = pt.positions.mean(axis=0)
        r = np.linalg.norm(pt.positions - cen, axis=1).max() + cfg.head_shell
        c = f"inside sphere {cen[0]:.2f} {cen[1]:.2f} {cen[2]:.2f} {r:.2f}"
    elif cfg.head_mode == "free":
        return ""
    else:
        raise ValueError("head_mode must be graphene, pt or free")
    return f"  atoms {ids}\n    {c}\n  end atoms\n"


def pack_system(cfg, solid, pt, z_top_layer, Lx, Ly, workdir):
    tol = cfg.tol
    naf = read(cfg.nafion_xyz) if cfg.n_nafion > 0 else None
    z_floor = z_top_layer + cfg.water_floor
    z_top = pt.positions[:, 2].max() + cfg.water_above
    v_excl = np.pi / 6 * (extent(pt, cfg.a_pt) + tol) ** 3
    if naf is not None:
        L_naf = max(np.linalg.norm(naf.positions - q, axis=1).max() for q in naf.positions)
        z_top = max(z_top, z_floor + L_naf + 2.0)           # room for the whole chain
        v_excl += cfg.n_nafion * 1.6605 * naf.get_masses().sum() / cfg.nafion_density
    n_w = int(cfg.water_fill * RHO_W * (Lx * Ly * (z_top - z_floor) - v_excl))

    write(workdir / "solid.xyz", solid)
    write(workdir / "water.xyz", molecule("H2O"))
    box = f"{tol/2} {tol/2} {z_floor} {Lx - tol/2} {Ly - tol/2} {z_top}"
    naf_block = ""
    if naf is not None:
        naf_block = (f"structure {Path(cfg.nafion_xyz).resolve()}\n"
                     f"  number {cfg.n_nafion}\n  inside box {box}\n"
                     f"{head_block(cfg, head_indices(naf), pt, z_top_layer)}end structure\n\n")
    (workdir / "pack.inp").write_text(
        f"tolerance {tol}\nfiletype xyz\nseed {cfg.seed}\noutput liquid.xyz\n\n"
        f"structure solid.xyz\n  number 1\n  fixed 0. 0. 0. 0. 0. 0.\nend structure\n\n"
        f"{naf_block}"
        f"structure water.xyz\n  number {n_w}\n  inside box {box}\nend structure\n")

    if cfg.run_packmol:
        subprocess.run("packmol < pack.inp > pack.log", shell=True, check=True, cwd=workdir)
    system = read(workdir / "liquid.xyz")
    system.set_cell(np.diag([Lx, Ly, z_top + cfg.vac]))
    system.set_pbc(True)
    return system, naf, n_w


def check(system, n_solid, n_naf, len_naf):
    """Inter-molecular contacts < 1.5 A (periodic). Solid = one group, Nafion/water per molecule."""
    n = len(system)
    mol = np.zeros(n, dtype=int)
    for k in range(n_naf):
        mol[n_solid + k * len_naf: n_solid + (k + 1) * len_naf] = 1 + k
    start = n_solid + n_naf * len_naf
    mol[start:] = 1000 + np.arange(n - start) // 3          # O,H,H per water
    i, j, d = neighbor_list("ijd", system, 1.5)
    bad = mol[i] != mol[j]
    return int(bad.sum()), (float(d[bad].min()) if bad.any() else None)


# =============================================================================
# Driver
# =============================================================================
def build_one(cfg, d_nm):
    label, np0 = build_np(cfg, d_nm)
    width = np.ptp(np0.positions[:, :2], axis=0)
    gr, Lx, Ly = build_graphene(cfg, width)
    solid, pt, z_top_layer, n_removed, dpc = assemble_solid(cfg, np0, gr, Lx, Ly)

    tag = f"{cfg.np_shape}_{len(pt):03d}pt"
    workdir = Path(cfg.outdir) / tag
    workdir.mkdir(parents=True, exist_ok=True)

    system, naf, n_w = pack_system(cfg, solid, pt, z_top_layer, Lx, Ly, workdir)
    n_naf, len_naf = (cfg.n_nafion, len(naf)) if naf is not None else (0, 0)
    nbad, dmin = check(system, len(solid), n_naf, len_naf)

    write(workdir / "data.lammps", system, format="lammps-data", atom_style="atomic",
          specorder=list(cfg.specorder), masses=True)
    if cfg.view:
        from ase.visualize import view
        view(system)

    print(f"{d_nm:5.2f} {len(pt):5d} {vol_eq_nm(len(pt), cfg.a_pt):6.2f} "
          f"{extent(pt, cfg.a_pt)/10:6.2f} {n_removed:5d} {dpc:6.2f} "
          f"{Lx:6.1f}x{Ly:5.1f} {len(system):7d}   contacts<1.5A: {nbad}"
          + (f" (min {dmin:.2f})" if dmin else "") + f"   [{label}]")


def main():
    cfg = Config()          # override here if you like, e.g. Config(np_shape="icosahedron")
    print(f"{'d_req':>5} {'N_Pt':>5} {'d_vol':>6} {'d_ext':>6} {'C_rm':>5} "
          f"{'minPtC':>6} {'cell':>12} {'atoms':>7}")
    for d in cfg.np_sizes_nm:
        build_one(cfg, d)


if __name__ == "__main__":
    main()