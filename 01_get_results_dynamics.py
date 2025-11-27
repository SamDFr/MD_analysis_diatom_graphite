#!/usr/bin/env python3

#!/usr/bin/env python3
"""
NO / diatomic scattering analysis on graphite (batch, cluster-ready).

This script:
  - Reads runtime parameters from an INI configuration file (see example no_scattering.ini).
  - Scans a base directory for trajectory files (.xyz, .lammpstrj, .dump, vasprun.xml, OUTCAR).
  - For each trajectory, computes:
      * COM kinematics (Vcm, scattering angle, height above slab)
      * Rotational / vibrational / total energies (via a diatomic model, optional Morse)
      * Geometric descriptors (bond length, orientations, etc.)
      * A binary outcome label (scattered vs stuck) using a detector height and a minimum-frame criterion.
  - Writes:
      * Per-trajectory CSV files: <traj>_metrics.csv (all frames)
      * A global batch_summary.csv file (one line per trajectory) in OUT_DIR.

Usage on cluster:
    python NO_scattering_analysis.py CONFIG.ini

The INI file controls: molecule, temperatures, incidence, detector height,
LAMMPS specorder and time-step, Morse parameters, base_dir, and output directory.
"""

import os
import sys
from configparser import ConfigParser

import numpy as np
from math import sqrt
from ase.io import read
from ase.data import atomic_masses, chemical_symbols

# Optional tabulate helper (same behavior as original)
sys.path.append(os.path.expanduser('~/bin/tabulate'))
try:
    from tabulate import tabulate
except Exception:
    tabulate = None

###########################################
# Units and constants
###########################################
# Conversions FROM atomic units
autoev   = 27.211383858491185
autofs   = 0.02418884326505      # a.u. time → fs
autoan   = 0.52917720859         # bohr → Å
autoamu  = 0.0005485799092659504
autoanf  = 21.876912541518593    # a.u. velocity → Å/fs

# Conversions TO atomic units
evtoau   = 0.03674932540
fstoau   = 41.341373336561364
antoau   = 1.8897261328856432
amutoau  = 1822.888485540950
anftoau  = 0.045710289242239875  # Å/fs → a.u. velocity

###########################################
# Helpers  (PBC-aware, NO only outputs)
###########################################

# Will be overridden from config in main()
SPECORDER = ['N', 'O', 'C']


def file_kind(path):
    n = os.path.basename(path).lower()
    if n.endswith('.lammpstrj') or n.endswith('.dump') or 'lammpstrj' in n:
        return 'lammps-dump-text'
    if n == 'outcar':
        return 'vasp-outcar'
    if n.endswith('.xml') or n == 'vasprun.xml':
        return 'vasp-xml'
    if n.endswith('.xyz'):
        return 'xyz'
    return 'auto'


def load_frames(path):
    """Return list[Atoms]. Uses ASE readers. Adds no extra arrays."""
    kind = file_kind(path)

    if kind == 'lammps-dump-text':
        # Positions + types via ASE
        frames = read(path, format='lammps-dump-text', index=":", specorder=SPECORDER)

        # Velocities via MDAnalysis if present
        try:
            import MDAnalysis as mda
        except ImportError:
            print("MDAnalysis not available: velocities from LAMMPS dump will not be used.", file=sys.stderr)
            return frames if isinstance(frames, list) else [frames]

        u = mda.Universe(path, format="LAMMPSDUMP")
        n_frames = len(u.trajectory)
        n_atoms = u.atoms.n_atoms
        if not u.trajectory.ts.has_velocities:
            print("No velocities in LAMMPS dump; will use finite-difference if requested.", file=sys.stderr)
            return frames if isinstance(frames, list) else [frames]

        V = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
        for i, ts in enumerate(u.trajectory):
            vel = u.atoms.velocities
            if vel is None:
                print("Frame has no velocities; falling back to finite-difference later.", file=sys.stderr)
                V = None
                break
            V[i] = vel

        frames_list = frames if isinstance(frames, list) else [frames]
        if V is not None and len(frames_list) == len(V):
            for i, atoms in enumerate(frames_list):
                atoms.set_velocities(V[i])
        return frames_list

    if kind == 'vasp-xml':
        frames = read(path, index=":")
        return frames if isinstance(frames, list) else [frames]

    if kind == 'xyz':
        frames = read(path, index=":")
        return frames if isinstance(frames, list) else [frames]

    # fallback
    frames = read(path, index=":")
    return frames if isinstance(frames, list) else [frames]


def get_velocities(at, vel_units='A_per_ps'):
    """
    Return per-atom velocities in Å/fs.

    Order of preference:
      1) ASE internal velocities (Atoms.get_velocities)
      2) vx,vy,vz arrays (LAMMPS, usually Å/ps)
      3) 'velocities' or 'velocity' arrays (Å/fs)
      4) 'momenta' (amu·Å/fs) divided by atomic masses
    """
    import numpy as np

    # static attribute to avoid re-printing
    if not hasattr(get_velocities, "_ase_vel_printed"):
        get_velocities._ase_vel_printed = False

    arr = at.arrays
    v = None

    # 1) ASE internal velocities
    try:
        v = at.get_velocities()
        if v is not None and np.any(v):
            if not get_velocities._ase_vel_printed:
                print("Using ASE internal velocities.")
                get_velocities._ase_vel_printed = True

            if vel_units == 'A_per_ps':  # convert Å/ps → Å/fs
                v = np.asarray(v, float) / 1000.0
            return np.asarray(v, float)
        else:
            if not get_velocities._ase_vel_printed:
                print("No ASE velocities found — choosing another method.")
                get_velocities._ase_vel_printed = True
    except Exception:
        if not get_velocities._ase_vel_printed:
            print("No ASE velocities found — choosing another method.")
            get_velocities._ase_vel_printed = True

    # 2) explicit vx,vy,vz (LAMMPS)
    if all(k in arr for k in ('vx', 'vy', 'vz')):
        if not get_velocities._ase_vel_printed:
            print("Using 'vx','vy','vz' arrays.")
            get_velocities._ase_vel_printed = True
        v = np.column_stack((arr['vx'], arr['vy'], arr['vz'])).astype(float)
        if vel_units == 'A_per_ps':
            v /= 1000.0
        return v

    # 3) velocities arrays (ASE)
    if 'velocities' in arr or 'velocity' in arr:
        if not get_velocities._ase_vel_printed:
            print("Using 'velocities' or 'velocity' array.")
            get_velocities._ase_vel_printed = True
        v = np.asarray(arr.get('velocities', arr.get('velocity')), float)
        if vel_units == 'A_per_ps':
            v /= 1000.0
        return v

    # 4) momenta → velocities
    if 'momenta' in arr:
        if not get_velocities._ase_vel_printed:
            print("Using 'momenta' array to compute velocities.")
            get_velocities._ase_vel_printed = True
        m = at.get_masses()[:, None]
        p = np.asarray(arr['momenta'], float)
        v = p / m
        if vel_units == 'A_per_ps':
            v /= 1000.0
        return v

    if not get_velocities._ase_vel_printed:
        print("No velocity data found in any source.")
        get_velocities._ase_vel_printed = True

    return None


def _mic(dr, cell, pbc):
    if cell is None or not np.any(pbc):
        return dr
    from ase.geometry import find_mic
    mic_dr, _ = find_mic(dr, cell=cell, pbc=pbc)
    return mic_dr


def finite_diff_vel(frames, dt_fs, vel_units='A_per_fs'):
    print("CAREFUL - using finite-difference velocities!")
    cell = frames[0].get_cell()
    pbc  = frames[0].get_pbc()
    pos = [f.get_positions() for f in frames]
    v = [np.zeros_like(pos[0])]
    for k in range(1, len(frames)-1):
        dr = pos[k+1] - pos[k-1]
        for i in range(dr.shape[0]):
            dr[i] = _mic(dr[i], cell, pbc)
        v.append(dr/(2.0*dt_fs))
    if len(frames) > 1:
        dr = pos[-1] - pos[-2]
        for i in range(dr.shape[0]):
            dr[i] = _mic(dr[i], cell, pbc)
        v.append(dr/dt_fs)
    if vel_units == 'A_per_ps':  # convert Å/ps → Å/fs
        v = [vi / 1000.0 for vi in v]
    return v


def _lz(at) -> float:
    # Works for non-orthorhombic too
    return float(np.linalg.norm(at.cell[2]))


def slab_reference_z(at, simple=False, top_tol=0.10, probe_z=None):
    """
    Reference z for a graphite slab.

    If probe_z is None:
        Original behavior (simple=True: mean near global zmax; else k=3 z-clustering).
    If probe_z is not None:
        Return the average z of C atoms within top_tol Å of the highest C layer
        that is *below or equal to* probe_z, using minimum-image wrapping on z.
        Excludes atoms 0 and 1.
    """
    pos = at.get_positions()
    Z   = at.get_atomic_numbers()

    Cidx = np.where(Z == 6)[0]
    Cidx = Cidx[(Cidx != 0) & (Cidx != 1)]
    if Cidx.size == 0:
        return float(np.percentile(pos[:, 2], 10))

    Cz = pos[Cidx, 2].astype(float)

    # ---- New path: reference layer below a probe, PBC-safe
    if probe_z is not None:
        pbc = at.get_pbc()
        Lz  = _lz(at)
        dz = Cz - float(probe_z)
        if pbc[2] and Lz > 0.0:
            dz -= np.rint(dz / Lz) * Lz  # minimum-image along z

        below = dz <= 0.0
        if not np.any(below):
            # Edge case: everything wrapped above. Shift once.
            dz -= Lz
            below = dz <= 0.0
            if not np.any(below):
                # Fallback: use classic top-of-C around global max
                zmax = float(Cz.max())
                band = Cz[Cz >= zmax - float(top_tol)]
                return float(band.mean() if band.size else zmax)

        dz_below = dz[below]
        z_ref = float(probe_z + np.max(dz_below))

        # Average a thin band for robustness
        Cz_unwrapped_below = probe_z + dz_below
        band = np.abs(Cz_unwrapped_below - z_ref) <= float(top_tol)
        return float(Cz_unwrapped_below[band].mean()) if np.any(band) else z_ref

    # ---- Old behavior unchanged below
    if Cz.size < 3:
        return float(Cz.mean())

    if simple:
        zmax = float(Cz.max())
        top = Cz[Cz >= zmax - float(top_tol)]
        return float(top.mean()) if top.size else zmax

    # k=3 clustering on z
    k = 3
    cent = np.percentile(Cz, [10, 50, 90]).astype(float)
    for _ in range(15):
        d = np.abs(Cz[:, None] - cent[None, :])
        lab = np.argmin(d, axis=1)
        new_cent = np.array([Cz[lab == i].mean() if np.any(lab == i) else cent[i] for i in range(k)])
        if np.allclose(new_cent, cent):
            break
        cent = new_cent
    top_cluster = int(np.argmax(cent))
    top_vals = Cz[lab == top_cluster]
    if top_vals.size == 0:
        top_vals = Cz[Cz >= np.percentile(Cz, 80)]
    return float(top_vals.mean() if top_vals.size else Cz.mean())


def pick_NO_indices(at, id_pair=(1, 2), type_pair=(1, 2), elem_pair=('N', 'O')):
    ids   = at.arrays.get('id')
    types = at.arrays.get('type')
    if ids is not None and id_pair is not None:
        wN = np.where(ids == id_pair[0])[0]
        wO = np.where(ids == id_pair[1])[0]
        if wN.size and wO.size:
            return int(wN[0]), int(wO[0])
    if types is not None and type_pair is not None:
        wN = np.where(types == type_pair[0])[0]
        wO = np.where(types == type_pair[1])[0]
        if wN.size and wO.size:
            return int(wN[0]), int(wO[0])
    if elem_pair is not None:
        sym = at.get_chemical_symbols()
        try:
            iN = next(i for i, s in enumerate(sym) if s == elem_pair[0])
            iO = next(i for i, s in enumerate(sym) if s == elem_pair[1])
            return (iN, iO)
        except StopIteration:
            pass
    return None


def com(masses, pos):
    M = np.sum(masses)
    return (pos * masses[:, None]).sum(0) / M


def safe_acos_deg(x):
    return np.degrees(np.arccos(np.clip(x, -1.0, 1.0)))


def diatomic_properties(Apos, Bpos, Avel, Bvel, mA_amu, mB_amu, slab_z_ref,
                        detector_A=None, cell=None, pbc=None, De=None, a=None, re=None):
    # MIC bond in Å
    dAB = Bpos - Apos
    rAB_A = dAB if (cell is None or pbc is None) else _mic(dAB, cell, pbc)
    Bpos_contig = Apos + rAB_A

    # to a.u.
    qA, qB = Apos * antoau, Bpos_contig * antoau
    vA, vB = Avel * anftoau, Bvel * anftoau
    mA, mB = mA_amu * amutoau, mB_amu * amutoau

    # geometry + COM
    rAB  = qB - qA
    r    = np.linalg.norm(rAB) if np.any(rAB) else 1.0
    rhat = rAB / r
    M    = mA + mB
    Rcm  = (mA * qA + mB * qB) / M
    Vcm  = (mA * vA + mB * vB) / M

    # reduced motion
    mu    = (mA * mB) / M
    vrel  = vB - vA
    p_rel = mu * vrel
    p_par = float(np.dot(p_rel, rhat))
    L_vec = np.cross(rAB, p_rel)
    L2    = float(np.dot(L_vec, L_vec))

    # energies (a.u.)
    Ecm_au  = 0.5 * M * np.dot(Vcm, Vcm)
    Evib_au = 0.5 * (p_par**2) / mu
    Erot_au = 0.5 * L2 / (mu * r**2)

    # optional Morse (eV): r in Å here
    if De is not None and a is not None and re is not None:
        mol_Epot_eV = De * (1.0 - np.exp(-a * (r / antoau - re)))**2
    else:
        mol_Epot_eV = 0.0

    # back to lab
    com_A   = (Rcm / antoau).tolist()
    vcm_Af  = (Vcm / anftoau).tolist()
    d_NO1_A = r / antoau
    heightA = (Rcm / antoau)[2] - slab_z_ref

    def cart2sph(v):
        R = np.linalg.norm(v)
        if R == 0:
            return (0.0, 0.0, 0.0)
        theta = np.degrees(np.arccos(np.clip(v[2] / R, -1.0, 1.0)))  # angle from +z
        phi   = np.degrees(np.arctan2(v[1], v[0])) % 360.0
        return (R, theta, phi)

    # orientation of O relative to COM
    O_rel_cm = qB - Rcm
    _, NO1_theta_deg, NO1_phi_deg = cart2sph(O_rel_cm)

    # bond-based orientation
    NO1_theta_a = safe_acos_deg(rAB[2] / r)
    NO1_phi_a = float((np.degrees(np.arctan2(rAB[1], rAB[0])) + 360.0) % 360.0) if (rAB[0] or rAB[1]) else 0.0

    # velocity orientation from COM velocity
    _, vel_theta_deg, vel_phi_deg = cart2sph(Vcm)
    Vcm_norm = np.linalg.norm(Vcm)
    vel_theta_a = safe_acos_deg(Vcm[2] / Vcm_norm) if Vcm_norm > 0 else 0.0
    vel_phi_a = float((np.degrees(np.arctan2(Vcm[1], Vcm[0])) + 360.0) % 360.0) if (Vcm[0] or Vcm[1]) else 0.0

    # j from |L| (ħ=1 in a.u.)
    L_norm    = np.sqrt(L2)
    jrot_cont = 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * L_norm**2))
    jrot      = float(np.rint(jrot_cont))  # round to nearest

    # channel: 1 emitted, 0 adsorbed
    output_CH = 1 if (detector_A is not None and heightA >= detector_A) else 0

    return {
        'output'          : output_CH,
        'mol_massCenter'  : com_A,
        'mol_mCenter_vel' : vcm_Af,
        'mol_jrot'        : jrot,
        'NO1_theta'       : round(NO1_theta_deg, 2),
        'NO1_phi'         : round(NO1_phi_deg, 2),
        'NO1_theta_a'     : round(NO1_theta_a, 2),
        'NO1_phi_a'       : round(NO1_phi_a, 2),
        'd_NO1'           : round(d_NO1_A, 4),
        'd_mol_L3'        : round(heightA, 4),
        'mol_vel_theta'   : round(vel_theta_deg, 2),
        'mol_vel_phi'     : round(vel_phi_deg, 2),
        'mol_vel_theta_a' : round(vel_theta_a, 2),
        'mol_vel_phi_a'   : round(vel_phi_a, 2),
        'mol_Ekin'        : round(Ecm_au * autoev, 4),  # COM kinetic (Ecm)
        'mol_Epot'        : round(mol_Epot_eV, 4),
        'mol_Evib_cin'    : round(Evib_au * autoev, 4),
        'mol_Erot'        : round(Erot_au * autoev, 4),
        'mol_Evib_tot'    : round(Evib_au * autoev + mol_Epot_eV, 4)
    }

###########################################
# Config parsing
###########################################

def parse_config(path):
    cfg = ConfigParser()
    with open(path, 'r') as f:
        cfg.read_file(f)

    # [system]
    incidence   = cfg.get('system', 'incidence', fallback='Normal')
    temperature = cfg.getint('system', 'temperature', fallback=300)
    energy      = cfg.get('system', 'energy', fallback='1.0')
    file_format = cfg.get('system', 'file_format', fallback='VASP')
    molecule    = cfg.get('system', 'molecule', fallback='NO').strip().upper()
    detector_A  = cfg.getfloat('system', 'detector_A', fallback=7.0)
    n_min_frames = cfg.getint('system', 'n_min_frames', fallback=50)

    diatom_str = cfg.get('system', 'diatom_ids', fallback='').strip()
    if diatom_str:
        parts = diatom_str.split(',')
        if len(parts) != 2:
            raise ValueError("diatom_ids must be of the form 'iA,iB' or empty.")
        diatom_ids = (int(parts[0]), int(parts[1]))
    else:
        diatom_ids = None

    # [paths]
    base_dir = cfg.get('paths', 'base_dir')
    out_dir_raw = cfg.get('paths', 'out_dir', fallback='').strip()
    if out_dir_raw:
        out_dir = out_dir_raw
    else:
        out_dir = f"results/{incidence}_{int(temperature)}K_{energy}eV_results"

    # [lammps]
    specorder_str = cfg.get('lammps', 'specorder', fallback='N,O,C')
    specorder = [s.strip() for s in specorder_str.split(',') if s.strip()]
    timestep_fs = cfg.getfloat('lammps', 'timestep_fs', fallback=1.0)
    dump_stride = cfg.getint('lammps', 'dump_stride', fallback=1)
    vel_units   = cfg.get('lammps', 'vel_units', fallback='A_per_ps')

    # [morse]
    def _get_float_or_none(section, key):
        val = cfg.get(section, key, fallback='').strip()
        return float(val) if val else None

    De = _get_float_or_none('morse', 'De')
    a  = _get_float_or_none('morse', 'a')
    re = _get_float_or_none('morse', 're')

    return {
        'incidence': incidence,
        'temperature': temperature,
        'energy': energy,
        'file_format': file_format,
        'molecule': molecule,
        'detector_A': detector_A,
        'diatom_ids': diatom_ids,
        'n_min_frames': n_min_frames,
        'base_dir': base_dir,
        'out_dir': out_dir,
        'specorder': specorder,
        'timestep_fs': timestep_fs,
        'dump_stride': dump_stride,
        'vel_units': vel_units,
        'De': De,
        'a': a,
        're': re,
    }

###########################################
# Main analysis
###########################################

def main(config_path):
    cfg = parse_config(config_path)

    incidence    = cfg['incidence']
    temperature  = cfg['temperature']
    energy       = cfg['energy']
    file_format  = cfg['file_format']
    molecule     = cfg['molecule']
    DETECTOR_A   = cfg['detector_A']
    DIATOM_IDS   = cfg['diatom_ids']
    N_MIN_FRAMES = cfg['n_min_frames']

    base_dir = cfg['base_dir']
    OUT_DIR  = cfg['out_dir']

    # LAMMPS parameters
    global SPECORDER
    SPECORDER   = cfg['specorder']
    TIMESTEP_FS = cfg['timestep_fs']
    DUMP_STRIDE = cfg['dump_stride']
    VEL_UNITS   = cfg['vel_units']
    DT_FS       = TIMESTEP_FS * DUMP_STRIDE

    # Morse parameters
    De = cfg['De']
    a  = cfg['a']
    re = cfg['re']

    # Define element composition from molecule string
    if len(molecule) == 2 and not molecule[1].isdigit():  # e.g. 'NO'
        elemA, elemB = molecule[0], molecule[1]
    elif len(molecule) == 2 and molecule[1].isdigit():    # e.g. 'O2'
        elemA, elemB = molecule[0], molecule[0]
    else:
        raise ValueError(f"Unsupported diatomic format: {molecule}")
    print(f"Molecule: {elemA}-{elemB}")

    mA_default = atomic_masses[chemical_symbols.index(elemA)]
    mB_default = atomic_masses[chemical_symbols.index(elemB)]
    print(f"Masses: {elemA}={mA_default:.4f} amu, {elemB}={mB_default:.4f} amu")

    # Collect input files
    INPUT_FILES = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.endswith(('.xyz', '.lammpstrj', '.xml', 'OUTCAR', '.dump'))
    ]
    print(f"Found {len(INPUT_FILES)} input files. format={file_format}")

    os.makedirs(OUT_DIR, exist_ok=True)

    summary_rows = []

    for INPUT_FILE in INPUT_FILES:
        print(f"\nProcessing: {os.path.basename(INPUT_FILE)}")
        kind   = file_kind(INPUT_FILE)
        frames = load_frames(INPUT_FILE)
        print()

        if frames is None:
            print("Skip: no frames loaded.")
            continue

        if len(frames) < 2:
            print("Skip: need ≥2 frames.")
            continue

        # --- Identify NO atoms
        if DIATOM_IDS is not None:
            iA, iB = DIATOM_IDS
        else:
            ids = pick_NO_indices(frames[0], id_pair=(1, 2), type_pair=(1, 2), elem_pair=(elemA, elemB))
            if ids is None:
                raise RuntimeError("Diatomic atoms not found")
            iA, iB = ids

        sym = frames[0].get_chemical_symbols()
        print(f"Using diatomic atoms {iA},{iB} ({sym[iA]}-{sym[iB]})")

        # --- Masses (amu)
        mA_amu = mA_default
        mB_amu = mB_default

        # --- Velocities (Å/fs)
        vel_list, need_fd = [], []
        for k, at in enumerate(frames):
            v = get_velocities(at, vel_units=VEL_UNITS)
            bad = (v is None) or (not np.isfinite(v).all()) or np.allclose(v, 0.0)
            vel_list.append(None if bad else v)
            need_fd.append(bad)
        if any(need_fd):
            print("Computing finite-difference velocities for missing frames.")
            fd = finite_diff_vel(frames, DT_FS)
            for k, bad in enumerate(need_fd):
                if bad:
                    vel_list[k] = fd[k]

        # --- Helper: probe_z = lower NO atom or COM
        def probe_z(at, use_min_atom=True):
            zA, zB = at.positions[iA, 2], at.positions[iB, 2]
            if use_min_atom:
                return float(min(zA, zB))
            masses = np.array([mA_amu, mB_amu], float)
            return float(com(masses, at.positions[[iA, iB]])[2])

        # --- Reference z for first frame (for reporting)
        zref0 = slab_reference_z(frames[0], top_tol=0.10, probe_z=probe_z(frames[0]))

        # --- Heights of probe over the slab (per frame)
        heights = []
        for at in frames:
            z_probe = probe_z(at, use_min_atom=True)
            zref_k  = slab_reference_z(at, top_tol=0.10, probe_z=z_probe)
            heights.append(max(0.0, z_probe - zref_k))
        heights  = np.array(heights, float)
        turn_idx = int(np.argmin(heights))

        cell = frames[0].get_cell()
        pbc  = frames[0].get_pbc()

        # --- Frame-wise properties
        def props_at(k):
            at = frames[k]
            pos = at.get_positions()
            Apos, Bpos = pos[iA], pos[iB]
            Avel, Bvel = vel_list[k][iA], vel_list[k][iB]
            z_probe = probe_z(at, use_min_atom=True)
            zref_k  = slab_reference_z(at, top_tol=0.10, probe_z=z_probe)
            return diatomic_properties(
                Apos, Bpos, Avel, Bvel, mA_amu, mB_amu, zref_k,
                detector_A=DETECTOR_A, cell=cell, pbc=pbc,
                De=De, a=a, re=re
            )

        props = [props_at(k) for k in range(len(frames))]
        pf = props[-1]

        # --- Scatter criterion
        far_enough    = pf["d_mol_L3"] >= DETECTOR_A
        after_min_frames = len(frames) > N_MIN_FRAMES
        outcome = 1 if (far_enough and after_min_frames) else 0  # 1=scatter, 0=stick

        # --- Output headers
        header = [
            "state", "t_idx", "d_NO(Å)", "COM_z-ztop(Å)", "NO_theta(°)", "NO_phi(°)",
            "Vcm_x(Å/fs)", "Vcm_y(Å/fs)", "Vcm_z(Å/fs)",
            "Vcm_theta(°)", "Vcm_phi(°)",
            "Ecm(eV)", "E_pot(eV)", "Evib_cin(eV)", "Evib_tot(eV)",
            "Erot(eV)", "j_rot", "output"
        ]

        # --- Row builder
        def row(label, idx, p):
            out_flag = 1 if (p["d_mol_L3"] >= DETECTOR_A and idx > N_MIN_FRAMES) else 0
            return [
                label, idx,
                round(p["d_NO1"], 4), round(p["d_mol_L3"], 3),
                round(p["NO1_theta"], 2), round(p["NO1_phi"], 2),
                round(p["mol_mCenter_vel"][0], 5),
                round(p["mol_mCenter_vel"][1], 5),
                round(p["mol_mCenter_vel"][2], 5),
                round(p["mol_vel_theta"], 2), round(p["mol_vel_phi"], 2),
                round(p["mol_Ekin"], 4), round(p["mol_Epot"], 4),
                round(p["mol_Evib_cin"], 4), round(p["mol_Evib_tot"], 4),
                round(p["mol_Erot"], 4),
                int(p["mol_jrot"]), out_flag
            ]

        rows_all = [
            row(
                "Start" if k == 0 else "Turn" if k == turn_idx else "End" if k == len(frames) - 1 else "Frame",
                k, props[k]
            )
            for k in range(len(frames))
        ]
        rows_key = [rows_all[0], rows_all[turn_idx], rows_all[-1]]

        # --- Summary
        summary = [
            ["file", os.path.basename(INPUT_FILE)],
            ["molecule", f"{elemA}-{elemB}"],
            ["diatomic", f"{iA},{iB} ({sym[iA]}-{sym[iB]})"],
            ["detector_A", DETECTOR_A],
            ["graphite_zref_A", round(zref0, 3)],
            ["outcome", "scattered(1)" if outcome == 1 else "stuck(0)"],
        ]

        if tabulate:
            print("\n== Summary ==")
            print(tabulate(summary, tablefmt="plain"))
            print("\n== Trajectory metrics (key frames) ==")
            print(tabulate(rows_key, headers=header, tablefmt="plain"))
        else:
            print(summary)
            print([header] + rows_key)

        # --- Write per-frame CSV
        base = os.path.splitext(os.path.basename(INPUT_FILE))[0]
        out_csv = os.path.join(OUT_DIR, f"{base}_metrics.csv")
        if os.path.exists(out_csv):
            os.remove(out_csv)
            print(f"Removed existing file: {out_csv}")

        np.savetxt(
            out_csv,
            np.array(rows_all, dtype=object),
            fmt='%s',
            delimiter=',',
            header=','.join(header),
            comments='',
            encoding='utf-8'
        )

        # --- Batch summary row
        summary_rows.append([
            base, f"{elemA}-{elemB}", f"{sym[iA]}-{sym[iB]}", DETECTOR_A, round(zref0, 3),
            outcome,
            round(pf["d_mol_L3"], 3), round(pf["mol_Ekin"], 4), int(pf["mol_jrot"]),
            round(pf["d_NO1"], 4), round(pf["mol_mCenter_vel"][2], 5),
            round(pf["mol_vel_theta"], 2),
            round(pf["mol_Evib_tot"], 4)
        ])

    # --- Batch summary file
    if summary_rows:
        sum_csv = os.path.join(OUT_DIR, "batch_summary.csv")
        if os.path.exists(sum_csv):
            os.remove(sum_csv)
            print(f"Removed existing file: {sum_csv}")
        hdr = [
            "file", "molecule", "diatomic", "detector_A", "zref_A",
            "outcome(1=scat,0=stick)",
            "final_COMheight_A", "final_Ecm_eV", "final_jrot", "final_d(Å)",
            "final_Vcm_z(Å/fs)", "final_Vcm_θ°", "final_Evib_tot_eV"
        ]
        np.savetxt(
            sum_csv,
            np.array(summary_rows, dtype=object),
            fmt='%s',
            delimiter=',',
            header=','.join(hdr),
            comments='',
            encoding='utf-8'
        )
        print(f"\nWrote {len(summary_rows)} records to {sum_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} CONFIG.ini", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])