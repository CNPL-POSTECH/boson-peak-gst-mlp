#!/usr/bin/env python3
import os
import shutil
from dataclasses import dataclass
from typing import Optional

from ase.io import read
from ase import units
from ase.md.npt import NPT
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md import MDLogger
from ase.io.trajectory import Trajectory


# -----------------------
# Global settings
# -----------------------
MODEL_PATH = "mace_swa.model"
POSCAR_PATH = "POSCAR.vasp"

PRESSURE_GPA = 0.0      # target pressure (GPa)  (isotropic)
TAU_T_FS = 1000.0       # thermostat time constant (fs)
TAU_P_FS = 10000.0      # barostat time constant (fs)

# Compressibility in 1/GPa (used in pfactor formula)
COMPRESSIBILITY_1GPA = 2.0e-2

DEFAULT_DT_FS = 10.0
LOG_INTERVAL = 10
PRINT_INTERVAL = 200

# Trajectory thinning:
SAVE_EVERY = 100         # dt=10 fs -> 100 steps = 1 ps

# Velocity init only at the beginning
INIT_VELOCITIES = True
INIT_TEMPERATURE_K = 1100.0


@dataclass
class Stage:
    folder: str
    ensemble: str   # "nvt" or "npt"
    mode: str       # "hold" or "ramp"
    time_ps: float
    dt_fs: float
    T_hold_K: Optional[float] = None
    T_start_K: Optional[float] = None
    T_end_K: Optional[float] = None

    def validate(self):
        if self.ensemble not in ("nvt", "npt"):
            raise ValueError(f"Invalid ensemble: {self.ensemble}")
        if self.mode not in ("hold", "ramp"):
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.time_ps <= 0:
            raise ValueError("time_ps must be > 0")
        if self.dt_fs <= 0:
            raise ValueError("dt_fs must be > 0")
        if self.mode == "hold":
            if self.T_hold_K is None:
                raise ValueError("hold mode requires T_hold_K")
        else:
            if self.T_start_K is None or self.T_end_K is None:
                raise ValueError("ramp mode requires T_start_K and T_end_K")


def build_mace_calculator(model_path: str):
    from mace.calculators import MACECalculator  # type: ignore
    device = "cuda"
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    return MACECalculator(model_paths=[model_path], device=device, default_dtype="float32")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def steps_from_time(time_ps: float, dt_fs: float) -> int:
    return int(round((time_ps * 1000.0) / dt_fs))


def set_dyn_temperature(dyn, T_K: float):
    """
    Update target temperature during a ramp.
    - NPT: dyn.set_temperature(temperature_K=...)
    - NoseHooverChainNVT: has set_temperature in recent ASE versions; fallback included.
    """
    if hasattr(dyn, "set_temperature"):
        dyn.set_temperature(temperature_K=T_K)  # type: ignore
        return

    # Fallback for older implementations
    kBT = units.kB * T_K
    if hasattr(dyn, "temperature"):
        dyn.temperature = kBT  # type: ignore
    elif hasattr(dyn, "temp"):
        dyn.temp = kBT  # type: ignore
    else:
        raise RuntimeError("Cannot set temperature on this dynamics object.")


def make_npt_dynamics(atoms, dt_fs: float, T_K: float, txt_log_path: str):
    dt = dt_fs * units.fs
    ttime = TAU_T_FS * units.fs
    extstress = PRESSURE_GPA * units.GPa  # scalar for isotropic

    pfactor = ((TAU_P_FS**2) / COMPRESSIBILITY_1GPA) * units.GPa * (units.fs**2)

    dyn = NPT(
        atoms,
        timestep=dt,
        temperature_K=T_K,
        externalstress=extstress,
        ttime=ttime,
        pfactor=pfactor,
        logfile=None,
    )

    logger = MDLogger(dyn, atoms, txt_log_path, header=True, stress=True, peratom=False, mode="w")
    dyn.attach(logger, interval=LOG_INTERVAL)
    return dyn


def make_nvt_dynamics_nose_hoover(atoms, dt_fs: float, T_K: float, txt_log_path: str):
    """
    NVT mixing with Nosé–Hoover chain.
    tdamp: thermostat time scale.
    """
    dt = dt_fs * units.fs
    tdamp = TAU_T_FS * units.fs

    dyn = NoseHooverChainNVT(
        atoms=atoms,
        timestep=dt,
        temperature_K=T_K,
        tdamp=tdamp,
        logfile=None,
    )

    logger = MDLogger(dyn, atoms, txt_log_path, header=True, stress=True, peratom=False, mode="w")
    dyn.attach(logger, interval=LOG_INTERVAL)
    return dyn


def run_stage(atoms, stage: Stage):
    stage.validate()
    ensure_dir(stage.folder)

    # Copy model for provenance
    if os.path.exists(MODEL_PATH):
        dst = os.path.join(stage.folder, os.path.basename(MODEL_PATH))
        if not os.path.exists(dst):
            shutil.copy2(MODEL_PATH, dst)

    # Write starting structure
    atoms.write(os.path.join(stage.folder, "start.vasp"), format="vasp")

    nsteps = steps_from_time(stage.time_ps, stage.dt_fs)
    txt_log_path = os.path.join(stage.folder, "md.log")
    traj_path = os.path.join(stage.folder, "md.traj")

    # Determine starting temperature
    if stage.mode == "hold":
        T0 = float(stage.T_hold_K)
    else:
        T0 = float(stage.T_start_K)

    # Create dynamics
    if stage.ensemble == "nvt":
        dyn = make_nvt_dynamics_nose_hoover(atoms, stage.dt_fs, T0, txt_log_path)
    else:
        dyn = make_npt_dynamics(atoms, stage.dt_fs, T0, txt_log_path)

    # Temperature ramp (works for both NVT and NPT as long as set_temperature is supported)
    if stage.mode == "ramp":
        T_start = float(stage.T_start_K)
        T_end = float(stage.T_end_K)

        def _update_temperature():
            frac = min(max(dyn.nsteps / max(nsteps, 1), 0.0), 1.0)
            T_now = T_start + (T_end - T_start) * frac
            set_dyn_temperature(dyn, T_now)

        dyn.attach(_update_temperature, interval=1)

    # Trajectory thinning
    traj = Trajectory(traj_path, "w", atoms)
    dyn.attach(traj.write, interval=SAVE_EVERY)

    # Status print
    def _print_status():
        temp = atoms.get_temperature()
        vol = atoms.get_volume()
        stress = atoms.get_stress(voigt=True) / units.GPa  # GPa
        pressure = -(stress[0] + stress[1] + stress[2]) / 3.0
        print(f"[{stage.folder}] step {dyn.nsteps:7d}/{nsteps}  "
              f"T {temp:8.2f} K  P {pressure: .4f} GPa  V {vol: .3f} Å^3")

    dyn.attach(_print_status, interval=PRINT_INTERVAL)

    # Run
    dyn.run(nsteps)
    traj.close()

    # Write final structure
    atoms.write(os.path.join(stage.folder, "final.vasp"), format="vasp")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(POSCAR_PATH):
        raise FileNotFoundError(f"Structure not found: {POSCAR_PATH}")

    atoms = read(POSCAR_PATH, format="vasp")
    atoms.calc = build_mace_calculator(MODEL_PATH)

    # Initialize velocities once (assumes POSCAR lattice already set to target density)
    if INIT_VELOCITIES:
        MaxwellBoltzmannDistribution(atoms, temperature_K=INIT_TEMPERATURE_K)
        Stationary(atoms)
        ZeroRotation(atoms)

    stages = [
        Stage("0.mix_nvt",   ensemble="nvt", mode="hold", time_ps=100.0,   dt_fs=DEFAULT_DT_FS, T_hold_K=1100.0),
        Stage("1.melt_npt",  ensemble="npt", mode="hold", time_ps=500.0,  dt_fs=DEFAULT_DT_FS, T_hold_K=1100.0),

        Stage("2.quench",    ensemble="npt", mode="ramp", time_ps=50.0,   dt_fs=DEFAULT_DT_FS, T_start_K=1100.0, T_end_K=300.0),
        Stage("3.heat",      ensemble="npt", mode="ramp", time_ps=50.0,   dt_fs=DEFAULT_DT_FS, T_start_K=300.0,  T_end_K=600.0),

        Stage("4.crys_1ns",  ensemble="npt", mode="hold", time_ps=1000.0, dt_fs=DEFAULT_DT_FS, T_hold_K=600.0),
        Stage("5.crys_4ns",  ensemble="npt", mode="hold", time_ps=4000.0, dt_fs=DEFAULT_DT_FS, T_hold_K=600.0),
        Stage("6.crys_5ns",  ensemble="npt", mode="hold", time_ps=5000.0, dt_fs=DEFAULT_DT_FS, T_hold_K=600.0),
    ]

    for st in stages:
        print(f"\n=== Running stage: {st.folder} ===")
        run_stage(atoms, st)

    atoms.write("final_all.vasp", format="vasp")
    print("\nAll stages completed. Wrote final_all.vasp")


if __name__ == "__main__":
    main()

