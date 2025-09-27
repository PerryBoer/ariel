import os, math, time
import numpy as np
import torch, mujoco

# import evoTorch
from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.logging import StdOutLogger
from evotorch.core import SolutionBatch

# import ariel and fitness function
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.renderers import video_renderer
from ariel.simulation.tasks.targeted_locomotion import distance_to_target

# set seeds and config values
SEED = 3 # random seed
torch.manual_seed(SEED); np.random.seed(SEED) 
TASK_NAME   = "Targeted Locomotion" # sets the fitness function
TARGET_XY   = (0.0, -10.0) # target to walk towards
SIM_SECONDS = 10.0           # 10 s episodes
ALPHA       = 0.12           # smoothing toward target (prevents curl-up)
PHASE_MIN, PHASE_MAX = -math.pi, math.pi # phase bounds
AMP_MIN, AMP_MAX     = 0.0, 1.0        # amplitude bounds
FREQ_MIN, FREQ_MAX   = 0.5, 2.0        # frequency bounds
POPSIZE     = 100 # population size
GENERATIONS = 50 # number of generations
SIGMA_INIT  = 0.30 # CMA-ES initial stdev
MAKE_VIDEO  = True # record video of best individual
VIDEO_DIR   = "./__videos__" # output directory for videos
LOG_CSV   = True # log progress to CSV
CSV_DIR   = "./__logs__" # output directory for CSV log
CSV_FILE  = os.path.join(CSV_DIR, "cmaes_cpg_progress.csv") # CSV filename
G_MODEL = None  # global mujoco model, initialized once
G_STEPS_PER_SEC = None # global sim steps/sec
G_NU = None # global number of actuators
G_ACT_LOW = G_ACT_HIGH = None # global actuator ctrlrange
G_CORE_GEOM_ID = None # global core geom id

# set up mujoco enviroment according to the examples given
def _find_core_geom_id(model):
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and "core" in name.lower():
            return gid
    raise RuntimeError("No geom with 'core' found")

# build mujoco env once give the model and info and spawn the gecko
def build_env_once():
    global G_MODEL, G_STEPS_PER_SEC, G_NU, G_ACT_LOW, G_ACT_HIGH, G_CORE_GEOM_ID
    if G_MODEL is not None:
        return
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    world.spawn(gecko().spec, spawn_position=[0.0, 0.0, 0.0])
    model = world.spec.compile()
    _ = mujoco.MjData(model)

    G_MODEL = model
    G_STEPS_PER_SEC = int(round(1.0 / model.opt.timestep))
    G_NU = int(model.nu)
    G_ACT_LOW  = model.actuator_ctrlrange[:, 0].copy()
    G_ACT_HIGH = model.actuator_ctrlrange[:, 1].copy()
    G_CORE_GEOM_ID = _find_core_geom_id(model)

# set the central pattern generator (CPG) controller for the gecko
def make_cpg_controller(params, model, alpha=ALPHA):
    """
    params = [phase_0..phase_{nu-1}, AMP, FREQ] # the parameters to optimize
    y_i(t)  = AMP * sin(2pi FREQ t + phase_i) ∈ [-AMP, AMP] # oscillator output
    target  = center + half_span * y_i # target position for actuator i
    ctrl    = ctrl + alpha * (target - ctrl) # smooth toward target
    """
    
    nu = int(model.nu) # number of actuators
    phases = np.asarray(params[:nu], dtype=np.float64) # per-actuator phases
    AMP    = float(params[nu])  # shared amplitude
    FREQ   = float(params[nu + 1]) # shared frequency

    # safety (bounds are also given to the optimizer)
    phases = np.clip(phases, PHASE_MIN, PHASE_MAX) 
    AMP    = float(np.clip(AMP, AMP_MIN, AMP_MAX))
    FREQ   = float(np.clip(FREQ, FREQ_MIN, FREQ_MAX))

    # init actuator limits low and high
    lo = model.actuator_ctrlrange[:, 0] 
    hi = model.actuator_ctrlrange[:, 1] 

    # compute center and half_span for each actuator, so we can scale the sine wave to the actuator range
    center    = 0.5 * (hi + lo) 
    half_span = 0.5 * (hi - lo)

    # the actual controller callback
    def cb(m, d):
        # current time
        t = d.time

        # generate sine wave for each actuator at time t shifted by the actuator's phase
        y = AMP * np.sin(2.0 * math.pi * FREQ * t + phases)   # [-AMP, AMP]

        # keep y inside limit to avoid saturating the actuators
        y = np.clip(y, -1.0, 1.0)

        # rescale y to the actuator limits and smooth toward target
        target = center + half_span * y

        # clip again to be safe
        np.clip(target, lo, hi, out=target)

        # move current control values toward target, where alpha controls the smoothing
        d.ctrl[:] = d.ctrl + alpha * (target - d.ctrl)

    # return the callback that mujoco can use every timestep
    return cb

# ================= Rollout & fitness =================
def rollout_fitness(params, seconds=SIM_SECONDS):
    # built the environment once
    build_env_once()
    model = G_MODEL
    data  = mujoco.MjData(model)

    # reset the simulation
    mujoco.mj_resetData(model, data)

    # initialize controls to zero
    if data.ctrl is not None: data.ctrl[:] = 0.0

    # create and set the controller callback and let mujoco run the controller at each timestep
    cb = make_cpg_controller(params, model)
    mujoco.set_mjcb_control(cb)

    # keep track of the last position of the gecko
    xy_last = None

    try:
        # step the simulation for the given number of seconds
        horizon = int(round(seconds * G_STEPS_PER_SEC))

        # run the simulation for the given horizon
        for _ in range(horizon):
            mujoco.mj_step(model, data) # step the sim
            xy_last = data.geom_xpos[G_CORE_GEOM_ID, :2].copy() # keep track of the last position
    finally:
        # remove the controller callback
        mujoco.set_mjcb_control(None)

    # compute and return the euclidean distance to the target as fitness
    return float(distance_to_target((float(xy_last[0]), float(xy_last[1])), TARGET_XY))

# video recorder, viewer and saving video
def record_video(params, outdir=VIDEO_DIR, duration=SIM_SECONDS):
    build_env_once()
    os.makedirs(outdir, exist_ok=True)
    model = G_MODEL
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if data.ctrl is not None: data.ctrl[:] = 0.0
    mujoco.set_mjcb_control(make_cpg_controller(params, model))
    recorder = VideoRecorder(output_folder=outdir)
    try:
        video_renderer(model, data, duration=duration, video_recorder=recorder)
    finally:
        mujoco.set_mjcb_control(None)

# =================== EvoTorch Problem =================
class GeckoCPGProblem(Problem):
    def __init__(self):

        # build environment once to get model info
        build_env_once()
        nu = G_NU # number of actuators
        L  = nu + 2  # phases + AMP + FREQ
        
        # define lower and upper bounds for each parameter
        lo = np.concatenate([np.full(nu, PHASE_MIN), [AMP_MIN], [FREQ_MIN]]).astype(np.float64) 
        hi = np.concatenate([np.full(nu, PHASE_MAX), [AMP_MAX], [FREQ_MAX]]).astype(np.float64)

        # initialize the evoTorch Problem
        super().__init__(
            objective_sense="min",
            solution_length=L,
            dtype=torch.float64,
            device="cpu",
            initial_bounds=(torch.from_numpy(lo), torch.from_numpy(hi)),
        )

    
    def evaluate(self, X):
        # Usual path: X is a SolutionBatch
        if isinstance(X, SolutionBatch):
            vals = X.access_values()  # (pop, L) torch tensor
            fits = []
            for row in vals:
                # convert candidate to numpy array and evaluate
                theta = row.detach().cpu().numpy()

                # run a rollout and get the fitness
                f = rollout_fitness(theta)

                # store the fitness
                fits.append(f)

            # Write back to the batch
            fits_t = torch.as_tensor(fits, dtype=vals.dtype, device=vals.device)
            X.set_evals(fits_t)
            return fits_t

        # Fallback: plain tensor (testing)
        elif isinstance(X, torch.Tensor):
            fits = [rollout_fitness(row.detach().cpu().numpy()) for row in X]
            return torch.as_tensor(fits, dtype=X.dtype, device=X.device)

        else:
            raise TypeError(f"Unsupported input to evaluate(): {type(X)}")

# ========================= Main =========================
def main():
    # define the problem
    prob = GeckoCPGProblem()
    nu = G_NU
    # sensible center: phases=0, AMP=0.5, FREQ=1.0
    center = np.concatenate([np.zeros(nu), [0.5], [1.0]]).astype(np.float64)

    # define the CMA-ES solver
    solver = CMAES(
        prob,
        popsize=POPSIZE,
        stdev_init=SIGMA_INIT,
        center_init=torch.from_numpy(center),
    )
    logger = StdOutLogger(solver, interval=1)

    # keep track of best solution
    best_params, best_fit = None, float("inf")

    for gen in range(GENERATIONS):
        # step the solver and get the new population
        solver.step()
        pop = solver.population
        vals = pop.access_values().detach().cpu().numpy()
        
        # get fitness values as numpy array
        fits_t = pop.get_evals() if hasattr(pop, "get_evals") else pop.evals
        fits = fits_t.detach().cpu().numpy()

        # compute mean and best fitness in the population
        mean_eval = float(np.mean(fits))
        pop_best_eval = float(np.min(fits))

        # find the best individual in the population
        i = int(np.argmin(fits))
        if fits[i] < best_fit:
            best_fit = float(fits[i])
            best_params = vals[i].copy()

        # log to CSV
        mean_for_csv = float(solver.status.get("mean_eval", mean_eval))
        best_for_csv = float(solver.status.get("pop_best_eval", pop_best_eval))

        print(f"[gen {gen:02d}] best={best_fit:.4f}  avg={mean_eval:.4f}")

        if LOG_CSV:
            os.makedirs(CSV_DIR, exist_ok=True)
            if not os.path.exists(CSV_FILE):
                with open(CSV_FILE, "w", newline="") as f:
                    f.write("iter,mean_eval,pop_best_eval\n")
            with open(CSV_FILE, "a", newline="") as f:
                f.write(f"{gen},{mean_for_csv},{best_for_csv}\n")

    # fallback if no best found
    if best_params is None:
        i = int(np.argmin(fits))
        best_fit = float(fits[i])
        best_params = vals[i].copy()

    # viewer and video
    WATCH = True
    if WATCH and best_params is not None:
        from mujoco import viewer
        build_env_once()
        model = G_MODEL
        data  = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.set_mjcb_control(make_cpg_controller(best_params, model))
        try:
            viewer.launch(model=model, data=data)   # blocks until closed
        finally:
            mujoco.set_mjcb_control(None)

    if MAKE_VIDEO and best_params is not None:
        record_video(best_params, outdir=VIDEO_DIR, duration=SIM_SECONDS)
        print(f"[video] saved to: {os.path.abspath(VIDEO_DIR)}")

if __name__ == "__main__":
    # total running time:
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total running time: {end_time - start_time:.2f} seconds")
