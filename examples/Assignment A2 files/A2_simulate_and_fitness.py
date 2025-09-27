import mujoco
from mujoco import viewer
import numpy as np
import matplotlib.pyplot as plt

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder

HISTORY = [] #keep track of gecko position
STEP_COUNTER = {"step": 0}  #global step counter

def simulate_movement(model, data, to_track, movements) -> None:
    """simulate movements from a given movement sequence"""
    global HISTORY, STEP_COUNTER

    step = STEP_COUNTER["step"]
    if step < len(movements):
        data.ctrl[:] += movements[step] * 0.05 #YOU CAN CHANGE THIS DELTA PARAMETER
        data.ctrl[:] = np.clip(data.ctrl[:], -np.pi/2, np.pi/2)
        HISTORY.append(to_track[0].xpos.copy())
        STEP_COUNTER["step"] += 1  
    else:
        mujoco.set_mjcb_control(None) #stop controlling when movement seq ends

def evaluate_fitness(movements, target):
    """Evaluate the fitness of a given movement sequence as the euclidean distance to a target"""
    global HISTORY, STEP_COUNTER

    #reset global variables
    mujoco.set_mjcb_control(None)
    HISTORY = []
    STEP_COUNTER = {"step": 0}

    #setup world and robot
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    #simulate gecko using movement array
    mujoco.set_mjcb_control(lambda m, d: simulate_movement(m, d, to_track, movements))
    for _ in range(len(movements)):
        mujoco.mj_step(model, data)

    #calculate euclidean distance to target
    target = np.array(target)
    final_pos = HISTORY[-1][:2]
    fitness = np.linalg.norm(final_pos - target)
    
    return fitness

def save_video(movements, output_folder="./__videos__", duration=10):
    """Save the simulation video of a given movement sequence"""
    global HISTORY, STEP_COUNTER

    #reset global variables
    mujoco.set_mjcb_control(None)
    HISTORY = []
    STEP_COUNTER = {"step": 0}

    #setup world and robot
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    #simulate gecko using movement array
    mujoco.set_mjcb_control(lambda m, d: simulate_movement(m, d, to_track, movements))

    #record the simulation video
    video_recorder = VideoRecorder(output_folder=output_folder)
    video_renderer(
        model,
        data,
        duration=duration, #length of video in sec
        video_recorder=video_recorder,
    )