import mujoco
from mujoco import viewer
import numpy as np

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder


def play_movement_sequence(movements, output_folder="./__videos__"):
    """Run a MuJoCo simulation with a given sequence of movements and save it as a video.
    movements: list of movement vectors, dimensions: simulation duration, n_hinges)
    """
    
    duration = len(movements[0])

    #setup world and robot
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    #controller that applies the movement sequence
    step = 0
    def controller(m, d):
        nonlocal step
        if step < duration:
            d.ctrl[:] = movements[step]
        step += 1

    mujoco.set_mjcb_control(controller)

    #record the simulation video
    video_recorder = VideoRecorder(output_folder=output_folder)
    video_renderer(
        model,
        data,
        duration=duration,
        video_recorder=video_recorder,
    )


if __name__ == "__main__":
    duration = 10 #number of steps in simulation
    num_joints = 8  #get this from model.nu
    example_movements = np.random.uniform(-np.pi/2, np.pi/2, size=(duration, num_joints)).tolist() 
    #na een paar stappen gaan de movements out of bounds van de hinges, maar dit is alleen ff om de video werkend te krijgen

    play_movement_sequence(example_movements, output_folder="./__videos__")
