# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates different legged robots.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/quadrupeds.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different legged robots.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from isaacsim.core.api.objects import DynamicCuboid

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# print("ISAAC_NUCLEUS_DIR: ", ISAAC_NUCLEUS_DIR)
DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Yahboom/Dofbot/dofbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 음수: 앞쪽으로 기움, range: [-1.571, 1.571]
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.3,
            "joint3": -1.1,
            "joint4": -1.5,  # -: 앞쪽
        },
        pos=(0, 0, 0.0),
    ),
    actuators={
        "front_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint3_act": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint4_act": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
    },
)

def add_cube_to_scene():
    # CUBE_COLORS = [[1,0,0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
    # CUBE_POS = [[0.08, 0.35, 0.0],[0.04, 0.35, 0.0],[0, 0.35, 0.0],[-0.04, 0.35, 0.0]]
    # for i in range(4):
    #     DynamicCuboid(
    #         prim_path=f"/World/cube_{i}",
    #         position=CUBE_POS[i],
    #         size=0.03,  # m 단위
    #         color=np.array(CUBE_COLORS[i]),
    #     )

    DynamicCuboid(
        prim_path=f"/World/cube_{0}",
        position=[1,0,0],
        size=0.03,  # m 단위
        color=np.array([0.08, 0.35, 0.0]),
    )

def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origins = [[0,0,0], [0,0,0.015]]

    sim_utils.create_prim("/World/Dofbot", "Xform")
    dofbot = Articulation(DOFBOT_CONFIG.replace(prim_path="/World/Dofbot/Robot"))

    add_cube_to_scene()
    # sim_utils.create_prim("/World/Objects", "Xform")
    scene_entities = {
        "dofbot": dofbot,
        # "red_cube": red_cube
    }
    # return scene_entities, origins
    return scene_entities, [origins[0]]

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset robots
            for index, robot in enumerate(entities.values()):
                # root state
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                # joint state
                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                # reset the internal state
                robot.reset()
            print("[INFO]: Resetting robots state...")
        # apply default actions to the quadrupedal robots
        for robot in entities.values():
            # generate random joint positions
            joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
            # apply action to the robot
            robot.set_joint_position_target(joint_pos_target)
            # write data to sim
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)

def print_robot_hierarchy(robot_articulation):
    # env_0 기준의 실제 경로를 가져옵니다.
    # "{ENV_REGEX_NS}" 부분이 실제 경로(예: /World/envs/env_0)로 치환된 상태여야 함
    root_path = robot_articulation.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    
    stage = simulation_app.context.get_stage()
    print(f"\n--- Hierarchy for {root_path} ---")
    
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if path.startswith(root_path):
            # 프림의 타입(Camera, Mesh, Joint 등)과 함께 출력
            prim_type = prim.GetTypeName()
            print(f"[{prim_type}] {path}")


def main():
    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()

    print("-------------------------------------------------------")
    dofbot = scene_entities["dofbot"]
    sim.step()
    print(dofbot.cfg.prim_path)
    print(dofbot.num_joints)
    print(dofbot.joint_names)
    print(dofbot.body_names)
    print(dofbot.data.joint_pos)
    print(dofbot.data.joint_vel)
    print("-------------------------------------------------------")
    # 사용 예시
    print_robot_hierarchy(dofbot)
    print("-------------------------------------------------------")
    cam = scene["camera"]
    rgb = cam.data.output["rgb"]             # (N, H, W, 4) 또는 (N, H, W, 3)
    print(rgb.shape)
    print("-------------------------------------------------------")
    torch.Size([1, 480, 640, 3])

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator``
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
