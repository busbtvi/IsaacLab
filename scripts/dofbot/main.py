"""
LIVESTREAM=2 PUBLIC_IP=10.126.36.101 ./isaaclab.sh -p scripts/dofbot/main.py
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="This script demonstrates different legged robots.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils

# from isaaclab.assets import Articulation

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class DofBotSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    # dofbot: ArticulationCfg = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # print("ISAAC_NUCLEUS_DIR: ", ISAAC_NUCLEUS_DIR)
    dofbot: ArticulationCfg = ArticulationCfg(
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
    ).replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/Objects/Cube3cm",                 # 스테이지 경로
        spawn=sim_utils.CuboidCfg(                          # ← DynamicCuboid에 대응되는 생성부
            size=(0.03, 0.03, 0.03),                       # 변 3cm
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.015),                         # 바닥(z=0) 위에 놓이게 중심을 0.015m로
        ),
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    dofbot = scene["dofbot"]
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            count = 0

            # reset robots
            # root_state = dofbot.data.default_root_state.clone()
            # dofbot.write_root_pose_to_sim(root_state[:, :7])
            # dofbot.write_root_velocity_to_sim(root_state[:, 7:])
            # joint state
            joint_pos, joint_vel = dofbot.data.default_joint_pos.clone(), dofbot.data.default_joint_vel.clone()
            dofbot.write_joint_state_to_sim(joint_pos, joint_vel)

            # dofbot.reset()  # scene이 일괄 관리
            scene.reset()
            print("[INFO]: Resetting robots state...")
        
        # apply default actions to the quadrupedal robots
        joint_pos_target = dofbot.data.default_joint_pos + torch.randn_like(dofbot.data.joint_pos) * 0.1
        dofbot.set_joint_position_target(joint_pos_target)
        dofbot.write_data_to_sim()

        # update sim-time
        count += 1

        # perform step
        sim.step()
        # update buffers
        dofbot.update(sim_dt)


def main():
    # Initialize the simulation context
    # sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:1"))
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # design scene

    # scene_cfg = DofBotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene_cfg = DofBotSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # scene_entities, scene_origins = design_scene()
    # scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()

    # print("-------------------------------------------------------")
    # dofbot = scene_entities["dofbot"]
    # sim.step()
    # print(dofbot.num_joints)
    # print(dofbot.joint_names)
    # print(dofbot.body_names)
    # print(dofbot.data.joint_pos)
    # print(dofbot.data.joint_vel)
    # print("-------------------------------------------------------")

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator``
    # run_simulator(sim, scene_entities, scene_origins)
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
