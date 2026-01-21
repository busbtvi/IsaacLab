# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from isaaclab.sensors import CameraCfg

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

# import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp
# import isaaclab.envs.mdp as mdp
import scripts.dofbot.mdp as mdp

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4"]

##
# Scene definition
##
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
    # print("ISAAC_NUCLEUS_DIR: ", ISAAC_NUCLEUS_DIR)
    dofbot: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Yahboom/Dofbot/dofbot.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=16, solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # 음수: 앞쪽으로 기움, range: [-1.571, 1.571]
            # joint_pos={
            #     "joint1": 0.0,
            #     "joint2": 0.3,
            #     "joint3": -1.1,
            #     "joint4": -1.5,  # -: 앞쪽
            # },
            joint_pos={"joint1": 0.0, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0,},
            pos=(0, 0, 0.0),
        ),
        actuators={
            "front_joints": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-2]"],
                effort_limit_sim=50.0,
                velocity_limit_sim=10.0,
                stiffness=50,
                damping=5,
            ),
            "joint3_act": ImplicitActuatorCfg(
                joint_names_expr=["joint3"],
                effort_limit_sim=50.0,
                velocity_limit_sim=10.0,
                stiffness=50,
                damping=5,
            ),
            "joint4_act": ImplicitActuatorCfg(
                joint_names_expr=["joint4"],
                effort_limit_sim=50.0,
                velocity_limit_sim=10.0,
                stiffness=50,
                damping=5,
            ),
        },
    ).replace(prim_path="{ENV_REGEX_NS}/Robot")
    camera = CameraCfg(
        # /World/Dofbot/Robot/link4/Camera
        prim_path="{ENV_REGEX_NS}/Robot/link4/Camera",  # USD 내 실제 카메라 경로와 일치해야 함
        spawn=None, # 이미 USD에 카메라가 있다면 None으로 설정하여 중복 생성 방지
        data_types=["rgb"],
        # height=480,
        # width=640,
        height=60,
        width=80,
    )
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube3cm",                 # 스테이지 경로
        # prim_path="/World/Cube3cm",                 # 스테이지 경로
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),                       # 변 3cm
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.15, 0.015),                         # 바닥(z=0) 위에 놓이게 중심을 0.015m로
        ),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # refer: humanoid_env_cfg.py(multiple scale?)

    #  joint_names: joint1, joint2, joint3, joint4, Wrist_Twist_RevoluteJoint, Finger_Left_01_RevoluteJoint, Finger_Right_01_RevoluteJoint, Finger_Left_02_RevoluteJoint, Finger_Right_02_RevoluteJoint, Finger_Left_03_RevoluteJoint, Finger_Right_03_RevoluteJoint
    joint_efforts = mdp.JointPositionActionCfg(
        asset_name="dofbot", scale=3.0,
        joint_names=ARM_JOINTS, 
        clip={
            "joint1": (-1.5, 1.5),
            "joint2": (-1.5, 1.5),
            "joint3": (-1.5, 1.5),
            "joint4": (-1.5, 1.5),
        }
    )
    # gripper_effort = mdp.JointEffortActionCfg(asset_name="dofbot", joint_names=["Finger_Right_01_RevoluteJoint"], scale=1.0)
    

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, 
            params={"asset_cfg": SceneEntityCfg("dofbot")}
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, 
            params={"asset_cfg": SceneEntityCfg("dofbot")}
        )
        rgb = ObsTerm(
            func=mdp.cam_rgb,
            params={"sensor_cfg": SceneEntityCfg("camera")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.history_length = 1
            self.concatenate_terms = False   # for MultiInputPolicy(Dict 관측)

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_joint_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("dofbot", joint_names=["joint1", "joint2", "joint3", "joint4", "Finger_Right_01_RevoluteJoint"]),
            "position_range": (0, 0),
            "velocity_range": (0, 0),
        },
    )
    reset_cube_position = EventTerm(
        func=mdp.reset_cube_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "x_range" : (0.0, 0.0),
            "y_range" : (0.3, 0.3),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    joint_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("dofbot", joint_names=ARM_JOINTS)},
    )

    red_cube_area = RewTerm(
        func=mdp.red_cube_area_reward,   # rgb에서 빨강 비율
        weight=5,
        params={"sensor_cfg": SceneEntityCfg("camera"), "r_min": 140.0},
    )

    # 2. 추가: 큐브에 다가가도록 유도 (중요!)
    # 로봇 손(ee)과 큐브 사이의 거리가 가까워지면 점수를 줌
    approach_cube = RewTerm(
        func=mdp.object_ee_distance, # Isaac Lab 기본 제공 함수 또는 직접 구현
        weight=-0.1, # 거리가 멀면 마이너스
        params={"asset_cfg": SceneEntityCfg("cube"), "camera": SceneEntityCfg("camera")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    cube_moved = DoneTerm(
        func=mdp.cube_moved_from_reset,
        params={
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


##
# Environment configuration
##
@configclass
class DofbotRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene: DofBotSceneCfg = DofBotSceneCfg(num_envs=4, env_spacing=1.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 8
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 100
        self.sim.render_interval = self.decimation
