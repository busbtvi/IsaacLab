# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def lift_cube_without_move(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cube: RigidObject = env.scene[asset_cfg.name]
    
    v = cube.data.root_lin_vel_b    # 또는 cube.data.root_lin_vel
    p = cube.data.root_pos_w          # (..., 3) [m]

    eps_v = 1e-3   # 속도 임계값 (m/s)
    eps_z = 1e-6   # 위치 마진

    xy_speed_0 = (v[..., 0].abs() <= eps_v) & (v[..., 1].abs() <= eps_v)   # vx, vy 거의 0
    z_above_3cm  = (p[..., 2] >= 0.03 - eps_z)                               # z ≥ 3 cm

    return xy_speed_0 & z_above_3cm

def arm_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)