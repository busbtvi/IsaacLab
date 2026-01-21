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

def red_cube_area_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    r_min: float = 120.0,       # 절대 밝기 임계
    ratio_min: float = 0.42,    # R/(R+G+B) 임계: 흰 조명 영향 줄임
    margin: float = 30.0,       # R - max(G,B) 최소 여유(밝기 단위)
    normalize: bool = False
):
    camera = env.scene.sensors[sensor_cfg.name]
    rgb = camera.data.output["rgb"][..., :3].to(torch.float32)  # (N,H,W,3) uint8 → float
    r, g, b = rgb.unbind(dim=-1)

    sum_rgb = r + g + b + 1e-6
    r_ratio = r / sum_rgb
    dom = r - torch.maximum(g, b)

    red_mask = (r >= r_min) & (r_ratio >= ratio_min) & (dom >= margin)

    red_count = red_mask.sum(dim=(-1, -2)).to(torch.float32)
    if normalize:
        H, W = rgb.shape[1], rgb.shape[2]
        return red_count / (H * W)
    else:
        return red_count

def object_ee_distance(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    camera: SceneEntityCfg
) -> torch.Tensor:
    cube: RigidObject = env.scene[asset_cfg.name]
    cube_pos = cube.data.root_pos_w

    camera = env.scene.sensors[camera.name]
    camera_pos = camera.data.pos_w

    # 3. L2 Norm (유클리드 거리) 계산
    return torch.norm(cube_pos - camera_pos, dim=-1)