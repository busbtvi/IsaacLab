import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject

def reset_cube_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-0.3, 0.3),
    y_range: tuple[float, float] = ( 0.1, 0.4),
    z_floor_center: float = 0.015,   # 바닥이 z=0, 큐브 변 3cm → 중심 z=0.015
):
    obj: RigidObject = env.scene[asset_cfg.name]
    N, device = env.num_envs, env.device

    # xy만 균일 샘플
    x = torch.empty(N, device=device).uniform_(*x_range)
    y = torch.empty(N, device=device).uniform_(*y_range)
    z = torch.full((N,), z_floor_center, device=device)

    pos = torch.stack([x, y, z], dim=-1)  # (N,3)
    # 멀티 env면 각 env 원점 더해주기
    if hasattr(env.scene, "env_origins"):
        pos = pos + env.scene.env_origins

    # 정방향 단위 쿼터니언, 속도 0
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(N, 1)
    lin_vel = torch.zeros(N, 3, device=device)
    ang_vel = torch.zeros(N, 3, device=device)

    root = obj.data.default_root_state.clone()  # (N,13)
    root[:, 0:3]   = pos
    root[:, 3:7]   = quat
    root[:, 7:10]  = lin_vel
    root[:, 10:13] = ang_vel

    obj.write_root_pose_to_sim(root[:, :7])
    obj.write_root_velocity_to_sim(root[:, 7:])
