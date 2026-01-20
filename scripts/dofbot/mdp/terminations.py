# mdp.py
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject

def cube_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    bounds: dict[str, tuple[float, float]],
    use_env_frame: bool = True,
) -> torch.Tensor:
    """
    bounds 예: {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.30)}
    반환: [num_envs] Bool (벗어나면 True)
    """
    cube: RigidObject = env.scene[asset_cfg.name]
    p = cube.data.root_pos_w  # (N,3), 월드 좌표

    if use_env_frame and hasattr(env.scene, "env_origins"):
        # 각 env의 원점 기준으로 평가하고 싶으면 원점 빼기
        p = p - env.scene.env_origins

    x_ok = (p[:, 0] >= bounds["x"][0]) & (p[:, 0] <= bounds["x"][1]) if "x" in bounds else torch.ones_like(p[:,0], dtype=torch.bool)
    y_ok = (p[:, 1] >= bounds["y"][0]) & (p[:, 1] <= bounds["y"][1]) if "y" in bounds else torch.ones_like(p[:,0], dtype=torch.bool)
    z_ok = (p[:, 2] >= bounds["z"][0]) & (p[:, 2] <= bounds["z"][1]) if "z" in bounds else torch.ones_like(p[:,0], dtype=torch.bool)

    inside = x_ok & y_ok & z_ok
    return ~inside  # 바깥이면 True → 종료

def cube_moved_from_reset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    dist_threshold: float = 0.01,   # 1 cm 등, 노이즈 여유 포함
    use_xy_only: bool = True,       # 바닥 위니까 XY만 비교 (권장)
) -> torch.Tensor:
    """
    리셋 시점 위치(랜덤 자리)에서 일정 거리 이상 이동했으면 종료(True) 반환.
    반환 shape: (num_envs,) bool
    """
    cube: RigidObject = env.scene[asset_cfg.name]
    cur_pos = cube.data.root_pos_w  # (N,3), 월드 좌표

    # 리셋 이벤트에서 기록한 기준 위치가 없으면, 지금 위치를 기준으로 잡고 종료하지 않음
    if not hasattr(cube, "_reset_pos"):
        cube._reset_pos = cur_pos.detach().clone()
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    ref_pos = cube._reset_pos  # (N,3)
    if use_xy_only:
        dist = torch.linalg.norm(cur_pos[:, :2] - ref_pos[:, :2], dim=-1)
    else:
        dist = torch.linalg.norm(cur_pos - ref_pos, dim=-1)

    return dist > dist_threshold
