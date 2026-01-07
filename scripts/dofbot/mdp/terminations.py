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
    p = cube.data.root_pos  # (N,3), 월드 좌표

    if use_env_frame and hasattr(env.scene, "env_origins"):
        # 각 env의 원점 기준으로 평가하고 싶으면 원점 빼기
        p = p - env.scene.env_origins

    x_ok = (p[:, 0] >= bounds["x"][0]) & (p[:, 0] <= bounds["x"][1]) if "x" in bounds else torch.ones_like(p[:,0], dtype=torch.bool)
    y_ok = (p[:, 1] >= bounds["y"][0]) & (p[:, 1] <= bounds["y"][1]) if "y" in bounds else torch.ones_like(p[:,0], dtype=torch.bool)
    z_ok = (p[:, 2] >= bounds["z"][0]) & (p[:, 2] <= bounds["z"][1]) if "z" in bounds else torch.ones_like(p[:,0], dtype=torch.bool)

    inside = x_ok & y_ok & z_ok
    return ~inside  # 바깥이면 True → 종료
