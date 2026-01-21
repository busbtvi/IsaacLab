import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def cam_rgb(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg
):
    # https://isaac-sim.github.io/IsaacLab/main/source/tutorials/04_sensors/add_sensors_on_robot.html
    cam = env.scene.sensors[sensor_cfg.name]
    rgb = cam.data.output["rgb"]  # (N, H, W, 3) 또는 (N, H, W, 4)
    # print(rgb.shape)  # torch.Size([1, 480, 640, 3])

    # uint8 [0,255]로 맞추기
    if rgb.dtype != torch.uint8:
        scale = 255.0 if (rgb.max() <= 1.0) else 1.0
        rgb = (rgb * scale).clamp(0, 255).to(torch.uint8)

    # CHW로 변환
    return rgb.permute(0, 3, 1, 2)  # (N, 3, H, W)
