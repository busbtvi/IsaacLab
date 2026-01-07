"""
LIVESTREAM=2 PUBLIC_IP=10.126.36.101 ./isaaclab.sh -p scripts/dofbot/main.py --enable_cameras --num_envs 4
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="This script demonstrates different legged robots.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from scripts.dofbot.dofbot_enf_cfg import DofbotRLEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
import torch

def main():
    """Main function."""
    # create environment configuration
    env_cfg = DofbotRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 100 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            # print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            print("[Env 0]: Rew: ", rew[:args_cli.num_envs].detach().cpu().numpy())
            # update counter
            count += 1

    # close the environment
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()