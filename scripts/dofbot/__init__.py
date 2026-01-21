# refer /isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/__init__.py

import gymnasium as gym
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Dofbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 아마도, env config를 어디서 찾을 수 있는지, 경로를 묘사하는 듯?
        # {__name__}/dofbot_env_cfg.py 파일의 DofbotRLEnvCfg class
        "env_cfg_entry_point": f"{__name__}.dofbot_env_cfg:DofbotRLEnvCfg",

        # 특정 agent의 설정이 어디에 있는지
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
# 실행 시, `--task Isaac-Cartpole-v0`를 하기 때문에, 해당 gym을 활용
# 실행 시, `--agent`가 default로 sb3_cfg_entry_point이기 때문에,
# 해당 경로의 agents(agents.__name__) 하위의 sb3_ppo_cfg.yaml을 읽음