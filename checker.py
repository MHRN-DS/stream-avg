import numpy as np
from env_factory import make_train_env

env = make_train_env(
    env_name="dm_control/cartpole-swingup-v0",
    
    backend="dmcontrol",
    gamma=0.99,
    render=False,
    use_reward_scaling=False,
)

returns = []
for ep in range(5):
    obs, _ = env.reset(seed=ep)
    done = False
    ep_ret = 0.0
    while not done:
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        ep_ret += r
    returns.append(ep_ret)
    print(f"episode {ep}: return = {ep_ret}")

env.close()
print("mean return:", np.mean(returns))