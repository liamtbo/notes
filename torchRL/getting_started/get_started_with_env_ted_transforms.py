from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")
reset = env.reset()
# print(reset)

# samples random action
reset_with_action = env.rand_action(reset)
# print(reset_with_action)
# print(reset_with_action["action"])

# take action
stepped_data = env.step(reset_with_action)
# print(stepped_data)

# filters out unneeded data and retrusn data structure in MDP format
from torchrl.envs import step_mdp
data = step_mdp(stepped_data)
# print(data)

# --------------------------------------
# env rollouts
## computing and recording each step
rollout = env.rollout(max_steps=10)
# print(step_mdp(rollout))
# print(rollout)

# indexing transitions
transition = rollout[3]
# print(transition)

# --------------------------------------
# Transfroming an environment
from torchrl.envs import StepCounter, TransformedEnv
## after 10 steps, the trajectory gets truncated. Full 100 steps never happens
transformed_env = TransformedEnv(env, StepCounter(max_steps=10))
rollout = transformed_env.rollout(max_steps=100)
# print(rollout)
# "next"allows us to see last truncated entry which is True
# print(rollout["next", "truncated"])


