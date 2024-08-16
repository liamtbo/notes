import torch
from tensordict.nn import TensorDictModule
from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")
module = torch.nn.LazyLinear(out_features=env.action_spec.shape[-1])
policy = TensorDictModule(
    module,
    in_keys=["observation"],
    out_keys=["action"],
)

rollout = env.rollout(max_steps=10, policy=policy)
# print(rollout)

# ------------------------------
# Specialized wrappers
from torchrl.modules import Actor
# actor looks in dict default in keys (observations) and creates out keys (actions)
# does same as above, just automatically provides obs and action inout keys
policy = Actor(module)
rollout = env.rollout(max_steps=10, policy=policy)
# print(rollout)

# ------------------------------
# Networks
## torchrl also has regular modules that dont need tensordict mods
from torchrl.modules import MLP
module = MLP( # type of common nn
    out_features=env.action_spec.shape[-1],
    num_cells=[32,64],
    activation_class=torch.nn.Tanh,
)
policy = Actor(module)
rollout = env.rollout(max_steps=10, policy=policy)
# print(rollout)

# -------------------------------
# Probabilistic Policies
## ProbabilitcActor - read loc and scale as in_keys, create dis with them, 
                        # and populate our tensordict with samples and logs probs
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Normal
from torchrl.modules import ProbabilisticActor

backbone = MLP(in_features=3, out_features=2) # rets tensor
extractor = NormalParamExtractor() # split MLP ret into 2 chunks - loc (mean) and scale (sdev)
module = torch.nn.Sequential(backbone, extractor)
td_module = TensorDictModule(module, in_keys=["observation"], out_keys=["loc", "scale"])
policy = ProbabilisticActor( # read in loc,scale, create dis's, populate tensordict with samples and log-probs
    td_module, # takes observation, rets loc and scale
    in_keys=["loc", "scale"],
    out_keys=["action"],
    distribution_class=Normal,
    return_log_prob=True
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
with set_exploration_type(ExplorationType.MEAN):
    # takes the mean as action
    rollout = env.rollout(max_steps=10, policy=policy)
with set_exploration_type(ExplorationType.RANDOM):
    # Samples actions according to the dist
    rollout = env.rollout(max_steps=10, policy=policy)

# ---------------------------------
# Exploration
from tensordict.nn import TensorDictSequential
from torchrl.modules import EGreedyModule

policy = Actor(MLP(3, 1, num_cells=[32, 64]))

exploration_module = EGreedyModule(
    # epsilon starts at 0.5, takes 1000 steps to each e_end
    spec=env.action_spec, annealing_num_steps=1000, eps_init=0.5
)
# tensordictsequential == sequential more or less
# policy gives action, then egreedy choses that or random
# this is how egreedy gets access to action space
exploration_policy = TensorDictSequential(policy, exploration_module)
# for continuoous action spaces
with set_exploration_type(ExplorationType.MEAN):
    rollout = env.rollout(max_steps=10, policy=exploration_policy)
with set_exploration_type(ExplorationType.RANDOM):
    rollout = env.rollout(max_steps=10, policy=exploration_policy)

# ------------------------------
# Q-Value actors
env = GymEnv("CartPole-v1")
# print(env.action_spec)
num_actions = 2
value_net = TensorDictModule(
    MLP(out_features=num_actions, num_cells=[32,32]),
    in_keys=["observation"],
    out_keys=["action_value"],
)
from torchrl.modules import QValueModule
policy = TensorDictSequential(
    value_net, # writes action values in our tensordict
    QValueModule(spec=env.action_spec), # reads the action value entry by default, argmax by default
)
rollout = env.rollout(max_steps=3, policy=policy)
# print(rollout)
policy_explore = TensorDictSequential(policy, EGreedyModule(env.action_spec))
with set_exploration_type(ExplorationType.RANDOM):
    rollout_explore = env.rollout(max_steps=3, policy=policy_explore)