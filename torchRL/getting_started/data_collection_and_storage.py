# Data Collectors
## rollout resets env after episode (batch)
## collectors don't reset, meaning multiple batches can be from same trajectory/episode
import torch
torch.manual_seed(0)

from torchrl.collectors import SyncDataCollector
from torchrl.envs import GymEnv
from torchrl.envs.utils import RandomPolicy

env = GymEnv("CartPole-v1")
env.set_seed(0)

policy = RandomPolicy(env.action_spec)
# delivers batches of size 200, may have part of multiple episdoes in each batch
# total frames is how long collected should be, -1 will produce a never ending collector
collector = SyncDataCollector(env, policy, frames_per_batch=200, total_frames=-1)
# each iteration will collect 200 env interactions
# data[collector, fields] records the transitions corresponding trajectory number
for data in collector:
    print(data)
    break

# ----------------------------------------
# Replay Buffer
from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer
buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=1000))
indices = buffer.extend(data) # or use .add for single element
assert len(buffer) == collector.frames_per_batch
sample = buffer.sample(batch_size=30)
# print(sample)