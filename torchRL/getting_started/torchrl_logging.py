# loggers
## different kinds: wandb, tensorboard, csv logger
from torchrl.record import CSVLogger
logger = CSVLogger(exp_name="my_exp")
logger.log_scalar("my_scalar", 0.4)

# -------------------------------------
# Recording Videos
from torchrl.envs import GymEnv
env = GymEnv("CartPole-v1", from_pixels=True, pixels_only=False)
# print(env.rollout(max_steps=3))

from torchrl.envs import TransformedEnv
from torchrl.record import VideoRecorder
# logger is what saves the video
recorder = VideoRecorder(logger, tag="my_video")
record_env = TransformedEnv(env, recorder)
rollout = record_env.rollout(max_steps=3)
# uncomment this line to save the video on disk, otherwise its all saved on RAM
# recorder.dump()