
import base64
import imageio
import io
import matplotlib
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import zipfile
import IPython
import numpy as np
import os
import time
import logging
import PIL
import gym

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
import matplotlib.animation as animation
import matplotlib.pyplot as plt
mpl.rc('animation', html='jshtml')

import tf_agents.environments.wrappers
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments.wrappers import ActionRepeat
from tf_agents.environments import suite_gym
from functools import partial
from gym.wrappers import TimeLimit
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories.trajectory import to_transition
from tf_agents.utils.common import function
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.environments.tf_py_environment import TFPyEnvironment

# Before here is libraries from TF Page

import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

game_name = "SpaceInvaders-v0"

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

logging.getLogger().setLevel(logging.INFO)
    
tf.random.set_seed(42)
np.random.seed(42)

environment_name = "SpaceInvaders-v0"
max_episode_steps = 27000 

class AtariPreprocessingWithSkipStart(AtariPreprocessing):
    def skip_frames(self, num_skip):
        for _ in range(num_skip):
          super().step(0) # NOOP for num_skip steps
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.skip_frames(40)
        return obs
    def step(self, action):
        lives_before_action = self.ale.lives()
        obs, rewards, done, info = super().step(action)
        if self.ale.lives() < lives_before_action and not done:
            self.skip_frames(40)
        return obs, rewards, done, info

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")
            
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim

def create_gif(frames):
    gif_name = "myAgentPlays-2.gif" 
    image_path = os.path.join("images", "rl", gif_name)
    # frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
    frame_images = [PIL.Image.fromarray(frame) for frame in frames]
    frame_images[0].save(image_path, format='GIF',
                         append_images=frame_images[1:],
                         save_all=True,
                         duration=30,
                         loop=0)
    print("GIF saved in the following directory: ")
    print("\nGIF directory: ", os.getcwd() + image_path + '/' + gif_name)
    print("GIF name is changed so as not to coincide with original GIF submission: myAgentPlays.gif")


env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessingWithSkipStart, FrameStack4])

env.seed(42)
env.reset()
for _ in range(4):
    time_step = env.step(2) # 
    
tf_env = TFPyEnvironment(env)
policy_dir = os.path.join(os.getcwd(), 'savedPolicy')
saved_policy = tf.saved_model.load(policy_dir)

frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

num_frames = 1000
    
watch_driver = DynamicStepDriver(
    tf_env,
    saved_policy,
    observers=[save_frames, ShowProgress(num_frames)],
    num_steps=num_frames)

final_time_step, final_policy_state = watch_driver.run()
create_gif(frames)

# anim = plot_animation(frames)
print("\n------------Program Execution Complete-------------\n")