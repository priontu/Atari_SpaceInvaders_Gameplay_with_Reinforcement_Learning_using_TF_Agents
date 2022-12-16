import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os
import PIL
import time

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

import gym
# gym.envs.registry.all()

from tf_agents.environments.wrappers import ActionRepeat

game_name = "SpaceInvaders-v0"

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
   
tf.random.set_seed(42)
np.random.seed(42)

from tf_agents.environments import suite_gym
import ale_py

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

import logging

env = gym.make(game_name)
# print(env)

env.seed(42)
env.reset()

# env.step(1) # Fire
repeating_env = ActionRepeat(env, times=4)
repeating_env.unwrapped

       
limited_repeating_env = suite_gym.load(
    game_name,
    gym_env_wrappers=[partial(TimeLimit, max_episode_steps=100)],
    env_wrappers=[partial(ActionRepeat, times=4)],
)

max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "SpaceInvaders-v0"

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
    print("\nGIF directory: ", os.getcwd() + image_path + '/' + gif_name)
    print("GIF name is changed so as not to coincide with original GIF submission.")

def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)
            # train_checkpointer.save(agent.train_step_counter)
        # print("iteration: ", iteration)

env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessingWithSkipStart, FrameStack4])

env.seed(42)
env.reset()
# for _ in range(4):
#     time_step = env.step(3) # LEFT

# Building the Agent:
tf_env = TFPyEnvironment(env)
preprocessing_layer = keras.layers.Lambda(
                          lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[512]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)


train_step = tf.Variable(0) # globak_step substitute
update_period = 4 # run a training step every 4 collect steps
optimizer = keras.optimizers.RMSprop(learning_rate=2.5e-4, rho=0.95, momentum=0.0,
                                     epsilon=0.00001, centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, # initial ε
    decay_steps=25000// update_period, # <=> 1,000,000 ALE frames
    end_learning_rate=0.01) # final ε

agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000, # <=> 32,000 ALE frames
                 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                 gamma=0.99, # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()

# Building the Replay Buffer:
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=100000) # reduce if OOM error

checkpoint_dir = os.path.join(os.getcwd(), 'lastModelCheckpoint')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step
)

train_checkpointer.initialize_or_restore()
global_step = tf.compat.v1.train.get_global_step()


replay_buffer_observer = replay_buffer.add_batch

# Creating Train Metrics:
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]


# Log Metrics:
logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)

collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period) # collect 4 steps for each training iteration


trajectories, buffer_info = next(iter(replay_buffer.as_dataset(
    sample_batch_size=2,
    num_steps=3,
    single_deterministic_pass=False)))

time_steps, action_steps, next_time_steps = to_transition(trajectories)

dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

start_time = time.time()
train_agent(n_iterations=10)
print("\nTime Taken: ", time.time() - start_time)
# beep_when_finished()

eval_policy = agent.policy
eval_policy_dir = os.path.join(os.getcwd(), 'eval_policy')
policy_dir = os.path.join(os.getcwd(), 'savedPolicy')
tf_policy_saver = policy_saver.PolicySaver(eval_policy)
tf_policy_saver.save(policy_dir)
tf_policy_saver.save(eval_policy_dir)
train_checkpointer.save(train_step)
checkpoint_path = os.path.join(os.getcwd(), "savedModel", "cpkt")
os.makedirs(checkpoint_path, exist_ok=True)
model_checkpoint = tf.train.Checkpoint(model = agent, step = train_step)
saved_path = model_checkpoint.save(file_prefix = checkpoint_path)
print("Model saced in: \n", saved_path)

frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

num_frames = 1000
    
watch_driver = DynamicStepDriver(
    tf_env,
    agent.policy,
    observers=[save_frames, ShowProgress(num_frames)],
    num_steps=num_frames)
final_time_step, final_policy_state = watch_driver.run()

# anim = plot_animation(frames)

create_gif(frames)

print("\n------------Program Execution Complete-------------\n")
