from tf_env.env1 import TagEnv
import pygame
import time
import numpy as np
import random
#tf_agents stuff
import tensorflow as tf
from tf_agents.networks.q_network import QNetwork
from tf_agents.networks.sequential import Sequential
from tf_agents.specs import tensor_spec
from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import random_tf_policy
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics

from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common


from game.Player import Player,IT
screen_size=(50,50)
init_it_pos=(16.7,16.7)
init_player_pos=(25,25)
max_walls=7


#testing
player_size=3
wall_thickness=2.67
walls=[[8.333333333333332, 8.333333333333332, 8.333333333333332, 2.666666666666667], [8.333333333333332, 8.333333333333332, 2.666666666666667, 8.333333333333332], [0, 25.0, 16.666666666666664, 2.666666666666667], [38.333333333333336, 18.333333333333332, 10.0, 2.666666666666667], [38.333333333333336, 5.0, 2.666666666666667, 13.333333333333334], [16.666666666666664, 38.333333333333336, 15.0, 2.666666666666667], [31.666666666666664, 30.0, 2.666666666666667, 8.333333333333332]]
player_a=0.2
pygame.init()
# Set up the display
pygame.display.set_caption("Tag")
screen = pygame.display.set_mode((screen_size[0], screen_size[1]))
#set up the players
it_player=IT(init_it_pos[0],init_it_pos[1],[255,0,0],player_size)
player1=Player(init_player_pos[0],init_player_pos[1],[0,0,255],player_size)
#init tf_agents environment
train_env = TagEnv(it_player,player1,walls,screen_size,acceleration=player_a,screen=screen)


#*****testing the environment with random movements***

# time_step = train_env.reset()
# print(time_step)
# cumulative_reward = time_step.reward

# for _ in range(2000):
#   time_step = train_env.step(np.array(random.randint(0,4),dtype=np.int32))
#   train_env.render()
#   if train_env._episode_ended:
#       print("ep ended")
#       time.sleep(2)
#       break
#   else:
#       time.sleep(0.01)
#       print(time_step)
#       cumulative_reward += time_step.reward

# print(time_step)
# cumulative_reward += time_step.reward
# print('Final Reward = ', cumulative_reward)

#hyperparameters
num_iterations=25000
init_collect_steps=200
collect_steps_per_iter=10
replay_buffer_max_size=10000

batch_size=32
learning_rate=0.025
log_interval=5000

num_eval_ep=5
eval_interval=2500

frames_skip=3 #so we don't train at every single frame

class TagQNet(QNetwork):
    def call(self,observation,step_type=None,network_state=(),training=False):
        state=tf.cast(observation,tf.float32)
        #normalise values to be between 0 and +-1 because NNets tend to prefer small numbers
        state=state/screen_size[0]
        return super(TagQNet, self).call(state,step_type=step_type,network_state=network_state,training=training)

#NETWORK
fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
obs_tensor_spec=tensor_spec.from_spec(train_env.observation_spec())
num_actions = 5

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# it's output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = Sequential(dense_layers + [q_values_layer])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)
#AGENT
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    action_tensor_spec,
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
)

agent.initialize()

#POLICIES
eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                tensor_spec.from_spec(train_env.action_spec()))


# time_step = train_env.reset()
# for t in range(10000):
#         train_env.render()
#         time.sleep(0.01)
#         action_step = random_policy.action(time_step)
#         print(action_step.action)
#         time_step = train_env.step(action_step.action)
#         print (time_step)
#         if train_env._episode_ended:
#             print("Finished after {} timesteps".format(t+1))
#             break

#REPLAY BUFFER
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=batch_size,
    max_length=replay_buffer_max_size)
#COLLECTION
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    batch = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), traj)

    print('batch:',batch)
    # Add trajectory to the replay buffer
    buffer.add_batch(batch)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, init_collect_steps)
iter(replay_buffer.as_dataset()).next()

# q_net = TagQNet(
#             train_env.observation_spec(),
#             train_env.action_spec(),
#             fc_layer_params=fc_layer_params)
# optimizer = tf.compat.v1.train.RMSPropOptimizer(
#     learning_rate=learning_rate,
#     decay=0.95,
#     momentum=0.0,
#     epsilon=0.00001,
#     centered=True)
#
# train_step_counter = tf.Variable(0)
#
# observation_spec = tensor_spec.from_spec(train_env.observation_spec())
# time_step_spec = ts.time_step_spec(observation_spec)
#
# action_spec = tensor_spec.from_spec(train_env.action_spec())
# target_update_period=32000  # ALE frames
# update_period=16  # ALE frames
# _update_period = update_period / frames_skip
# _global_step = tf.compat.v1.train.get_or_create_global_step()
#
# agent = dqn_agent.DqnAgent(
#     time_step_spec,
#     action_spec,
#     q_network=q_net,
#     optimizer=optimizer,
#     epsilon_greedy=0.01,
#     n_step_update=1.0,
#     target_update_tau=1.0,
#     target_update_period=(
#         target_update_period / frames_skip / _update_period),
#     td_errors_loss_fn=common.element_wise_huber_loss,
#     gamma=0.99,
#     reward_scale_factor=1.0,
#     gradient_clipping=None,
#     debug_summaries=False,
#     summarize_grads_and_vars=False,
#     train_step_counter=_global_step)
#
#
#
# agent.initialize()