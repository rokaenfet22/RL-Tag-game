from models.DQN_tf.dqn_tools import GameWrapper,build_q_network,ReplayBuffer,Agent
import numpy as np
import pygame
import tensorflow as tf
from game.Player import Runner,Catcher
import time
#No walls, no momentum. Random runner and catcher start positions 5x5. Navigation

#EVALUATION
# hyper
EVAL_LENGTH = 900  # Number of frames to evaluate for

MEM_SIZE = 1000000  # The maximum size of the replay buffer
MAX_EP_LEN=30
eps_annealing_frames = 100000
use_per=False
INPUT_SHAPE = (24,)  # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
BATCH_SIZE = 4  # Number of samples the agent learns from at once
LEARNING_RATE = 0.001
screen_size = (5, 5)
render_scale_factor=100   #scale everything up when rendering on screen for better visibilty
MIN_REPLAY_BUFFER_SIZE = 400  # The minimum size the replay buffer must be before we start to update the agent
layer_structure = (50, 5, 3)
# ENV details
init_catcher_pos = (1, 1)
init_runner_pos = (4, 4)
max_walls = 7 #need in order for the full state to have consistent shape
player_size = 1
wall_thickness = 1
walls = [[8.333333333333332, 8.333333333333332, 8.333333333333332, 2.666666666666667],
         [8.333333333333332, 8.333333333333332, 2.666666666666667, 8.333333333333332],
         [0, 25.0, 16.666666666666664, 2.666666666666667],
         [38.333333333333336, 18.333333333333332, 10.0, 2.666666666666667],
         [38.333333333333336, 5.0, 2.666666666666667, 13.333333333333334],
         [16.666666666666664, 38.333333333333336, 15.0, 2.666666666666667],
         [31.666666666666664, 30.0, 2.666666666666667, 8.333333333333332]]
player_a = 1
pygame.init()
# Set up the display
pygame.display.set_caption("Tag")
screen = pygame.display.set_mode((screen_size[0]*render_scale_factor, screen_size[1]*render_scale_factor))

# set up the players
catcher = Catcher(init_catcher_pos[0], init_catcher_pos[1], [255, 0, 0], player_size)
runner = Runner(init_runner_pos[0], init_runner_pos[1], [0, 0, 255], player_size)
# init tf_agents environment
game_wrapper = GameWrapper(catcher, runner, wall_list=[], screen_size=screen_size, acceleration=player_a, screen=screen,
                            init_catcher_pos=init_catcher_pos, init_runner_pos=init_runner_pos)

# Build main and target networks
MAIN_DQN = build_q_network(layer_structure=layer_structure,n_actions=game_wrapper.action_space.n, learning_rate=LEARNING_RATE,screen_size=screen_size, input_shape=INPUT_SHAPE)

TARGET_DQN = build_q_network(layer_structure=layer_structure,n_actions=game_wrapper.action_space.n,learning_rate=LEARNING_RATE,screen_size=screen_size ,input_shape=INPUT_SHAPE)

replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE)
agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, game_wrapper.action_space.n, input_shape=INPUT_SHAPE,max_frames=1000000,
              batch_size=BATCH_SIZE,use_per=use_per,eps_annealing_frames=eps_annealing_frames,replay_buffer_start_size=MIN_REPLAY_BUFFER_SIZE)
model_path='saved_models/tag2/navigation/save-00025059'
print('Loading model...')
agent.load(model_path)
print('Loaded')

terminal = True
eval_rewards = []
wins=0

for game in range(10):
    game_wrapper.reset(rand=True)
    game_wrapper.render(len_scale_factor=render_scale_factor)
    time.sleep(0.1)
    episode_reward_sum = 0
    if not terminal:
        eval_rewards.append(0)

    for frame in range(MAX_EP_LEN):
        game_wrapper.render(len_scale_factor=render_scale_factor)
        state=np.array(game_wrapper.get_state(catcher=True))
        state=np.reshape(state,(1,20))
        action=agent.get_action(0, state, evaluation=True)
        print(action)
        # Step action
        new_state, reward, terminal, info = game_wrapper.step(action)
        print(f'reward:{reward}')
        game_wrapper.render(len_scale_factor=render_scale_factor)
        episode_reward_sum += reward
        time.sleep(0.1)
        # On game-over
        if terminal: #if the catcher caught the runner
            wins+=1
            print(f'Game over, reward: {episode_reward_sum}')
            eval_rewards.append(episode_reward_sum)
            break

print(f'wins:{wins}; losses:{10-wins}')
print('Average reward:', np.mean(eval_rewards))