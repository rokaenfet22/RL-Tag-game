
'''
Classes and functions for DQN
adapted from
https://medium.com/analytics-vidhya/building-a-powerful-dqn-in-tensorflow-2-0-explanation-tutorial-d48ea8f3177a
'''

import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import ( Dense, Input,
                                     Lambda, )
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from gym_envs.gym_env import TagEnv

import json

import os
import random
import numpy as np


class GameWrapper(TagEnv):
    def __init__(self,catcher,runner,wall_list,screen_size,acceleration,screen,init_catcher_pos,init_runner_pos,no_op_steps):
        #no_op_steps is the number of random steps the catcher makes when environment is reset to add some randomness
        super(GameWrapper, self).__init__(catcher,runner,wall_list,screen_size,acceleration,screen,init_catcher_pos=init_catcher_pos,init_runner_pos=init_runner_pos)
        self.no_op_steps = no_op_steps

    def reset(self, evaluation=False):
        """Resets the environment
        Arguments:
            evaluation: Set to True when the agent is being evaluated. Takes a random number of no-op steps if True.
        """

        super(GameWrapper, self).reset()

        # If evaluating, take a random number of no-op steps.
        # This adds an element of randomness, so that the each
        # evaluation is slightly different.
        if evaluation:
            for _ in range(self.no_op_steps):
                self.step(random.randint(0,3),catcher=True)
            #runner random start position
            runner_pos=(random.randint(0,self.screen_size[0]),random.randint(0,self.screen_size[0]))
            while runner_pos==self.catcher.get_pos():#make sure the position of runner isn't the same as pos of catcher
                runner_pos = (random.randint(0, self.screen_size[0]), random.randint(0, self.screen_size[0]))
            self.runner.set_pos(runner_pos[0],runner_pos[1])

def dense_layer(num_units):
  return Dense(
      num_units,
      activation='tanh',
      kernel_initializer=VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

def build_q_network(n_actions, learning_rate, input_shape, screen_size):
    """Builds a  DQN as a Keras model
    Arguments:
        n_actions: Number of possible action the agent can take
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed frame the model sees
    Returns:
        A compiled Keras model
    """
    model_input = Input(shape=(input_shape[0],))
    x = Lambda(lambda layer: layer / screen_size[0])(model_input)  # normalize by screen size

    x = dense_layer(50)(model_input)
    x = dense_layer(50)(x)
    x = dense_layer(50)(x)
    # Split into value and advantage streams
    # val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 1))(x)  # custom splitting layer
    #
    # val_stream = Flatten()(val_stream)
    # val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)
    #
    #
    # adv_stream = Flatten()(adv_stream)
    #adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

    # Combine streams into Q-Values
   # reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean
    #q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])

    q_vals=Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(x)
    # Build model
    model = Model(model_input, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model

class ReplayBuffer:
    """Replay Buffer to store transitions.
    This implementation was heavily inspired by Fabio M. Graetz's replay buffer
    here: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb"""
    def __init__(self, size, input_shape,use_per=False):
        """
        Arguments:
            size: Integer, Number of stored transitions
            input_shape: Shape of the preprocessed frame
            use_per:User Priority Experience Replay instead of simple
        """
        self.use_per=use_per
        self.size = size
        self.input_shape = input_shape
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.input_shape[0]), dtype=np.int8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)

    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        """Saves a transition to the replay buffer
        Arguments:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (8,1) frame of the game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != self.input_shape:
            raise ValueError('Dimension of frame is wrong!')

        if clip_reward:
            reward = np.sign(reward)

        # Write memory
        self.actions[self.current] = action
        self.frames[self.current] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size
        self.priorities[self.current] = max(self.priorities.max(), 1)  # make the most recent experience important


    def get_minibatch(self, batch_size,priority_scale=0.0):
        """Returns a minibatch of self.batch_size = 2transitions
        Arguments:
            batch_size: How many samples to return
            priority_scale: How much to weight priorities. 0 = completely random, 1 = completely based on priority
        Returns:
            A tuple of states, actions, rewards, new_states, and terminals
        """
        # Get sampling probabilities from priority list
        if self.use_per:
            scaled_priorities = self.priorities ** priority_scale
            sample_probabilities = scaled_priorities / sum(scaled_priorities)
        # Get a list of valid indices
        indices = []
        for i in range(batch_size):
            while True:
                index = random.randint(1, self.count - 1)

                # We check that all frames are from same episode with the two following if statements.  If either are True, the index is invalid.
                if index >= self.current and index - 1 <= self.current:
                    continue
                if self.terminal_flags[index - 1:index].any():
                    continue
                break
            indices.append(index)

        # Retrieve states from memory
        states = []
        new_states = []
        for idx in indices:
            states.append(self.frames[idx-1:idx][0])
            new_states.append(self.frames[idx:idx+1][0])

        if self.use_per:
            # Get importance weights from probabilities calculated earlier

            importance = (1 / self.count) * (1 / sample_probabilities[[index for index in indices]])
            importance = importance / importance.max()

            return (np.array(states), self.actions[indices], self.rewards[indices], np.array(new_states),
                    self.terminal_flags[indices]), importance, indices
        else:
            return np.array(states), self.actions[indices], self.rewards[indices], np.array(new_states), self.terminal_flags[indices]

    def set_priorities(self, indices, errors, offset=0.1):
        """Update priorities for PER
        Arguments:
            indices: Indices to update
            errors: For each index, the error between the target Q-vals and the predicted Q-vals
        """
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def save(self, folder_name):
        """Save the replay buffer to a folder"""

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/frames.npy', self.frames)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/terminal_flags.npy', self.terminal_flags)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.actions = np.load(folder_name + '/actions.npy')
        self.frames = np.load(folder_name + '/frames.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.terminal_flags = np.load(folder_name + '/terminal_flags.npy')

class Agent(object):
    """Implements a standard DQN agent"""
    def __init__(self,
                 dqn,
                 target_dqn,
                 replay_buffer,
                 n_actions,
                 input_shape,
                 batch_size,
                 eps_initial=1,
                 use_per=False,
                 eps_final=0.1,
                 eps_final_frame=0.01,
                 eps_evaluation=0.0,
                 eps_annealing_frames=1000000,
                 replay_buffer_start_size=50000,
                 max_frames=25000000):
        """
        Arguments:
            dqn: A DQN (returned by the DQN function) to predict moves
            target_dqn: A DQN (returned by the DQN function) to predict target-q values.  This can be initialized in the same way as the dqn argument
            replay_buffer: A ReplayBuffer object for holding all previous experiences
            n_actions: Number of possible actions for the given environment
            input_shape: Tuple/list describing the shape of the pre-processed environment
            batch_size: Number of samples to draw from the replay memory every updating session
            eps_initial: Initial epsilon value.
            eps_final: The "half-way" epsilon value.  The epsilon value decreases more slowly after this
            eps_final_frame: The final epsilon value
            eps_evaluation: The epsilon value used during evaluation
            eps_annealing_frames: Number of frames during which epsilon will be annealed to eps_final, then eps_final_frame
            replay_buffer_start_size: Size of replay buffer before beginning to learn (after this many frames, epsilon is decreased more slowly)
            max_frames: Number of total frames the agent will be trained for
        """

        self.n_actions = n_actions
        self.input_shape = input_shape

        # Memory information
        self.replay_buffer_start_size = replay_buffer_start_size
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.use_per=use_per
        self.replay_buffer = replay_buffer

        # Epsilon information
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames

        # Slopes and intercepts for exploration decrease
        # (Credit to Fabio M. Graetz for this and calculating epsilon based on frame number)
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope*self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (self.max_frames - self.eps_annealing_frames - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_frames

        # DQN
        self.DQN = dqn
        self.target_dqn = target_dqn

    def calc_epsilon(self, frame_number, evaluation=False):
        """Get the appropriate epsilon value from a given frame number
        Arguments:
            frame_number: Global frame number (used for epsilon)
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            The appropriate epsilon value
        """
        if evaluation:
            return self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            return self.eps_initial
        elif frame_number >= self.replay_buffer_start_size and frame_number < self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope*frame_number + self.intercept
        elif frame_number >= self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope_2*frame_number + self.intercept_2

    def get_action(self, frame_number, state, evaluation=False):
        """Query the DQN for an action given a state
        Arguments:
            frame_number: Global frame number (used for epsilon)
            state: State to give an action for
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            An integer as the predicted move
        """

        # Calculate epsilon based on the frame number
        eps = self.calc_epsilon(frame_number, evaluation)

        # With chance epsilon, take a random action
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)

        # Otherwise, query the DQN for an action
        q_vals = self.DQN.predict(state)[0]
        return q_vals.argmax()

    def get_intermediate_representation(self, state, layer_names=None):
        """
        Get the output of a hidden layer inside the model.  This will be/is used for visualizing model
        Arguments:
            state: The input to the model to get outputs for hidden layers from
            layer_names: Names of the layers to get outputs from.  This can be a list of multiple names, or a single name
        Returns:
            Outputs to the hidden layers specified, in the order they were specified.
        """
        # Prepare list of layers
        if isinstance(layer_names, list) or isinstance(layer_names, tuple):
            layers = [self.DQN.get_layer(name=layer_name).output for layer_name in layer_names]
        else:
            layers = self.DQN.get_layer(name=layer_names).output

        # Model for getting intermediate output
        temp_model = tf.keras.Model(self.DQN.inputs, layers)

        # Put it all together
        return temp_model.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], 1)))

    def update_target_network(self):
        """Update the target Q network"""
        self.target_dqn.set_weights(self.DQN.get_weights())

    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        """Wrapper function for adding an experience to the Agent's replay buffer"""
        self.replay_buffer.add_experience(action, frame, reward, terminal, clip_reward)

    def learn(self, batch_size, gamma, frame_number,priority_scale=1.0):
        """Sample a batch and use it to improve the DQN
        Arguments:
            batch_size: How many samples to draw for an update
            gamma: Reward discount
            frame_number: Global frame number (used for calculating importances)
            priority_scale: How much to weight priorities when sampling the replay buffer. 0 = completely random, 1 = completely based on priority
        Returns:
            The loss between the predicted and target Q as a float
        """

        if self.use_per:
            (states, actions, rewards, new_states,terminal_flags), importance, indices = self.replay_buffer.get_minibatch(batch_size=self.batch_size,
                                                                       priority_scale=priority_scale)
            importance = importance ** (1 - self.calc_epsilon(frame_number))
        else:
            states, actions, rewards, new_states, terminal_flags = self.replay_buffer.get_minibatch(batch_size=self.batch_size)
        # Main DQN estimates best action in new states
        arg_q_max = self.DQN.predict(new_states).argmax(axis=1)

        # Target DQN estimates q-vals for new states
        future_q_vals = self.target_dqn.predict(new_states)
        double_q = future_q_vals[range(batch_size), arg_q_max]

        # Calculate targets (bellman equation)
        target_q = rewards + (gamma*double_q * (1-terminal_flags))

        # Use targets to calculate loss (and use loss to calculate gradients)
        with tf.GradientTape() as tape:
            q_values = self.DQN(states)

            one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions, dtype=np.float32)  # using tf.one_hot causes strange errors
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            error = Q - target_q
            loss = tf.keras.losses.Huber()(target_q, Q)
            if self.use_per:
                # Multiply the loss by importance, so that the gradient is also scaled.
                # The importance scale reduces bias against situataions that are sampled
                # more frequently.
                loss = tf.reduce_mean(loss * importance)

        model_gradients = tape.gradient(loss, self.DQN.trainable_variables)
        self.DQN.optimizer.apply_gradients(zip(model_gradients, self.DQN.trainable_variables))
        if self.use_per:
            self.replay_buffer.set_priorities(indices, error)
        return float(loss.numpy()), error

    def save(self, folder_name, **kwargs):
        """Saves the Agent and all corresponding properties into a folder
        Arguments:
            folder_name: Folder in which to save the Agent
            **kwargs: Agent.save will also save any keyword arguments passed.  This is used for saving the frame_number
        """

        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN
        self.DQN.save(folder_name + '/dqn.h5')
        self.target_dqn.save(folder_name + '/target_dqn.h5')

        # Save replay buffer
        self.replay_buffer.save(folder_name + '/replay-buffer')

        # Save meta
        with open(folder_name + '/meta.json', 'w+') as f:
            f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current}, **kwargs}))  # save replay_buffer information and any other information

    def load(self, folder_name, load_replay_buffer=True):
        """Load a previously saved Agent from a folder
        Arguments:
            folder_name: Folder from which to load the Agent
        Returns:
            All other saved attributes, e.g., frame number
        """

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # Load DQNs
        self.DQN = tf.keras.models.load_model(folder_name + '/dqn.h5')
        self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')
        self.optimizer = self.DQN.optimizer

        # Load replay buffer
        if load_replay_buffer:
            self.replay_buffer.load(folder_name + '/replay-buffer')

        # Load meta
        with open(folder_name + '/meta.json', 'r') as f:
            meta = json.load(f)

        if load_replay_buffer:
            self.replay_buffer.count = meta['buff_count']
            self.replay_buffer.current = meta['buff_curr']

        del meta['buff_count'], meta['buff_curr']  # we don't want to return this information
        return meta