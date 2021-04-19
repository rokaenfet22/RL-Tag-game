import gym
import pygame
import time,random
from gym import error, spaces, utils
import numpy as np
from game.Player import Catcher, Runner




class TagEnv(gym.Env):
    def __init__(self,catcher,runner,wall_list,screen_size,acceleration,screen,init_catcher_pos,init_runner_pos):
        #each wall format : [top left x,top left y,width,height ]
        #wall_list=[wall1,wall2,wall3 ...]
        # There are 4 actions, corresponding to arrow keys UP,RIGHT,DOWN,LEFT
        self.catcher = catcher
        self.init_catcher_pos=init_catcher_pos
        self.init_runner_pos=init_runner_pos
        self.a=acceleration
        self.runner = runner
        self.wall_list = wall_list
        self.screen_size=screen_size
        self.screen=screen
        self.state=self.get_init_state()
        self.action_space = spaces.Discrete(4)
        ''' full state format:
        [wall1,
        wall2,
        wall3,
        .
        .
        .
        wall7,
        [catcher_x,catcher_y,runner_x,runner_y],
        [screen_width,screen_height,catcher_vx,catcher_vy],
        [runner_vx,runner_vy,runner_a,catcher_a]
        
        ]
        '''

        '''
        reduced state format, doesn't account for walls
        [catcher_x,catcher_y,runner_x,runner_y,catcher_vx,catcher_vy,runner_vx,runner_vy]
        '''

        self.observation_space = spaces.Box(-4,self.screen_size[0],[8,1])


    def step(self,action,catcher=True):
        if catcher==True:
            player=self.catcher
        else:
            player=self.runner

        if action==0:#UP
            player.accelerate(0, -self.a)
        elif action==1:#RIGHT
            player.accelerate(self.a, 0)
        elif action==2:#DOWN
            player.accelerate(0, self.a)
        elif action==3:#LEFT
            player.accelerate(-self.a, 0)

        has_collided= player.move(self.screen_size, self.wall_list)
        info={}
        if self.catcher.rect.colliderect(self.runner):
            done=True
        else:
            done=False


        self.state=self.get_state()
        if done:
            if catcher:
                reward=100000
            else:
                reward=-100000
        else:
            reward=self.calculate_reward(catcher=catcher)
            if has_collided: #if the agent collides with wall, give 0 reward
                reward=0
        return self.state,reward,done,info

    def reset(self):
        state=self.get_init_state()
        self.catcher.set_pos(self.init_catcher_pos[0],self.init_catcher_pos[1])
        self.runner.set_pos(self.init_runner_pos[0],self.init_runner_pos[1])
        self.state=state
        self.runner.reset_v()
        self.catcher.reset_v()
        return self.state

    def calculate_reward(self,catcher=True):
        catcher_pos = np.array(self.catcher.get_pos())
        runner_pos = np.array(self.runner.get_pos())
        distance=float((np.sum((catcher_pos - runner_pos) ** 2)) ** (1 / 2))
        if catcher:
            return float((self.screen_size[1]*1.5)) - distance
        else:
            return distance

    def get_state(self):
        '''reduced state format. Doesn't account for walls
        [catcher_x,catcher_y,runner_x,runner_y,catcher_vx,catcher_vy,runner_vx,runner_vy]
        '''
        state = [self.catcher.get_pos()[0], self.catcher.get_pos()[1], self.runner.get_pos()[0],
                 self.runner.get_pos()[1], self.catcher.get_v()[0], self.catcher.get_v()[1],
                 self.runner.get_v()[0], self.runner.get_v()[1]]
        return state
    def get_init_state(self):
        '''reduced state format. Doesn't account for walls
        [catcher_x,catcher_y,runner_x,runner_y,catcher_vx,catcher_vy,runner_vx,runner_vy]
        '''
        state = [self.init_catcher_pos[0], self.init_catcher_pos[1], self.init_runner_pos[0], self.init_runner_pos[1], 0, 0, 0, 0]
        return state

    def render(self,mode='human'):
        # Draw the scene
        self.screen.fill((255, 255, 255))
        # fill screen excess black since there is a minimum width a screen in pygame can be
        if self.screen.get_size() != (self.screen_size[0],self.screen_size[1]):
            r = pygame.Rect(self.screen_size[0], 0, self.screen.get_size()[0] - self.screen_size[0],
                            self.screen_size[1])
            pygame.draw.rect(self.screen, (0, 0, 0), r)
        for wall in self.wall_list:
            #wall format : [top left x,top left y,width,height ]
            wall_rect = pygame.Rect(wall[0], wall[1], wall[2], wall[3])
            pygame.draw.rect(self.screen, (0, 0, 0), wall_rect)
        pygame.draw.rect(self.screen, self.catcher.color, self.catcher.rect)
        pygame.draw.rect(self.screen, self.runner.color, self.runner.rect)
        pygame.display.flip()

    def close(self):
        pass
# screen_size=(10,10)
# init_it_pos=(3,3)
# incatcher_pos=(7,7)
# max_walls=7
#
#
# #game parameters
# player_size=1
# wall_thickness=1
# walls=[[8.333333333333332, 8.333333333333332, 8.333333333333332, 2.666666666666667], [8.333333333333332, 8.333333333333332, 2.666666666666667, 8.333333333333332], [0, 25.0, 16.666666666666664, 2.666666666666667], [38.333333333333336, 18.333333333333332, 10.0, 2.666666666666667], [38.333333333333336, 5.0, 2.666666666666667, 13.333333333333334], [16.666666666666664, 38.333333333333336, 15.0, 2.666666666666667], [31.666666666666664, 30.0, 2.666666666666667, 8.333333333333332]]
# player_a=1
# #set up the player objects
# catcher=IT(init_it_pos[0],init_it_pos[1],[255,0,0],player_size)
# player1=Player(incatcher_pos[0],incatcher_pos[1],[0,0,255],player_size)
# pygame.init()
# # Set up the display
# pygame.display.set_caption("Tag")
# screen = pygame.display.set_mode((screen_size[0], screen_size[1]))
# #init gym environment
# env = TagEnv(catcher,player1,[],screen_size,acceleration=player_a,screen=screen,incatcher_pos=incatcher_pos,init_it_pos=init_it_pos)
#
# observation = env.reset()
# for t in range(10000):
#         env.render()
#         time.sleep(0.01)
#         action = env.action_space.sample()
#         print(action)
#         observation, reward, done, info = env.step(action)
#         print (observation, reward, done, info)
#         if done:
#             print("Finished after {} timesteps".format(t+1))
#             env.reset()
#             env.render()
#             time.sleep(2)
#             break
#
