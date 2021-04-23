import gym
import pygame
from gym import error, spaces, utils
import numpy as np
import time,random
from game.Player import Runner,Catcher




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
        self.screen_size=np.array(screen_size)
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
        prev_state=self.state
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
            reward=self.calculate_reward(catcher=catcher,prev_state=prev_state)
            if has_collided: #if the agent collides with wall, give 0 reward
                reward=-100
        return self.state,reward,done,info

    def reset(self):
        state=self.get_init_state()
        self.catcher.set_pos(self.init_catcher_pos[0],self.init_catcher_pos[1])
        self.runner.set_pos(self.init_runner_pos[0],self.init_runner_pos[1])
        self.state=state
        self.runner.reset_v()
        self.catcher.reset_v()
        return self.state



    # def calculate_reward(self,catcher=True):
    #     catcher_pos = np.array(self.catcher.get_pos())
    #     runner_pos = np.array(self.runner.get_pos())
    #     distance=float((np.sum((catcher_pos - runner_pos) ** 2)) ** (1 / 2))
    #     if catcher:
    #         return float((self.screen_size[1]*1.5)) - distance
    #     else:
    #         return distance
    def calculate_reward(self,prev_state,catcher=True):
        reward=float(np.sqrt(np.sum(self.screen_size**2)))
        prev_catcher_pos=prev_state[[0,1]]
        prev_runner_pos=prev_state[[2,3]]
        prev_distance=float((np.sum((prev_catcher_pos - prev_runner_pos) ** 2)) ** (1 / 2))
        catcher_pos = np.array(self.catcher.get_pos())
        runner_pos = np.array(self.runner.get_pos())
        distance = float((np.sum((catcher_pos - runner_pos) ** 2)) ** (1 / 2))
        if prev_distance>distance:
            if catcher:
                return reward
            else:
                return -reward
        elif prev_distance<distance:
            if catcher:
                return -reward
            else:
                return reward
        else:
            if catcher:
                return -10
            else:
                return 10

    def get_state(self):
        '''reduced state format. Doesn't account for walls
        [catcher_x,catcher_y,runner_x,runner_y,catcher_vx,catcher_vy,runner_vx,runner_vy]
        '''
        state = [self.catcher.get_pos()[0], self.catcher.get_pos()[1], self.runner.get_pos()[0],
                 self.runner.get_pos()[1], self.catcher.get_v()[0], self.catcher.get_v()[1],
                 self.runner.get_v()[0], self.runner.get_v()[1]]
        return np.array(state)
    def get_init_state(self):
        '''reduced state format. Doesn't account for walls
        [catcher_x,catcher_y,runner_x,runner_y,catcher_vx,catcher_vy,runner_vx,runner_vy]
        '''
        state = [self.init_catcher_pos[0], self.init_catcher_pos[1], self.init_runner_pos[0], self.init_runner_pos[1], 0, 0, 0, 0]
        return np.array(state)

    def render(self,mode='human',len_scale_factor=1):
        # Draw the scene
        self.screen.fill((255, 255, 255))

        # fill screen excess black since there is a minimum width a screen in pygame can be
        if self.screen.get_size() != (self.screen_size[0]*len_scale_factor,self.screen_size[1]*len_scale_factor):
            r = pygame.Rect(self.screen_size[0]*len_scale_factor, 0, self.screen.get_size()[0] - self.screen_size[0]*len_scale_factor,
                            self.screen_size[1]*len_scale_factor)
            pygame.draw.rect(self.screen, (0, 0, 0), r)
        for wall in self.wall_list:
            #wall format : [top left x,top left y,width,height ]
            wall_rect = pygame.Rect(wall[0]*len_scale_factor, wall[1]*len_scale_factor, wall[2]*len_scale_factor, wall[3]*len_scale_factor)
            pygame.draw.rect(self.screen, (0, 0, 0), wall_rect)

        catcher_rect=pygame.Rect(self.catcher.x*len_scale_factor, self.catcher.y*len_scale_factor, self.catcher.size*len_scale_factor, self.catcher.size*len_scale_factor)
        runner_rect=pygame.Rect(self.runner.x*len_scale_factor, self.runner.y*len_scale_factor, self.runner.size*len_scale_factor, self.runner.size*len_scale_factor)
        pygame.draw.rect(self.screen, self.catcher.color, catcher_rect)
        pygame.draw.rect(self.screen, self.runner.color, runner_rect)
        pygame.display.flip()

    def close(self):
        pass
# screen_size=(10,10)
# init_catcher_pos=(8,8)
# init_runner_pos=(9,9)
# max_walls=7
# len_scale_factor=10
#
# #game parameters
# player_size=1
# wall_thickness=1
# walls1=np.array([[0, 2, 3, 1], [2, 3, 1, 2], [4, 4,1, 3], [6, 2, 4, 1], [2,8, 3, 1]])  #constructed on a 10x10 grid
# walls=(walls1/10)*screen_size[0]   #scale the walls according to screen size
# player_a=1
# #set up the player objects
# catcher=Catcher(init_catcher_pos[0],init_catcher_pos[1],[255,0,0],player_size)
# runner=Runner(init_runner_pos[0],init_runner_pos[1],[0,0,255],player_size)
# pygame.init()
# # Set up the display
# pygame.display.set_caption("Tag")
# screen = pygame.display.set_mode((screen_size[0]*len_scale_factor, screen_size[1]*len_scale_factor))
# #init gym environment
# env = TagEnv(catcher,runner,wall_list=walls,screen_size=screen_size,acceleration=player_a,screen=screen,init_catcher_pos=init_catcher_pos,init_runner_pos=init_runner_pos)
#
# observation = env.reset()
# env.render(len_scale_factor=len_scale_factor)
# time.sleep(2)
# print(catcher.rect.colliderect(runner.rect))
# for t in range(10000):
#         time.sleep(0.1)
#         action = env.action_space.sample()
#         print(action)
#         observation, reward, done, info = env.step(action)
#         print (observation, reward, done, info)
#         env.render(len_scale_factor=len_scale_factor)
#
#         if done:
#             print("Finished after {} timesteps".format(t+1))
#             env.reset()
#             time.sleep(2)
#             break
#
