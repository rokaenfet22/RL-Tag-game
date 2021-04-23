import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import math

"""
NOTE: this is navigation agent for chaser -> target.
ACTION: up,down,left,right = 0,1,2,3
STATE: [cy,cx,ry,rx,direc] = chaser x, chaser y, runner x, runner y, direction looking from chaser to runner
RESULT: Agent reliably taking optimal route to target/runner pos after around 1~3k episodes of training. (haven't tested when, but it does work :))

+ = open space
X = wall
S = start
G = Goal
"""

#walls = [[0,3],[1,1],[2,1],[3,3],[2,3],[2,2]]
walls = []

class Nav(chainer.Chain): #class for the DQN
    def __init__(self,obs_size,n_actions,n_hidden_channels):
        super(Nav,self).__init__()
        with self.init_scope(): #defining the layers
            self.l1=L.Linear(obs_size,n_hidden_channels)
            self.l2=L.Linear(n_hidden_channels,n_hidden_channels)
            self.l3=L.Linear(n_hidden_channels,n_hidden_channels)
            self.l4=L.Linear(n_hidden_channels,n_actions)
    def __call__(self,x,test=False): #activation function = sigmoid
        h1=F.sigmoid(self.l1(x))
        h2=F.sigmoid(self.l2(h1))
        h3=F.sigmoid(self.l3(h2))
        y=chainerrl.action_value.DiscreteActionValue(self.l4(h3)) #agent chooses distinct action from finite action set
        return y

def get_dist(s): #returns euclidiean distance between (s[0],s[1]) and (s[2],s[3])
    return math.sqrt((s[0]-s[2])**2+(s[1]-s[3])**2)

def get_direc(s): #get state representation of direction from (s[0],s[1]) to (s[2],s[3])
    [ay,ax]=s[:2] #[y,x] form
    [by,bx]=s[2:4]
    #coords r in pygame style, top left = [0,0] going down adds positively on the y value i.e. [+1,0]
    #representing cardinal directions in clockwise order, starting with N
    s=0
    if ax==bx:
        if ay>by:s=1 #N
        else:s=5 #S
    elif ay==by:
        if ax<bx:s=3 #E
        else:s=7 #W
    elif ax>bx:
        if ay>by:s=8 #NW
        else:s=6 #SW
    elif ax<bx:
        if ay>by:s=2 #NE
        else:s=4 #SE
    return s

def find_index(arr,t): #arr.index(t), but with numpy arrays. Returns False if t not in arr
    for i,n in enumerate(arr):
        if np.array_equal(t,n):
            return i
    return False

def disp_wall(s): #ASCII representation of the current grid. See top of code for terminology of letters
    grid=[["+" for a in range(5)] for _ in range(5)]
    for y,x in walls:
        grid[y][x]="X"
    grid[s[0]][s[0]]="S"
    grid[s[1]][s[0]]="G"
    for n in grid:
        print("".join(n))

def random_action(): #returns random action, used by "explorer"
    return np.random.choice([0,1,2,3])

#this is where optimizers, explorers, replay buffers, network structure, and agent type is defined
def setup(gamma,obs_size,n_actions,n_hidden_channels,start_epsilon=1,end_epsilon=0.1,num_episodes=1): #the skeletal structure of my agent.
    func=Nav(obs_size,n_actions,n_hidden_channels) #model's structure defined here
    optimizer=chainer.optimizers.Adam(eps=1e-8) #optimizer chosen
    optimizer.setup(func)
    #explorer setup
    explorer=chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=start_epsilon,end_epsilon=end_epsilon,decay_steps=num_episodes,random_action_func=random_action)
    replay_buffer=chainerrl.replay_buffer.PrioritizedReplayBuffer(capacity=10**6) #experience replay buffer setup
    phi=lambda x: x.astype(np.float32,copy=False) #must be float32 for chainer
    #defining network type and setting up agent. Required parameter differs for most networks (e.g. DDPG, AC3, DQN)
    agent=chainerrl.agents.DQN(func,optimizer,replay_buffer,gamma,explorer,replay_start_size=300,update_interval=1,target_update_interval=50,phi=phi)
    return agent

#given current state s, and action a from network, returns next state and reward for that action a
def step(s,a):
    r=0
    #record chaser position
    py,px=s[0],s[1]
    if a==0 and s[0]!=0:#up
        s[0]-=1
    elif a==1 and s[0]!=4:#down
        s[0]+=1
    elif a==2 and s[1]!=0:#left
        s[1]-=1
    elif a==3 and s[1]!=4:#right
        s[1]+=1
    else: #if no movement observed, penalize
        r=-100
    #if collided into wall, penalize
    if [s[0],s[1]] in walls: s[0],s[1]=py,px; r=-100
    return s,r #return new state, and reward

def r_both_pos(state): #generate random pos for both entities, which is not overlapping with themselves nor the walls
    state[0]=np.random.choice([0,1,2,3,4])
    state[1]=np.random.choice([0,1,2,3,4])
    state[2]=np.random.choice([0,1,2,3,4])
    state[3]=np.random.choice([0,1,2,3,4])
    while [state[0],state[1]]==[state[2],state[3]] or [n for n in state[:2]] in walls or [n for n in state[2:4]] in walls:
        state[0]=np.random.choice([0,1,2,3,4])
        state[1]=np.random.choice([0,1,2,3,4])
        state[2]=np.random.choice([0,1,2,3,4])
        state[3]=np.random.choice([0,1,2,3,4])
    return state

def train(num_episodes,name,save=True,load=True,interv_save=True,disp_interval=1,save_interval=100): #actual train algo
    #agent setup below
    agent=setup(0.3,5,4,50,1,0.1,num_episodes) #gamma,state_dim,action_num,hidden_num,start_epsilon=1,end_epsilon=0.01,num_episodes=1
    #loading prexisting agent
    if load:
        agent.load("nav_agent"+name)

    longest_dist=get_dist(np.array([0,4,4,0])) #longest possible dist
    max_step=30 #time/step limit
    timeout_states=[] #for recording starting states which resulted in a failed catch (timeout)
    mem=10 #number of past steps to store

    for episode in range(num_episodes): #episode loop
        state=np.array([4,0,0,4,0]) #cx,cy,rx,ry,direc
        
        #Choose a state which resulted in failed catch, or random pos
        if len(timeout_states)>0:
            if np.random.uniform(0,1)>0.5:
                state=r_both_pos(state)
            else:
                state=np.copy(timeout_states.pop(0))
        else:
            state=r_both_pos(state)

        state[4]=get_direc(state[:4]) #set initial direction
        prev_states=[] #used with "mem", for recording previous steps
        start_state=[n for n in state] #record starting state

        #set vars
        r=0 #cur reward
        total_r=0 #total reward of that episode
        step_taken=0 #step/time counter

        #distance vars
        dist=get_dist(state) #current distance
        og_dist=get_dist(state) #distance at initial state

        #caught or not
        caught=True

        while [state[0],state[1]]!=[state[2],state[3]] and caught: #step loop
            action=agent.act_and_train(state,r) #get action from agent, and update network (train)

            #recording "mem" number of past states from [newest,...,oldest]
            if len(prev_states)<=mem: 
                prev_states.insert(0,state[:4])
            else:
                prev_states.pop()
                prev_states.insert(0,state[:4])

            state,r=step(state,action) #update state from given action

            #distance rewarding/penalizing and update
            d=get_dist(state)
            if d<dist: r+=longest_dist #if current distance is less than previous, reward
            elif d>dist: r-=longest_dist #otherwise penalize
            dist=d

            #penalizing for repeated pos in previous "mem" steps
            i=find_index(prev_states,state[:4])
            if i: r-=longest_dist*((mem-i+1)*(5/mem)) #The more recent the repeated position, the more you are penalized for

            total_r+=r #update current episode's total reward
            step_taken+=1 #increment step counter

            #update direc
            state[4]=get_direc(state[:4])

            #time limit, detect failed catch
            if step_taken>max_step:
                r-=100 #penalize
                print("***********************************************")
                timeout_states.append(np.copy(start_state)) #record starting state of this episode as a "failed catch"
                caught=False
        if caught:
            r+=10**5 #larger reward for successful catch
        total_r+=r
        agent.stop_episode_and_train(state,r,caught) #final train of episode
        #displaying data
        if episode%disp_interval==0:
            print(f"episode: {episode}, chaser reward: {total_r}")
            print(f"steps taken: {step_taken}, start state: {start_state}, length of timeouts: {len(timeout_states)}")
            print()
        #saving agent models in intervals
        if episode%save_interval==0 and interv_save:
            agent.save("nav_agent"+name)
    #saving agent models after all episode ran
    if save:
        agent.save("nav_agent"+name)
    #display basic values of network
    print(agent.get_statistics())

train(5000,"Test",False,False,False)