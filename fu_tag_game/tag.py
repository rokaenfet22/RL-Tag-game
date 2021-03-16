def tag(wall_rects):

    import pygame
    from random import randint

    window_x,window_y=300,300
    pygame.init()
    window=pygame.display.set_mode((window_x,window_y))

    #modes of movement, key, random, basic
    seeker_move_mode="random"
    runner_move_mode="key"
    #size of both
    seeker_size=20
    runner_size=10

    class Seeker():
        def __init__(self,x,y,size,color="black",velocity=1):
            self.coords=[x,y,size,size]
            self.prev_pos=[x,y]
            self.color=color
            self.velocity=velocity
        def draw(self):
            draw_rect(self.coords,self.color)

    class Runner():
        def __init__(self,x,y,size,color="black",velocity=1):
            self.coords=[x,y,size,size]
            self.prev_pos=[x,y]
            self.color=color
            self.velocity=velocity
        def draw(self):
            draw_rect(self.coords,self.color)

    def draw_rect(c,color="black",w=1):
        pygame.draw.rect(window,pygame.Color(color),(c[0],c[1],c[2],c[3]),w)

    def rect_collision(rect1,rect2): #rect1 = [x,y,x+dx,y+dy]
        if ((rect1[0] <= rect2[0]+rect2[2]) and (rect1[0] >= rect2[0])) or ((rect1[0]+rect1[2] <= rect2[0]+rect2[2]) and (rect1[0]+rect1[2] >= rect2[0])):
            if ((rect1[1] <= rect2[1]+rect2[3]) and (rect1[1] >= rect2[1])) or ((rect1[1]+rect1[3] <= rect2[1]+rect2[3]) and (rect1[1]+rect1[3] >= rect2[1])):
                return True

    def out_of_bounds(r): #r = Rect
        if r[0]<=0: r[0] = 0
        elif r[0]+r[2]>=window_x: r[0]=window_x-r[2]
        if r[1]<=0: r[1]=1
        elif r[1]+r[3]>=window_y: r[1]=window_y-r[3]
        return r

    def key_move(key,i): #key mode, i=instance
        if key[pygame.K_w]: i.coords[1]-=i.velocity
        elif key[pygame.K_s]: i.coords[1]+=i.velocity
        if key[pygame.K_a]: i.coords[0]-=i.velocity
        elif key[pygame.K_d]: i.coords[0]+=i.velocity

    def random_move(key,i): #random mode, 0=move in negative, 1=move in positive, 2= don't move
        r=randint(0,2)
        if r==2: i.coords[1]-=i.velocity
        elif r: i.coords[1]+=i.velocity
        r=randint(0,2)
        if r==2: i.coords[0]-=i.velocity
        elif r: i.coords[0]+=i.velocity

    def basic_move(key,a,b): #basic mode, a=moved instance, b=based instance
        [ax,ay]=a.coords[:2]
        [bx,by]=b.coords[:2]
        #y
        if ax<bx:a.coords[0]+=a.velocity
        elif ax>bx:a.coords[0]-=a.velocity
        #x
        if ay<by:a.coords[1]+=a.velocity
        elif ay>by:a.coords[1]-=a.velocity

    def moves(key,s,r): #s=seeker, r=runner
        if seeker_move_mode=="key":
            key_move(key,s)
        elif seeker_move_mode=="random":
            random_move(key,s)
        elif seeker_move_mode=="basic":
            basic_move(key,s,r)

        if runner_move_mode=="key":
            key_move(key,r)
        elif runner_move_mode=="random":
            random_move(key,r)
        elif runner_move_mode=="basic":
            basic_move(key,r,s)

    #create instances of seeker and runner, generate in open space
    while True:
        c=[randint(0,window_x-seeker_size),randint(0,window_y-seeker_size)]
        if not any([rect_collision(c+[seeker_size,seeker_size],n) for n in wall_rects]):
            break
    seeker=Seeker(c[0],c[1],seeker_size,"red")
    while True: 
        c = [randint(0,window_x-runner_size),randint(0,window_y-runner_size)]
        if not any([rect_collision(c+[runner_size,runner_size],n) for n in wall_rects+[seeker.coords]]):
            break
    runner=Runner(c[0],c[1],10,"green")

    run=True
    caught=False
    while run:
        pygame.time.delay(3)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                run=False
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE:
                    run=False

        #keep track of where it was previously
        seeker.prev_pos=seeker.coords[:2]
        runner.prev_pos=runner.coords[:2]

        key=pygame.key.get_pressed()
        moves(key,seeker,runner)

        #bound within screen
        seeker.coords=out_of_bounds(seeker.coords)
        runner.coords=out_of_bounds(runner.coords)

        #collided into a wall
        for n in wall_rects:
            if rect_collision(seeker.coords,n):
                t=seeker.coords
                if rect_collision([seeker.prev_pos[0]]+t[1:],n): #if reverting x fails
                    if rect_collision([t[0]]+[seeker.prev_pos[1]]+t[2:],n): #if reverting y fails
                        seeker.coords=seeker.prev_pos+[seeker_size,seeker_size]
                    else:
                        seeker.coords=[t[0],seeker.prev_pos[1],seeker_size,seeker_size]
                else:
                    seeker.coords=[seeker.prev_pos[0],t[1],seeker_size,seeker_size]
                break
        
        for n in wall_rects:
            if rect_collision(runner.coords,n):
                t=runner.coords
                if rect_collision([runner.prev_pos[0]]+t[1:],n): #if reverting x fails
                    if rect_collision([t[0]]+[runner.prev_pos[1]]+t[2:],n): #if reverting y fails
                        runner.coords=runner.prev_pos+[runner_size,runner_size]
                    else:
                        runner.coords=[t[0],runner.prev_pos[1],runner_size,runner_size]
                else:
                    runner.coords=[runner.prev_pos[0],t[1],runner_size,runner_size]

        #check collision between seeker and runner
        if rect_collision(seeker.coords,runner.coords):
            caught=True
            run=False #quit game when caught for now

        window.fill(pygame.Color("white"))

        seeker.draw()
        runner.draw()

        for n in wall_rects:
            draw_rect(n)

        pygame.display.flip()
    pygame.quit()

    if caught: print("GOTCHA BITCH")
    else: print("uncaught sad face :/")