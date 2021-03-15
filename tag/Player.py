import pygame
length=16
class Player(object):

    def __init__(self,x,y,color,direction=0):
        self.x,self.y,self.color,self.dir=x,y,color,direction
        self.rect = pygame.Rect(self.x, self.y, length, length)
        self.is_it=False

    def move(self, dx, dy,walls):

        if dx != 0:
            self.move_single_axis(dx, 0,walls)
        if dy != 0:
            self.move_single_axis(0, dy,walls)

    def move_single_axis(self, dx, dy,walls):

        # Move the rect
        self.rect.x += dx
        self.rect.y += dy

        #If you collide with a wall, move out based on velocity
        for wall in walls:
            if self.rect.colliderect(wall.rect):
                if dx > 0:  # Moving right; Hit the left side of the wall
                    self.rect.right = wall.rect.left
                if dx < 0:  # Moving left; Hit the right side of the wall
                    self.rect.left = wall.rect.right
                if dy > 0:  # Moving down; Hit the top side of the wall
                    self.rect.bottom = wall.rect.top
                if dy < 0:  # Moving up; Hit the bottom side of the wall
                    self.rect.top = wall.rect.bottom
class IT(Player):
    def __init__(self,x,y,color,direction=0):
        super().__init__(x,y,color,direction)
        self.is_it=True