import pygame,os
from Player import Player,IT
from Game import Game

SCREEN_SIZE=(700,700)
#Initialise pygame
os.environ["SDL_VIDEO_CENTERED"] = "1"
pygame.init()

# Set up the display
pygame.display.set_caption("Tag")
screen = pygame.display.set_mode((SCREEN_SIZE[0], SCREEN_SIZE[1]))

clock = pygame.time.Clock()
it_player=IT(200,200,"red")
player1=Player(300,300,"blue")
game=Game(it_player,player1,"a")
game.run(clock,screen)
