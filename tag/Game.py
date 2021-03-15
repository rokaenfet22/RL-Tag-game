import pygame
class Game:
    def __init__(self,it_player,player,wall_list):
        self.it_player=it_player
        self.player=player
        self.wall_list=wall_list
    def get_state(self):
            pass
    def run(self,clock,screen):
        running = True
        while running:

            clock.tick(60)

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    running = False

            # Move the player if an arrow key is pressed
            key = pygame.key.get_pressed()
            if key[pygame.K_LEFT]:
                self.player.move(-2, 0)

            if key[pygame.K_RIGHT]:
                self.player.move(2, 0)
            if key[pygame.K_UP]:
                self.player.move(0, -2)
            if key[pygame.K_DOWN]:
                self.player.move(0, 2)

            if self.it_player.rect.colliderect(self.player):
                raise SystemExit("You win!")

            # Draw the scene
            screen.fill((0, 0, 0))
           # for wall in walls:
           #     pygame.draw.rect(screen, (255, 255, 255), wall.rect)
            pygame.draw.rect(screen, (255, 0, 0), self.it_player.rect)
            pygame.draw.rect(screen, (255, 200, 0), self.player.rect)
            pygame.display.flip()

class Wall:
    def __init__(self):