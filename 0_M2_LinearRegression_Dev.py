import pygame
import sys
import os

# Constants (Existing code)
WIDTH, HEIGHT = 800, 600
PLAYER_SIZE = 70
FPS = 60

# Colors (Existing code)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# New code: Asset loading
def load_image(name, size=None):
    fullname = os.path.join("assets", name)
    image = pygame.image.load(fullname)
    if size:
        return pygame.transform.scale(image, size)
    return image.convert_alpha()

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        # Modified code: Use image instead of colored surface
        self.image = load_image("M2_Player.png", (PLAYER_SIZE, PLAYER_SIZE))
        self.rect = self.image.get_rect(center=(WIDTH // 2, HEIGHT - PLAYER_SIZE))
        self.speed = 5

    def move(self, direction):
        if direction == 'left':
            self.rect.x -= self.speed
        elif direction == 'right':
            self.rect.x += self.speed
        
        # Keep player within the screen bounds
        self.rect.x = max(0, min(WIDTH - PLAYER_SIZE, self.rect.x))

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Casual Mobile Game")
        self.clock = pygame.time.Clock()
        
        self.player = Player()
        self.all_sprites = pygame.sprite.Group(self.player)
        
        # New code: Load background
        self.background = load_image("M2_BG_Space.png", (WIDTH, HEIGHT))
        
        self.running = True

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.player.move('left')
        if keys[pygame.K_RIGHT]:
            self.player.move('right')
    def update(self):
        self.all_sprites.update()

    def draw(self):
        # Modified code: Draw background
        self.screen.blit(self.background, (0, 0))
        self.all_sprites.draw(self.screen)
        pygame.display.flip()

def main():
    game = Game()
    game.run()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()