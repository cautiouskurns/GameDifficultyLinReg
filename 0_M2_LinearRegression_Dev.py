import pygame
import sys
import os
import random
import csv  # New import for Lesson 4

# Existing constants
WIDTH, HEIGHT = 800, 600
PLAYER_SIZE = 70
FPS = 60
POWERUP_SIZE = (50, 50)
WIN_SCORE = 10
GAME_DURATION = 10  # seconds

# Existing colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# Existing function
def load_image(name, size=None):
    fullname = os.path.join("assets", name)
    image = pygame.image.load(fullname)
    if size:
        return pygame.transform.scale(image, size)
    return image.convert_alpha()

# Existing Player class
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = load_image("M2_Player.png", (PLAYER_SIZE, PLAYER_SIZE))
        self.rect = self.image.get_rect(center=(WIDTH // 2, HEIGHT - PLAYER_SIZE))
        self.speed = 0
        self.acceleration = 0.5
        self.max_speed = 7

    def move(self, direction):
        if direction == 'left':
            self.speed = max(self.speed - self.acceleration, -self.max_speed)
        elif direction == 'right':
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        else:
            self.speed *= 0.9  # Deceleration when no key is pressed

        self.rect.x += self.speed
        self.rect.x = max(0, min(WIDTH - PLAYER_SIZE, self.rect.x))

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.move('left')
        elif keys[pygame.K_RIGHT]:
            self.move('right')
        else:
            self.move(None)

# Existing Powerup class
class Powerup(pygame.sprite.Sprite):
    def __init__(self, speed_multiplier):
        super().__init__()
        self.image = load_image("M2_PowerUp.png", POWERUP_SIZE)
        self.rect = self.image.get_rect(center=(random.randint(0, WIDTH), 0))
        self.speed = 3 * speed_multiplier

    def update(self):
        self.rect.y += self.speed
        if self.rect.top > HEIGHT:
            self.kill()

# New class for Lesson 4
class DataCollector:
    def __init__(self):
        self.data = []
        self.current_game_data = {
            'playtime': 0,
            'actions': 0,
            'powerups_collected': 0,
            'score': 0
        }

    def update(self, dt, action_taken=False):
        self.current_game_data['playtime'] += dt
        if action_taken:
            self.current_game_data['actions'] += 1

    def record_powerup(self):
        self.current_game_data['powerups_collected'] += 1

    def set_score(self, score):
        self.current_game_data['score'] = score

    def save_game_data(self):
        self.data.append(self.current_game_data.copy())
        self.reset_current_game_data()

    def reset_current_game_data(self):
        self.current_game_data = {
            'playtime': 0,
            'actions': 0,
            'powerups_collected': 0,
            'score': 0
        }

    def save_to_csv(self, filename='game_data.csv'):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['playtime', 'actions', 'powerups_collected', 'score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for game_data in self.data:
                writer.writerow(game_data)
        print(f"Data saved to {filename}")

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Casual Mobile Game")
        self.clock = pygame.time.Clock()
        self.data_collector = DataCollector()  # New for Lesson 4
        self.reset_game()

    def reset_game(self):
        self.player = Player()
        self.all_sprites = pygame.sprite.Group(self.player)
        self.powerups = pygame.sprite.Group()
        self.background = load_image("M2_BG_Space.png", (WIDTH, HEIGHT))
        self.score = 0
        self.font = pygame.font.Font(None, 36)
        self.speed_multiplier = 1.0
        self.start_time = pygame.time.get_ticks()
        self.game_over = False
        self.data_saved = False  # New flag to prevent multiple saves
        self.data_collector.reset_current_game_data()  # New for Lesson 4

    def run(self):
        self.running = True
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0  # Modified for Lesson 4
            self.handle_events()
            if not self.game_over:
                self.update(dt)  # Modified for Lesson 4
                self.draw()
            else:
                self.draw_game_over()
                if not self.data_saved:  # Check if data has already been saved
                    self.data_collector.save_game_data()  # New for Lesson 4
                    self.data_collector.save_to_csv()  # New for Lesson 4
                    self.data_saved = True  # Set flag to True after saving

    def handle_events(self):
        action_taken = False  # New for Lesson 4
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                action_taken = True  # New for Lesson 4
                if event.key == pygame.K_r and self.game_over:
                    self.reset_game()
                elif event.key == pygame.K_UP:
                    self.speed_multiplier = min(self.speed_multiplier * 1.1, 2.0)
                elif event.key == pygame.K_DOWN:
                    self.speed_multiplier = max(self.speed_multiplier / 1.1, 0.5)
        
        # New for Lesson 4
        self.data_collector.update(self.clock.get_time() / 1000.0, action_taken)

    def update(self, dt):  # Modified for Lesson 4
        self.all_sprites.update()
        self.powerups.update()
        
        if random.random() < 0.02 * self.speed_multiplier:
            self.spawn_powerup()
        
        self.collect_powerups()
        
        elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000
        if self.score >= WIN_SCORE or elapsed_time >= GAME_DURATION:
            self.game_over = True

        self.data_collector.set_score(self.score)  # New for Lesson 4

    def draw(self):
        self.screen.blit(self.background, (0, 0))
        self.all_sprites.draw(self.screen)
        self.powerups.draw(self.screen)
        
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        time_left = max(0, GAME_DURATION - (pygame.time.get_ticks() - self.start_time) / 1000)
        time_text = self.font.render(f"Time: {time_left:.1f}", True, (255, 255, 255))
        self.screen.blit(time_text, (10, 50))
        
        speed_text = self.font.render(f"Speed: {self.speed_multiplier:.1f}x", True, (255, 255, 255))
        self.screen.blit(speed_text, (WIDTH - 150, 10))
        
        pygame.display.flip()

    def draw_game_over(self):
        self.screen.blit(self.background, (0, 0))
        game_over_text = self.font.render("Game Over!", True, (255, 0, 0))
        score_text = self.font.render(f"Final Score: {self.score}", True, (255, 255, 255))
        restart_text = self.font.render("Press 'R' to restart", True, (255, 255, 255))
        
        self.screen.blit(game_over_text, (WIDTH // 2 - 70, HEIGHT // 2 - 50))
        self.screen.blit(score_text, (WIDTH // 2 - 70, HEIGHT // 2))
        self.screen.blit(restart_text, (WIDTH // 2 - 100, HEIGHT // 2 + 50))
        
        pygame.display.flip()

    def spawn_powerup(self):
        powerup = Powerup(self.speed_multiplier)
        self.powerups.add(powerup)
        self.all_sprites.add(powerup)

    def collect_powerups(self):
        collected = pygame.sprite.spritecollide(self.player, self.powerups, True)
        for powerup in collected:
            self.score += 10
            self.data_collector.record_powerup()  # New for Lesson 4

# Existing main function
def main():
    game = Game()
    game.run()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()