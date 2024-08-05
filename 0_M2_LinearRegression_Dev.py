import pygame
import sys
import os
import random
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Existing constants
WIDTH, HEIGHT = 800, 600
PLAYER_SIZE = 70
FPS = 60
POWERUP_SIZE = (50, 50)
WIN_SCORE = 10
GAME_DURATION = 60  # seconds

# Existing colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Existing function
def load_image(name, size=None):
    fullname = os.path.join("assets", name)
    image = pygame.image.load(fullname)
    if size:
        return pygame.transform.scale(image, size)
    return image.convert_alpha()

# Existing Player class (unchanged)
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

# Existing Powerup class (unchanged)
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

# Existing DataCollector class (unchanged)
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

# Existing ScorePredictor class (unchanged)
class ScorePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        self.mse = None
        self.r2 = None
        self.feature_importance = None

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate evaluation metrics
        y_pred = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)
        
        print(f"Model R² score: {self.r2:.2f}")
        print(f"Mean Squared Error: {self.mse:.2f}")
        
        # Calculate feature importance
        self.feature_importance = self.calculate_feature_importance()
        
        # Create scatter plot
        self.plot_actual_vs_predicted(y_test, y_pred)
        
        # Create feature importance plot
        self.plot_feature_importance()

    def predict(self, X):
        if not self.is_trained:
            return None
        return self.model.predict(X)

    def plot_actual_vs_predicted(self, y_true, y_pred):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel("Actual Scores")
        plt.ylabel("Predicted Scores")
        plt.title("Actual vs Predicted Scores")
        plt.tight_layout()
        plt.savefig("actual_vs_predicted.png")
        plt.close()

    def calculate_feature_importance(self):
        feature_names = ['playtime', 'actions', 'powerups_collected']
        importances = self.model.coef_
        feature_importance = dict(zip(feature_names, importances))
        return feature_importance

    def plot_feature_importance(self):
        plt.figure(figsize=(10, 6))
        features = list(self.feature_importance.keys())
        importances = list(self.feature_importance.values())
        plt.bar(features, importances)
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()

    def interpret_feature_importance(self):
        interpretation = "Feature Importance Interpretation:\n"
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, importance in sorted_features:
            interpretation += f"- {feature}: {importance:.4f}\n"
            if importance > 0:
                interpretation += f"  Positive impact on score. Increasing {feature} tends to increase the score.\n"
            else:
                interpretation += f"  Negative impact on score. Increasing {feature} tends to decrease the score.\n"
        return interpretation

# Modified Game class for Lesson 9
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Casual Mobile Game")
        self.clock = pygame.time.Clock()
        self.data_collector = DataCollector()
        self.score_predictor = ScorePredictor()
        self.data_saved = False
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
        self.data_collector.reset_current_game_data()
        self.data_saved = False
        self.predicted_score = 0
        self.prediction_accuracy = 0

    def run(self):
        self.running = True
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            if not self.game_over:
                self.update(dt)
                self.draw()
            else:
                if not self.data_saved:
                    self.data_collector.save_game_data()
                    self.data_collector.save_to_csv()
                    self.data_saved = True
                self.draw_game_over()

    def handle_events(self):
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                raise SystemExit
            elif event.type == pygame.KEYDOWN:
                action_taken = True
                if event.key == pygame.K_r and self.game_over:
                    self.reset_game()
                elif event.key == pygame.K_UP:
                    self.speed_multiplier = min(self.speed_multiplier * 1.1, 2.0)
                elif event.key == pygame.K_DOWN:
                    self.speed_multiplier = max(self.speed_multiplier / 1.1, 0.5)
                elif event.key == pygame.K_q:
                    self.running = False
                    pygame.quit()
                    raise SystemExit

        self.data_collector.update(self.clock.get_time() / 1000.0, action_taken)

    def update(self, dt):
        self.all_sprites.update()
        self.powerups.update()
        
        if random.random() < 0.02 * self.speed_multiplier:
            self.spawn_powerup()
        
        self.collect_powerups()
        
        elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000
        if self.score >= WIN_SCORE or elapsed_time >= GAME_DURATION:
            self.game_over = True

        self.data_collector.set_score(self.score)

        # Real-time prediction
        if self.score_predictor.is_trained:
            current_data = self.data_collector.current_game_data
            features = [[current_data['playtime'], current_data['actions'], current_data['powerups_collected']]]
            self.predicted_score = self.score_predictor.predict(features)[0]
            
            # Calculate prediction accuracy
            if self.score > 0:
                self.prediction_accuracy = 1 - abs(self.predicted_score - self.score) / self.score
            else:
                self.prediction_accuracy = 1

        # Adjust difficulty based on prediction
        if self.predicted_score > self.score * 1.2:  # If predicted score is 20% higher than current score
            self.speed_multiplier = min(self.speed_multiplier * 1.05, 2.0)  # Increase difficulty
        elif self.predicted_score < self.score * 0.8:  # If predicted score is 20% lower than current score
            self.speed_multiplier = max(self.speed_multiplier / 1.05, 0.5)  # Decrease difficulty

    def draw(self):
        self.screen.blit(self.background, (0, 0))
        self.all_sprites.draw(self.screen)
        self.powerups.draw(self.screen)
        
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        time_left = max(0, GAME_DURATION - (pygame.time.get_ticks() - self.start_time) / 1000)
        time_text = self.font.render(f"Time: {time_left:.1f}", True, WHITE)
        self.screen.blit(time_text, (10, 50))
        
        speed_text = self.font.render(f"Speed: {self.speed_multiplier:.1f}x", True, WHITE)
        self.screen.blit(speed_text, (WIDTH - 150, 10))
        
        if self.score_predictor.is_trained:
            prediction_text = self.font.render(f"Predicted Score: {self.predicted_score:.0f}", True, YELLOW)
            self.screen.blit(prediction_text, (WIDTH - 250, 50))
            
            # Visual indicator for prediction accuracy
            accuracy_color = self.get_accuracy_color(self.prediction_accuracy)
            accuracy_text = self.font.render(f"Prediction Accuracy: {self.prediction_accuracy:.2f}", True, accuracy_color)
            self.screen.blit(accuracy_text, (WIDTH - 250, 90))
        
        pygame.display.flip()

    def draw_game_over(self):
        self.screen.blit(self.background, (0, 0))
        game_over_text = self.font.render("Game Over!", True, RED)
        score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
        restart_text = self.font.render("Press 'R' to restart", True, WHITE)
        
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
            self.data_collector.record_powerup()

    def get_accuracy_color(self, accuracy):
        if accuracy > 0.9:
            return GREEN
        elif accuracy > 0.7:
            return YELLOW
        else:
            return RED

# Modified main function for Lesson 9
def main():
    game = Game()
    
    # Play a few games to collect initial data
    for _ in range(3):
        game.run()
        game.reset_game()
    
    # Train the model with multiple features
    data = game.data_collector.data
    X = np.array([[game['playtime'], game['actions'], game['powerups_collected']] for game in data])
    y = np.array([game['score'] for game in data])
    game.score_predictor.train(X, y)
    
    # Display evaluation metrics
    print(f"Mean Squared Error: {game.score_predictor.mse:.2f}")
    print(f"R-squared Score: {game.score_predictor.r2:.2f}")
    
    # Display feature importance interpretation
    print(game.score_predictor.interpret_feature_importance())
    
    # Continue playing with predictions and enhanced gameplay
    while True:
        game.run()
        if not game.running:
            break
        game.reset_game()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()