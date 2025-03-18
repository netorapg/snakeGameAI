import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# Hiperparâmetros
MAX_MEMORY = 100_000
BATCH_SIZE = 2000  
LR = 0.0005  

# Verifica se há GPU disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0  # Inicialmente, exploração alta
        self.gamma = 0.95  
        self.memory = deque(maxlen=MAX_MEMORY)
        
        self.model = Linear_QNet(11, 256, 3).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        # Estado do jogo - agora com 11 variáveis
        state = [
            # Perigos ao redor
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Perigo à direita
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Perigo à esquerda
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Direção atual
            dir_l, dir_r, dir_u, dir_d,

            # Posição relativa da comida (x e y)
            (game.food.x - game.snake[0].x) / game.w,  
            (game.food.y - game.snake[0].y) / game.h,

            # Distância Euclidiana até a comida (normalizada)
            np.linalg.norm([game.food.x - game.snake[0].x, game.food.y - game.snake[0].y]) / (game.w + game.h),

            # Adiciona uma variável extra, como a distância até a borda mais próxima
            min(game.snake[0].x, game.w - game.snake[0].x, game.snake[0].y, game.h - game.snake[0].y) / (game.w + game.h)
    ]

        return np.array(state, dtype=np.float32) 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        self.epsilon = max(0.05, 0.2 * (0.98 ** self.n_games))  # Decay exponencial
        final_move = [0, 0, 0]

        if random.random() < self.epsilon:  # Exploração correta
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)  # Enviando para GPU/CPU
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        
        final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        # Obtém o estado atual
        state_old = agent.get_state(game)

        # Decide ação
        final_move = agent.get_action(state_old)

        # Executa a ação
        reward, done, score = game.play_step(final_move)

        # Obtém o novo estado
        state_new = agent.get_state(game)

        # Treina a memória de curto prazo
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Armazena na memória de experiência
        agent.remember(state_old, final_move, reward, state_new, done)

        # Se o jogo acabou, treina memória de longo prazo
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Salva checkpoints periodicamente
            if score > record or agent.n_games % 50 == 0:
                record = max(record, score)
                agent.model.save(f"model_checkpoint_{agent.n_games}.pth")

            print('Jogo', agent.n_games, 'Pontuação', score, 'Recorde', record)

            # Plota evolução da IA
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
