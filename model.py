import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

    def save(self, file_path="./models", file_name="snake.pt"):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        path = os.path.join(file_path, file_name)
        torch.save(self.state_dict(), path)

class QTrainer:
    def __init__(self, model, learning_rate, discount_rate):
        self.model = model
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, new_state, game_over):
        state = torch.tensor(np.array(state), dtype=torch.float)
        new_state = torch.tensor(np.array(new_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        game_over = torch.tensor(game_over, dtype=torch.bool)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            new_state = torch.unsqueeze(new_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = torch.unsqueeze(game_over, 0)

        prediction = self.model(state)

        target = prediction.clone()
        for i in range(len(game_over)):
            Q_new = reward[i]
            if not game_over[i]:
                Q_new = reward[i] + self.discount_rate * torch.max(self.model(new_state[i]))

            target[i][torch.argmax(action[i]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()
