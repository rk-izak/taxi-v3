from dataclasses import dataclass
import random, torch
import torch.nn as nn, torch.optim as optim
from .replay_buffer import ReplayBuffer

class DNN(nn.Module):
    """Extremely simple DNN uusing ReLU + sequential layers"""
    def __init__(self, state_dim, action_dim, hidden_sizes,
                 use_embedding, embedding_dim):
        super().__init__()
        self.use_embedding = use_embedding

        if use_embedding:
            self.embed = nn.Embedding(state_dim, embedding_dim)
            in_dim = embedding_dim
        else:
            in_dim = state_dim

        layers = []
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
    
        layers.append(nn.Linear(in_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_embedding:
            x = self.embed(x.long()).squeeze(1)
        return self.model(x)

# dataclass for more pythonic storage
@dataclass
class AgentConfig:
    state_dim: int
    action_dim: int
    hidden_sizes: list[int]
    use_embedding: bool
    embedding_dim: int
    gamma: float
    lr: float
    batch_size: int
    buffer_size: int
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    device: str

class DQNAgent:
    """DQN Agent"""
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.device = cfg.device

        self.policy_net = DNN(cfg.state_dim, cfg.action_dim,
                                     cfg.hidden_sizes, cfg.use_embedding,
                                     cfg.embedding_dim).to(cfg.device)
    
        self.target_net = DNN(cfg.state_dim, cfg.action_dim,
                                     cfg.hidden_sizes, cfg.use_embedding,
                                     cfg.embedding_dim).to(cfg.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.buffer_size)
        self.epsilon = cfg.epsilon

    # epsilon-based choosing
    def choose_action(self, state: int, exploit: bool = False) -> int:
    
        if (not exploit) and (random.random() < self.epsilon):
            return random.randrange(self.cfg.action_dim)
    
        with torch.no_grad():
            s = torch.tensor([[state]], device=self.device)
            return self.policy_net(s).argmax(dim=1).item()

    # store transition
    def push(self, *trans):
        self.memory.push(*trans)

    # single update logic
    def update(self):
        if len(self.memory) < self.cfg.batch_size:
            return None
        
        # naming convetion in line with PyTorch guides
        s, a, r, s2, d = self.memory.sample(self.cfg.batch_size)
        s = torch.tensor(s, dtype=torch.long, device=self.device).unsqueeze(1)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(s2, dtype=torch.long, device=self.device).unsqueeze(1)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.policy_net(s).gather(1, a)

        with torch.no_grad():
            max_q_s2 = self.target_net(s2).max(dim=1, keepdim=True)[0]
            target = r + (1 - d) * self.cfg.gamma * max_q_s2
        loss = nn.functional.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # random utils
    def decay_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save({"model": self.policy_net.state_dict()}, path)

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device)["model"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = 0.0   # force greedy!
