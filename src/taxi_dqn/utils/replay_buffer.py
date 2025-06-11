from collections import deque
import random

class ReplayBuffer:
    """Basic Memory buffer for DQN Agent."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, *transition):
        self.buffer.append(tuple(transition))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return [list(x) for x in zip(*batch)]

