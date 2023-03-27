import random
from collections import deque


class ReplatBuffer(object):
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, transition):
        """
        Saves a transition.
        :param transition: (state, action, reward, next_state, done)
        """
        self.buffer.append(transition)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)