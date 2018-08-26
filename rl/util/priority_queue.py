import numpy as np


# https://github.com/jaara/AI-blog/blob/master/SumTree.py
class PriorityQueue:

    def __init__(self, capacity):
        super().__init__()

        self.size = 0
        self.capacity = capacity
        self.priorities = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.data_index = 0

    def add(self, priority, data):
        index = self.data_index + self.capacity - 1

        self.data[self.data_index] = data
        self.update(index, priority)

        self.data_index = (self.data_index + 1) % self.capacity
        self.size += 1

    def update(self, index, priority):
        delta = priority - self.priorities[index]
        self.priorities[index] = priority

        while index != 0:
            index = (index - 1) // 2
            self.priorities[index] += delta

    def get(self, priority):
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1

            if left >= len(self.priorities):
                node_index = parent
                break
            else:
                if priority < self.priorities[left]:
                    parent = left
                else:
                    priority -= self.priorities[left]
                    parent = right
        data_index = node_index - self.capacity + 1

        return node_index, self.priorities[node_index], self.data[data_index]

    @property
    def max_priority(self):
        return np.min(self.priorities[-self.capacity:])

    @property
    def total_priority(self):
        return self.priorities[0]

    def __len__(self):
        return min(self.size, self.capacity)
