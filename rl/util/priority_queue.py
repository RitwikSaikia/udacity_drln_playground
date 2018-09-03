# MIT License
#
# Copyright (c) 2018 Jaromír Janisch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# https://github.com/jaara/AI-blog/blob/master/SumTree.py

import numpy as np


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
