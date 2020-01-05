from unittest import TestCase
import sys
from ReplayBuffer import ReplayBuffer


class TestReplayBuffer(TestCase):
    def test_memory(self):
        inputShape = (84, 84, 4)
        memorySlots = 90000
        numberOfActions = 4
        buffer = ReplayBuffer(memorySlots, inputShape, numberOfActions)
        memories = [sys.getsizeof(buffer.stateMemory), sys.getsizeof(buffer.outcomeStateMemory),
                    sys.getsizeof(buffer.actionMemory), sys.getsizeof(buffer.rewardMemory),
                    sys.getsizeof(buffer.isDoneMemory)]
        for m in memories:
            print(m / 1024 ** 3)
        print(sum(memories) / 1024 ** 3)
        self.assertTrue(sum(memories) / 1024 ** 3 <= 5)
