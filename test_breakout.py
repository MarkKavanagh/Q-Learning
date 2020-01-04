from unittest import TestCase
import numpy as np
# noinspection PyUnresolvedReferences
from ReplayBuffer import ReplayBuffer
# noinspection PyUnresolvedReferences
from Policy import QNetBuilder
import sys

# noinspection PyUnresolvedReferences
from breakout import main


class Test_breakout(TestCase):
    def test_memory(self):
        inputShape = (84, 84, 4)
        memorySlots = 90000
        numberOfActions = 4
        buffer = ReplayBuffer(memorySlots, inputShape, numberOfActions)
        memories = [sys.getsizeof(buffer.stateMemory), sys.getsizeof(buffer.outcomeStateMemory),
                    sys.getsizeof(buffer.actionMemory), sys.getsizeof(buffer.rewardMemory),
                    sys.getsizeof(buffer.isDoneMemory)]
        for m in memories:
            print(m / 1024**3)
        print(sum(memories) / 1024**3)
        self.assertTrue(sum(memories) / 1024**3 <= 5)

    def test_QNet(self):
        inputDimensions = (84, 84, 4)
        numberOfActions = 4
        learningRate = 0.0001
        net = QNetBuilder(learningRate, numberOfActions, inputDimensions, False).getModel()
        gameState = np.zeros((1, *inputDimensions), dtype=np.uint8)
        action = net.predict(gameState)
        self.assertEqual(len(action[0]), numberOfActions)
