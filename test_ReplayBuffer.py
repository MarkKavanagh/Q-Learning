from unittest import TestCase
import sys
import numpy as np
from ReplayBuffer import ReplayBuffer


class TestReplayBuffer(TestCase):
    def setUp(self):
        self.inputShape = (84, 84, 4)
        self.numberOfActions = 4
        self.memorySlots = 2000

    def test_create(self):
        buffer = ReplayBuffer.Builder() \
            .setMemorySlots(self.memorySlots) \
            .setInputShape(self.inputShape) \
            .setNumberOfActions(self.numberOfActions) \
            .build()
        self.assertEqual(buffer.stateMemory.shape, (self.memorySlots, *self.inputShape))
        self.assertEqual(buffer.outcomeStateMemory.shape, (self.memorySlots, *self.inputShape))
        self.assertEqual(buffer.actionMemory.shape, (self.memorySlots, self.numberOfActions))
        self.assertEqual(buffer.rewardMemory.shape, (self.memorySlots,))
        self.assertEqual(buffer.isDoneMemory.shape, (self.memorySlots,))

    def test_memory(self):
        memorySlots = 90000
        buffer = ReplayBuffer.Builder() \
            .setMemorySlots(memorySlots) \
            .setInputShape(self.inputShape) \
            .setNumberOfActions(self.numberOfActions) \
            .build()
        memories = [sys.getsizeof(buffer.stateMemory), sys.getsizeof(buffer.outcomeStateMemory),
                    sys.getsizeof(buffer.actionMemory), sys.getsizeof(buffer.rewardMemory),
                    sys.getsizeof(buffer.isDoneMemory)]
        self.assertTrue(sum(memories) / 1024 ** 3 <= 5)

    def test_storeTransition(self):
        buffer = ReplayBuffer.Builder() \
            .setMemorySlots(self.memorySlots) \
            .setInputShape(self.inputShape) \
            .setNumberOfActions(self.numberOfActions) \
            .build()
        for i in range(2 * self.memorySlots):
            state = np.random.randint(100, size = self.inputShape)
            action = np.random.randint(self.numberOfActions, size = self.numberOfActions)
            reward = np.random.randint(100)
            outcomeState = np.random.randint(100, size = self.inputShape)
            isDone = np.random.randint(2)
            buffer.storeTransition(state, action, reward, outcomeState, isDone)
        self.assertEqual(buffer.stateMemory.shape, (self.memorySlots, *self.inputShape))
        self.assertEqual(buffer.outcomeStateMemory.shape, (self.memorySlots, *self.inputShape))
        self.assertEqual(buffer.actionMemory.shape, (self.memorySlots, self.numberOfActions))
        self.assertEqual(buffer.rewardMemory.shape, (self.memorySlots,))
        self.assertEqual(buffer.isDoneMemory.shape, (self.memorySlots,))

    def test_sampleBuffer(self):
        inputShape = (84, 84, 4)
        memorySlots = 100
        numberOfActions = 4
        buffer = ReplayBuffer.Builder() \
            .setMemorySlots(memorySlots) \
            .setInputShape(inputShape) \
            .setNumberOfActions(numberOfActions) \
            .build()
        for i in range(memorySlots):
            state = np.random.randint(100, size = inputShape)
            action = np.random.randint(numberOfActions, size = numberOfActions)
            reward = np.random.randint(100)
            outcomeState = np.random.randint(100, size = inputShape)
            isDone = 0
            buffer.storeTransition(state, action, reward, outcomeState, isDone)

        states, actions, rewards, outComeStates, terminal = buffer.sampleBuffer(64)
        print(states.shape)
        print(actions.shape)
        print(rewards.shape)
        print(outComeStates.shape)
        print(terminal.shape)
