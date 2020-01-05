from unittest import TestCase
import numpy as np
from numpy import uint8, int8
from DQNAgent import DDQNAgent


class TestDDQNAgent(TestCase):
    def setUp(self):
        self.inputDimensions = (84, 84, 4)
        self.memorySlots = 100
        self.numberOfActions = 4
        self.agent = DDQNAgent(memorySlots=self.memorySlots, decisionFactor=1.0,
                               batchSize=64,
                               inputDimensions=self.inputDimensions, modelName="modelName", useMaxPooling=False,
                               decisionFactorDecayRate=0.99996, numberOfActions=self.numberOfActions,
                               decisionFactorMinimum=0.1, discountFactor=0.99,
                               updateTargetModelFrequency=1,
                               learningRate=0.0001)

    def test_chooseAction(self):
        self.randomizeReplayBuffer(self.agent, self.inputDimensions, self.memorySlots, self.numberOfActions)
        state = self.agent.memory.stateMemory[0]
        action = self.agent.chooseAction(state)
        self.assertTrue(0 <= action < self.numberOfActions)
        self.assertTrue(type(action) == uint8)

    def test_learn(self):
        self.randomizeReplayBuffer(self.agent, self.inputDimensions, self.memorySlots, self.numberOfActions)
        self.agent.learn()

    @staticmethod
    def randomizeReplayBuffer(agent, inputDimensions, memorySlots, numberOfActions):
        agent.memory.memorySlotCounter = memorySlots
        agent.memory.stateMemory = np.random.randint(100, size=(memorySlots, *inputDimensions)).astype(uint8)
        agent.memory.outcomeStateMemory = np.random.randint(100, size=(memorySlots, *inputDimensions)).astype(uint8)
        actions = np.random.randint(numberOfActions, size=memorySlots)
        agent.memory.actionMemory[np.arange(actions.size), actions] = 1
        agent.memory.rewardMemory = np.random.randint(100, size=memorySlots).astype(int8)
        agent.memory.isDoneMemory = np.random.randint(2, size=memorySlots).astype(uint8)
