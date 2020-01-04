from unittest import TestCase
import numpy as np
from numpy import uint8, int8
# noinspection PyUnresolvedReferences
from DQNAgent import DDQNAgent


class TestDDQNAgent(TestCase):
    def test_learn(self):
        inputDimensions = (84, 84, 4)
        memorySlots = 100
        numberOfActions = 4
        agent = DDQNAgent(memorySlots=memorySlots, decisionFactor=1.0,
                          batchSize=64,
                          inputDimensions=inputDimensions, modelName="modelName", useMaxPooling=False,
                          decisionFactorDecayRate=0.99996, numberOfActions=numberOfActions,
                          decisionFactorMinimum=0.1, discountFactor=0.99,
                          updateTargetModelFrequency=1,
                          learningRate=0.0001)
        self.randomizeReplayBuffer(agent, inputDimensions, memorySlots, numberOfActions)
        agent.learn()

    @staticmethod
    def randomizeReplayBuffer(agent, inputDimensions, memorySlots, numberOfActions):
        agent.memory.memorySlotCounter = memorySlots
        agent.memory.stateMemory = np.random.randint(100, size=(memorySlots, *inputDimensions)).astype(uint8)
        agent.memory.outcomeStateMemory = np.random.randint(100, size=(memorySlots, *inputDimensions)).astype(uint8)
        # agent.memory.actionMemory = np.random.randint(4, size=(memorySlots, numberOfActions)).astype(int8)
        actions = np.random.randint(numberOfActions, size=memorySlots)
        # b = np.zeros((actions.size, actions.max() + 1))
        agent.memory.actionMemory[np.arange(actions.size), actions] = 1
        agent.memory.rewardMemory = np.random.randint(100, size=memorySlots).astype(int8)
        agent.memory.isDoneMemory = np.random.randint(2, size=memorySlots).astype(uint8)
