from unittest import TestCase
import numpy as np
from numpy import uint8, int8
from DQNAgent import DDQNAgent


class TestDDQNAgent(TestCase):
    def setUp(self):
        self.inputDimensions = (84, 84, 4)
        self.memorySlots = 100
        self.numberOfActions = 4
        self.agent = DDQNAgent.Builder()\
            .setMemorySlots(self.memorySlots) \
            .setInputDimensions(self.inputDimensions) \
            .setNumberOfActions(self.numberOfActions) \
            .setDecisionFactor(1.0) \
            .setDecisionFactorDecayRate(0.99996) \
            .setDecisionFactorMinimum(0.05) \
            .setDiscountFactor(0.99) \
            .setModelName("modelName") \
            .setUseMaxPooling(False) \
            .setUpdateTargetModelFrequency(1) \
            .setLearningRate(0.0001) \
            .setLearningFrequency(1) \
            .setBatchSize(64) \
            .build()

    def test_chooseAction(self):
        self.randomizeReplayBuffer(self.agent, self.inputDimensions, self.memorySlots, self.numberOfActions)
        state = self.agent.memory.stateMemory[0]
        action = self.agent.chooseAction(state)
        self.assertTrue(0 <= action < self.numberOfActions)
        self.assertTrue(type(action) == uint8)

    def test_learn(self):
        self.randomizeReplayBuffer(self.agent, self.inputDimensions, self.memorySlots, self.numberOfActions)
        self.agent.actionCount = 1
        for i in range(4):
            self.agent.learn()
            print(self.agent.accuracy)

    def test_updateParameters(self):
        self.agent.targetQNetModel.set_weights(self.agent.trainingQNetModel.get_weights())

    @staticmethod
    def randomizeReplayBuffer(agent, inputDimensions, memorySlots, numberOfActions):
        agent.memory.memorySlotCounter = memorySlots
        agent.memory.stateMemory = np.random.randint(100, size=(memorySlots, *inputDimensions)).astype(uint8)
        agent.memory.outcomeStateMemory = np.random.randint(100, size=(memorySlots, *inputDimensions)).astype(uint8)
        actions = np.random.randint(numberOfActions, size=memorySlots)
        agent.memory.actionMemory[np.arange(actions.size), actions] = 1
        agent.memory.rewardMemory = np.random.randint(100, size=memorySlots).astype(int8)
        agent.memory.isDoneMemory = np.random.choice(a = [True, False], size=memorySlots).astype(uint8)
