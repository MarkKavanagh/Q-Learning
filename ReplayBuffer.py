import numpy as np


class ReplayBuffer(object):
    __slots__ = ["memorySlots", "memorySlotCounter", "discreteActions", "stateMemory", "outcomeStateMemory",
                 "actionMemory", "rewardMemory", "isDoneMemory"]

    def __init__(self, memorySlots, inputShape, numberOfActions, discreteActions = True):
        self.memorySlots = memorySlots
        self.memorySlotCounter = 0
        self.discreteActions = discreteActions
        self.stateMemory = np.zeros((self.memorySlots, *inputShape), dtype = np.uint8)
        self.outcomeStateMemory = np.zeros((self.memorySlots, *inputShape), dtype = np.uint8)
        dTypeForActionMemory = np.int8 if self.discreteActions else np.float32
        self.actionMemory = np.zeros((self.memorySlots, numberOfActions), dtype = dTypeForActionMemory)
        self.rewardMemory = np.zeros(self.memorySlots, dtype = np.int8)
        self.isDoneMemory = np.zeros(self.memorySlots, dtype = np.uint8)

    def storeTransition(self, state, action, reward, outcomeState, isDone):
        index = self.memorySlotCounter % self.memorySlots
        self.stateMemory[index] = state
        self.outcomeStateMemory[index] = outcomeState
        if self.discreteActions:
            actions = np.zeros(self.actionMemory.shape[1])
            actions[action] = 1.0
            self.actionMemory[index] = actions
        else:
            self.actionMemory[index] = action
        self.rewardMemory[index] = reward
        self.isDoneMemory[index] = 1 - isDone
        self.memorySlotCounter += 1

    def sampleBuffer(self, batchSize):
        max_mem = min(self.memorySlotCounter, self.memorySlots)
        batch = np.random.choice(max_mem, batchSize)

        states = self.stateMemory[batch]
        actions = self.actionMemory[batch]
        rewards = self.rewardMemory[batch]
        outComeStates = self.outcomeStateMemory[batch]
        terminal = self.isDoneMemory[batch]

        return states, actions, rewards, outComeStates, terminal

    class Builder:
        def __init__(self):
            self.memorySlots = 1000000
            self.inputShape = None
            self.numberOfActions = None
            self.discreteActions = True

        def setMemorySlots(self, memorySlots):
            self.memorySlots = memorySlots
            return self

        def setInputShape(self, inputShape):
            self.inputShape = inputShape
            return self

        def setNumberOfActions(self, numberOfActions):
            self.numberOfActions = numberOfActions
            return self

        def setDiscreteActions(self, discreteActions):
            self.discreteActions = discreteActions
            return self

        def build(self):
            return ReplayBuffer(self.memorySlots, self.inputShape, self.numberOfActions, self.discreteActions)
