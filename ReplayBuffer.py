import numpy as np


class ReplayBuffer(object):
    def __init__(self, memorySlots, inputShape, numberOfActions, discreteActions=True):
        self.memorySlots = memorySlots
        self.memorySlotCounter = 0
        self.discreteActions = discreteActions
        self.stateMemory = np.zeros((self.memorySlots, *inputShape), dtype=np.uint8)
        self.outcomeStateMemory = np.zeros((self.memorySlots, *inputShape), dtype=np.uint8)
        dTypeForActionMemory = np.int8 if self.discreteActions else np.float32
        self.actionMemory = np.zeros((self.memorySlots, numberOfActions), dtype=dTypeForActionMemory)
        self.rewardMemory = np.zeros(self.memorySlots, dtype=np.int8)
        self.isDoneMemory = np.zeros(self.memorySlots, dtype=np.uint8)

    def store_transition(self, state, action, reward, outcomeState, isDone):
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

    def sample_buffer(self, batchSize):
        max_mem = min(self.memorySlotCounter, self.memorySlots)
        batch = np.random.choice(max_mem, batchSize)

        states = self.stateMemory[batch]
        actions = self.actionMemory[batch]
        rewards = self.rewardMemory[batch]
        outComeStates = self.outcomeStateMemory[batch]
        terminal = self.isDoneMemory[batch]

        return states, actions, rewards, outComeStates, terminal
