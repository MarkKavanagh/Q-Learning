import numpy as np
# noinspection PyUnresolvedReferences
from Policy import QNetBuilder
# noinspection PyUnresolvedReferences
from ReplayBuffer import ReplayBuffer


class DDQNAgent(object):
    __slots__ = ["actionSpace", "numberOfActions", "discountFactor", "decisionFactor", "decisionFactorDecayRate",
                 "decisionFactorMinimum", "batchSize", "modelName", "updateTargetModelFrequency", "memory",
                 "trainingQNetModel", "targetQNetModel", "scoreHistory", "decisionFactorHistory", "avgScoreHistory"]

    def __init__(self, learningRate, discountFactor, numberOfActions, decisionFactor, batchSize,
                 inputDimensions, decisionFactorDecayRate=0.996, decisionFactorMinimum=0.01,
                 memorySlots=1000000, modelName='ddqn_model.h5', updateTargetModelFrequency=100,
                 useMaxPooling=False):
        self.actionSpace = np.arange(numberOfActions, dtype=np.uint8)
        self.numberOfActions = numberOfActions
        self.discountFactor = discountFactor
        self.decisionFactor = decisionFactor
        self.decisionFactorDecayRate = decisionFactorDecayRate
        self.decisionFactorMinimum = decisionFactorMinimum
        self.batchSize = batchSize
        self.modelName = modelName
        self.updateTargetModelFrequency = updateTargetModelFrequency
        self.memory = ReplayBuffer(memorySlots, inputDimensions, numberOfActions, discreteActions=True)
        self.trainingQNetModel = QNetBuilder(learningRate, numberOfActions, inputDimensions, useMaxPooling).getModel()
        self.targetQNetModel = QNetBuilder(learningRate, numberOfActions, inputDimensions, useMaxPooling).getModel()
        self.scoreHistory = np.array([0])
        self.decisionFactorHistory = np.array([1])
        self.avgScoreHistory = np.array([0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.storeTransition(state, action, reward, new_state, done)

    def chooseAction(self, state):
        rand = np.random.random()
        if rand < self.decisionFactor:
            action = np.random.choice(self.actionSpace)
        else:
            state = state[np.newaxis, :]
            actions = self.trainingQNetModel.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.memorySlotCounter > self.batchSize:
            state, action, reward, newState, done = self.memory.sampleBuffer(self.batchSize)

            # action_values = np.array(self.actionSpace, dtype=np.uint8)
            # action_indices = np.dot(action, action_values).astype(np.uint8)

            # trainingPredictNextAction = self.trainingQNetModel.predict(newState)
            # targetPredictNextAction = self.targetQNetModel.predict(newState)
            # trainingPredictCurrentAction = self.trainingQNetModel.predict(state)
            #
            # bestAction = np.argmax(trainingPredictNextAction, axis=1)
            #
            # targetPredictCurrentAction = trainingPredictCurrentAction
            #
            # batchIndex = np.arange(self.batchSize, dtype=np.int32)
            #
            # targetPredictCurrentAction[batchIndex, action_indices] = reward + \
            #     self.discountFactor * targetPredictNextAction[batchIndex, bestAction.astype(int)] * done

            # _ = self.trainingQNetModel.fit(state, targetPredictCurrentAction, verbose=0)

            QValuesForActionsOnFutureStates = self.targetQNetModel.predict(newState)
            QValuesForActionsOnFutureStates[done] = 0

            maximumFutureQValue = np.max(QValuesForActionsOnFutureStates, axis=1)
            TargetModelQValue = reward + self.discountFactor * maximumFutureQValue
            _ = self.trainingQNetModel.fit(
                state, action * TargetModelQValue[:, None],
                epochs=1, batch_size=self.batchSize, verbose=0
            )

            self.__updateDecisionFactor()

            if self.memory.memorySlotCounter % self.updateTargetModelFrequency == 0:
                self.__updateNetworkParameters()

    def __updateDecisionFactor(self):
        self.decisionFactor = self.decisionFactor * self.decisionFactorDecayRate if self.decisionFactor > \
            self.decisionFactorMinimum else self.decisionFactorMinimum

    def __updateNetworkParameters(self):
        tau = 0.0001
        self.targetQNetModel.set_weights(
            [(1 - tau) * w for w in self.trainingQNetModel.get_weights()] +
            [tau * w for w in self.trainingQNetModel.get_weights()])

    def saveModel(self):
        self.trainingQNetModel.save(self.modelName)

    def loadModel(self):
        self.trainingQNetModel = QNetBuilder.loadModel(self.modelName)
        if self.decisionFactor == 0.0:
            self.__updateNetworkParameters()

    def getModelSummary(self):
        modelSummary = []
        self.trainingQNetModel.summary(print_fn=lambda x: modelSummary.append(x))
        return "\n".join(modelSummary)
