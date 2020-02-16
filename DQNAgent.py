import numpy as np
from Policy import QNetBuilder
from ReplayBuffer import ReplayBuffer


class DDQNAgent(object):
    __slots__ = ["actionSpace", "numberOfActions", "discountFactor", "decisionFactor", "decisionFactorDecayRate",
                 "decisionFactorMinimum", "batchSize", "modelName", "updateTargetModelFrequency", "memory",
                 "trainingQNetModel", "targetQNetModel", "scoreHistory", "decisionFactorHistory", "avgScoreHistory",
                 "actionCount", "learningFrequency", "loss", "accuracy", "lossHistory", "accuracyHistory",
                 "avgLossHistory", "avgAccuracyHistory"]

    def __init__(self, learningRate, discountFactor, numberOfActions, decisionFactor, batchSize,
                 inputDimensions, decisionFactorDecayRate=0.996, decisionFactorMinimum=0.01,
                 memorySlots=1000000, modelName='ddqn_model.h5', updateTargetModelFrequency=100,
                 learningFrequency=1, useMaxPooling = False):
        self.actionSpace = np.arange(numberOfActions, dtype = np.uint8)
        self.numberOfActions = numberOfActions
        self.discountFactor = discountFactor
        self.decisionFactor = decisionFactor
        self.decisionFactorDecayRate = decisionFactorDecayRate
        self.decisionFactorMinimum = decisionFactorMinimum
        self.batchSize = batchSize
        self.modelName = modelName
        self.updateTargetModelFrequency = updateTargetModelFrequency
        self.memory = ReplayBuffer(memorySlots, inputDimensions, numberOfActions, discreteActions = True)
        self.trainingQNetModel = QNetBuilder(learningRate, numberOfActions,
                                             inputDimensions, useMaxPooling, 2).getModel()
        self.targetQNetModel = QNetBuilder(learningRate, numberOfActions,
                                           inputDimensions, useMaxPooling, 2).getModel()
        self.scoreHistory = np.array([0])
        self.decisionFactorHistory = np.array([1])
        self.avgScoreHistory = np.array([0])
        self.learningFrequency = learningFrequency
        self.actionCount = 0
        self.lossHistory = np.array([])
        self.accuracyHistory = np.array([])
        self.avgLossHistory = np.array([])
        self.avgAccuracyHistory = np.array([])
        self.loss = 0
        self.accuracy = 0

    def remember(self, state, action, reward, new_state, done):
        self.memory.storeTransition(state, action, reward, new_state, done)

    def chooseAction(self, state):
        rand = np.random.random()
        if rand < self.decisionFactor:
            action = np.random.choice(self.actionSpace)
        else:
            actions = self.trainingQNetModel.predict(state[np.newaxis, :])
            action = np.argmax(actions)
        self.actionCount += 1
        return action

    def learn(self):
        if self.actionCount % self.learningFrequency != 0:
            return
        if self.memory.memorySlotCounter > self.batchSize:
            state, action, reward, newState, done = self.memory.sampleBuffer(self.batchSize)
            actionEncodings = np.array(self.actionSpace, dtype = np.uint8)
            actionTaken = np.dot(action, actionEncodings).astype(np.uint8)
            QValues = self.targetQNetModel.predict(state)
            futureQValues = self.__getMaximumQScoresFromNextStates(newState)
            self.__updateQValues(QValues, reward, futureQValues, actionTaken, done)
            self.__trainTheModelWithUpdatedQValues(QValues, state)

            self.__updateDecisionFactor()

            if self.memory.memorySlotCounter % self.updateTargetModelFrequency == 0:
                self.__updateNetworkParameters()

    def __getMaximumQScoresFromNextStates(self, newState):
        nextStatePredictions = self.targetQNetModel.predict(newState)
        return np.max(nextStatePredictions, axis = 1)

    def __updateQValues(self, q_values, reward, predictedNextQValues, actionTaken, done):
        for i in range(self.batchSize):
            if done[i]:
                q_values[i][actionTaken[i]] = reward[i]
            else:
                q_values[i][actionTaken[i]] = reward[i] + self.discountFactor * predictedNextQValues[i]

    def __trainTheModelWithUpdatedQValues(self, q_values, state):
        fit = self.trainingQNetModel.fit(state, q_values, batch_size = self.batchSize, verbose = 0)
        self.loss = fit.history["loss"][0]
        try:
            self.accuracy = fit.history["acc"][0] * 100
        except KeyError:
            self.accuracy = 0

    def __updateDecisionFactor(self):
        self.decisionFactor = self.decisionFactor * self.decisionFactorDecayRate if self.decisionFactor > \
            self.decisionFactorMinimum else self.decisionFactorMinimum

    def __updateNetworkParameters(self):
        if self.updateTargetModelFrequency == 1:
            tau = 0.0001
            self.targetQNetModel.set_weights(
                [(1 - tau) * w for w in self.trainingQNetModel.get_weights()] +
                [tau * w for w in self.trainingQNetModel.get_weights()])
        else:
            self.targetQNetModel.set_weights(self.trainingQNetModel.get_weights())

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

    def appendStats(self, gameNumber, gameScore):
        self.scoreHistory = np.append(self.scoreHistory, gameScore)
        avgScore = np.mean(self.scoreHistory[max(0, gameNumber - 100):(gameNumber + 1)])
        self.avgScoreHistory = np.append(self.avgScoreHistory, avgScore)
        self.lossHistory = np.append(self.lossHistory, self.loss)
        avgLoss = np.mean(self.lossHistory[max(0, gameNumber - 100):(gameNumber + 1)])
        self.avgLossHistory = np.append(self.avgLossHistory, avgLoss)
        self.accuracyHistory = np.append(self.accuracyHistory, self.accuracy)
        avgAcc = np.mean(self.accuracyHistory[max(0, gameNumber - 100):(gameNumber + 1)])
        self.avgAccuracyHistory = np.append(self.avgAccuracyHistory, avgAcc)