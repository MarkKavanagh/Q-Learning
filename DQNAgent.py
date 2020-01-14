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
            state = state[np.newaxis, :]
            actions = self.trainingQNetModel.predict(state)
            action = np.argmax(actions)
        self.actionCount += 1
        return action

    def learn(self):
        if self.actionCount % self.learningFrequency != 0:
            return
        version = 1
        if self.memory.memorySlotCounter > self.batchSize:
            state, action, reward, newState, done = self.memory.sampleBuffer(self.batchSize)
            if version == 1:
                action_values = np.array(self.actionSpace, dtype = np.uint8)
                action_indices = np.dot(action, action_values).astype(np.uint8)

                trainingPredictNextAction = self.trainingQNetModel.predict(newState)
                targetPredictNextAction = self.targetQNetModel.predict(newState)
                trainingPredictCurrentAction = self.trainingQNetModel.predict(state)

                bestAction = np.argmax(trainingPredictNextAction, axis = 1)

                targetPredictCurrentAction = trainingPredictCurrentAction

                batchIndex = np.arange(self.batchSize, dtype = np.int32)

                targetPredictCurrentAction[batchIndex, action_indices] = reward + \
                    self.discountFactor * targetPredictNextAction[batchIndex, bestAction.astype(int)] * done

                fit = self.trainingQNetModel.fit(state, targetPredictCurrentAction, verbose = 0)
            elif version == 2:
                QValuesForActionsOnFutureStates = self.targetQNetModel.predict(newState)
                maximumFutureQValue = np.max(QValuesForActionsOnFutureStates, axis = 1)
                TargetModelQValue = reward + self.discountFactor * (maximumFutureQValue * (1 - done)) - (100.0 * done)
                fit = self.trainingQNetModel.fit(
                    state, action * TargetModelQValue[:, None],
                    epochs = 1, batch_size = self.batchSize, verbose = 0
                )
            else:
                current_states = []
                q_values = []
                action_values = np.array(self.actionSpace, dtype = np.uint8)
                actionTaken = np.dot(action, action_values).astype(np.uint8)
                for i in range(self.batchSize):
                    current_states.append(state[i])
                    next_state_prediction = self.targetQNetModel.predict(np.expand_dims(newState[i], axis = 0))
                    next_q_value = np.max(next_state_prediction)
                    q = list(self.trainingQNetModel.predict(np.expand_dims(state[i], axis = 0))[0])
                    if done[i]:
                        q[actionTaken[i]] = reward[i]
                    else:
                        q[actionTaken[i]] = reward[i] + self.discountFactor * next_q_value
                    q_values.append(q)
                fit = self.trainingQNetModel.fit(np.asarray(current_states),
                                                 np.asarray(q_values),
                                                 batch_size = self.batchSize,
                                                 verbose = 0)

            self.loss = fit.history["loss"][0]
            try:
                self.accuracy = fit.history["acc"][0] * 100
            except KeyError:
                self.accuracy = 0

            self.__updateDecisionFactor()

            if self.memory.memorySlotCounter % self.updateTargetModelFrequency == 0:
                self.__updateNetworkParameters()

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
