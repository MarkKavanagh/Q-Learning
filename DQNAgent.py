import numpy as np
from Policy import QNetBuilder
from ReplayBuffer import ReplayBuffer


class DDQNAgent(object):
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
        self.trainingQNetModel = QNetBuilder(learningRate, numberOfActions, inputDimensions, useMaxPooling).get_model()
        self.targetQNetModel = QNetBuilder(learningRate, numberOfActions, inputDimensions, useMaxPooling).get_model()
        self.scoreHistory = np.array([0])
        self.decisionFactorHistory = np.array([1])
        self.avgScoreHistory = np.array([0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
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
            state, action, reward, newState, done = self.memory.sample_buffer(self.batchSize)

            action_values = np.array(self.actionSpace, dtype=np.uint8)
            action_indices = np.dot(action, action_values).astype(np.uint8)

            trainingPredictNextAction = self.trainingQNetModel.predict(newState)
            targetPredictNextAction = self.targetQNetModel.predict(newState)
            trainingPredictCurrentAction = self.trainingQNetModel.predict(state)

            bestAction = np.argmax(trainingPredictNextAction, axis=1)

            targetPredictCurrentAction = trainingPredictCurrentAction

            batchIndex = np.arange(self.batchSize, dtype=np.int32)

            targetPredictCurrentAction[batchIndex, action_indices] = reward + \
                self.discountFactor * targetPredictNextAction[batchIndex, bestAction.astype(int)] * done

            _ = self.trainingQNetModel.fit(state, targetPredictCurrentAction, verbose=0)

            # QValuesForActionsOnFutureStates = self.targetQNetModel.predict(newState)
            # QValuesForActionsOnFutureStates[done] = 0
            #
            # maximumFutureQValue = np.max(QValuesForActionsOnFutureStates, axis=1)
            # TargetModelQValue = reward + self.discountFactor * maximumFutureQValue
            # print(bestAction.shape)
            # print(maximumFutureQValue.shape)
            # print(state.shape)
            # print(TargetModelQValue[:, None].shape)
            # self.trainingQNetModel.fit(
            #     state, TargetModelQValue[:, None],
            #     epochs=1, batch_size=len(state), verbose=0
            # )

            self.update_decisionFactor()

            if self.memory.memorySlotCounter % self.updateTargetModelFrequency == 0:
                self.update_network_parameters()

    def update_decisionFactor(self):
        self.decisionFactor = self.decisionFactor * self.decisionFactorDecayRate if self.decisionFactor > \
            self.decisionFactorMinimum else self.decisionFactorMinimum

    def update_network_parameters(self):
        tau = 0.0001
        self.targetQNetModel.set_weights(
            [(1 - tau) * w for w in self.trainingQNetModel.get_weights()] +
            [tau * w for w in self.trainingQNetModel.get_weights()])

    def save_model(self):
        self.trainingQNetModel.save(self.modelName)

    def load_model(self):
        self.trainingQNetModel = QNetBuilder.load_model(self.modelName)
        if self.decisionFactor == 0.0:
            self.update_network_parameters()

    def getModelSummary(self):
        modelSummary = []
        self.trainingQNetModel.summary(print_fn=lambda x: modelSummary.append(x))
        return "\n".join(modelSummary)