from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Lambda, Input
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras import backend as K

import numpy as np
import gym
from gym import wrappers

import curses
from IPython.display import display, clear_output
import matplotlib.pylab as plt
import cv2
import psutil
import os

process = psutil.Process(os.getpid())
isNotebook = False
useMaxPooling = False


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


class QNetBuilder(object):
    def __init__(self, learningRate, numberOfActions, inputDimensions):
        self.model = self.buildNeuralNet(learningRate, numberOfActions, inputDimensions)

    def get_model(self):
        return self.model

    def buildNeuralNet(self, learningRate, numberOfActions, inputDimensions):
        inputs = Input(shape=inputDimensions, name="Input_Layer")
        if len(inputDimensions) > 1:
            cnnLayers = self.buildCnnLayers(inputs)
            fullyConnectedLayers = self.buildFullyConnectedLayers(cnnLayers)
        else:
            fullyConnectedLayers = self.buildFullyConnectedLayers(inputs)
        outputs = Dense(numberOfActions, name="Output_Layer")(fullyConnectedLayers)
        model = Model(inputs=inputs, outputs=outputs, name="Deep Q-Learning CNN Model")
        model.compile(optimizer=Adam(lr=learningRate), loss=self.huber_loss)
        return model

    @staticmethod
    def buildCnnLayers(previousLayer):
        cnnFilters = (32, 64, 64)
        cnnSizes = ((8, 8), (4, 4), (2, 2))
        cnnStrides = ((4, 4), (2, 2), (2, 2))
        prev = Lambda(lambda x: x / 255.0, name="Normalized_RGB")(previousLayer)
        for i in range(len(cnnFilters)):
            prev = Conv2D(filters=cnnFilters[i],
                          kernel_size=cnnSizes[i],
                          strides=cnnStrides[i],
                          activation="relu",
                          name="CNN-{n}".format(n=i + 1))(prev)
            if useMaxPooling:
                prev = MaxPooling2D(pool_size=(2, 2), strides=2)
        return Flatten(name="Flatten_Layer")(prev)

    @staticmethod
    def buildFullyConnectedLayers(previousLayer):
        fcLayers = (256, 256)
        layer = Dense(fcLayers[0], name="Hidden-1", activation="relu")(previousLayer)
        for i in range(1, len(fcLayers)):
            layer = Dense(fcLayers[i], name="Hidden-{n}".format(n=i + 1), activation="relu")(layer)
        return layer

    @staticmethod
    def huber_loss(expected, predicted, in_keras=True):
        error = expected - predicted
        quadratic_term = error * error / 2
        linear_term = abs(error) - 1 / 2
        use_linear_term = (abs(error) > 1.0)
        if in_keras:
            use_linear_term = K.cast(use_linear_term, 'float32')
        return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term


class DDQNAgent(object):
    def __init__(self, learningRate, discountFactor, numberOfActions, decisionFactor, batchSize,
                 inputDimensions, decisionFactorDecayRate=0.996, decisionFactorMinimum=0.01,
                 memorySlots=1000000, modelName='ddqn_model.h5', updateTargetModelFrequency=100):
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
        self.trainingQNetModel = QNetBuilder(learningRate, numberOfActions, inputDimensions).get_model()
        self.targetQNetModel = QNetBuilder(learningRate, numberOfActions, inputDimensions).get_model()

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
        # self.targetQNetModel.set_weights(self.trainingQNetModel.get_weights())
        tau = 0.0001
        self.targetQNetModel.set_weights(
            [(1 - tau) * w for w in agent.trainingQNetModel.get_weights()] +
            [tau * w for w in agent.trainingQNetModel.get_weights()])

    def save_model(self):
        self.trainingQNetModel.save(self.modelName)

    def load_model(self):
        self.trainingQNetModel = load_model(self.modelName, custom_objects={'huber_loss': QNetBuilder.huber_loss})
        if self.decisionFactor == 0.0:
            self.update_network_parameters()

    def getModelSummary(self):
        modelSummary = []
        self.trainingQNetModel.summary(print_fn=lambda x: modelSummary.append(x))
        return "\n".join(modelSummary)


class GameProcessor(object):
    def __init__(self, gameSelection):
        self.gameName, self.videoName, self.modelName = self.selectGameFromLibrary(gameSelection)
        self.theGame = self.initializeGame()
        self.isDone = False
        self.frameCount = 0
        self.gameScore = 0
        self.gameFrame = None
        self.newGameFrame = None
        self.reward = 0
        self.info = ""
        self.endState = None
        self.gameState = None
        self.newGameState = None
        self.resetGame()

    @staticmethod
    def selectGameFromLibrary(gameSelection):
        print('selecting game', end="                            \r")
        gameLibrary = {1: 'LunarLander-v2', 2: 'Breakout-v0'}
        videoLibrary = {1: './lunar-lander-ddqn-2', 2: './breakout-ddqn-0'}
        modelLibrary = {1: './lunar-lander-ddqn_model.h5', 2: './breakout-ddqn_model.h5'}
        gameName = gameLibrary.get(gameSelection)
        videoName = videoLibrary.get(gameSelection)
        modelName = modelLibrary.get(gameSelection)
        return gameName, videoName, modelName

    def initializeGame(self):
        print("loading game", end="                            \r")
        theGame = gym.make(self.gameName).env
        if showVideo:
            theGame = wrappers.Monitor(self.theGame, self.videoName, video_callable=lambda episode_id: True, force=True)
        return theGame

    def resetGame(self):
        self.isDone = False
        self.frameCount = 0
        self.gameScore = 0
        self.gameFrame = self.theGame.reset()
        self.gameFrame = self.rgb2gray(self.gameFrame)
        self.gameState = np.stack([self.gameFrame] * 4, axis=2).astype(np.uint8)

    def playOneGame(self, avgScore):
        self.endState = None
        while not self.isDone:
            action = self.playFrame()
            self.processNewGameFrame()
            agent.remember(self.gameState, action, self.reward, self.newGameState, int(self.isDone))
            agent.learn()
            self.gameState = self.newGameState
            printScores(i + 1, self.frameCount, self.gameScore, self.info, avgScore, agent.decisionFactor,
                        numberOfGamesToPlay, agent.getModelSummary())

    def playFrame(self):
        action = agent.choose_action(GP.gameState)
        for j in range(3):
            _ = self.theGame.step(action)
        self.newGameFrame, self.reward, self.isDone, self.info = GP.theGame.step(action)
        self.frameCount += 1
        self.gameScore += self.reward
        return action

    def processNewGameFrame(self):
        if self.isDone:
            self.endState = self.newGameFrame
        self.newGameFrame = self.rgb2gray(self.newGameFrame)
        self.newGameState = np.append(self.gameState[:, :, 1:],
                                      np.expand_dims(self.newGameFrame + 2, 2), axis=2).astype(np.uint8)

    @staticmethod
    def rgb2gray(rgb):
        if len(rgb.shape) == 1:
            return rgb
        rgb = cv2.resize(rgb, dsize=(80, 105), interpolation=cv2.INTER_AREA)
        rgb = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return rgb


def printScores(episodeId, frameCount, gameScore, info, avgScore, decisionFactor, numberOfEpisodes, modelSummary):
    episodeLength = int(np.floor(np.log10(numberOfEpisodes)) + 1)
    scoreLength = int(np.floor(max(np.log10(max(abs(gameScore), 0.01)), 0)) + 1)
    avgScoreLength = int(np.floor(max(np.log10(max(abs(avgScore), 0.01)), 0)) + 4)
    maxRawLength = max(episodeLength, scoreLength, avgScoreLength)
    maxLength = max(episodeLength + len('      Episode: '),
                    scoreLength + len('Current Score: '),
                    avgScoreLength + len('Average Score: ')
                    )
    padding = int(max(maxLength - maxRawLength - len('      Episode: '), 0)) * ' '
    line1 = "      Episode: {id:{n}d}{padding} Frame: {frame:d}" \
        .format(id=episodeId, n=maxRawLength, padding=padding, frame=frameCount)
    padding = int(max(maxLength - maxRawLength - len('Current Score: '), 0)) * ' '
    line2 = "Current Score: {score:{n}d}{padding} info: {info}" \
        .format(score=int(gameScore), info=info, n=maxRawLength, padding=padding)
    padding = int(max(maxLength - maxRawLength - len('Average Score: '), 0)) * ' '
    line3 = "Average Score: {avg:{n}.2f}{padding} Decision Factor: {df:.3f}" \
        .format(avg=avgScore, n=maxRawLength, padding=padding, df=decisionFactor)
    line4 = "Process Memory: {memory:.3f} GB" \
        .format(memory=process.memory_info()[0] / 1024 ** 3)
    line5 = modelSummary
    if not isNotebook:
        stdOut.addstr(0, 0, line1)
        stdOut.addstr(1, 0, line2)
        stdOut.addstr(2, 0, line3)
        stdOut.addstr(3, 0, "")
        stdOut.addstr(4, 0, line4)
        stdOut.addstr(5, 0, line5)
        stdOut.refresh()
    else:
        display((line1,))
        display((line2,))
        display((line3,))
        display((line4,))
        display((line5,))
        clear_output(wait=True)
    # time.sleep(0.01)


if __name__ == '__main__':
    if not isNotebook:
        stdOut = curses.initscr()
        curses.noecho()
        curses.cbreak()
    else:
        stdOut = None
    print('selecting game', end="                            \r")
    gameLibrary = {1: 'LunarLander-v2', 2: 'Breakout-v0'}
    videoLibrary = {1: './lunar-lander-ddqn-2', 2: './breakout-ddqn-0'}
    modelLibrary = {1: './lunar-lander-ddqn_model.h5', 2: './breakout-ddqn_model.h5'}
    gameSelection = 2

    gameName = gameLibrary.get(gameSelection)
    videoName = videoLibrary.get(gameSelection)
    modelName = modelLibrary.get(gameSelection)

    print("loading game", end="                            \r")
    theGame = gym.make(gameName).env
    gameFrame = theGame.reset()
    gameFrame = rgb2gray(gameFrame)
    gameState = np.stack([gameFrame] * 4, axis=2).astype(np.uint8)
    inputDimensions = gameState.shape
    numberOfActions = theGame.action_space.n

    print("enter player 1", end="                            \r")
    agent = DDQNAgent(learningRate=0.001, discountFactor=0.99, numberOfActions=numberOfActions, memorySlots=200000,
                      decisionFactor=.10, batchSize=64, inputDimensions=inputDimensions, modelName=GP.modelName,
                      decisionFactorDecayRate=0.999996, updateTargetModelFrequency=1, decisionFactorMinimum=0.1)

    numberOfGamesToPlay = 20
    # ddqn_agent.load_model()
    scoreHistory = np.array([0])
    decisionFactorHistory = np.array([1])
    avgScoreHistory = np.array([0])

    modelSummary = []
    agent.trainingQNetModel.summary(print_fn=lambda x: modelSummary.append(x))
    modelSummary = "\n".join(modelSummary)

    showVideo = False
    if showVideo:
        theGame = wrappers.Monitor(theGame, videoName, video_callable=lambda episode_id: True, force=True)
    avg_score = 0
    plt.figure(figsize=(20, 8))
    plt.subplot(2, 2, 1)
    plt.title('End State of the Game')
    plt.subplot(2, 2, 2)
    plt.title('Score Distribution')
    plt.hist([0], [0, 1])
    plt.subplot(2, 2, 3)
    plt.title('Game Score Progression')
    plt.plot([x for x in range(len(scoreHistory))], scoreHistory, 'b')
    plt.plot([x for x in range(len(avgScoreHistory))], avgScoreHistory, 'k')
    plt.legend(('Scores', 'Trendline'), loc='best')
    plt.subplot(2, 2, 4)
    plt.title('Decsion Factor Decay')
    plt.plot([x for x in range(len(decisionFactorHistory))], decisionFactorHistory, 'b')
    for i in range(numberOfGamesToPlay):
        # print('starting game: ', str(i+1), end="                            \r")
        isDone = False
        endState = None
        frameCount = 0
        gameScore = 0
        gameFrame = theGame.reset()
        gameFrame = rgb2gray(gameFrame)
        gameState = np.stack([gameFrame] * 4, axis=2).astype(np.uint8)
        while not isDone:
            action = agent.choose_action(gameState)
            for j in range(3):
                _ = theGame.step(action)
            newGameFrame, reward, isDone, info = theGame.step(action)
            frameCount += 1
            if isDone:
                endState = newGameFrame
            newGameFrame = rgb2gray(newGameFrame)
            newGameState = np.append(gameState[:, :, 1:], np.expand_dims(newGameFrame + 2, 2), axis=2).astype(np.uint8)
            gameScore += reward
            agent.remember(gameState, action, reward, newGameState, int(isDone))
            gameState = newGameState
            agent.learn()

            # time.sleep(0.1)

            printScores(i + 1, frameCount, gameScore, info, avg_score, agent.decisionFactor, numberOfGamesToPlay,
                        modelSummary)
            # time.sleep(.01)

        decisionFactorHistory = np.append(decisionFactorHistory, agent.decisionFactor)
        # agent.update_decisionFactor()

        scoreHistory = np.append(scoreHistory, gameScore)
        avg_score = np.mean(scoreHistory[max(0, i - 100):(i + 1)])
        avgScoreHistory = np.append(avgScoreHistory, avg_score)

        plt.clf()
        plt.subplot(2, 2, 1)
        plt.cla()
        plt.title('End State of the Game')
        plt.imshow(endState)
        plt.subplot(2, 2, 2)
        plt.cla()
        plt.title('Score Distribution')
        plt.hist(scoreHistory, [x for x in range(max(max(scoreHistory), 10))], color='b', align='left')
        plt.subplot(2, 2, 3)
        plt.cla()
        plt.title('Game Score Progression')
        plt.plot([x for x in range(len(scoreHistory))], scoreHistory, 'b')
        plt.plot([x for x in range(len(avgScoreHistory))], avgScoreHistory, 'k')
        plt.legend(('Scores', 'Trendline'), loc='best')
        plt.subplot(2, 2, 4)
        plt.cla()
        plt.title('Decsion Factor Decay')
        plt.plot([x for x in range(len(decisionFactorHistory))], decisionFactorHistory, 'b')
        plt.savefig('./thePlot.jpg', dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    metadata=None)
        if i % 1000 == 0:
            agent.save_model()
        if not isNotebook:
            stdOut.clear()
    agent.save_model()
