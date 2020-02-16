import numpy as np
import gym
from gym import wrappers
import cv2
import gc
from OutputUtils import OutputUtils


class GameProcessor(object):
    __slots__ = ["showVideo", "numberOfGamesToPlay", "gameName", "videoName", "modelName",
                 "theGame", "isDone", "gameNumber", "frameCount", "gameScore", "gameFrame",
                 "newGameFrame", "reward", "info", "endState", "gameState", "newGameState",
                 "agent", "plotter"]

    def __init__(self, gameSelection, numberOfGamesToPlay, showVideo = False):
        self.numberOfGamesToPlay = numberOfGamesToPlay
        self.showVideo = showVideo
        self.agent = None
        self.isDone = False
        self.frameCount = 0
        self.gameScore = 0
        self.gameFrame = None
        self.gameState = None
        self.endState = None
        self.__resetVariables(gameSelection, numberOfGamesToPlay, showVideo)

    def selectNewGameToPlay(self, game):
        self.__resetVariables(game, self.numberOfGamesToPlay, self.showVideo)

    def __resetVariables(self, gameSelection, numberOfGamesToPlay, showVideo):
        self.showVideo = showVideo
        self.numberOfGamesToPlay = numberOfGamesToPlay
        self.gameName, self.videoName, self.modelName = self.__selectGameFromLibrary(gameSelection)
        self.theGame = self.__initializeGame()
        self.isDone = False
        self.gameNumber = 0
        self.frameCount = 0
        self.gameScore = 0
        self.gameFrame = None
        self.newGameFrame = None
        self.reward = 0
        self.info = ""
        self.endState = None
        self.gameState = None
        self.newGameState = None
        self.agent = None
        self.plotter = OutputUtils()
        self.resetGame()

    def setNumberOfGamesToPlay(self, numberOfGamesToPlay):
        self.numberOfGamesToPlay = numberOfGamesToPlay

    def setShowVideo(self, showVideo):
        self.showVideo = showVideo

    def addAgent(self, agent):
        self.agent = agent

    @staticmethod
    def __selectGameFromLibrary(gameSelection):
        gameLibrary = {1: 'LunarLander-v2', 2: 'Breakout-v0'}
        videoLibrary = {1: './lunar-lander-ddqn-2', 2: './breakout-ddqn-0'}
        modelLibrary = {1: './lunar-lander-ddqn_model.h5', 2: './breakout-ddqn_model.h5'}
        gameName = gameLibrary.get(gameSelection)
        videoName = videoLibrary.get(gameSelection)
        modelName = modelLibrary.get(gameSelection)
        return gameName, videoName, modelName

    def __initializeGame(self):
        theGame = gym.make(self.gameName).env
        if self.showVideo:
            theGame = wrappers.Monitor(self.theGame, self.videoName,
                                       video_callable = lambda episode_id: True, force = True)
        return theGame

    def resetGame(self):
        self.isDone = False
        self.frameCount = 0
        self.gameScore = 0
        self.gameFrame = self.theGame.reset()
        self.gameFrame = self.__rgb2gray(self.gameFrame)
        try:
            self.gameState = np.stack([self.gameFrame] * 4, axis = 2).astype(np.uint8)
        except np.AxisError:
            self.gameState = np.stack([self.gameFrame] * 4, axis = 1).astype(np.uint8)

    def playGames(self):
        gc.enable()
        for gameNumber in range(self.numberOfGamesToPlay):
            self.resetGame()
            self.playOneGame()
            self.agent.decisionFactorHistory = np.append(self.agent.decisionFactorHistory, self.agent.decisionFactor)
            self.agent.appendStats(gameNumber, self.gameScore)
            self.plotter.updatePlot(self, self.agent)
            self.plotter.clearOutput()
            gc.collect()
            if gameNumber % 1000 == 0:
                self.agent.saveModel()

    def playOneGame(self):
        self.endState = None
        self.gameNumber += 1
        while not self.isDone:
            action = self.__playFrame()
            self.__processNewGameFrame()
            self.agent.remember(self.gameState, action, self.reward, self.newGameState, int(self.isDone))
            self.agent.learn()
            self.agent.update()
            self.gameState = self.newGameState
            # self.plotter.printScores(self.gameNumber, self.frameCount, self.gameScore, self.info,
            #                          self.agent.avgScoreHistory[-1], self.agent.decisionFactor,
            #                          self.numberOfGamesToPlay, self.agent.getModelSummary(),
            #                          self.agent.accuracy, self.agent.loss)

    def __playFrame(self):
        action = self.agent.chooseAction(self.gameState)
        self.newGameFrame, self.reward, self.isDone, self.info = self.theGame.step(action)
        self.frameCount += 1
        self.gameScore += self.reward
        return action

    def __processNewGameFrame(self):
        if self.isDone:
            self.endState = self.newGameFrame
        self.newGameFrame = self.__rgb2gray(self.newGameFrame)
        self.newGameState = np.append(self.gameState[:, :, 1:],
                                      np.expand_dims(self.newGameFrame + 2, 2), axis = 2).astype(np.uint8)

    @staticmethod
    def __rgb2gray(rgb):
        if len(rgb.shape) == 1:
            return rgb
        rgb = cv2.resize(rgb, dsize = (80, 105), interpolation = cv2.INTER_AREA)
        rgb = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return rgb

    class Builder:
        def __init__(self):
            self.gameSelection = None
            self.numberOfGamesToPlay = None
            self.showVideo = False

        def setGameSelection(self, gameSelection):
            self.gameSelection = gameSelection
            return self

        def setNumberOfGamesToPlay(self, numberOfGamesToPlay):
            self.numberOfGamesToPlay = numberOfGamesToPlay
            return self

        def setShowVideo(self, showVideo):
            self.showVideo = showVideo
            return self

        def build(self):
            return GameProcessor(self.gameSelection, self.numberOfGamesToPlay, self.showVideo)