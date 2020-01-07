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

    def __init__(self, gameSelection, numberOfGamesToPlay, showVideo=False):
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
        # self.plotter = OutputUtils()
        self.resetGame()

    def setNumberOfGamesToPlay(self, numberOfGamesToPlay):
        self.numberOfGamesToPlay = numberOfGamesToPlay

    def setShowVideo(self, showVideo):
        self.showVideo = showVideo

    def addAgent(self, agent):
        self.agent = agent

    @staticmethod
    def __selectGameFromLibrary(gameSelection):
        print('selecting game', end="                            \r")
        gameLibrary = {1: 'LunarLander-v2', 2: 'Breakout-v0'}
        videoLibrary = {1: './lunar-lander-ddqn-2', 2: './breakout-ddqn-0'}
        modelLibrary = {1: './lunar-lander-ddqn_model.h5', 2: './breakout-ddqn_model.h5'}
        gameName = gameLibrary.get(gameSelection)
        videoName = videoLibrary.get(gameSelection)
        modelName = modelLibrary.get(gameSelection)
        return gameName, videoName, modelName

    def __initializeGame(self):
        print("loading game", end="                            \r")
        theGame = gym.make(self.gameName).env
        if self.showVideo:
            theGame = wrappers.Monitor(self.theGame, self.videoName, video_callable=lambda episode_id: True, force=True)
        return theGame

    def resetGame(self):
        self.isDone = False
        self.frameCount = 0
        self.gameScore = 0
        self.gameFrame = self.theGame.reset()
        self.gameFrame = self.__rgb2gray(self.gameFrame)
        try:
            self.gameState = np.stack([self.gameFrame] * 4, axis=2).astype(np.uint8)
        except np.AxisError:
            self.gameState = np.stack([self.gameFrame] * 4, axis=1).astype(np.uint8)

    def playGames(self):
        gc.enable()
        avgScore = 0
        for gameNumber in range(self.numberOfGamesToPlay):
            self.resetGame()
            self.playOneGame(avgScore)
            self.agent.decisionFactorHistory = np.append(self.agent.decisionFactorHistory, self.agent.decisionFactor)
            self.agent.scoreHistory = np.append(self.agent.scoreHistory, self.gameScore)
            avgScore = np.mean(self.agent.scoreHistory[max(0, gameNumber - 100):(gameNumber + 1)])
            self.agent.avgScoreHistory = np.append(self.agent.avgScoreHistory, avgScore)
            self.plotter.updatePlot(self, self.agent)
            self.plotter.clearOutput()
            gc.collect()
            if gameNumber % 1000 == 0:
                self.agent.saveModel()

    def playOneGame(self, avgScore):
        self.endState = None
        self.gameNumber += 1
        while not self.isDone:
            action = self.__playFrame()
            self.__processNewGameFrame()
            self.agent.remember(self.gameState, action, self.reward, self.newGameState, int(self.isDone))
            self.agent.learn()
            self.gameState = self.newGameState
            self.plotter.printScores(self.gameNumber, self.frameCount, self.gameScore, self.info, avgScore,
                                     self.agent.decisionFactor, self.numberOfGamesToPlay, self.agent.getModelSummary())

    def __playFrame(self):
        action = self.agent.chooseAction(self.gameState)
        for j in range(3):
            _ = self.theGame.step(action)
        self.newGameFrame, self.reward, self.isDone, self.info = self.theGame.step(action)
        self.frameCount += 1
        self.gameScore += self.reward
        return action

    def __processNewGameFrame(self):
        if self.isDone:
            self.endState = self.newGameFrame
        self.newGameFrame = self.__rgb2gray(self.newGameFrame)
        self.newGameState = np.append(self.gameState[:, :, 1:],
                                      np.expand_dims(self.newGameFrame + 2, 2), axis=2).astype(np.uint8)

    @staticmethod
    def __rgb2gray(rgb):
        if len(rgb.shape) == 1:
            return rgb
        rgb = cv2.resize(rgb, dsize=(80, 105), interpolation=cv2.INTER_AREA)
        rgb = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return rgb
