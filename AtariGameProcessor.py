import numpy as np
import gym
from gym import wrappers
import cv2
from OutputUtils import OutputUtils


class GameProcessor(object):
    def __init__(self, gameSelection, numberOfGamesToPlay, showVideo=False):
        self.showVideo = showVideo
        self.numberOfGamesToPlay = numberOfGamesToPlay
        self.gameName, self.videoName, self.modelName = self.selectGameFromLibrary(gameSelection)
        self.theGame = self.initializeGame()
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

    def addAgent(self, agent):
        self.agent = agent

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
        if self.showVideo:
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
        self.gameNumber += 1
        while not self.isDone:
            action = self.playFrame()
            self.processNewGameFrame()
            self.agent.remember(self.gameState, action, self.reward, self.newGameState, int(self.isDone))
            self.agent.learn()
            self.gameState = self.newGameState
            self.plotter.printScores(self.gameNumber, self.frameCount, self.gameScore, self.info, avgScore,
                                     self.agent.decisionFactor, self.numberOfGamesToPlay, self.agent.getModelSummary())

    def playFrame(self):
        action = self.agent.choose_action(self.gameState)
        for j in range(3):
            _ = self.theGame.step(action)
        self.newGameFrame, self.reward, self.isDone, self.info = self.theGame.step(action)
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