import numpy as np

from DQNAgent import DDQNAgent
from AtariGameProcessor import GameProcessor

isNotebook = False
useMaxPooling = False
loadPreviousModel = False
showVideo = False
numberOfGamesToPlay = 20


def createAgent(inputDimensions, numberOfActions):
    print("enter player 1", end="                            \r")
    agent = DDQNAgent(learningRate=0.001, discountFactor=0.99, numberOfActions=numberOfActions, memorySlots=200000,
                      decisionFactor=.10, batchSize=64, inputDimensions=inputDimensions, modelName=GP.modelName,
                      decisionFactorDecayRate=0.999996, updateTargetModelFrequency=1, decisionFactorMinimum=0.1,
                      useMaxPooling=useMaxPooling)
    if loadPreviousModel:
        agent.load_model()
    return agent


if __name__ == '__main__':
    GP = GameProcessor(2, numberOfGamesToPlay)
    GP.plotter.setIsNotebook(isNotebook)
    inputDimensions = GP.gameState.shape
    numberOfActions = GP.theGame.action_space.n
    agent = createAgent(inputDimensions, numberOfActions)
    GP.addAgent(agent)
    avgScore = 0
    GP.plotter.initializePlot(agent)
    for gameNumber in range(numberOfGamesToPlay):
        GP.resetGame()
        GP.playOneGame(avgScore)
        agent.decisionFactorHistory = np.append(agent.decisionFactorHistory, agent.decisionFactor)
        agent.scoreHistory = np.append(agent.scoreHistory, GP.gameScore)
        avgScore = np.mean(agent.scoreHistory[max(0, gameNumber - 100):(gameNumber + 1)])
        agent.avgScoreHistory = np.append(agent.avgScoreHistory, avgScore)
        GP.plotter.updatePlot(GP, agent)
        GP.plotter.clearOutput()
        if gameNumber % 1000 == 0:
            agent.save_model()
    agent.save_model()
