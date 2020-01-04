# noinspection PyUnresolvedReferences
from DQNAgent import DDQNAgent
# noinspection PyUnresolvedReferences
from AtariGameProcessor import GameProcessor

isNotebook = False
useMaxPooling = False
loadPreviousModel = False
showVideo = False
numberOfGamesToPlay = 50000
memorySlots = 90000
learningRate = 0.0001
decisionFactor = 1.00
decisionFactorDecayRate = 0.999996
decisionFactorMinimum = 0.1
discountFactor = 0.99
batchSize = 64
gameNumberForLibrary = 2
updateTargetModelFrequency = 1


def createAgent(inputDimensions, numberOfActions, modelName):
    agent = DDQNAgent(learningRate=learningRate, discountFactor=discountFactor, numberOfActions=numberOfActions,
                      memorySlots=memorySlots, decisionFactor=decisionFactor, batchSize=batchSize,
                      inputDimensions=inputDimensions, modelName=modelName, useMaxPooling=useMaxPooling,
                      decisionFactorDecayRate=decisionFactorDecayRate, decisionFactorMinimum=decisionFactorMinimum,
                      updateTargetModelFrequency=updateTargetModelFrequency)
    if loadPreviousModel:
        agent.loadModel()
    return agent


def main():
    GP = GameProcessor(gameNumberForLibrary, numberOfGamesToPlay)
    GP.plotter.setIsNotebook(isNotebook)
    inputDimensions = GP.gameState.shape
    numberOfActions = GP.theGame.action_space.n
    GP.addAgent(createAgent(inputDimensions, numberOfActions, GP.modelName))
    GP.plotter.initializePlot(GP.agent)
    GP.playGames()
    GP.agent.saveModel()


if __name__ == '__main__':
    main()
