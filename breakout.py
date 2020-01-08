from DQNAgent import DDQNAgent
from AtariGameProcessor import GameProcessor
from ArgumentParser import ArgumentParser


def createAgent(parser, inputDimensions, numberOfActions, modelName):
    agent = DDQNAgent(memorySlots=parser.memorySlots, decisionFactor=parser.decisionFactor, batchSize=parser.batchSize,
                      inputDimensions=inputDimensions, modelName=modelName, useMaxPooling=parser.useMaxPooling,
                      decisionFactorDecayRate=parser.decisionFactorDecayRate, numberOfActions=numberOfActions,
                      decisionFactorMinimum=parser.decisionFactorMinimum, discountFactor=parser.discountFactor,
                      updateTargetModelFrequency=parser.updateTargetModelFrequency, learningRate=parser.learningRate,
                      learningFrequency=parser.learningFrequency)
    if parser.loadPreviousModel:
        agent.loadModel()
    return agent


def main():
    parser = ArgumentParser()
    GP = GameProcessor(parser.gameNumberForLibrary, parser.numberOfGamesToPlay)
    GP.plotter.setIsNotebook(parser.isNotebook)
    inputDimensions = GP.gameState.shape
    numberOfActions = GP.theGame.action_space.n
    GP.addAgent(createAgent(parser, inputDimensions, numberOfActions, GP.modelName))
    GP.plotter.initializePlot(GP.agent)
    GP.playGames()
    GP.agent.saveModel()


if __name__ == '__main__':
    main()
