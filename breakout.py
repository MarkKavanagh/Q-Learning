from DQNAgent import DDQNAgent
from AtariGameProcessor import GameProcessor
from ArgumentParser import ArgumentParser


def createAgent(parser, inputDimensions, numberOfActions, modelName):
    agent = DDQNAgent.Builder() \
        .setMemorySlots(parser.memorySlots) \
        .setInputDimensions(inputDimensions) \
        .setNumberOfActions(numberOfActions) \
        .setDecisionFactor(parser.decisionFactor) \
        .setDecisionFactorDecayRate(parser.decisionFactorDecayRate) \
        .setDecisionFactorMinimum(parser.decisionFactorMinimum) \
        .setDiscountFactor(parser.discountFactor) \
        .setModelName(modelName) \
        .setUseMaxPooling(parser.useMaxPooling) \
        .setUpdateTargetModelFrequency(parser.updateTargetModelFrequency) \
        .setLearningRate(parser.learningRate) \
        .setLearningFrequency(parser.learningFrequency) \
        .setBatchSize(parser.batchSize) \
        .build()
    if parser.loadPreviousModel:
        agent.loadModel()
    return agent


def main():
    parser = ArgumentParser()
    GP = GameProcessor.Builder()\
        .setGameSelection(parser.gameNumberForLibrary)\
        .setNumberOfGamesToPlay(parser.numberOfGamesToPlay)\
        .setShowVideo(parser.showVideo)\
        .build()
    GP.plotter.setIsNotebook(parser.isNotebook)
    inputDimensions = GP.gameState.shape
    numberOfActions = GP.theGame.action_space.n
    GP.addAgent(createAgent(parser, inputDimensions, numberOfActions, GP.modelName))
    GP.plotter.initializePlot(GP.agent)
    GP.playGames()
    GP.agent.saveModel()


if __name__ == '__main__':
    main()
