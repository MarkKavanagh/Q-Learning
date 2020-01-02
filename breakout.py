import numpy as np

from DQNAgent import DDQNAgent
from AtariGameProcessor import GameProcessor

isNotebook = False
useMaxPooling = False
loadPreviousModel = False
showVideo = False
numberOfGamesToPlay = 20

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
    agent.save_model()
