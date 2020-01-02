import curses
import numpy as np
from IPython.display import display, clear_output
import matplotlib.pylab as plt
import psutil
import os


class OutputUtils(object):
    def __init__(self):
        self.isNotebook = False
        self.stdOut = self.__getStdOut(self.isNotebook)
        self.process = psutil.Process(os.getpid())

    def setIsNotebook(self, isNotebook):
        self.isNotebook = isNotebook
        self.stdOut = self.__getStdOut(isNotebook)

    @staticmethod
    def __getStdOut(isNotebook):
        if not isNotebook:
            stdOut = curses.initscr()
            curses.noecho()
            curses.cbreak()
        else:
            stdOut = None
        return stdOut

    def printScores(self, episodeId, frameCount, gameScore, info, avgScore,
                    decisionFactor, numberOfEpisodes, modelSummary):
        maxRawLength, maxLength = self.__getMaxLengths(numberOfEpisodes, gameScore, avgScore)
        lines = self.__formatOutput(maxLength, maxRawLength, episodeId, gameScore, avgScore,
                                    frameCount, info, decisionFactor, modelSummary, self.process)
        if not self.isNotebook:
            for i in range(len(lines)):
                self.stdOut.addstr(i, 0, lines[i])
            self.stdOut.refresh()
        else:
            for i in range(len(lines)):
                display((lines[i],))
            clear_output(wait=True)

    @staticmethod
    def __getMaxLengths(numberOfEpisodes, gameScore, avgScore):
        episodeLength = int(np.floor(np.log10(numberOfEpisodes)) + 1)
        scoreLength = int(np.floor(max(np.log10(max(abs(gameScore), 0.01)), 0)) + 1)
        avgScoreLength = int(np.floor(max(np.log10(max(abs(avgScore), 0.01)), 0)) + 4)
        maxRawLength = max(episodeLength, scoreLength, avgScoreLength)
        maxLength = max(episodeLength + len('      Episode: '),
                        scoreLength + len('Current Score: '),
                        avgScoreLength + len('Average Score: ')
                        )
        return maxRawLength, maxLength

    @staticmethod
    def __formatOutput(maxLength, maxRawLength, episodeId, gameScore, avgScore,
                       frameCount, info, decisionFactor, modelSummary, process):
        lines = []
        padding = int(max(maxLength - maxRawLength - len('      Episode: '), 0)) * ' '
        lines.append("      Episode: {id:{n}d}{padding} Frame: {frame:d}"
                     .format(id=episodeId, n=maxRawLength, padding=padding, frame=frameCount))
        padding = int(max(maxLength - maxRawLength - len('Current Score: '), 0)) * ' '
        lines.append("Current Score: {score:{n}d}{padding} info: {info}"
                     .format(score=int(gameScore), info=info, n=maxRawLength, padding=padding))
        padding = int(max(maxLength - maxRawLength - len('Average Score: '), 0)) * ' '
        lines.append("Average Score: {avg:{n}.2f}{padding} Decision Factor: {df:.3f}"
                     .format(avg=avgScore, n=maxRawLength, padding=padding, df=decisionFactor))
        lines.append("")
        lines.append("Process Memory: {memory:.3f} GB"
                     .format(memory=process.memory_info()[0] / 1024 ** 3))
        lines.append(modelSummary)
        return lines

    @staticmethod
    def initializePlot(agent):
        plt.figure(figsize=(20, 8))
        plt.subplot(2, 2, 1)
        plt.title('End State of the Game')
        plt.subplot(2, 2, 2)
        plt.title('Score Distribution')
        plt.hist([0], [0, 1])
        plt.subplot(2, 2, 3)
        plt.title('Game Score Progression')
        plt.plot([x for x in range(len(agent.scoreHistory))], agent.scoreHistory, 'b')
        plt.plot([x for x in range(len(agent.avgScoreHistory))], agent.avgScoreHistory, 'k')
        plt.legend(('Scores', 'Trendline'), loc='best')
        plt.subplot(2, 2, 4)
        plt.title('Decsion Factor Decay')
        plt.plot([x for x in range(len(agent.decisionFactorHistory))], agent.decisionFactorHistory, 'b')

    @staticmethod
    def updatePlot(GP, agent):
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.cla()
        plt.title('End State of the Game')
        plt.imshow(GP.endState)
        plt.subplot(2, 2, 2)
        plt.cla()
        plt.title('Score Distribution')
        plt.hist(agent.scoreHistory, [x for x in range(max(max(agent.scoreHistory), 10))], color='b', align='left')
        plt.subplot(2, 2, 3)
        plt.cla()
        plt.title('Game Score Progression')
        plt.plot([x for x in range(len(agent.scoreHistory))], agent.scoreHistory, 'b')
        plt.plot([x for x in range(len(agent.avgScoreHistory))], agent.avgScoreHistory, 'k')
        plt.legend(('Scores', 'Trendline'), loc='best')
        plt.subplot(2, 2, 4)
        plt.cla()
        plt.title('Decsion Factor Decay')
        plt.plot([x for x in range(len(agent.decisionFactorHistory))], agent.decisionFactorHistory, 'b')
        try:
            plt.savefig('./thePlot.jpg', dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1,
                        metadata=None)
        except PermissionError:
            pass

    def clearOutput(self):
        if not self.isNotebook:
            self.stdOut.clear()
