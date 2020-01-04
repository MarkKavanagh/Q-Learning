import argparse


class ArgumentParser(object):
    __slots__ = ["isNotebook", "useMaxPooling", "loadPreviousModel", "showVideo", "numberOfGamesToPlay", "memorySlots",
                 "learningRate", "decisionFactor", "decisionFactorDecayRate", "decisionFactorMinimum", "discountFactor",
                 "batchSize", "gameNumberForLibrary", "updateTargetModelFrequency", "parser"]

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.isNotebook = False
        self.useMaxPooling = False
        self.loadPreviousModel = False
        self.showVideo = False
        self.numberOfGamesToPlay = 50000
        self.memorySlots = 90000
        self.learningRate = 0.0001
        self.decisionFactor = 1.00
        self.decisionFactorDecayRate = 0.99996
        self.decisionFactorMinimum = 0.1
        self.discountFactor = 0.99
        self.batchSize = 64
        self.gameNumberForLibrary = 2
        self.updateTargetModelFrequency = 1
        self.__addArguments()
        self.__parseArguments()

    def __addArguments(self):
        self.parser.add_argument("-n", "--notebook", type=bool,
                                 help="[boolean] Use notebook display (default: {nb})"
                                 .format(nb=self.isNotebook))
        self.parser.add_argument("-p", "--maxpooling", type=bool,
                                 help="[boolean]Use max pooling layer in CNN (default: {mp})"
                                 .format(mp=self.useMaxPooling))
        self.parser.add_argument("-l", "--load", type=bool,
                                 help="[boolean] Load previous Q-Net model (default: {l})"
                                 .format(l=self.loadPreviousModel))
        self.parser.add_argument("-r", "--render", type=bool,
                                 help="[boolean] Render gameplay (default: {r})"
                                 .format(r=self.showVideo))
        self.parser.add_argument("-g", "--games", type=int,
                                 help="[int] Number of games to play before exit (default: {g})"
                                 .format(g=self.numberOfGamesToPlay))
        self.parser.add_argument("-m", "--buffersize", type=int,
                                 help="[int] Number of replay experiences stored in buffer(default: {b})"
                                 .format(b=self.memorySlots))
        self.parser.add_argument("-a", "--learningrate", type=float,
                                 help="[float] Learning rate of Q-Net model (default: {r}"
                                 .format(r=self.learningRate))
        self.parser.add_argument("-d", "--decisionstart", type=float,
                                 help="[float] Starting decision factor for Agent (default: {d}"
                                 .format(d=self.decisionFactor))
        self.parser.add_argument("-y", "--decisiondecay", type=float,
                                 help="[float] Rate of decay for the Agent's decision factor (default: {r})"
                                 .format(r=self.decisionFactorDecayRate))
        self.parser.add_argument("-f", "--decisionfinal", type=float,
                                 help="[float] Final value of the Agent's decision factor (default: {f})"
                                 .format(f=self.decisionFactorMinimum))
        self.parser.add_argument("-c", "--discount", type=float,
                                 help="[float] The Agent's discount factor on future rewards (default: {d})"
                                 .format(d=self.discountFactor))
        self.parser.add_argument("-b", "--batch", type=int,
                                 help="[int] Batch size for choosing replay experiences for learning (default: {b})"
                                 .format(b=self.batchSize))
        self.parser.add_argument("-s", "--selection", type=int,
                                 help="[int] Game to play from library"
                                 .format(s=self.gameNumberForLibrary))
        self.parser.add_argument("-u", "--update", type=int,
                                 help="[int] how often to update the Agent's target model (default: {u})"
                                 .format(u=self.updateTargetModelFrequency))

    def __parseArguments(self):
        args = self.parser.parse_args()
        self.parser = argparse.ArgumentParser()
        self.isNotebook = self.parseField(args.notebook, self.isNotebook)
        self.useMaxPooling = self.parseField(args.maxpooling, self.useMaxPooling)
        self.loadPreviousModel = self.parseField(args.load, self.loadPreviousModel)
        self.showVideo = self.parseField(args.render, self.showVideo)
        self.numberOfGamesToPlay = self.parseField(args.games, self.numberOfGamesToPlay)
        self.memorySlots = self.parseField(args.buffersize, self.memorySlots)
        self.learningRate = self.parseField(args.learningrate, self.learningRate)
        self.decisionFactor = self.parseField(args.decisionstart, self.decisionFactor)
        self.decisionFactorDecayRate = self.parseField(args.decisiondecay, self.decisionFactorDecayRate)
        self.decisionFactorMinimum = self.parseField(args.decisionfinal, self.decisionFactorMinimum)
        self.discountFactor = self.parseField(args.discount, self.discountFactor)
        self.batchSize = self.parseField(args.batch, self.batchSize)
        self.gameNumberForLibrary = self.parseField(args.selection, self.gameNumberForLibrary)
        self.updateTargetModelFrequency = self.parseField(args.update, self.updateTargetModelFrequency)

    @staticmethod
    def parseField(argument, field):
        return argument if argument is not None else field
