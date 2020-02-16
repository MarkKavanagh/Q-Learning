from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Lambda, Input
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop


class QNet(object):
    __slots__ = ["model", "optimizer", "learningRate", "numberOfActions", "inputDimensions", "useMaxPooling"]

    def __init__(self, learningRate, numberOfActions, inputDimensions, optimizer, useMaxPooling = False):
        self.optimizer = optimizer
        self.learningRate = learningRate
        self.numberOfActions = numberOfActions
        self.inputDimensions = inputDimensions
        self.useMaxPooling = useMaxPooling

    def getModel(self):
        model = self.__buildNeuralNet()
        return model

    @staticmethod
    def loadModel(modelName):
        return load_model(modelName)

    def __buildNeuralNet(self):
        inputs = Input(shape = self.inputDimensions, name = "Input_Layer")
        if len(self.inputDimensions) > 1:
            cnnLayers = self.__buildCnnLayers(inputs, self.useMaxPooling)
            fullyConnectedLayers = self.__buildFullyConnectedLayers(cnnLayers)
        else:
            fullyConnectedLayers = self.__buildFullyConnectedLayers(inputs)
        outputs = Dense(self.numberOfActions, name="Output_Layer")(fullyConnectedLayers)
        model = Model(inputs = inputs, outputs = outputs, name = "Deep Q-Learning CNN Model")
        model.compile(optimizer = self.optimizer, loss = "mean_squared_error", metrics = ["acc"])
        return model

    @staticmethod
    def __buildCnnLayers(previousLayer, useMaxPooling):
        cnnFilters = (32, 64, 64)
        cnnSizes = ((8, 8), (4, 4), (3, 3))
        cnnStrides = ((4, 4), (2, 2), (1, 1))
        prev = Lambda(lambda x: x / 255.0, name = "Normalized_RGB")(previousLayer)
        for i in range(len(cnnFilters)):
            prev = Conv2D(filters = cnnFilters[i],
                          kernel_size = cnnSizes[i],
                          strides = cnnStrides[i],
                          padding = "valid",
                          activation = "relu",
                          name = "CNN-{n}".format(n=i + 1))(prev)
            if useMaxPooling:
                prev = MaxPooling2D(pool_size = (2, 2), strides = 2)
        return Flatten(name = "Flatten_Layer")(prev)

    @staticmethod
    def __buildFullyConnectedLayers(previousLayer):
        fcLayers = (512, 256, 128)
        layer = Dense(fcLayers[0], name = "Hidden-1", activation = "relu")(previousLayer)
        for i in range(1, len(fcLayers)):
            layer = Dense(fcLayers[i], name = "Hidden-{n}".format(n = i + 1), activation = "relu")(layer)
        return layer

    class Builder:
        def __init__(self):
            self.optimizer = None
            self.learningRate = None
            self.numberOfActions = None
            self.inputDimensions = None
            self.useMaxPooling = False

        def useRmsPropOptimizer(self, learningRate):
            self.learningRate = learningRate
            self.optimizer = RMSprop(lr = learningRate, rho = 0.95, epsilon = 0.01)
            return self

        def useAdamOptimizer(self, learningRate):
            self.learningRate = learningRate
            self.optimizer = Adam(lr = learningRate)
            return self

        def setNumberOfActions(self, numberOfActions):
            self.numberOfActions = numberOfActions
            return self

        def setInputDimensions(self, inputDimensions):
            self.inputDimensions = inputDimensions
            return self

        def setUseMaxPooling(self, useMaxPooling):
            self.useMaxPooling = useMaxPooling
            return self

        def build(self):
            return QNet(self.learningRate, self.numberOfActions, self.inputDimensions, self.optimizer, self.useMaxPooling)
