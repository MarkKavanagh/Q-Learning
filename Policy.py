from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Lambda, Input
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop


class QNetBuilder(object):
    __slots__ = ["model", "version", "learningRate", "numberOfActions", "inputDimensions", "useMaxPooling"]

    def __init__(self, learningRate, numberOfActions, inputDimensions, useMaxPooling = False, version = 1):
        self.version = version
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
        if self.version == 1:
            optimizer = Adam(lr = self.learningRate)
        else:
            optimizer = RMSprop(lr = self.learningRate, rho = 0.95, epsilon = 0.01)
        model.compile(optimizer = optimizer, loss = "mean_squared_error", metrics = ["acc"])
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
        fcLayers = (512,)
        layer = Dense(fcLayers[0], name = "Hidden-1", activation = "relu")(previousLayer)
        for i in range(1, len(fcLayers)):
            layer = Dense(fcLayers[i], name = "Hidden-{n}".format(n = i + 1), activation = "relu")(layer)
        return layer
