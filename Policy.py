from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Lambda, Input
from keras.models import Model, load_model
from keras.optimizers import Adam


class QNetBuilder(object):
    __slots__ = ["model"]

    def __init__(self, learningRate, numberOfActions, inputDimensions, useMaxPooling=False):
        self.model = self.__buildNeuralNet(learningRate, numberOfActions, inputDimensions, useMaxPooling)

    def getModel(self):
        return self.model

    @staticmethod
    def loadModel(modelName):
        return load_model(modelName)

    def __buildNeuralNet(self, learningRate, numberOfActions, inputDimensions, useMaxPooling):
        inputs = Input(shape=inputDimensions, name="Input_Layer")
        if len(inputDimensions) > 1:
            cnnLayers = self.__buildCnnLayers(inputs, useMaxPooling)
            fullyConnectedLayers = self.__buildFullyConnectedLayers(cnnLayers)
        else:
            fullyConnectedLayers = self.__buildFullyConnectedLayers(inputs)
        outputs = Dense(numberOfActions, name="Output_Layer")(fullyConnectedLayers)
        model = Model(inputs=inputs, outputs=outputs, name="Deep Q-Learning CNN Model")
        model.compile(optimizer=Adam(lr=learningRate), loss="logcosh")
        return model

    @staticmethod
    def __buildCnnLayers(previousLayer, useMaxPooling):
        cnnFilters = (32, 64, 64)
        cnnSizes = ((8, 8), (4, 4), (3, 3))
        cnnStrides = ((4, 4), (2, 2), (2, 2))
        prev = Lambda(lambda x: x / 255.0, name="Normalized_RGB")(previousLayer)
        for i in range(len(cnnFilters)):
            prev = Conv2D(filters=cnnFilters[i],
                          kernel_size=cnnSizes[i],
                          strides=cnnStrides[i],
                          activation="relu",
                          name="CNN-{n}".format(n=i + 1))(prev)
            if useMaxPooling:
                prev = MaxPooling2D(pool_size=(2, 2), strides=2)
        return Flatten(name="Flatten_Layer")(prev)

    @staticmethod
    def __buildFullyConnectedLayers(previousLayer):
        fcLayers = (512,)
        layer = Dense(fcLayers[0], name="Hidden-1", activation="relu")(previousLayer)
        for i in range(1, len(fcLayers)):
            layer = Dense(fcLayers[i], name="Hidden-{n}".format(n=i + 1), activation="relu")(layer)
        return layer
