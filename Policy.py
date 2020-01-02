from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Lambda, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K


class QNetBuilder(object):
    def __init__(self, learningRate, numberOfActions, inputDimensions, useMaxPooling=False):
        self.model = self.buildNeuralNet(learningRate, numberOfActions, inputDimensions, useMaxPooling)

    def get_model(self):
        return self.model

    @staticmethod
    def load_model(modelName):
        return load_model(modelName, custom_objects={'huber_loss': QNetBuilder.huber_loss})

    def buildNeuralNet(self, learningRate, numberOfActions, inputDimensions, useMaxPooling):
        inputs = Input(shape=inputDimensions, name="Input_Layer")
        if len(inputDimensions) > 1:
            cnnLayers = self.buildCnnLayers(inputs, useMaxPooling)
            fullyConnectedLayers = self.buildFullyConnectedLayers(cnnLayers)
        else:
            fullyConnectedLayers = self.buildFullyConnectedLayers(inputs)
        outputs = Dense(numberOfActions, name="Output_Layer")(fullyConnectedLayers)
        model = Model(inputs=inputs, outputs=outputs, name="Deep Q-Learning CNN Model")
        model.compile(optimizer=Adam(lr=learningRate), loss=self.huber_loss)
        return model

    @staticmethod
    def buildCnnLayers(previousLayer, useMaxPooling):
        cnnFilters = (32, 64, 64)
        cnnSizes = ((8, 8), (4, 4), (2, 2))
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
    def buildFullyConnectedLayers(previousLayer):
        fcLayers = (256, 256)
        layer = Dense(fcLayers[0], name="Hidden-1", activation="relu")(previousLayer)
        for i in range(1, len(fcLayers)):
            layer = Dense(fcLayers[i], name="Hidden-{n}".format(n=i + 1), activation="relu")(layer)
        return layer

    @staticmethod
    def huber_loss(expected, predicted, in_keras=True):
        error = expected - predicted
        quadratic_term = error * error / 2
        linear_term = abs(error) - 1 / 2
        use_linear_term = (abs(error) > 1.0)
        if in_keras:
            use_linear_term = K.cast(use_linear_term, 'float32')
        return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term
