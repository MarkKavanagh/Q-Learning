from unittest import TestCase
from Policy import QNet
import numpy as np
from keras.optimizers import Adam, RMSprop


class TestQNetBuilder(TestCase):
    def test_QNet(self):
        inputDimensions = (84, 84, 4)
        numberOfActions = 4
        learningRate = 0.0001
        netBuilder = QNet.Builder().useAdamOptimizer(learningRate)\
            .setNumberOfActions(numberOfActions)\
            .setInputDimensions(inputDimensions)
        net = netBuilder.build().getModel()
        gameState = np.zeros((1, *inputDimensions), dtype = np.uint8)
        action = net.predict(gameState)
        self.assertEqual(net.name, "Deep Q-Learning CNN Model")
        self.assertEqual(net.input_shape, (None, *inputDimensions))
        self.assertEqual(len(net.layers), 10)
        self.assertEqual(net.loss_functions[0].name, "mean_squared_error")
        self.assertEqual(net.loss_functions[0].reduction, "sum_over_batch_size")
        self.assertEqual(net.optimizer.__class__, Adam)
        self.assertEqual(len(action[0]), numberOfActions)

        net = netBuilder.useRmsPropOptimizer(learningRate).build().getModel()
        self.assertEqual(net.optimizer.__class__, RMSprop)
