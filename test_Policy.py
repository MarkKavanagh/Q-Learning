from unittest import TestCase
from Policy import QNetBuilder


class TestQNetBuilder(TestCase):
    def test_QNet(self):
        inputDimensions = (84, 84, 4)
        numberOfActions = 4
        learningRate = 0.0001
        net = QNetBuilder(learningRate, numberOfActions, inputDimensions, False).getModel()
        gameState = np.zeros((1, *inputDimensions), dtype=np.uint8)
        action = net.predict(gameState)
        self.assertEqual(len(action[0]), numberOfActions)
