from unittest import TestCase, mock
from AtariGameProcessor import GameProcessor


class TestGameProcessor(TestCase):
    def setUp(self):
        with mock.patch("AtariGameProcessor.OutputUtils") as plotterMock:
            plotterMock.return_value.getStdOut.return_value = 1
            plotterMock.return_value.printScores.return_value = 1
            plotterMock.__getStdOut.return_value = 1
            plotterMock.printScores.return_value = 1
            self.processor = GameProcessor.Builder().setGameSelection(2).setNumberOfGamesToPlay(10).build()

    def test_select_new_game_to_play(self):
        with mock.patch("AtariGameProcessor.OutputUtils") as plotterMock:
            plotterMock.return_value.getStdOut.return_value = 1
            self.processor.selectNewGameToPlay(1)
            self.assertTrue(self.processor.gameName, "LunarLander-v2")

    def test_set_number_of_games_to_play(self):
        self.processor.setNumberOfGamesToPlay(20)
        self.assertTrue(self.processor.numberOfGamesToPlay, 20)

    def test_set_show_video(self):
        self.processor.setShowVideo(True)
        self.assertTrue(self.processor.showVideo)

    def test_add_agent(self):
        self.processor.addAgent({"Name": "theAgent", "Note": "This should really be an DDQNAgent Object"})
        self.assertEqual(self.processor.agent["Name"], "theAgent")
        self.assertEqual(self.processor.agent["Note"], "This should really be an DDQNAgent Object")

    def test_reset_game(self):
        self.processor.gameScore = 100
        self.processor.resetGame()
        self.assertEqual(self.processor.gameScore, 0)

    def test_play_one_game(self):
        self.processor.isDone = True
        self.processor.playOneGame()
        self.assertEqual(self.processor.gameNumber, 1)
