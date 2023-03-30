import random


class MockQ:
    def getAction(self, observation):
        return random.randrange(4)

    def updateQValues(self, reward):
        pass

    def endEpisode(self):
        pass
