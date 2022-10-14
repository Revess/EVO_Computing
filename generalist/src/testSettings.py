import os
def testSettings(settings):
    assert settings["settings"]["popSize"] > 0
    assert settings["settings"]["popInit"] == "random" or type(settings["settings"]["popInit"]) == type([])
    assert settings["settings"]["cxpb"] >= 0 and settings["settings"]["cxpb"] <= 1
    assert settings["settings"]["mutpb"] >= 0 and settings["settings"]["cxpb"] <= 1
    assert settings["settings"]["tournamentSize"] > 0
    assert settings["settings"]["ngens"] > 0
    assert min(settings["enemies"]) >= 1 and max(settings["enemies"]) <=8
    assert len(settings["enemies"]) <=8