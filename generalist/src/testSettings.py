import os
def testSettings(settings):
    # name = settings["name"].split("_")
    # print(name)
    # assert (len(name) == 4 and int(name[0]) > 0 and int(name[0]) < 3 and int(name[1]) > 0 and int(name[1]) < 3 and (name[2] == "S" or name[2] == "R") and int(name[0]) >= 0) or (len(name) == 3 and name[0] == "S" and int(name[1]) > 0 and int(name[1]) <= 8 and int(name[2]) > 0 )
    assert settings["settings"]["popSize"] > 0
    assert settings["settings"]["popInit"] == "random" or type(settings["settings"]["popInit"]) == type([])
    assert settings["settings"]["cxpb"] >= 0 and settings["settings"]["cxpb"] <= 1
    assert settings["settings"]["mutpb"] >= 0 and settings["settings"]["cxpb"] <= 1
    assert settings["settings"]["selType"] =="tournament" or settings["settings"]["selType"] =="roulette"
    assert settings["settings"]["tournamentSize"] > 0
    assert settings["settings"]["ngens"] > 0
    assert min(settings["enemies"]) >= 1 and max(settings["enemies"]) <=8
    assert len(settings["enemies"]) <=8