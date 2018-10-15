import yaml
from nupic.frameworks.opf.model_factory import ModelFactory
import os
from itertools import islice
import csv
import datetime

__PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
__PARAM_PATH = os.path.join(__PROJECT_DIR, 'params.yml')
__INPUT_FILE = os.path.join(__PROJECT_DIR,'gym.csv')


def create_mode():
    with open(__PARAM_PATH, "r") as f:
        model_params = yaml.safe_load(f)
    return ModelFactory.create(model_params)


def runHotGym(numRecords):
    model = create_mode()
    model.enableInference({"predictedField":"consumption"})
    with open(__INPUT_FILE,"r") as f:
        reader = csv.reader(f)
        headers = reader.next()
        reader.next()
        reader.next()

        results = []
        for record in islice(reader, numRecords):
            modelInput = dict(zip(headers,record))
            modelInput["consumption"] = float(modelInput["consumption"])
            modelInput["timestamp"] = datetime.datetime.strptime(
                modelInput["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
            del modelInput['attendeeCount']

            result = model.run(modelInput)
            bestPredictions = result.inferences["multiStepBestPredictions"]
            allPredictions = result.inferences["multiStepPredictions"]
            oneStep = bestPredictions[1]
            oneStepConfidence = allPredictions[1][oneStep]
            fiveStep = bestPredictions[5]
            fiveStepConfidence = allPredictions[5][fiveStep]

            result = (oneStep, oneStepConfidence * 100,
                      fiveStep, fiveStepConfidence * 100)
            print "1-step: {:16} ({:4.4}%)\t 5-step: {:16} ({:4.4}%)".format(*result)
            results.append(result)

if __name__ == "__main__":
    runHotGym(3000)