from nupic.swarming import permutations_runner

SWARM_CONFIG = {
    "includedFields": [
        {
            "fieldName": "sine",
            "fieldType": "float",
            "maxValue": 1.0,
            "minValue": -1.0
        }
    ],
    "streamDef": {
        "info": "sine",
        "version": 1,
        "streams": [
            {
                "info": "sine.csv",
                "source": "file://sine.csv",
                "columns": [
                    "*"
                ]
            }
        ]
    },
    "inferenceType": "TemporalAnomaly",
    "inferenceArgs": {
        "predictionSteps": [
            1
        ],
        "predictedField": "sine"
    },
    "swarmSize": "medium"
}


def swarm_over_data():
    permutations_runner.runWithConfig(SWARM_CONFIG,
                                      {'maxWorkers': 8, 'overwrite': True})


if __name__ == "__main__":
    swarm_over_data()
