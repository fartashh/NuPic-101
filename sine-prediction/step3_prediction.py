import csv
from nupic.frameworks.opf.model_factory import ModelFactory
from nupic_output import NuPICFileOutput, NuPICPlotOutput
from nupic.swarming import permutations_runner


def predict():
    from model_0 import model_params
    input_file = "sine.csv"
    model = ModelFactory.create(model_params.MODEL_PARAMS)
    model.enableInference({"predictedField": "sine"})

    output = NuPICFileOutput("sine_output", show_anomaly_score=True)

    with open(input_file, "rb") as sine_input:
        csv_reader = csv.reader(sine_input)

        # skip header rows
        csv_reader.next()
        csv_reader.next()
        csv_reader.next()

        # the real data
        for row in csv_reader:
            angle = float(row[0])
            sine_value = float(row[1])
            result = model.run({"sine": sine_value})
            output.write(angle, sine_value, result, prediction_step=1)
    output.close()

if __name__ == "__main__":
    predict()