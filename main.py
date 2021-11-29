import numpy as np


class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def string_input(
        self, outlook="sunny", temperature="hot", humidity="high", wind="strong"
    ):
        training_input = []

        outlook = outlook.lower()
        temperature = temperature.lower()
        humidity = humidity.lower()
        wind = wind.lower()

        training_input.append(1) if outlook == "sunny" else training_input.append(0)
        training_input.append(1) if outlook == "overcast" else training_input.append(0)
        training_input.append(1) if outlook == "rain" else training_input.append(0)

        training_input.append(1) if temperature == "hot" else training_input.append(0)
        training_input.append(1) if temperature == "mild" else training_input.append(0)
        training_input.append(1) if temperature == "cold" else training_input.append(0)

        training_input.append(1) if humidity == "high" else training_input.append(0)
        training_input.append(1) if humidity == "normal" else training_input.append(0)

        training_input.append(1) if wind == "strong" else training_input.append(0)
        training_input.append(1) if wind == "weak" else training_input.append(0)

        return training_input

    def string_output(self, play="yes"):
        play = play.lower()
        return 1 if play == "yes" else 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):

        for _ in range(training_iterations):

            output = self.think(training_inputs)
            error = training_outputs - output

            adjustments = np.dot(
                training_inputs.T, error * self.sigmoid_derivative(output)
            )
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print(neural_network.synaptic_weights)

    training_inputs = np.array(
        [
            neural_network.string_input(
                outlook="sunny", temperature="hot", humidity="high", wind="weak"
            ),
            neural_network.string_input(
                outlook="sunny", temperature="hot", humidity="high", wind="strong"
            ),
            neural_network.string_input(
                outlook="overcast", temperature="hot", humidity="high", wind="weak"
            ),
            neural_network.string_input(
                outlook="rain", temperature="mild", humidity="high", wind="weak"
            ),
            neural_network.string_input(
                outlook="rain", temperature="cold", humidity="normal", wind="weak"
            ),
            neural_network.string_input(
                outlook="rain", temperature="cold", humidity="normal", wind="strong"
            ),
            neural_network.string_input(
                outlook="overcast", temperature="cold", humidity="normal", wind="strong"
            ),
            neural_network.string_input(
                outlook="sunny", temperature="mild", humidity="high", wind="weak"
            ),
            neural_network.string_input(
                outlook="sunny", temperature="cold", humidity="normal", wind="weak"
            ),
            neural_network.string_input(
                outlook="rain", temperature="mild", humidity="normal", wind="weak"
            ),
            neural_network.string_input(
                outlook="sunny", temperature="mild", humidity="normal", wind="strong"
            ),
            neural_network.string_input(
                outlook="overcast", temperature="mild", humidity="high", wind="strong"
            ),
            neural_network.string_input(
                outlook="overcast", temperature="hot", humidity="normal", wind="weak"
            ),
            neural_network.string_input(
                outlook="rain", temperature="mild", humidity="high", wind="strong"
            ),
        ]
    )

    training_outputs = np.array(
        [
            [
                neural_network.string_output("no"),
                neural_network.string_output("no"),
                neural_network.string_output("yes"),
                neural_network.string_output("yes"),
                neural_network.string_output("yes"),
                neural_network.string_output("no"),
                neural_network.string_output("yes"),
                neural_network.string_output("no"),
                neural_network.string_output("yes"),
                neural_network.string_output("yes"),
                neural_network.string_output("yes"),
                neural_network.string_output("yes"),
                neural_network.string_output("yes"),
                neural_network.string_output("no"),
            ]
        ]
    ).T
    
    while(True):
        outlook = input("\nInput Outlook: ")
        temperature = input("Input Temperature: ")
        humidity = input("Input Humidity: ")
        wind = input("Input Wind: ")

        print(f"\nInputs = {outlook}, {temperature}, {humidity}, {wind}")
        print("Play = {0}")

    
