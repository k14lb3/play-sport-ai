import numpy as np
import mediapipe as mp


class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)

        self._synaptic_weights = 2 * np.random.random((10, 1)) - 1

    def get_attributes(
        self, outlook="sunny", temperature="hot", humidity="high", wind="strong"
    ):
        """Converts given string attributes into binary.

        Arguments:
            outlook: Whether sunny, overcast, or rain.
            temperature: Whether hot, mild, or cold.
            humidity: Whether high, or normal.
            wind: Whether strong, or weak.

        Return:
            List of the given parameters in binary. (e.g. outlook="sunny" => 1 0 0, temperature="hot" => 1 0 0, humidity="high" => 1 0, wind="strong" => 1 0)
        """
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

    def can_play(self, play="yes"):
        """Converts given string into binary.

        Arguments:
            play: Wether can play or not.

        Return:
            Binary of the given answer. (1 if yes, and 2 if no)
        """

        return 1 if play.lower() == "yes" else 0

    def sigmoid(self, x):
        """Calculates using the sigmoid function from a given value x.

        Arguments:
            x: Given value.

        Return:
            Solution.
        """

        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Calculates the derivative of the sigmoid function.

        Arguments:
            x : Solution from the calculation using the sigmoid function from its given value x.

        Return:
            Derivative of the sigmoid function.
        """

        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        """Trains the neural network.

        Arguments:
            training_inputs: Training inputs.
            training_outputs: Training outputs.
            training_iterations: Number of iterations for the training of the neural network.
        """

        for _ in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output

            adjustments = np.dot(
                training_inputs.T, error * self.sigmoid_derivative(output)
            )

            self._synaptic_weights += adjustments

    def think(self, inputs):
        """Predicts the output/s from the given input/s.

        Arguments:
            inputs:  Given input/s.

        Return:
            Predicted output/s.
        """

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self._synaptic_weights))

        return output

    def get_synaptic_weights(self):
        """Returns the synaptic weights.

        Return:
            Synaptic weights.
        """

        return self._synaptic_weights


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("Random starting weights: ")
    print(neural_network.get_synaptic_weights())

    training_inputs = np.array(
        [
            neural_network.get_attributes(
                outlook="sunny", temperature="hot", humidity="high", wind="weak"
            ),
            neural_network.get_attributes(
                outlook="sunny", temperature="hot", humidity="high", wind="strong"
            ),
            neural_network.get_attributes(
                outlook="overcast", temperature="hot", humidity="high", wind="weak"
            ),
            neural_network.get_attributes(
                outlook="rain", temperature="mild", humidity="high", wind="weak"
            ),
            neural_network.get_attributes(
                outlook="rain", temperature="cold", humidity="normal", wind="weak"
            ),
            neural_network.get_attributes(
                outlook="rain", temperature="cold", humidity="normal", wind="strong"
            ),
            neural_network.get_attributes(
                outlook="overcast", temperature="cold", humidity="normal", wind="strong"
            ),
            neural_network.get_attributes(
                outlook="sunny", temperature="mild", humidity="high", wind="weak"
            ),
            neural_network.get_attributes(
                outlook="sunny", temperature="cold", humidity="normal", wind="weak"
            ),
            neural_network.get_attributes(
                outlook="rain", temperature="mild", humidity="normal", wind="weak"
            ),
            neural_network.get_attributes(
                outlook="sunny", temperature="mild", humidity="normal", wind="strong"
            ),
            neural_network.get_attributes(
                outlook="overcast", temperature="mild", humidity="high", wind="strong"
            ),
            neural_network.get_attributes(
                outlook="overcast", temperature="hot", humidity="normal", wind="weak"
            ),
            neural_network.get_attributes(
                outlook="rain", temperature="mild", humidity="high", wind="strong"
            ),
        ]
    )

    training_outputs = np.array(
        [
            [
                neural_network.can_play("no"),
                neural_network.can_play("no"),
                neural_network.can_play("yes"),
                neural_network.can_play("yes"),
                neural_network.can_play("yes"),
                neural_network.can_play("no"),
                neural_network.can_play("yes"),
                neural_network.can_play("no"),
                neural_network.can_play("yes"),
                neural_network.can_play("yes"),
                neural_network.can_play("yes"),
                neural_network.can_play("yes"),
                neural_network.can_play("yes"),
                neural_network.can_play("no"),
            ]
        ]
    ).T

    neural_network.train(
        training_inputs=training_inputs,
        training_outputs=training_outputs,
        training_iterations=100000,
    )

    print("\nWeights after training: ")
    print(neural_network.get_synaptic_weights())

    while True:
        outlook = input("\nInput Outlook [Sunny / Overcast / Rain]\n→ ")
        temperature = input("Input Temperature [Hot / Mild / Cold]\n→ ")
        humidity = input("Input Humidity [High / Normal]\n→ ")
        wind = input("Input Wind [Weak / Strong]\n→ ")

        inputs = np.array(
            neural_network.get_attributes(
                outlook=outlook,
                humidity=humidity,
                temperature=temperature,
                wind=wind,
            )
        )

        print(f"\nInputs = {outlook}, {temperature}, {humidity}, {wind}")

        output = neural_network.think(inputs=inputs)

        output_word = "Yes" if neural_network.think(inputs=inputs) > 0.5 else "No"

        print(f"Play = {output_word}")
        print(f"Numerical output = {output}")

        while True:
            repeat = input("\nDo you want try again? [y / n] : ").lower()
            if repeat == "y":
                break
            elif repeat == "n":
                exit()
