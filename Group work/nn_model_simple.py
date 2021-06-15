from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F

from chemprop.nn_utils import get_activation_function, initialize_weights


class NeuralNetwork(nn.Module):
    """
    linear neural network
    """
    def __init__(self):
        super(Model, self).__init__()


    def create_ffn(self, hidden_size, dropout, num_layers, activation):
        """
        creates the ffed forward network for the model
        :param hidden_size:
        :param dropout:
        :param num_layers:
        :param activation:
        :return:
        """
        first_linear_dim = 23
        output_size = 1

        if num_layers >= 2:


        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, output_size)
            ]
            for _ in range(num_layers - 3):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(hidden_size, hidden_size),
                ])

            ffn.extend([
                activation,
                dropout,
                nn.Linear(hidden_size, hidden_size),
            ])

            ffn.extend([
                activation,
                dropout,
                nn.Linear(hidden_size, output_size),
            ])
        # Create FFN model
        self.ffn = nn.Sequential(*ffn)



    def forward(self, *input):
        """
        Runs the MoleculeModel on input.
        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        output = self.ffn(*input)

        return output


def build_model(hidden_size : int, dropout : float, num_layers : int, activation : F) -> nn.Module:
    """
    Builds a NeuralNetwork, which is a message passing neural network + feed-forward layers.
    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """

    model = NeuralNetwork()
    model.create_ffn(hidden_size, dropout, num_layers, activation)

    initialize_weights(model)

    return model


class EvaluationDropout(nn.Dropout):
    def forward(self, input):
        return nn.functional.dropout(input, p = self.p)