import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): the size of the input vectors
            hidden_dim (int): the output size of the first Linear layer
            output_dim (int): the output size of the second Linear layer
        """
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the MLP

        Args:
            x_in (torch.Tensor): an input data tensor
                x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim).
        """
        intermediate = F.relu(self.fc1(x_in))
        #use dropout here
        output = self.fc2(F.dropout(intermediate, p=0.5)).squeeze()

        #the softmax function is optionally applied to make sure the outputs sum to 1; that is, are interpreted as “probabilities.” The reason it is optional has to do with the mathematical formulation of the loss function we use—the cross-entropy loss, introduced in “Loss Functions”. Recall that cross-entropy loss is most desirable for multiclass classification, but computation of the softmax during training is not only wasteful but also not numerically stable in many situations.
        if apply_softmax:
            output = F.softmax(output, dim=1)
        return output