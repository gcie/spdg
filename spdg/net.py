from collections import defaultdict

import numpy
import torch


class Net(torch.nn.Module):
    """Neural net with dual parameters"""

    def __init__(self, ngram, architecture=None, output_size=10, device='cpu'):
        super(Net, self).__init__()

        self.ngram = ngram
        self.n = ngram.n
        self.cnt = 1
        self.output_size = output_size
        self.device = device

        for idx in ngram:
            ngram[idx]

        if architecture is None:
            self.primal = torch.nn.Sequential(
                torch.nn.Linear(28*28, 300),
                torch.nn.ReLU(),
                torch.nn.Linear(300, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, output_size)
            ).to(self.device)
        else:
            self.primal = architecture.to(self.device)

        self.dual = defaultdict(int)

        for idx in self.ngram:
            self.dual[idx] = torch.tensor(0.).uniform_(-1, 0).to(self.device).requires_grad_()

        self.to(self.device)
        self.init_weights()

    def forward_sequences(self, x):
        """Forward sequences through a model.

        Parameters:
        - x: batch_size x sequence_length (ngram.n) x net_input_size (784) tensor

        Returns:
        - output: batch_size x sequence_length (ngram.n) x output_size tensor
        """
        return torch.nn.functional.softmax(self.primal.forward(x), dim=2)

    def forward(self, x):
        """Forward data through a model.

        Parameters:
        - x: batch_size x net_input_size (784) tensor

        Returns:
        - output: batch_size x output_size tensor
        """
        return torch.nn.functional.softmax(self.primal.forward(x), dim=0)

    def loss(self, output, target):
        return torch.mean(-torch.log(torch.gather(output, 1, target.unsqueeze(1))))

    def loss_primal(self, output):
        """Eval primal loss on given output"""
        loss = torch.tensor(0, dtype=torch.float32).to(self.device)
        for i in self.ngram:
            loss += torch.sum(output[:, numpy.arange(self.n), i].prod(dim=1) * self.dual[i] * self.ngram[i])
            # use torch.nn.functional.conv1d ?
        return loss / output.shape[0]

    def loss_dual(self, output):
        """Eval dual loss on given output"""
        loss = self.loss_primal(output)
        for i in self.ngram:
            loss += torch.log(-self.dual[i]) * self.ngram[i]
        return -loss

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.startswith('primal') and name.endswith('weight'):
                with torch.no_grad():
                    param.data.uniform_(-1.0/28,  1.0/28)
            if name.startswith('primal') and name.endswith('bias'):
                with torch.no_grad():
                    param.zero_()
