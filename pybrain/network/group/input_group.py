import torch
from .group import Group


class InputGroup(Group):
    def __init__(self, n_neurons, traces=False, trace_tc=5e-2):
        super().__init__()
        self.n_neurons = n_neurons
        self.traces = traces
        self.spikes = torch.zeros_like(torch.Tensor(n_neurons))

        if self.traces:
            self.x = torch.zeros_like(torch.Tensor(n_neurons))  # Firing traces
            self.trace_tc = trace_tc

    def step(self, inputs, mode, dt):
        if self.traces:
            self.x -= dt * self.trace_tc * self.x

        self.s = inputs

        if self.traces:
            self.x[self.s] = 1.0
