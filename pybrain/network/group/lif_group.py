import torch
from .group import Group


class LIFGroup(Group):

    def __init__(self, n_neurons,
                 traces=False,
                 rest=-65.0,
                 reset=-65.0,
                 threshold=-52.0,
                 refractory=5,
                 voltage_decay=1e-2,
                 trace_tc=5e-2):

        super().__init__()
        self.n_neurons = n_neurons
        self.traces = traces
        self.rest = rest
        self.reset = reset
        self.threshold = threshold
        self.refractory = refractory
        self.voltage_decay = voltage_decay

        self.v = self.rest * torch.ones_like(torch.Tensor(n_neurons))
        self.spikes = torch.zeros_like(torch.Tensor(n_neurons))

        if traces:
            self.x = torch.zeros_like(torch.Tensor(n_neurons))
            self.trace_tc = trace_tc

        self.refrac_count = torch.zeros_like(torch.Tensor(n_neurons))

    def step(self, inputs, mode, dt):

        self.v -= dt * self.voltage_decay * (self.v - self.rest)

        if self.traces:
            self.x -= dt * self.trace_tc * self.x

        self.refrac_count[self.refrac_count != 0] -= dt

        self.spikes = (self.v >= self.threshold) * (self.refrac_count == 0)
        self.refrac_count[self.spikes] = self.refractory
        self.v[self.spikes] = self.reset

        self.v += inputs
        if self.traces:
            self.x[self.spikes] = 1.0


class AdaptiveLIFGroup(Group):
    '''
    Group of leaky integrate-and-fire neurons with adaptive thresholds.
    '''

    def __init__(self, n_neurons,
                 traces=False,
                 rest=-65.0,
                 reset=-65.0,
                 threshold=-52.0,
                 refractory=5,
                 voltage_decay=1e-2,
                 theta_plus=0.05,
                 theta_decay=1e-7,
                 trace_tc=5e-2):

        super().__init__()

        self.n_neurons = n_neurons
        self.traces = traces
        self.rest = rest
        self.reset = reset
        self.threshold = threshold
        self.refractory = refractory
        self.voltage_decay = voltage_decay
        self.theta_plus = theta_plus
        self.theta_decay = theta_decay

        # Neuron voltages.
        self.v = self.rest * torch.ones_like(torch.Tensor(n_neurons))
        self.s = torch.zeros_like(torch.Tensor(n_neurons))  # Spike occurences.
        # Adaptive threshold parameters.
        self.theta = torch.zeros_like(torch.Tensor(n_neurons))

        if traces:
            # Firing traces.
            self.x = torch.zeros_like(torch.Tensor(n_neurons))
            # Rate of decay of spike trace time constant.
            self.trace_tc = trace_tc

        # Refractory period counters.
        self.refrac_count = torch.zeros_like(torch.Tensor(n_neurons))

    def step(self, inpts, mode, dt):
        # Decay voltages.
        self.v -= dt * self.voltage_decay * (self.v - self.rest)

        if self.traces:
            # Decay spike traces and adaptive thresholds.
            self.x -= dt * self.trace_tc * self.x
            self.theta -= dt * self.theta_decay * self.theta

        # Decrement refractory counters.
        self.refrac_count -= dt

        # Check for spiking neurons.
        self.s = (self.v >= self.threshold + self.theta) * \
            (self.refrac_count <= 0)
        self.refrac_count[self.s] = dt * self.refractory
        self.v[self.s] = self.reset

        # Choose only a single neuron to spike (ETH replication).
        if torch.sum(self.s) > 0:
            s = torch.zeros_like(torch.Tensor(self.s.size()))
            s[torch.multinomial(self.s.float(), 1)] = 1

        # Integrate inputs.
        self.v += inpts

        if self.traces:
            # Update adaptive thresholds, synaptic traces.
            self.theta[self.s] += self.theta_plus
            self.x[self.s] = 1.0
