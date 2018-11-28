import torch

from .group.group import Group
from ..neuron.synapse import Synapse


class Network(object):
    def __init__(self, dt=1):
        self.dt = dt
        self.groups = {}
        self.synapses = {}
        self.monitors = {}

    def add_group(self, group, name):
        self.groups[name] = group

    def add_synapse(self, synapse, source, target):
        self.synapses[(source, target)] = synapse

    def add_monitor(self, monitor, name):
        self.monitors[name] = monitor

    @property
    def inputs(self):
        inputs = {}
        for key in self.synapses:
            weights = self.synapses[key].w

            source = self.synapses[key].source
            target = self.synapses[key].target

            if not key[1] in inputs:
                inputs[key[1]] = torch.zeros_like(torch.Tensor(target.n))

            inputs[key[1]] += source.s.float() @ weights

        return inputs

    def get_weights(self, name):
        return self.synapses[name].weights
    
    def get_theta(self, name):
        return self.groups[name].theta
