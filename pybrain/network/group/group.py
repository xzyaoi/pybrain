import torch
from abc import ABC, abstractmethod

class Group(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def step(self, inputs, mode):
        pass
    
    @property
    def spikes(self):
        return self.spikes
    
    @property
    def voltages(self):
        return self.voltages
    
    @property
    def traces(self):
        return self.traces
    
    