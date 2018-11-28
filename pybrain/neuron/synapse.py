import torch


class Synapse(object):
    def __init__(self, source, target, w=None):
        self.source = source
        self.target = target
        if w is None:
            self.w = torch.rand(source.n, target.n)
        else:
            self.w = w

    @property
    def weight(self):
        return self.w

    @weight.setter
    def weight(self, w):
        self.w = w

    @property
    def source(self):
        return self.source

    @source.setter
    def source(self, source):
        self.source = source

    @property
    def target(self):
        return self.target

    @target.setter
    def target(self, target):
        self.target = target


class STDPSynapse(Synapse):
    def __init__(self, source, target, w=None, nu_pre=1e-4, nu_post=1e-2, w_max=1.0, norm=78.0):
        super(STDPSynapse, source, target, w).__init__()
        self.nu_pre = nu_pre
        self.nu_post = nu_post
        self.w_max = w_max
        self.norm = norm

    def normalize(self):
        self.w *= self.norm / self.w.sum(0).view(1, -1)

    def update(self):
        '''
        STDP Weight Update
        '''
        self.w += self.nu_post * \
            (self.source.x.view(self.source.n, 1) *
             self.target.s.float().view(1, self.target.n))
        # Pre-synaptic.
        self.w -= self.nu_pre * \
            (self.source.s.float().view(self.source.n, 1)
             * self.target.x.view(1, self.target.n))

        # Ensure that weights are within [0, self.wmax].
        self.w = torch.clamp(self.w, 0, self.w_max)
