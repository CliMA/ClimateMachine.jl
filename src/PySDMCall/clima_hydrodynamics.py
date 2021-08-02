
"""
Class for utilizing ClimateMachine's hydrodynamics
"""


class ClimateMachine:

    def __init__(self, clima_fields):
        self.fields = {}
        self.fields['qv'] = clima_fields['q_v']
        self.fields['th'] = clima_fields['th']
        self.core = None

    def register(self, builder):
        self.core = builder.core

    def __call__(self):
        self.core.env.get_predicted('qv').download(self.core.env.get_qv(), reshape=True)
        self.core.env.get_predicted('thd').download(self.core.env.get_thd(), reshape=True)

    def set_qv(self, qv):
        self.fields['qv'] = qv

    def set_th(self, th):
        self.fields['th'] = th
