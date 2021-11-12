from PySDM.environments._moist import _Moist
from PySDM.state.mesh import Mesh
from PySDM.state import arakawa_c
import numpy as np
from PySDM.initialisation.r_wet_init import r_wet_init, default_rtol
from PySDM.initialisation.multiplicities import discretise_n

"""
    Kinematic2DMachine(_Moist)

PySDM's Kinematic 2D environment coupled with ClimateMachine.    
"""
class Kinematic2DMachine(_Moist):

    def __init__(self, dt, grid, size, clima_rhod):
        super().__init__(dt, Mesh(grid, size), [])

        self.clima_rhod = clima_rhod
        self.fields = {}

    def register(self, builder):
        super().register(builder)
        self.formulae = builder.particulator.formulae
        rhod = builder.particulator.Storage.from_ndarray(self.clima_rhod.ravel())
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod

    @property
    def dv(self):
        return self.mesh.dv

    def init_attributes(self, *,
                        spatial_discretisation,
                        spectral_discretisation,
                        kappa,
                        rtol=default_rtol
                        ):
        
        super().sync()
        self.notify()

        attributes = {}
        with np.errstate(all='raise'):
            positions = spatial_discretisation.sample(self.mesh.grid, self.particulator.n_sd)
            attributes['cell id'], attributes['cell origin'], attributes['position in cell'] = \
                self.mesh.cellular_attributes(positions)
            r_dry, n_per_kg = spectral_discretisation.sample(self.particulator.n_sd)
            attributes['dry volume'] = self.formulae.trivia.volume(radius=r_dry)
            attributes['kappa times dry volume'] = kappa*attributes['dry volume']
            r_wet = r_wet_init(r_dry, self, kappa_times_dry_volume=kappa*attributes['dry volume'], rtol=rtol, cell_id=attributes['cell id'])
            rhod = self['rhod'].to_ndarray()
            cell_id = attributes['cell id']
            domain_volume = np.prod(np.array(self.mesh.size))

        attributes['n'] = discretise_n(n_per_kg * rhod[cell_id] * domain_volume)
        attributes['volume'] = self.formulae.trivia.volume(radius=r_wet)

        return attributes

    def get_thd(self):
        return self.fields['thd']

    def get_qv(self):
        return self.fields['qv']

    def set_thd(self, thd):
        self.fields['thd'] = thd

    def set_qv(self, qv):
        self.fields['qv'] = qv    

    def sync(self):
        super().sync()