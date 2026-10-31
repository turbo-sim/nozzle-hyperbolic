"""General-EOS annular physics implementing the conservative FVM interface.

The model owns angular-momentum storage, transverse-frame characteristic
information, source physics and coupled boundary laws. Reconstruction,
numerical-flux algebra, area weighting, residual assembly and solution are
provided by the generic engine, never selected here by a reconstruction name.
"""

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.models.euler_annular.boundaries import make_boundary_residual
from nozzle_hyperbolic.models.euler_annular.closures.wall import heat_flux_wall, shear_stress_wall
from nozzle_hyperbolic.models.euler_annular.equations import characteristic_speeds, transverse_flux
from nozzle_hyperbolic.models.euler_annular.equations import physical_flux
from nozzle_hyperbolic.models.euler_annular.flux_jax import conservative_projectors
from nozzle_hyperbolic.models.euler_annular.initialization import uniform_initial_state
from nozzle_hyperbolic.models.euler_annular.state import (
    admissible as admissible_state, primitive_fields, primitive_to_conservative,
    state_property, thermodynamic_state, transverse_conservative,
)
from nozzle_hyperbolic.thermodynamics.adapters import as_jax_fluid



class AnnularModel:
    nvar = 4
    equation_names = ("Mass", "Meridional momentum", "Angular momentum", "Energy")
    name = "euler_annular"
    source_dependency_radius = 0
    boundary_dependency_radius = 2
    boundary_coupling = "local"

    def __init__(self, mesh, native_mesh, fluid, *, inlet_data, outlet_pressure,
                 inlet_regime="subsonic", outlet_regime="subsonic",
                 outlet_formulation="acoustic", initial_state=None, tau_w=0.,
                 q_w=0., wall_closure=None):
        self.mesh, self.native_mesh = mesh, native_mesh
        self.inlet_regime, self.outlet_regime = inlet_regime, outlet_regime
        self.outlet_formulation = outlet_formulation
        self.fluid = as_jax_fluid(fluid)
        self._initial_state = initial_state
        self.inlet_data = dict(inlet_data)
        self.outlet_pressure = float(outlet_pressure)
        if inlet_regime == "supersonic":
            datum = np.asarray(self.inlet_data["state"], dtype=float)
            if datum.shape != (4,) or not np.all(np.isfinite(datum)):
                raise ValueError("Supersonic inlet state must contain four finite primitives")
            self.inlet_data["state"] = datum
            p_ref, h_ref = datum[[0, 3]]
            self.pressure_total = float(self.inlet_data.get("p0", p_ref))
        elif inlet_regime == "subsonic":
            p_ref = float(self.inlet_data["p0"])
            reservoir = self.fluid.get_state(jxp.PT_INPUTS, p_ref, float(self.inlet_data["T0"]))
            self.inlet_data.setdefault("h0", float(state_property(reservoir, "h")))
            self.inlet_data.setdefault("s0", float(state_property(reservoir, "s")))
            h_ref = self.inlet_data["h0"]
            self.pressure_total = p_ref
        else:
            raise ValueError("inlet_regime must be subsonic or supersonic")
        ref_state = self.fluid.get_state(jxp.HmassP_INPUTS, h_ref, p_ref)
        self.reference_density = float(state_property(ref_state, "rho"))
        self.reference_sound_speed = float(state_property(ref_state, "a"))
        self.reference_pressure, self.reference_enthalpy = float(p_ref), float(h_ref)
        temperature = float(state_property(ref_state, "T"))
        rho, sound, radius = (self.reference_density, self.reference_sound_speed,
                              float(native_mesh.r_face[0]))
        self.variable_scales = np.array([p_ref, sound, radius*sound, sound**2])
        self.conservative_scales = rho*np.array([1., sound, radius*sound, sound**2])
        self.reference_time = float(mesh.faces[-1]-mesh.faces[0])/sound
        self.global_flux_scales = float(mesh.area_faces[0])*sound*self.conservative_scales
        self.boundary_scales = dict(p=float(p_ref), a=sound, ell=radius*sound,
                                    h=sound**2, s=sound**2/temperature)
        self._boundary = make_boundary_residual(
            self.fluid, mesh.centers, mesh.faces, native_mesh.r_face,
            self.inlet_data, self.outlet_pressure, self.boundary_scales,
            self.inlet_regime, self.outlet_regime,outlet_formulation=outlet_formulation)
        self.outlet=self._boundary.outlet
        self._r = jnp.asarray(native_mesh.r_center)
        self._r_face = jnp.asarray(native_mesh.r_face)
        self._r_full = jnp.asarray(np.r_[native_mesh.r_face[0], native_mesh.r_center,
                                        native_mesh.r_face[-1]])
        self._radius_prime = jnp.asarray(native_mesh.r_prime)
        self._height = jnp.asarray(native_mesh.b)
        self._volumes = jnp.asarray(mesh.volumes)
        self._pressure_measure = jnp.asarray(mesh.area_derivative_integrals)
        self.tau_wall = self._wall_array(tau_w, nonnegative=True)
        self.heat_wall = self._wall_array(q_w)
        self.wall_closure = wall_closure
        self.source_dependency_radius = int(getattr(wall_closure, "dependency_radius", 0))

    def _wall_array(self, value, *, nonnegative=False):
        values = np.broadcast_to(np.asarray(value, dtype=float), (self.mesh.n_cells,)).copy()
        if not np.all(np.isfinite(values)) or (nonnegative and np.any(values < 0)):
            raise ValueError("Wall inputs must be finite, with nonnegative shear magnitude")
        return jnp.asarray(values)

    def cell_conserved(self, physical):
        return primitive_to_conservative(physical, self._r, self.fluid)

    def face_conserved(self, face_state):
        return self.face_data(face_state)["conserved"]

    def face_physical_flux(self, face_state):
        return self.face_data(face_state)["physical_flux"]

    def face_speed_bounds(self, face_state):
        return self.face_data(face_state)["speed_bounds"]

    def face_data(self, face_state):
        """One EOS call, reusing the canonical physical state/flux helpers."""
        state=thermodynamic_state(face_state,self.fluid)
        rho,sound=(state_property(state,key) for key in ("rho","a"))
        q=transverse_conservative(face_state,self._r_face,self.fluid,rho=rho)
        flux=transverse_flux(face_state,self._r_face,self.fluid,rho=rho)
        velocity=face_state[...,1]
        return dict(conserved=q,physical_flux=flux,
                    speed_bounds=(velocity-sound,velocity+sound),auxiliary=sound)

    def face_characteristics(self, left, right, delta_factor):
        sound_left = state_property(thermodynamic_state(left, self.fluid), "a")
        sound_right = state_property(thermodynamic_state(right, self.fluid), "a")
        return self.face_characteristics_from_data(left,right,delta_factor,sound_left,sound_right)

    def face_characteristics_from_data(self,left,right,delta_factor,sound_left,sound_right):
        """Reuse side sound speeds and evaluate the average EOS only once."""
        average = .5*(left+right)
        mean_state = thermodynamic_state(average, self.fluid)
        rho, sound, gruneisen = (state_property(mean_state, key) for key in ("rho", "a", "G"))
        projectors = conservative_projectors(average, self._r_face, rho, sound, gruneisen)
        eigenvalues = characteristic_speeds(average, sound)
        spread = characteristic_speeds(right, sound_right)-characteristic_speeds(left, sound_left)
        base = delta_factor*sound[..., None]
        positive_spread = .5*(spread+jnp.sqrt(spread**2+base**2))
        return projectors, eigenvalues, base+.5*positive_spread

    def flux_to_conserved_frame(self, face_flux):
        return face_flux.at[..., 2].set(face_flux[..., 2]*self._r_face)

    def integrated_source(self, full):
        physical = full[1:-1]
        rho = state_property(thermodynamic_state(physical, self.fluid), "rho")
        vm, vtheta = physical[..., 1], physical[..., 2]/self._r
        momentum = (physical[..., 0]*self._pressure_measure
                    +self._volumes*rho*vtheta**2*self._radius_prime/self._r)
        speed_squared = vm**2+vtheta**2
        safe_speed = jnp.sqrt(jnp.where(speed_squared > 0., speed_squared, 1.))
        wall_measure = 2*self._volumes/self._height
        shear = wall_measure*shear_stress_wall(physical, self._r, self.fluid,
                                             prescribed=self.tau_wall)
        momentum = momentum-shear*vm/safe_speed
        torque = -self._r*shear*vtheta/safe_speed
        heating = wall_measure*heat_flux_wall(physical, self._r, self.fluid,
                                            prescribed=self.heat_wall)
        return jnp.stack((jnp.zeros_like(momentum), momentum, torque, heating), axis=-1)

    def boundary_residual(self, full, time=0.):
        return self._boundary(full)

    def boundary_face_flux(self,full,flux):
        if self.outlet_formulation!='finite_wave':return flux
        return flux.at[-1].set(jax.lax.cond(self.outlet.direct_flux(full[1:-1]),
            lambda:physical_flux(full[-1],self._r_face[-1],self.fluid),lambda:flux[-1]))

    def boundary_wave_speed(self,full):
        traces=full[jnp.array([0,-1])]
        speeds=jnp.abs(traces[:,1])+state_property(thermodynamic_state(traces,self.fluid),'a')
        return speeds.at[-1].set(jnp.maximum(speeds[-1],self.outlet.signal_speed(full[1:-1])))

    def boundary_diagnostics(self,full):
        if self.outlet_formulation=='finite_wave':
            wave=self.outlet.wave(full[1:-1])
            return dict(mode=wave.mode,direct_flux=wave.direct_flux,
                        head_speed=wave.head_speed,tail_speed=wave.tail_speed)
        return dict(mode=jnp.int32(0),direct_flux=jnp.asarray(False))

    def admissible(self, full, left, right):
        basic = (jnp.all(jnp.isfinite(full)) & jnp.all(full[..., 0] > 0)
                 & jnp.all(jnp.isfinite(left))
                 & jnp.all(jnp.isfinite(right)) & jnp.all(left[..., 0] > 0)
                 & jnp.all(right[..., 0] > 0))
        return jax.lax.cond(basic,
            lambda: (jnp.all(admissible_state(full, self._r_full, self.fluid))
                     & jnp.all(admissible_state(left, self._r_face, self.fluid))
                     & jnp.all(admissible_state(right, self._r_face, self.fluid))),
            lambda: jnp.asarray(False))

    def wave_speed(self, physical):
        return jnp.abs(physical[..., 1])+state_property(thermodynamic_state(physical, self.fluid), "a")

    def cell_admissible(self,cells):
        return jnp.all(cells[...,0]>0)&jnp.all(admissible_state(cells,self._r,self.fluid))

    def boundary_admissible(self,full):
        return jnp.all(full[...,0]>0)&jnp.all(admissible_state(full,self._r_full,self.fluid))

    def initial_state(self):
        if self._initial_state is not None:
            values=self._initial_state(self.mesh.points) if callable(self._initial_state) else self._initial_state
            values=jnp.asarray(values)
            if values.shape != (self.mesh.n_cells+2,self.nvar):
                raise ValueError("initial_state must provide the full (N+2,4) primitive state")
            return values
        if self.inlet_regime == "supersonic":
            datum = self.inlet_data["state"]
            return uniform_initial_state(self._r_full, datum[0], datum[3], datum[1],
                                         v_theta=datum[2]/float(self._r_face[0]))
        vm = .05*self.reference_sound_speed
        vtheta = (vm*np.tan(self.inlet_data["alpha_in"]) if "alpha_in" in self.inlet_data
                  else float(self.inlet_data["ell_in"])/float(self._r_face[0]))
        enthalpy = self.inlet_data["h0"]-.5*(vm**2+vtheta**2)
        return uniform_initial_state(self._r_full, self.outlet_pressure, enthalpy, vm,
                                     v_theta=vtheta)

    def fields(self, full):
        values = primitive_fields(full[1:-1], self._r, self.fluid)
        values["alpha_deg"] = jnp.rad2deg(jnp.arctan2(values["v_theta"], values["v_m"]))
        values["mdot"] = values["rho"]*values["v_m"]*jnp.asarray(self.mesh.area_centers)
        inlet_pressure = full[0, 0]
        values["Cp"] = (values["p"]-inlet_pressure)/(self.pressure_total-inlet_pressure)
        return values

    def quantities(self, full, face_flux):
        """Model-specific scalar monitors; the generic driver owns their history."""
        values = self.fields(full)
        length = jnp.sum(jnp.asarray(self.mesh.widths))
        return dict(mdot=face_flux[-1, 0],
                    Cp_mean=jnp.sum(jnp.asarray(self.mesh.widths)*values["Cp"])/length)

    def regime_metrics(self, full):
        boundary = full[jnp.array([0, -1])]
        mach = boundary[..., 1]/state_property(thermodynamic_state(boundary, self.fluid), "a")
        return dict(boundary_mach=mach)

    def regimes_match(self, full, *, sonic_tolerance=2e-3):
        if np.min(np.asarray(full)[1:-1,1]) < -1e-12*self.reference_sound_speed:return False
        mach = np.asarray(self.regime_metrics(full)["boundary_mach"])
        return all(value >= 0 and (True if index==1 and self.outlet_formulation=='finite_wave' else
                                   value <= 1+sonic_tolerance if regime == "subsonic"
                                   else value >= 1-sonic_tolerance)
                   for index,(value, regime) in enumerate(zip(mach, (self.inlet_regime, self.outlet_regime))))

    solution_audit = regimes_match

