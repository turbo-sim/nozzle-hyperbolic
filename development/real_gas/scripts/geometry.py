"""Area law for the CO2 nozzle benchmark."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Nozzle:
    x_in: float = -0.03201
    x_out: float = 0.06636
    throat_radius: float = 0.0015
    inlet_ratio: float = 17.1
    exit_ratio: float = 1.69

    @property
    def At(self):
        return np.pi * self.throat_radius**2

    def area(self, x):
        x = np.asarray(x)
        return self.At * np.where(x < 0, 1 + (self.inlet_ratio - 1) * (x / self.x_in)**2,
                                 1 + (self.exit_ratio - 1) * (x / self.x_out)**2)

    def derivative(self, x):
        x = np.asarray(x)
        return self.At * np.where(x < 0, 2 * (self.inlet_ratio - 1) * x / self.x_in**2,
                                 2 * (self.exit_ratio - 1) * x / self.x_out**2)
