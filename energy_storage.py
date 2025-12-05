# ============================================================================================
#   Class EnergyStorage
# ============================================================================================
class EnergyStorage:

    def __init__(self):
        self.es_id = -1         # bus number (positive integer)
        self.bus = -1           # bus number (positive integer)
        self.s = 0.0            # Apparent power, [MVA]
        self.e = 0.0            # Capacity (energy), [MWh]
        self.e_init = 0.0       # Initial energy stored, [MWh]
        self.e_min = 0.0        # Minimum energy stored, [MWh]
        self.e_max = 0.0        # Maximum energy stored, [MWh]
        self.eff_ch = 0.97      # Charging efficiency, [0-1]
        self.eff_dch = 0.96     # Discharging efficiency, [0-1]
        self.max_pf = 0.90      # Maximum power factor
        self.min_pf = -0.90     # Minimum power factor
