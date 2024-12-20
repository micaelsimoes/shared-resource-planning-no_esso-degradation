# ======================================================================================================================
#   Class SharedEnergyStorage
# ======================================================================================================================
class SharedEnergyStorage:

    def __init__(self):
        self.bus = -1                       # bus number (positive integer)
        self.dn_name = str()                # Distribution network name (used as identifier)
        self.s = 0.00                       # Apparent power, [MVA]
        self.e = 0.00                       # Capacity (energy), [MVAh]
        self.e_init = 0.00                  # Initial energy stored, [MVAh]
        self.e_min = 0.00                   # Minimum energy stored, [MVAh]
        self.e_max = 0.00                   # Maximum energy stored, [MVAh]
        self.eff_ch = 1.00                  # Charging efficiency, [0-1]
        self.eff_dch = 0.95                 # Discharging efficiency, [0-1]
        self.max_pf = 0.80                  # Maximum power factor
        self.min_pf = -0.80                 # Minimum power factor
        self.t_cal = 15                     # Calendar life of the ESS, [years]
        self.cl_nom = 10000                 # Cycle life, nominal, [number of cycles]
        self.dod_nom = 0.90                 # Depth-of-Discharge, nominal, [0-1]
        self.soh_min = 0.10                 # Minimum SoH, [0-1]
