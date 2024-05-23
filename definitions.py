ERROR_SPECIFICATION_FILE = -1
ERROR_MARKET_DATA_FILE = -2
ERROR_PARAMS_FILE = -3
ERROR_NETWORK_FILE = -4
ERROR_OPERATIONAL_DATA_FILE = -5
ERROR_NETWORK_MODEL = -6
ERROR_NETWORK_OPTIMIZATION = -7

BUS_PQ = 1
BUS_PV = 2
BUS_REF = 3
BUS_ISOLATED = 4

GEN_REFERENCE = 0
GEN_CONV = 1
GEN_RES_WIND = 2
GEN_RES_SOLAR = 3
GEN_RES_OTHER = 4
GEN_RES_CONTROLLABLE = 5
GEN_INTERCONNECTION = 6
GEN_CONTROLLABLE_TYPES = [GEN_REFERENCE, GEN_CONV, GEN_RES_CONTROLLABLE]
GEN_CURTAILLABLE_TYPES = [GEN_RES_WIND, GEN_RES_SOLAR, GEN_RES_OTHER, GEN_INTERCONNECTION]
GEN_RENEWABLE_TYPES = [GEN_RES_WIND, GEN_RES_SOLAR, GEN_RES_OTHER, GEN_RES_CONTROLLABLE]

BRANCH_UNKNOWN_RATING = 999.99
TRANSFORMER_MAXIMUM_RATIO = 1.17
TRANSFORMER_MINIMUM_RATIO = 0.83

ENERGY_STORAGE_MAX_ENERGY_STORED = 0.95
ENERGY_STORAGE_MIN_ENERGY_STORED = 0.05
ENERGY_STORAGE_RELATIVE_INIT_SOC = 0.50

OBJ_MIN_COST = 1
OBJ_CONGESTION_MANAGEMENT = 2

COST_GENERATION_CURTAILMENT = 50.00
COST_CONSUMPTION_CURTAILMENT = 250.00
PENALTY_SLACK = 1e3
PENALTY_VOLTAGE = 1e4
PENALTY_CURRENT = 1e3
PENALTY_SHARED_ESS = 1e6
PENALTY_GENERATION_CURTAILMENT = 1e3
PENALTY_LOAD_CURTAILMENT = 1e6

DATA_ACTIVE_POWER = 1
DATA_REACTIVE_POWER = 2
DATA_UPWARD_FLEXIBILITY = 3
DATA_DOWNWARD_FLEXIBILITY = 4
DATA_COST_FLEXIBILITY = 5

EQUALITY_TOLERANCE = 1e-6
SMALL_TOLERANCE = 1e-4
ADMM_ADAPTIVE_PENALTY_FACTOR = 0.05
ADMM_CONVERGENCE_REL_TOL = 10e-2
ERROR_PRECISION = 6     # Nth decimal place
