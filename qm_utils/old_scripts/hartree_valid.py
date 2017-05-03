
# Hartree Columns #

FILE_COLNAME = "File Name"
SOLVENT_COLNAME = "Solvent type"
STOI_COLNAME = "Stoichiometry"
CHARGE_COLNAME = "Charge"
MULT_COLNAME = "Mult"
FUNC_COLNAME = "Functional"
BASIS_COLNAME = "Basis Set"
ENERGY_COLNAME = "Energy (A.U.)"
DIPOLE_COLNAME = "dipole"
ZPE_COLNAME = "ZPE (kcal/mol)"
FREQ1_COLNAME = "Freq 1"
FREQ2_COLNAME = "Freq 2"

# Validation #

# Columns that should have the same value throughout a data set
SAME_COLS = [SOLVENT_COLNAME, STOI_COLNAME, CHARGE_COLNAME, MULT_COLNAME,
             FUNC_COLNAME, BASIS_COLNAME]


def verify_same_cols(pandas_dframe):
    errs = {}
    for col in SAME_COLS:
        unique = pandas_dframe[col].unique()
        if (len(unique) != 1):
            errs[col] = unique
    return errs


def verify_local_minimum(pandas_dframe):
    for col in [FREQ1_COLNAME, FREQ2_COLNAME]:
        if len(pandas_dframe[pandas_dframe[col] <= 0]) > 0:
            return False
    return True


def verify_transition_state(pandas_dframe):
    if len(pandas_dframe[pandas_dframe[FREQ1_COLNAME] >= 0]) > 0:
        return False
    if len(pandas_dframe[pandas_dframe[FREQ2_COLNAME] <= 0]) > 0:
        return False
    return True
