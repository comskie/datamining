import numpy as np
import pandas as pd
from enum import Enum


class Parameter(Enum):
    GENDER = 'gender'
    PROGRAM = 'program'
    STRAND = 'strand'
    TESDA = 'tesda'
    SCHOLAR = 'scholar'
    GWA = 'gwa'
    RESOURCES = 'resources'
    ABSENCES = 'absences'
    EXPERIENCE = 'experience'
    ACTIVE = 'active'
    TUITION = 'tuition'
    SATISFACTION = 'satisfaction'


SHIFT = 'shift'


_dataset = pd.read_csv('dataset.csv', index_col=False, keep_default_na=False)

_shift_yes = sum(_dataset[SHIFT] == 'yes')
_shift_no = sum(_dataset[SHIFT] == 'no')

_shift_yes_percent = np.divide(_shift_yes, np.add(_shift_yes, _shift_no))
_shift_no_percent = np.divide(_shift_no, np.add(_shift_yes, _shift_no))

_shifts = {}

for parameter in Parameter:
    parameter_values = _dataset[parameter.value].unique()

    _shifts[parameter.value] = {}

    for param_value in parameter_values:
        _shifts[parameter.value][param_value] = {
            'yes': np.divide((
                (_dataset[parameter.value] == param_value) & (
                    _dataset[SHIFT] == 'yes')
            ).sum(), _shift_yes),
            'no': np.divide((
                (_dataset[parameter.value] == param_value) & (
                    _dataset[SHIFT] == 'no')
            ).sum(), _shift_no)
        }

# Returns tuple percentage of shift yes and shift no
def predict_naive_bayes(data: dict):

    yes_values = [_shifts[parameter.value][data[parameter.value]]['yes']
                  for parameter in Parameter]
    no_values = [_shifts[parameter.value][data[parameter.value]]['no']
                 for parameter in Parameter]

    p_yes = np.multiply(np.product(yes_values), _shift_yes_percent)
    p_no = np.multiply(np.product(no_values), _shift_no_percent)

    p_total = np.sum([p_yes, p_no])

    return np.divide(p_yes, p_total), np.divide(p_no, p_total)


# print(predict_naive_bayes({
#     Parameter.GENDER.value: 'Male',
#     Parameter.PROGRAM.value: 'BSIT',
#     Parameter.STRAND.value: 'TVL',
#     Parameter.TESDA.value: 'yes',
#     Parameter.SCHOLAR.value: 'no',
#     Parameter.GWA.value: '84-80',
#     Parameter.RESOURCES.value: 'yes',
#     Parameter.ABSENCES.value: '1-4',
#     Parameter.EXPERIENCE.value: '0',
#     Parameter.ACTIVE.value: 'yes',
#     Parameter.TUITION.value: 'yes',
#     Parameter.SATISFACTION.value: 'yes',
# }))
# print(predict_naive_bayes({
#     Parameter.GENDER.value: 'Female',
#     Parameter.PROGRAM.value: 'BSIT',
#     Parameter.STRAND.value: 'ABM',
#     Parameter.TESDA.value: 'no',
#     Parameter.SCHOLAR.value: 'no',
#     Parameter.GWA.value: '84-80',
#     Parameter.RESOURCES.value: 'no',
#     Parameter.ABSENCES.value: '>10',
#     Parameter.EXPERIENCE.value: '0',
#     Parameter.ACTIVE.value: 'no',
#     Parameter.TUITION.value: 'yes',
#     Parameter.SATISFACTION.value: 'no',
# }))
# print(predict_naive_bayes({
#     Parameter.GENDER.value: 'Female',
#     Parameter.PROGRAM.value: 'BSCS',
#     Parameter.STRAND.value: 'TVL',
#     Parameter.TESDA.value: 'yes',
#     Parameter.SCHOLAR.value: 'no',
#     Parameter.GWA.value: '95-90',
#     Parameter.RESOURCES.value: 'yes',
#     Parameter.ABSENCES.value: '>10',
#     Parameter.EXPERIENCE.value: '2-4',
#     Parameter.ACTIVE.value: 'yes',
#     Parameter.TUITION.value: 'yes',
#     Parameter.SATISFACTION.value: 'no',
# }))
# for parameter in Parameter:
#     values = pd.DataFrame(_shifts[parameter.value]).transpose()
#     values = values.add_prefix('shifted_')
#     print(values)
