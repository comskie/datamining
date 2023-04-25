import os
import numpy as np
import pandas as pd
from enum import Enum

THRESHOLD = 0.6

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

_dataset = pd.read_csv(os.getenv('DATASET_PATH', 'dataset.csv'), index_col=False, keep_default_na=False)

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
#     print(parameter.value)
#     print(values)
# gender
#         shifted_yes  shifted_no
# Male       0.615385    0.647059
# Female     0.384615    0.352941
#
# program
#       shifted_yes  shifted_no
# BSCS     0.615385    0.823529
# BSIT     0.384615    0.176471
#
#  strand
#        shifted_yes  shifted_no
# TVL       0.230769    0.529412
# N/A       0.230769    0.176471
# HUMSS     0.153846    0.058824
# STEM      0.153846    0.058824
# GAS       0.153846    0.117647
# ABM       0.076923    0.058824
#
# tesda
#      shifted_yes  shifted_no
# no      0.769231    0.647059
# yes     0.230769    0.352941
#
# scholar
#      shifted_yes  shifted_no
# yes          0.0    0.235294
# no           1.0    0.764706
#
# gwa
#        shifted_yes  shifted_no
# 95-90     0.076923    0.294118
# 89-85     0.230769    0.470588
# 84-80     0.461538    0.176471
# 79-75     0.230769    0.058824
#
# resources
#      shifted_yes  shifted_no
# yes     0.538462    0.882353
# no      0.461538    0.117647
#
# absences
#       shifted_yes  shifted_no
# 0        0.076923    0.294118
# >10      0.384615    0.000000
# 1-4      0.153846    0.705882
# 5-10     0.384615    0.000000
#
# experience
#      shifted_yes  shifted_no
# 2-4     0.230769    0.647059
# 0       0.307692    0.117647
# 1       0.384615    0.117647
# >5      0.076923    0.117647
# 
# active
#      shifted_yes  shifted_no
# yes     0.230769    0.352941
# no      0.769231    0.647059
# 
# tuition
#      shifted_yes  shifted_no
# no      0.384615    0.352941
# yes     0.615385    0.647059
# 
# satisfaction
#      shifted_yes  shifted_no
# yes     0.538462    0.705882
# no      0.461538    0.294118