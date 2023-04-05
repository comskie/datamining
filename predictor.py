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


dataset = pd.read_csv('dataset.csv', index_col=False, keep_default_na=False)

shift_yes = sum(dataset[SHIFT] == 'yes')
shift_no = sum(dataset[SHIFT] == 'no')

shift_yes_percent = shift_yes / (shift_yes + shift_no)
shift_no_percent = shift_no / (shift_yes + shift_no)

shifts = {}

for parameter in Parameter:
    values = dataset[parameter.value].unique()

    shifts[parameter.value] = [{
        parameter.value: value,
        'yes': (
            (dataset[parameter.value] == value) & (
                dataset[SHIFT] == 'yes')
        ).sum() / shift_yes,
        'no': (
            (dataset[parameter.value] == value) & (
                dataset[SHIFT] == 'no')
        ).sum() / shift_no
    } for value in values]

def get_value(parameter: Parameter, data_value: str, yes_or_no: str):
    for shift in shifts[parameter.value]:
        if shift[parameter.value] == data_value:
            return shift[yes_or_no]


# Returns tuple percentage of shift yes and shift no
def predict_naive_bayes(data: dict):
    
    yes_values = [get_value(parameter, data[parameter.value], 'yes') for parameter in Parameter]
    no_values = [get_value(parameter, data[parameter.value], 'no') for parameter in Parameter]

    p_yes = np.product(yes_values) * shift_yes_percent
    p_no = np.product(no_values) * shift_no_percent

    return (p_yes / (p_yes + p_no)), (p_no / (p_yes + p_no))


print(predict_naive_bayes({
    Parameter.GENDER.value: 'Male',
    Parameter.PROGRAM.value: 'BSIT',
    Parameter.STRAND.value: 'TVL',
    Parameter.TESDA.value: 'yes',
    Parameter.SCHOLAR.value: 'no',
    Parameter.GWA.value: '84-80',
    Parameter.RESOURCES.value: 'yes',
    Parameter.ABSENCES.value: '1-4',
    Parameter.EXPERIENCE.value: '0',
    Parameter.ACTIVE.value: 'yes',
    Parameter.TUITION.value: 'yes',
    Parameter.SATISFACTION.value: 'yes',
}))
