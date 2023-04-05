import numpy as np
import pandas as pd
from enum import Enum


class Parameter(Enum):
    GENDER = 'gender'
    PROGRAM = 'program'
    YEAR_LEVEL = 'year_level'
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

shift_yes = sum(dataset[Parameter.SHIFT.value] == 'yes')
shift_no = sum(dataset[Parameter.SHIFT.value] == 'no')

shift_yes_percent = shift_yes / (shift_yes + shift_no)
shift_no_percent = shift_no / (shift_yes + shift_no)


# Genders
genders = dataset[Parameter.GENDER.value].unique()

shift_yes_by_genders = {}
shift_no_by_genders = {}

for gender in genders:
    shift_yes_by_genders[gender] = ((dataset[Parameter.GENDER.value] == gender) & (
        dataset[Parameter.SHIFT.value] == 'yes')).sum() / shift_yes
    shift_no_by_genders[gender] = ((dataset[Parameter.GENDER.value] == gender) & (
        dataset[Parameter.SHIFT.value] == 'no')).sum() / shift_no

# Programs
programs = dataset[Parameter.PROGRAM.value].unique()

shift_yes_by_programs = {}
shift_no_by_programs = {}

for program in programs:
    shift_yes_by_programs[program] = ((dataset[Parameter.PROGRAM.value] == program) & (
        dataset[Parameter.SHIFT.value] == 'yes')).sum() / shift_yes
    shift_no_by_programs[program] = ((dataset[Parameter.PROGRAM.value] == program) & (
        dataset[Parameter.SHIFT.value] == 'no')).sum() / shift_no

# SHS Strands
shs_strands = dataset[Parameter.STRAND.value].unique()
print(shs_strands)

shift_yes_by_shs_strands = {}
shift_no_by_shs_strands = {}

for shs_strand in shs_strands:
    shift_yes_by_shs_strands[shs_strand] = ((dataset[Parameter.STRAND.value] == shs_strand) & (
        dataset[Parameter.SHIFT.value] == 'yes')).sum() / shift_yes
    shift_no_by_shs_strands[shs_strand] = ((dataset[Parameter.STRAND.value] == shs_strand) & (
        dataset[Parameter.SHIFT.value] == 'no')).sum() / shift_no

# NC2 Passers
nc2_passers = dataset[Parameter.TESDA.value].unique()

shift_yes_by_nc2_passers = {}
shift_no_by_nc2_passers = {}

for nc2_passer in nc2_passers:
    shift_yes_by_nc2_passers[nc2_passer] = ((dataset[Parameter.TESDA.value] == nc2_passer) & (
        dataset[Parameter.SHIFT.value] == 'yes')).sum() / shift_yes
    shift_no_by_nc2_passers[nc2_passer] = ((dataset[Parameter.TESDA.value] == nc2_passer) & (
        dataset[Parameter.SHIFT.value] == 'no')).sum() / shift_no

# Scholars
scholars = dataset[Parameter.SCHOLAR.value].unique()

shift_yes_by_scholars = {}
shift_no_by_scholars = {}

for scholar in scholars:
    shift_yes_by_scholars[scholar] = ((dataset[Parameter.SCHOLAR.value] == scholar) & (
        dataset[Parameter.SHIFT.value] == 'yes')).sum() / shift_yes
    shift_no_by_scholars[scholar] = ((dataset[Parameter.SCHOLAR.value] == scholar) & (
        dataset[Parameter.SHIFT.value] == 'no')).sum() / shift_no

# General Averages
general_averages = dataset[Parameter.GWA.value].unique()

shift_yes_by_general_averages = {}
shift_no_by_general_averages = {}

for general_average in general_averages:
    shift_yes_by_general_averages[general_average] = ((dataset[Parameter.GWA.value] == general_average) & (
        dataset[Parameter.SHIFT.value] == 'yes')).sum() / shift_yes
    shift_no_by_general_averages[general_average] = ((dataset[Parameter.GWA.value] == general_average) & (
        dataset[Parameter.SHIFT.value] == 'no')).sum() / shift_no

# Have Tools
have_tools = dataset[Parameter.RESOURCES.value].unique()

shift_yes_by_have_tools = {}
shift_no_by_have_tools = {}

for have_tool in have_tools:
    shift_yes_by_have_tools[have_tool] = ((dataset[Parameter.RESOURCES.value] == have_tool) & (
        dataset[Parameter.SHIFT.value] == 'yes')).sum() / shift_yes
    shift_no_by_have_tools[have_tool] = ((dataset[Parameter.RESOURCES.value] == have_tool) & (
        dataset[Parameter.SHIFT.value] == 'no')).sum() / shift_no

# Total absences
total_absences = dataset[Parameter.ABSENCES.value].unique()

shift_yes_by_total_absences = {}
shift_no_by_total_absences = {}

for total_absence in total_absences:
    shift_yes_by_total_absences[total_absence] = ((dataset[Parameter.ABSENCES.value] == total_absence) & (
        dataset[Parameter.SHIFT.value] == 'yes')).sum() / shift_yes
    shift_no_by_total_absences[total_absence] = ((dataset[Parameter.ABSENCES.value] == total_absence) & (
        dataset[Parameter.SHIFT.value] == 'no')).sum() / shift_no

# Programming experience
programming_experiences = dataset[Parameter.EXPERIENCE.value] \
    .unique()

shift_yes_by_programming_experiences = {}
shift_no_by_programming_experiences = {}

for programming_experience in programming_experiences:
    shift_yes_by_programming_experiences[programming_experience] = ((dataset[Parameter.EXPERIENCE.value] == programming_experience) & (
        dataset[Parameter.SHIFT.value] == 'yes')).sum() / shift_yes
    shift_no_by_programming_experiences[programming_experience] = ((dataset[Parameter.EXPERIENCE.value] == programming_experience) & (
        dataset[Parameter.SHIFT.value] == 'no')).sum() / shift_no


# Participates extracurricular activities
participates_extracurricular_activities = dataset[Parameter.ACTIVE.value] \
    .unique()

shift_yes_by_participates_extracurricular_activities = {}
shift_no_by_participates_extracurricular_activities = {}

for participates_extracurricular_activity in participates_extracurricular_activities:
    shift_yes_by_participates_extracurricular_activities[participates_extracurricular_activity] = ((dataset[Parameter.ACTIVE.value] == participates_extracurricular_activity) & (
        dataset[Parameter.SHIFT.value] == 'yes')).sum() / shift_yes
    shift_no_by_participates_extracurricular_activities[participates_extracurricular_activity] = ((dataset[Parameter.ACTIVE.value] == participates_extracurricular_activity) & (
        dataset[Parameter.SHIFT.value] == 'no')).sum() / shift_no

# challenge in fulfilling tuition payment
challenge_in_fulfilling_tuition_payments = dataset[Parameter.TUITION.value] \
    .unique()

shift_yes_by_challenge_in_fulfilling_tuition_payments = {}
shift_no_by_challenge_in_fulfilling_tuition_payments = {}

for challenge_in_fulfilling_tuition_payment in challenge_in_fulfilling_tuition_payments:
    shift_yes_by_challenge_in_fulfilling_tuition_payments[challenge_in_fulfilling_tuition_payment] = ((dataset[Parameter.TUITION.value] == challenge_in_fulfilling_tuition_payment) & (
        dataset[Parameter.SHIFT.value] == 'yes')).sum() / shift_yes
    shift_no_by_challenge_in_fulfilling_tuition_payments[challenge_in_fulfilling_tuition_payment] = ((dataset[Parameter.TUITION.value] == challenge_in_fulfilling_tuition_payment) & (
        dataset[Parameter.SHIFT.value] == 'no')).sum() / shift_no

# Satisfied in learning environment
satisfied_in_learning_environments = dataset[Parameter.SATISFACTION.value] \
    .unique()

shift_yes_by_satisfied_in_learning_environments = {}
shift_no_by_satisfied_in_learning_environments = {}

for satisfied_in_learning_environment in satisfied_in_learning_environments:
    shift_yes_by_satisfied_in_learning_environments[satisfied_in_learning_environment] = ((dataset[Parameter.SATISFACTION.value] == satisfied_in_learning_environment) & (
        dataset[Parameter.SHIFT.value] == 'yes')).sum() / shift_yes
    shift_no_by_satisfied_in_learning_environments[satisfied_in_learning_environment] = ((dataset[Parameter.SATISFACTION.value] == satisfied_in_learning_environment) & (
        dataset[Parameter.SHIFT.value] == 'no')).sum() / shift_no



# Returns tuple percentage of shift yes and shift no
def predict_naive_bayes(data):
    gender = data[Parameter.GENDER.value]
    program = data[Parameter.PROGRAM.value]
    strand = data[Parameter.STRAND.value]
    tesda = data[Parameter.TESDA.value]
    scholar = data[Parameter.SCHOLAR.value]
    gwa = data[Parameter.GWA.value]
    resources = data[Parameter.RESOURCES.value]
    absences = data[Parameter.ABSENCES.value]
    experience = data[Parameter.EXPERIENCE.value]
    active = data[Parameter.ACTIVE.value]
    tuition = data[Parameter.TUITION.value]
    satisfaction = data[Parameter.SATISFACTION.value]

    p_yes = shift_yes_by_genders[gender] * \
        shift_yes_by_programs[program] * \
        shift_yes_by_shs_strands[strand] * \
        shift_yes_by_nc2_passers[tesda] * \
        shift_yes_by_scholars[scholar] * \
        shift_yes_by_general_averages[gwa] * \
        shift_yes_by_have_tools[resources] * \
        shift_yes_by_total_absences[absences] * \
        shift_yes_by_programming_experiences[experience] * \
        shift_yes_by_participates_extracurricular_activities[active] * \
        shift_yes_by_challenge_in_fulfilling_tuition_payments[tuition] * \
        shift_yes_by_satisfied_in_learning_environments[satisfaction] * \
        shift_yes_percent
    p_no = shift_no_by_genders[gender] * \
        shift_no_by_programs[program] * \
        shift_no_by_shs_strands[strand] * \
        shift_no_by_nc2_passers[tesda] * \
        shift_no_by_scholars[scholar] * \
        shift_no_by_general_averages[gwa] * \
        shift_no_by_have_tools[resources] * \
        shift_no_by_total_absences[absences] * \
        shift_no_by_programming_experiences[experience] * \
        shift_no_by_participates_extracurricular_activities[active] * \
        shift_no_by_challenge_in_fulfilling_tuition_payments[tuition] * \
        shift_no_by_satisfied_in_learning_environments[satisfaction] * \
        shift_no_percent

    print(p_yes, p_no)
    return (p_yes / (p_yes + p_no)), (p_no / (p_yes + p_no))


# print(shift_yes_by_total_absences)
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
