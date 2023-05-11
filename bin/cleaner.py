import os
import pandas as pd

DATASET_PATH = os.path.join('..', os.getenv('DATASET_PATH', 'dataset.csv'))

df = pd.read_csv(DATASET_PATH, keep_default_na=False)

# Remove unnecessary columns
dropFilter = df.filter(['Timestamp', 'Email Address', 'Participant\'s Agreement', 'ID Number', 'Year Level'])
df = df.drop(dropFilter, axis=1)

# Rename headers
simplifiedHeaders = [
    'gender',
    'program',
    'strand',
    'tesda',
    'scholar',
    'gwa',
    'resources',
    'absences',
    'experience',
    'active',
    'tuition',
    'satisfaction',
    'shift',
]
df.columns = simplifiedHeaders

# Replace values
df = df.replace(to_replace='95-90', value='100-90')
df = df.replace(to_replace='TVL.*', value='TVL', regex=True)

# Export cleaned dataset
df.to_csv(DATASET_PATH, index=False)

print('Done!')
