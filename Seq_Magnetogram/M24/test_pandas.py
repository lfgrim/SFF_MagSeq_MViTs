import pandas as pd
from datetime import datetime

flaresRAs = pd.read_csv('Seq16_flare_Mclass_24h_Test.txt_over', sep=' ', dtype='str')
print(flaresRAs.head())
print(flaresRAs.info())

RA_flares0 = flaresRAs[flaresRAs['0'] == '0']
RA_flares1 = flaresRAs[flaresRAs['0'] == '1']
print(len(RA_flares0))
print(len(RA_flares1))
