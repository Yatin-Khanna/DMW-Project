import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('Data/filtered.csv')

df_repeat = df[df['BookingsCanceled'] + df['BookingsNoShowed'] + df['BookingsCheckedIn'] >= 3]
df_no_repeat = df[df['BookingsCanceled'] + df['BookingsNoShowed'] + df['BookingsCheckedIn'] < 3]


colNames = ["SRHighFloor","SRLowFloor","SRAccessibleRoom","SRMediumFloor","SRBathtub","SRShower","SRCrib","SRKingSizeBed","SRTwinBed","SRNearElevator","SRAwayFromElevator","SRNoAlcoholInMiniBar","SRQuietRoom"]
repeat_support = [0 for i in range(len(colNames))]
no_repeat_support = [0 for i in range(len(colNames))]
repeat_count = 0
no_repeat_count = 0
for index, row in df_repeat.iterrows():
    repeat_count += 1
    for i in range(len(colNames)):
        if row[colNames[i]] == 1:
            repeat_support[i] += 1

for index, row in df_no_repeat.iterrows():
    no_repeat_count += 1
    for i in range(len(colNames)):
        if row[colNames[i]] == 1:
            no_repeat_support[i] += 1

for i in range(len(repeat_support)):
    repeat_support[i] /= repeat_count

for i in range(len(no_repeat_support)):
    no_repeat_support[i] /= no_repeat_count 


X = np.arange(len(colNames))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, repeat_support, color = 'b', width = 0.33)
ax.bar(X + 0.33, no_repeat_support, color = 'r', width = 0.33)
ax.set_xticks(X, colNames, rotation = 'vertical')
ax.legend(labels = ['Repeat', 'Non-Repeat'])

plt.savefig('repeat_non-repeat.png', bbox_inches = 'tight')
