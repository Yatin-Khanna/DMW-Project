import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('Data/filtered.csv')

df_corp = df[(df['DistributionChannel'] == "Corporate")]
df_no_corp = df[(df['DistributionChannel'] != "Corporate")]


df_corp.to_csv('Data/corporate.csv')
df_no_corp.to_csv('Data/no_corporate.csv')

colNames = ["SRHighFloor","SRLowFloor","SRAccessibleRoom","SRMediumFloor","SRBathtub","SRShower","SRCrib","SRKingSizeBed","SRTwinBed","SRNearElevator","SRAwayFromElevator","SRNoAlcoholInMiniBar","SRQuietRoom"]
corp_support = [0 for i in range(len(colNames))]
no_corp_support = [0 for i in range(len(colNames))]
corp_count = 0
no_corp_count = 0
for index, row in df_corp.iterrows():
    corp_count += 1
    for i in range(len(colNames)):
        if row[colNames[i]] == 1:
            corp_support[i] += 1

for index, row in df_no_corp.iterrows():
    no_corp_count += 1
    for i in range(len(colNames)):
        if row[colNames[i]] == 1:
            no_corp_support[i] += 1

for i in range(len(corp_support)):
    corp_support[i] /= corp_count

for i in range(len(no_corp_support)):
    no_corp_support[i] /= no_corp_count 


X = np.arange(len(colNames))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, corp_support, color = 'b', width = 0.33)
ax.bar(X + 0.33, no_corp_support, color = 'r', width = 0.33)
ax.set_xticks(X, colNames, rotation = 'vertical')
ax.legend(labels = ['Corp', 'Non-corp'])

plt.savefig('corp_behavioral.png', bbox_inches = 'tight')
