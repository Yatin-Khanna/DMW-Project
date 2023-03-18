import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
df = pd.read_csv('Data/behavioral.csv')

dataset = []
colNames = ["SRHighFloor","SRLowFloor","SRAccessibleRoom","SRMediumFloor","SRBathtub","SRShower","SRCrib","SRKingSizeBed","SRTwinBed","SRNearElevator","SRAwayFromElevator","SRNoAlcoholInMiniBar","SRQuietRoom"]
for index, row in df.iterrows():
    temp = []
    for name in colNames:
        if row[name] == 1:
            temp.append(name)
    dataset.append(temp)


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df, min_support=0.005, use_colnames=True)

print(frequent_itemsets)

res = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.05)

print(res)
res.to_csv('Results/behavioral_association_rules', index = False)
