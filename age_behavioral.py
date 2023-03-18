import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
df = pd.read_csv('Data/filtered.csv')

df2 = df.drop(df.columns[[i for i in range(3, 16)]], axis = 1)
df2 = df2.drop(df2.columns[[i for i in range(2)]], axis = 1)

df2.to_csv('Data/age_behavioral.csv', index = False)

dataset = []
colNames = ["Age","SRHighFloor","SRLowFloor","SRAccessibleRoom","SRMediumFloor","SRBathtub","SRShower","SRCrib","SRKingSizeBed","SRTwinBed","SRNearElevator","SRAwayFromElevator","SRNoAlcoholInMiniBar","SRQuietRoom"]
for index, row in df2.iterrows():
    temp = []
    for name in colNames:
        if name == "Age":
            age = int(row[name])
            age = 5 * (age // 5)
            temp.append(str(age))
        elif row[name] == 1:
            temp.append(name)
    dataset.append(temp)


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df2 = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df2, min_support=0.01, use_colnames=True)

print(frequent_itemsets)

res = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.1)

res.sort_values('confidence', inplace = True, ascending = False, ignore_index = True)
print(res)
res.to_csv('Results/age_behavioral_association_rules', index = False)


# print(pd.read_csv('Results/behavioral_association_rules'))