import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth

df = pd.read_csv("Data/filtered.csv")

behavioral =  ["SRHighFloor","SRLowFloor","SRAccessibleRoom","SRMediumFloor","SRBathtub","SRShower","SRCrib","SRKingSizeBed","SRTwinBed","SRNearElevator","SRAwayFromElevator","SRNoAlcoholInMiniBar","SRQuietRoom"]

df2 = df[sum([df[colName] for colName in behavioral]) >= 1]

df2.to_csv("Data/demanding.csv")