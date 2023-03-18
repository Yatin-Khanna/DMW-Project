import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth

# Separation of columns based on similarity
age = ["Age"]
nationality = ["Nationality"]
behavioral =  ["SRHighFloor","SRLowFloor","SRAccessibleRoom","SRMediumFloor","SRBathtub","SRShower","SRCrib","SRKingSizeBed","SRTwinBed","SRNearElevator","SRAwayFromElevator","SRNoAlcoholInMiniBar","SRQuietRoom"]
price = ["LodgingRevenue", "OtherRevenue"]
bookings = ["BookingsCanceled", "BookingsNoShowed", "BookingsCheckedIn", "TotalBookings"]
duration = ["AverageLeadTime", "PersonsNights", "RoomNights"]

# constants
repeatCustomerThreshold = 3
MIN_SUP_THRES = 0.01
MIN_CONF_THRES = 0.1

# functions
def chooseColumns(df, colNames):
    return df[colNames]

def round(val, len):
    return len * (int(val) // len)

def generateRules(df):
    dataset = []
    colNames = list(df.columns)

    for index, row in df.iterrows():
        temp = []
        for col in colNames:
            if col in age:
                ageLow = round(row[col], 5)
                temp.append("A[" + str(ageLow) + "]")
            if col in nationality:
                temp.append(row[col])
            if col in behavioral:
                if row[col] == 1:
                    temp.append(col[2:])
            if col == "LodgingRevenue":
                priceLow = round(row[col], 50)
                temp.append("LR[" + str(priceLow) + "]")
            if col == "OtherRevenue":
                priceLow = round(row[col], 10)
                temp.append("OR[" + str(priceLow) + "]")
            if col == "TotalBookings":
                if int(row[col]) >= repeatCustomerThreshold:
                    temp.append("RepeatCustomer")
            elif col in bookings:
                if int(row[col]) < 10:
                    temp.append(col + " " + str(row[col]))
                else:
                    temp.append(col + " >10")
            if col == "AverageLeadTime":
                daysLow = round(row[col], 3)
                temp.append("ALT[" + str(daysLow) + "]")
            if col == "PersonsNights":
                nightsLow = round(row[col], 3)
                temp.append("PN[" + str(nightsLow) + "]")
            if col == "RoomNights":
                nightsLow = round(row[col], 3)
                temp.append("RN[" + str(nightsLow) + "]")


        dataset.append(temp)
    
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = fpgrowth(df, min_support = MIN_SUP_THRES, use_colnames=True)
    return association_rules(frequent_itemsets, metric = "confidence", min_threshold = MIN_CONF_THRES)

        



df = pd.read_csv('Data/repeat.csv')
# add total bookings column
df = df.assign(TotalBookings = df.BookingsCanceled + df.BookingsNoShowed + df.BookingsCheckedIn)


# Choose columns
df2 = chooseColumns(df, behavioral)

# Generate association rules

rules = generateRules(df2)

# Print and Save wherever you want

rules.sort_values('confidence', inplace = True, ascending = False, ignore_index = True)
print(rules)
rules.to_csv('Results/demanding_behavioral.csv', index = False)




