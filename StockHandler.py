# Python code to
# demonstrate readlines()
import json as js

lp = -1
lv = -1
totalShare = 2002672605

def extractContent(stock):
    global lp
    global lv
    if 'data' not in stock:
        return

    data = stock['data']

    totalP = 0
    totalV = 0
    count = 0
    for d in data:
        p = d['p']
        v = d['v']
        totalP += p
        totalV += v
        count += 1

    avgP = totalP / count

    if lp != -1:
        diffP = avgP - lp
        changeRateP = diffP / lp

        diffV = totalV - lv
        changeRateV = diffV / lv
        volPercent = totalV / totalShare
        print(changeRateV, volPercent， changeRateP)


    lp = avgP
    lv = totalV

    # changeRateP, changeRateV, volPercent



# Using readlines()
file1 = open('PLTR.txt', 'r')
Lines = file1.readlines()

count = 0
# Strips the newline character
obj = ""
stocks = []
for line in Lines:
    if len(line) == 1:
        json = js.loads(obj)
        stocks.append(json)
        obj = ""
    else:
        obj += line


print(len(stocks))

for stock in stocks:
    extractContent(stock)
