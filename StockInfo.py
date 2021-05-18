import websocket
import requests
from datetime import timezone
import datetime

symbol = "SENS"


liveStockData = []

def writeToFile(symbol, data):
    my_file = open(symbol + "2.txt", "a")
    my_file.write(data)
    my_file.close()


def getTimestamp(y, mon, d, h, min, s):
    dt = datetime.datetime(y, mon, d, h, min, s)
    timestamp = dt.timestamp()
    timestamp = str(timestamp).replace('.0', '')
    return timestamp

def timestampToDatetime(timestamp):
    dateTime = datetime.datetime.fromtimestamp(timestamp)
    return dateTime

def on_message(ws, message):
    print(message)
    writeToFile(symbol, message + "\n\n")

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    ws.send('{"type":"subscribe","symbol":"' + symbol + '"}')

def streamPrice():
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("wss://ws.finnhub.io?token=c14nrh748v6st27554cg",
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close)
    ws.on_open = on_open
    ws.run_forever()


def getStockSentiment(symbol):
    r = requests.get('https://finnhub.io/api/v1/news-sentiment?symbol=' + symbol + '&token=c14nrh748v6st27554cg')
    r = r.json()
    print(r)
    return r

def getBasicFinancial(symbol):
    r = requests.get('https://finnhub.io/api/v1/stock/metric?symbol=' + symbol + '&metric=all&token=c14nrh748v6st27554cg')
    r = r.json()
    print(r)
    return r

def getStockCandles(symbol, resolution, start, end):
    r = requests.get('https://finnhub.io/api/v1/stock/candle?symbol=' + symbol + '&resolution=' + str(resolution) + '&from=' + start + '&to=' + end + '&token=c14nrh748v6st27554cg')
    r = r.json()
    print(r)
    return r

start = 1618407219606 / 1000#getTimestamp(2021, 4, 8, 11, 30, 0)
end = 1618417216510 / 1000#getTimestamp(2021, 4, 8, 11, 59, 0)


streamPrice()
#obj = getStockCandles("SENS", 1, str(int(start)), str(int(end)))
#print(obj)
