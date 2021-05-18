import json as js
import numpy as np
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')
file1 = open('/content/drive/My Drive/DL/Project/SENS.txt', 'r')

lp = -1
lv = -1
#totalShare = 81920000 #appl

#totalShare = 4280000 #pltr
# threashold=0.07  #急速升降的阈值
# threasholdmin=0.02 #升降的阈值
# inter=10  #每隔几个数据记录一次股价
# split=2600 #test train分界线

totalShare = 15530000 #sens
threashold=0.015
threasholdmin=0.005
inter=5
split=1200

windowsize=19
X=[]

Lines = file1.readlines()
count = 0
# Strips the newline character
obj = ""
stocks1 = []
Prices=[]
for line in Lines:
    if len(line) == 1:
        json = js.loads(obj)
        stocks1.append(json)
        obj = ""
    else:
        obj += line


inter_ct=0
for stock in stocks1:
    # print(inter_ct)
    if 'data' not in stock :
        inter_ct+=1
        continue
    if inter_ct<inter:
        inter_ct+=1
        continue
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
    Prices.append(avgP)
    inter_ct=0
    if lp != -1:
        diffP = avgP - lp
        changeRateP = diffP / lp
        diffV = totalV - lv
        changeRateVforLast = diffV / lv
        changeRateVforAll= diffV / totalShare
        volPercent = totalV / totalShare
        # print("changeRateV:",changeRateV, "volPercent:",volPercent, "changeRateP",changeRateP)
        X.append([changeRateVforLast,changeRateVforAll,volPercent,changeRateP])

    lp = avgP
    lv = totalV



def split_data(data, window,train_set_size):
  Data_Window=[]
  Y_Window=[]

  for index in range(len(data) - window-1):
    Data_Window.append(data[index: index + window])

    #PastPrice=Prices[index: index + window].mean()
    PastPrice=sum(Prices[index: index + window]) / window
    Price=Prices[index + window +1]

    diff=Price-PastPrice

    if abs( diff) > threashold:
      y= (diff)/abs( diff ) *2
    elif abs( diff) > threasholdmin:
      y= (diff)/abs( diff ) *1
    else:
      y=0

    y+=2

    Y_Window.append(y)

  train_data = []
  for i in range(len(Data_Window)):
    train_data.append([Data_Window[i], Y_Window[i]])

  print("0",Y_Window.count(0),"1",Y_Window.count(1),"2",Y_Window.count(2),"3",Y_Window.count(3),"4",Y_Window.count(4))

  return [train_data[:train_set_size], train_data[train_set_size:], Data_Window, Y_Window]

# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
# x_minmax = min_max_scaler.fit_transform(X)
#X=x_minmax
X=np.array(X)
train_data, test_data, x_train, y_train = split_data(X, window=windowsize, train_set_size=30000)

x_train = torch.from_numpy(np.array(x_train)).type(torch.Tensor)
y_train_lstm = torch.from_numpy(np.array(y_train)).type(torch.Tensor)

x_test=x_train[split:]
y_test_lstm=y_train_lstm[split:]
x_train=x_train[:split]
y_train_lstm=y_train_lstm[:split]


import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(range(len(Prices)),Prices,'-',linewidth=1,label='Prices')
plt.xlabel('epoch')
plt.ylabel('Prices')
plt.grid(True)
plt.legend()

import matplotlib.pyplot as plt

plt.figure(2)
plt.plot(range(len(X[:,0])),X[:,0],'-',linewidth=1,label='changeRateVforLast')
plt.xlabel('epoch')
plt.ylabel('??')
plt.grid(True)
plt.legend()

plt.figure(3)
plt.plot(range(len(X[:,1])),X[:,1],'-',linewidth=1,label='changeRateVforAll')
plt.xlabel('epoch')
plt.ylabel('??')
plt.grid(True)
plt.legend()

plt.figure(4)
plt.plot(range(len(X[:,2])),X[:,2],'-',linewidth=1,label='volPercent')
plt.xlabel('epoch')
plt.ylabel('??')
plt.grid(True)
plt.legend()


plt.figure(5)
plt.plot(range(len(X[:,3])),X[:,3],'-',linewidth=1,label='changeRateP')
plt.xlabel('epoch')
plt.ylabel('??')
plt.grid(True)
plt.legend()

plt.figure(6)
plt.plot(range(len(y_train)),y_train,'-',linewidth=1,label='Y')
plt.xlabel('epoch')
plt.ylabel('??')
plt.grid(True)
plt.legend()

print(len(x_train))
print(x_train.shape)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 20

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out

# input_dim = 4
# hidden_dim = 32
# num_layers = 2
# output_dim = 5
# num_epochs = 500

input_dim = 4
hidden_dim = 64
num_layers = 4
output_dim = 5
num_epochs = 500

class LSTMxun(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMxun, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sft=nn.Softmax(dim=1)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        #out= self.sft(out)
        return out

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out

model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
criterion = torch.nn.MSELoss(reduction='mean')
criterion = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)



import time
Loss =[]
start_time = time.time()
lstm = []
offset=20
TrainAcc=[]
TestAcc=[]

for t in range(num_epochs):
    y_train_lstm=y_train_lstm.long().to(device)
    x_train=x_train.to(device)
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_lstm)

    Loss.append(loss.item())
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    print("Epoch ", t, "MSE: ", loss.item())


    if t % offset ==0:
      acc=0
      lstm=[]
      for i in range(len(x_test)):
        x=x_test[i].reshape(-1,19,4).to(device)
        y_train_pred = model(x)
        y_train_pred=np.argmax(y_train_pred.cpu().detach().numpy())
        #print(y_train_pred.item())
        if y_test_lstm[i]==y_train_pred.item():
          acc+=1
        lstm.append(y_train_pred.item())
      TestAcc.append(acc/len(x_test))
      print("\n\n Acc: ", acc/len(x_test))

      acc=0
      for i in range(len(x_train)):
        x=x_train[i].reshape(-1,19,4)
        y_train_pred = model(x)
        y_train_pred=np.argmax(y_train_pred.cpu().detach().numpy())
        #print(y_train_pred.item())
        if y_train_lstm[i]==y_train_pred.item():
          acc+=1
        lstm.append(y_train_pred.item())
      TrainAcc.append(acc/len(x_train))


import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(range(len(TrainAcc)),TrainAcc,'-',linewidth=1,label='TrainAcc')
plt.plot(range(len(TestAcc)),TestAcc,'-',linewidth=1,label='TestAcc')
plt.xlabel('epoch')
plt.ylabel('Acc')
plt.grid(True)
plt.legend()

plt.figure(2)
plt.plot(range(len(Loss)),Loss,'-',linewidth=1,label='Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()


import time
import numpy as np
hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []
acc=0
for i in range(len(x_test)):
    x=x_test[i].reshape(-1,19,4)
    y_train_pred = model(x)
    y_train_pred=np.argmax(y_train_pred.detach().numpy())
    #print(y_train_pred.item())
    if y_test_lstm[i]==y_train_pred.item():
      acc+=1
    lstm.append(y_train_pred.item())


training_time = time.time()-start_time
print("Training time: {}".format(training_time))
print("Acc: ", acc/len(x_test))

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(range(len(lstm)),lstm,'-',linewidth=1,label='lstm')
#plt.plot(range(len(y_train_lstm)),y_train_lstm,'-',linewidth=1,label='y_train_lstm')
plt.xlabel('epoch')
plt.ylabel('??')
plt.grid(True)
plt.legend()

plt.figure(2)
#plt.plot(range(len(lstm)),lstm,'-',linewidth=4,label='lstm')
plt.plot(range(len(y_test_lstm)),y_test_lstm,'-',linewidth=1,label='y_lstm')
plt.xlabel('epoch')
plt.ylabel('??')
plt.grid(True)
plt.legend()


import time
import numpy as np
hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []
acc=0
for i in range(len(x_train)):
    x=x_train[i].reshape(-1,19,4)
    y_train_pred = model(x)
    y_train_pred=np.argmax(y_train_pred.detach().numpy())
    #print(y_train_pred.item())
    if y_train_lstm[i]==y_train_pred.item():
      acc+=1
    lstm.append(y_train_pred.item())


training_time = time.time()-start_time
print("Training time: {}".format(training_time))
print("Acc: ", acc/len(x_train))

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(range(len(lstm)),lstm,'-',linewidth=1,label='lstm')
#plt.plot(range(len(y_train_lstm)),y_train_lstm,'-',linewidth=1,label='y_train_lstm')
plt.xlabel('epoch')
plt.ylabel('??')
plt.grid(True)
plt.legend()

plt.figure(2)
#plt.plot(range(len(lstm)),lstm,'-',linewidth=4,label='lstm')
plt.plot(range(len(y_train_lstm)),y_train_lstm,'-',linewidth=1,label='y_lstm')
plt.xlabel('epoch')
plt.ylabel('??')
plt.grid(True)
plt.legend()
