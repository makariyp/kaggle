import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tensorflow import keras
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv('../input/diamondprices/DiamondsPrices.csv')
df.head()

df.info()
df['cut'].unique()
df['color'].unique()
df['clarity'].unique()

df.loc[(df.cut=='Ideal'), 'cut'] = 5
df.loc[(df.cut=='Premium'), 'cut'] = 4
df.loc[(df.cut=='Very Good'), 'cut'] = 3
df.loc[(df.cut=='Good'), 'cut'] = 2
df.loc[(df.cut=='Fair'), 'cut'] = 1
df['cut'] = df['cut'].astype(float)

# D, E, F, G, H, I, J
df.loc[(df.color=='D'), 'color'] = 7
df.loc[(df.color=='E'), 'color'] = 6
df.loc[(df.color=='F'), 'color'] = 5
df.loc[(df.color=='G'), 'color'] = 4
df.loc[(df.color=='H'), 'color'] = 3
df.loc[(df.color=='I'), 'color'] = 2
df.loc[(df.color=='J'), 'color'] = 1
df['color'] = df['color'].astype(float)

# IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1
df.loc[(df.clarity=='IF'), 'clarity'] = 8
df.loc[(df.clarity=='VVS1'), 'clarity'] = 7
df.loc[(df.clarity=='VVS2'), 'clarity'] = 6
df.loc[(df.clarity=='VS1'), 'clarity'] = 5
df.loc[(df.clarity=='VS2'), 'clarity'] = 4
df.loc[(df.clarity=='SI1'), 'clarity'] = 3
df.loc[(df.clarity=='SI2'), 'clarity'] = 2
df.loc[(df.clarity=='I1'), 'clarity'] = 1
df['clarity'] = df['clarity'].astype(float)

df = df.drop(columns='z') 
df.head()

df.describe()

df.info()

train_x = df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y']][:-5000].values
train_y = df['price'][:-5000].values
test_x = df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y']][-5000:].values
test_y = df['price'][-5000:].values
model = keras.Sequential()
model.add(Dense(units=34, input_shape=(8,), activation='relu'))
model.add(Dense(units=1, activation='relu'))
model.compile(loss='mean_absolute_percentage_error', optimizer=keras.optimizers.Adam(0.1))
log = model.fit(train_x, train_y, batch_size=32, epochs=5, validation_split=0.1)
model.evaluate(test_x, test_y)

plt.plot(log.history['loss'])
plt.show()
