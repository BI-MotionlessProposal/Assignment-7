import folium
import inline as inline
import matplotlib
from folium.plugins import HeatMap
% matplotlib inline
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from math import sqrt
from sklearn import linear_model, metrics

# part 1
dataframe = pd.read_csv("hn_items.csv",
                        names=['id', 'type', 'by', 'time', 'text', 'dead', 'parent', 'poll', 'kids', 'score', 'title',
                               'parts'])
# dataframe = [x for x in dataframe if pd.notnull(x['title'])]
dataframe = dataframe.dropna(subset=['text'])
print(dataframe['text'])

import nltk

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

model = SentimentIntensityAnalyzer()
import sklearn

dataset = [sklearn.datasets.base.Bunch(data=x) for x in dataframe["text"]]
len(dataset)
# scores = [model.polarity_scores(x) for x in dataset.data]
scores = [model.polarity_scores(x['data']) for x in dataset]

scores.sort(key=lambda x: x['pos'], reverse=True)
scores[:5]  # 5 most possitive posts

scores.sort(key=lambda x: x['neg'], reverse=True)
scores[:5]  # 5 most negative posts

# part 2

df = pd.DataFrame(data = scores)
kf = KFold(n_splits = 10)
XS = df['pos']
YS = df['neg']
MAES = []
RSMES = []
accs = []

scaledX = scale(XS)
scaledY = scale(YS)

for train, test in kf.split(XS, YS):
    X_Train, X_Test = scaledX[train], scaledX[test]
    Y_Train, Y_Test = scaledY[train], scaledY[test]
    model = linear_model.LinearRegression()
    model.fit(X_Train.reshape(-1, 1), Y_Train.reshape(-1, 1))

    plt.scatter(x=X_Train, y=Y_Train)
    plt.show()
    predict = model.predict(Y_Test.reshape(-1, 1))

    mae = str(metrics.mean_absolute_error(X_Test.reshape(-1, 1), predict))
    rsme = str(sqrt(metrics.mean_squared_error(X_Test.reshape(-1, 1), predict)))

    MAES.append(mae)
    RSMES.append(rsme)

# plt.scatter(scores['pos'], scores['pos'])
# x = pd.Series(scores['pos'])
# Y = pd.Series(scores['neg'])
print('MAE : ')
print(MAES)
print('RSME : ')
print(RSMES)

print('MAE Average : ')
# print(MAES)
print(sum([float(i) for i in MAES]))
print('RSME Average : ')
print(sum([float(i) for i in RSMES]))

# part 3
df = pd.read_csv('boliga_zealand.csv').drop(['Index', '_1', 'Unnamed: 0'], axis=1)
df = df[['lat', 'lon', 'price']].dropna()

norreport = (55.6836329, 12.5709413)
my_map = folium.Map(location=[55.6836329, 12.5709413], zoom_start=10)
folium.Marker(location=norreport, icon=folium.Icon(color='red', icon='home')).add_to(my_map)
data = []
for row in df.itertuples():
    data.append([row.lat, row.lon, row.price])
HeatMap(data, radius=7).add_to(my_map)

my_map.save('heatmap.html')


# According to the map data we see as the best the triangle area between Bronshøj, Islev and Vanløse. Not only that
# the HeatMap shows that as a reasonably priced area (however we do not consider the size of the apartments, so it
# could be that the apartments are just smaller in the area) but also the distance and connection to the city center
# is pretty good, especially around Vanløse where is also metro.


# Part 4: Housing price model
def haversine_distance(lat_dest, lon_dest):
    lat_orig, lon_orig = (55.6836722, 12.5693963)
    # lat_dest, lon_dest = destination
    radius = 6371

    dlat = math.radians(lat_dest - lat_orig)
    dlon = math.radians(lon_dest - lon_orig)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat_orig))
         * math.cos(math.radians(lat_dest)) * math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


X = []
Y = []
Z = []

df = pd.read_csv('./boliga/boliga_zealand.csv')
df = df[np.isfinite(df['lon'])]
df = df[np.isfinite(df['lat'])]
df = df[np.isfinite(df['price'])]
df = df[np.isfinite(df['size_in_sq_m'])]

df['dist'] = df.apply(lambda x: haversine_distance(x['lat'], x['lon']), axis=1)

XS = df['dist']
YS = df['price']
ZS = df['size_in_sq_m']

model = LinearRegression()

X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(XS, YS, ZS, test_size=0.2)
XZ_train = np.stack([X_train, Z_train], axis=1).reshape(-1, 2)
XZ_test = np.stack([X_test, Z_test], axis=1).reshape(-1, 2)
model.fit(XZ_train, Y_train)

kf = KFold(n_splits=10)

scaledX = scale(XS)
scaledY = scale(YS)
pearson = []
MAES = []
MSE = []
accs = []
for train, test in kf.split(XS, YS, ZS):
    XZ = np.stack([XS, ZS], axis=1).reshape(-1, 2)
    XZ_Train, XZ_Test = XZ[train], XZ[test]
    Y_Train, Y_Test = scaledY[train], scaledY[test]

    model = linear_model.LinearRegression()
    model.fit(XZ_train, Y_train)

    predict = model.predict(XZ_Test)

    mae = str(metrics.mean_absolute_error(Y_Test, predict))
    mse = str(metrics.mean_squared_error(Y_Test, predict))

    pearson.append(model.score(XZ_train, Y_train))

    MAES.append(mae)
    MSE.append(mse)
print('coef: ', str(model.coef_))
print('intercept', str(model.intercept_))

print('MAE : ', str(MAES))
print('MSE : ')
print(MSE)

print('MAE Average : ')
# print(MAES)
print(sum([float(i) for i in MAES]) / 10)
print('MSE Average : ')
print(sum([float(i) for i in MSE]) / 10)
print('Pearson R Average : ')
print(sum([float(i) for i in pearson]) / 10)
# Conclusion

# We created a linear model that tries to predict the price of the place, given the distance to Nørreport Station and price and  square meter.
# coef:  [-16423.23477145  16448.20355148] distance , square meter
# intercept 709106.186971

# MAE Average :
# 2237609.3088789997
# MSE Average :
# 3381080.7514172345
# Pearson R Average :
# 0.19807411204695002

# Then we get the model

# y = -16423.23477145 x1 + 16448.20355148x2 + 709106.186971
# x1 = distance and x2 is square meter
# y = price and x is distance in kilometer.

# Our model has Pearson R Average 0.19807411204695002  --- which tells us it's not vey good linear model.

# Howevery it tells us that and apartment with same square meter should be around 16000 dkk cheaper/km.
# This tells us that 0 km from Nørreport Station an appartment of 1 square meter cost 16448.20355148 + 709106 = 725554 DKK.
# 5 km from Nørreport Station an appartment of 1 square meter cost 82115 dkk less.
