import os
import pandas as pd
from datetime import datetime, timedelta
import util
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from math import sqrt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def parser(x):
    return datetime.fromtimestamp(float(x)).isoformat()


def datasetRead():
    return pd.read_csv(
        os.path.join(os.getcwd(), "2nd question", "datasets", "lbl-conn-dataset-30" + ".red"),
        sep=' ',
        header=0,
        index_col=0,
        parse_dates=[0],
        squeeze=True,
        date_parser=parser,
        names=["timestamp", "duration", "protocol", "bytesOriginator", "bytesResponder", "localhost", "remotehost", "A",
               "B"])


def q1remoteHosts():
    # Question 3
    # Copying the pandas df in order to further modify it.
    df = table.copy()

    # Group By the remote hosts and find the total connections of it with lbl localhosts
    # We count duplicates too since there is not any requirement for unique connections
    groupedDfMax = df.groupby("remotehost")["localhost"].sum().idxmax()

    # Getting all the unique localhosts the remotehosts has connected with
    connections = df.groupby("remotehost")["localhost"].unique().agg(lambda
                                                                         x: list(x))

    print("The remoteHost with the most connections is: ", groupedDfMax)
    print("The unique Localhost connections this host made are", connections[groupedDfMax])


def q2LocalHosts():
    df = table.copy()
    # Replacing the empty fields
    df.drop(df['bytesResponder'].loc[df['bytesResponder'] == '?'].index, inplace=True)

    df['bytesResponder'] = df['bytesResponder'].astype(str).astype(int)

    groupedDf = df.groupby("localhost")["bytesResponder"].sum().idxmax()

    print("The local host with the most largest number of inbound data is the localhost: ", groupedDf)


def q3ProtocolDuration():
    df = table.copy()
    # removing all the lines that have "?", undefined value in our case
    df.drop(df['duration'].loc[df['duration'] == '?'].index, inplace=True)

    out = df.groupby("protocol").agg({'duration': ['min', 'max']})

    print("The min and max duration values of each protocol are: \n", out)


def q4ReplaceWithAverage():
    df = table.copy()
    df2 = table.copy()
    df3 = table.copy()

    df.drop(df['bytesOriginator'].loc[df['bytesOriginator'] == '?'].index, inplace=True)
    df2.drop(df2['bytesResponder'].loc[df2['bytesResponder'] == '?'].index, inplace=True)

    df['bytesOriginator'] = df['bytesOriginator'].astype(str).astype(float)
    df2['bytesResponder'] = df2['bytesResponder'].astype(str).astype(float)

    meanOr = df.groupby('protocol').agg({'bytesOriginator': ['mean']}).reset_index()
    meanRe = df2.groupby('protocol').agg({'bytesResponder': ['mean']}).reset_index()

    dfm = df3.loc[df3['bytesResponder'] == "?"]
    df3.drop(df3['bytesResponder'].loc[df3['bytesResponder'] == '?'].index, inplace=True)

    dfm = dfm.merge(meanRe[['bytesResponder', 'protocol']], on='protocol', how="left")

    dfm.iloc[:, [6]] = dfm.iloc[:, [6]].fillna(dfm['bytesResponder'])

    dfm.drop(['bytesResponder'], inplace=True, axis=1)

    dfm.rename(columns={dfm.columns.values.tolist()[5]: 'bytesResponder'}, inplace=True)

    df3 = df3.append(dfm, ignore_index=True)
    df3 = df3.sort_values('bytesOriginator', ignore_index=True)

    print(df3.head(100).to_string())
    return df3
    # df.drop(df['bytesOriginator'].loc[df['bytesOriginator'] == '?'].index, inplace=True)
    #
    # df['bytesOriginator'] = df['bytesOriginator'].astype(str).astype(float)
    # # #geting the mean values
    # out = df.groupby("protocol").agg({'bytesOriginator': ['mean']})
    #
    # df2.loc[(df2["bytesOriginator"] == "?"), "bytesOriginator"] = 123
    #
    # print(df2.to_string())


def q5MinTraffic():
    df = table.copy()

    df.drop(df['bytesOriginator'].loc[df['bytesOriginator'] == '?'].index, inplace=True)
    df.drop(df['bytesResponder'].loc[df['bytesResponder'] == '?'].index, inplace=True)

    df['bytesOriginator'] = df['bytesOriginator'].astype(str).astype(int)
    df['bytesResponder'] = df['bytesResponder'].astype(str).astype(int)

    grouped = df.groupby('timestamp')["bytesOriginator", "bytesResponder"].agg(['sum'])

    total = grouped["bytesOriginator"] + grouped["bytesResponder"]

    print("The time of day with the less traffic is: ", total.idxmax()[0])


def q610Traffic():
    series = table.copy()

    series.drop(series['bytesOriginator'].loc[series['bytesOriginator'] == '?'].index, inplace=True)
    series.drop(series['bytesResponder'].loc[series['bytesResponder'] == '?'].index, inplace=True)

    series['bytesOriginator'] = series['bytesOriginator'].astype(str).astype(int)
    series['bytesResponder'] = series['bytesResponder'].astype(str).astype(int)

    series2 = series[["bytesResponder", "bytesOriginator"]]

    print(series2.resample('10min').sum())


def q7TrafficTimePlot():
    df = table.copy()

    df.drop(df['bytesOriginator'].loc[df['bytesOriginator'] == '?'].index, inplace=True)
    df.drop(df['bytesResponder'].loc[df['bytesResponder'] == '?'].index, inplace=True)

    df['bytesOriginator'] = df['bytesOriginator'].astype(str).astype(int)
    df['bytesResponder'] = df['bytesResponder'].astype(str).astype(int)

    series2 = df[["bytesResponder", "bytesOriginator"]]

    series2 = df["bytesResponder"] + df["bytesOriginator"]

    series2.plot()
    pyplot.show()


def q8DecomposeTrend():
    df = table.copy()

    df.drop(df['bytesOriginator'].loc[df['bytesOriginator'] == '?'].index, inplace=True)
    df.drop(df['bytesResponder'].loc[df['bytesResponder'] == '?'].index, inplace=True)

    df['bytesOriginator'] = df['bytesOriginator'].astype(str).astype(int)
    df['bytesResponder'] = df['bytesResponder'].astype(str).astype(int)

    result = seasonal_decompose(df["bytesResponder"] + df['bytesOriginator'], model="additive", period=60)
    result.plot()
    pyplot.show()


def q9ARMA():
    df = table.copy()

    df.drop(df['bytesOriginator'].loc[df['bytesOriginator'] == '?'].index, inplace=True)
    df.drop(df['bytesResponder'].loc[df['bytesResponder'] == '?'].index, inplace=True)

    df['bytesOriginator'] = df['bytesOriginator'].astype(str).astype(int)
    df['bytesResponder'] = df['bytesResponder'].astype(str).astype(int)

    X = df['bytesResponder'].values
    size = int(len(X) * 0.70)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()

    for t in range(len(test)):
        model = sm.tsa.arima.ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))

    return test, yhat


def q10ARMATest():
    test, predictions = q9ARMA()

    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()


if __name__ == "__main__":
    # Store it in global variable lines
    table = datasetRead()
    # Droping the last two necessary columns
    table.drop(['A', 'B'], axis=1, inplace=True)

    # Question 1
    # q1remoteHosts()

    # Question 2
    # q2LocalHosts()

    # Question 3
    # q3ProtocolDuration()

    # Question 4
    # q4ReplaceWithAverage()

    # Question 5
    # q5MinTraffic()

    # Question 6
    # q610Traffic()

    # Question7
    # q7TrafficTimePlot()

    # Question 8
    # q8DecomposeTrend()

    # Question 9
    q9ARMA()

    # Question 10
    # q10ARMATest()
