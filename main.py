import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
import seaborn as sns


# 2.1.1 acquire: daily trading data with open, close, high, low, volume, amount, yield, total market value
daily = pd.DataFrame()
for file in ["Final Exam Data/Daily Return 2007-01-01 to 2011-12-31/TRD_Dalyr.csv",
             "Final Exam Data/Daily Return 2007-01-01 to 2011-12-31/TRD_Dalyr1.csv",
             "Final Exam Data/Daily Return 2007-01-01 to 2011-12-31/TRD_Dalyr2.csv",

             "Final Exam Data/Daily Return 2012-01-01 to 2016-12-31/TRD_Dalyr.csv",
             "Final Exam Data/Daily Return 2012-01-01 to 2016-12-31/TRD_Dalyr1.csv",
             "Final Exam Data/Daily Return 2012-01-01 to 2016-12-31/TRD_Dalyr2.csv",
             "Final Exam Data/Daily Return 2012-01-01 to 2016-12-31/TRD_Dalyr3.csv",

             "Final Exam Data/Daily Return 2017-01-01 to 2021-12-31/TRD_Dalyr.csv",
             "Final Exam Data/Daily Return 2017-01-01 to 2021-12-31/TRD_Dalyr1.csv",
             "Final Exam Data/Daily Return 2017-01-01 to 2021-12-31/TRD_Dalyr2.csv",
             "Final Exam Data/Daily Return 2017-01-01 to 2021-12-31/TRD_Dalyr3.csv",
             "Final Exam Data/Daily Return 2017-01-01 to 2021-12-31/TRD_Dalyr4.csv",

             "Final Exam Data/Daily Return 2022-01-01 to 2023-10-27/TRD_Dalyr.csv",
             "Final Exam Data/Daily Return 2022-01-01 to 2023-10-27/TRD_Dalyr1.csv",
             "Final Exam Data/Daily Return 2022-01-01 to 2023-10-27/TRD_Dalyr2.csv"]:
    df = pd.read_csv(file, dtype={"Stkcd": str}, parse_dates=["Trddt"], index_col=["Trddt"])
    daily = pd.concat([daily, df])
daily.rename(columns={"Opnprc": "Open", "Hiprc": "High", "Loprc": "Low",
                      "Clsprc": "Close", "Dnshrtrd": "Volume", "Dretwd": "Yield",
                      "Dnvaltrd": "Value", "Dsmvtll": "TotalValue"}, inplace=True)
print(daily)


# 2.1.2 acquire: SHIBOR
shibor = pd.read_excel("SHIBOR/SHIBOR_LDAVGRATE.xlsx", skiprows=3, header=None)
shibor.columns = ["Trddt", "Market", "Term", "Currency", "InterestRate"]
shibor = shibor[shibor["Term"] == "O/N"]
shibor.drop(columns=["Term", "Market", "Currency"], inplace=True)
shibor["Trddt"] = pd.to_datetime(shibor["Trddt"], format='%Y-%m-%d', errors='coerce')
shibor = shibor.set_index(["Trddt"]).sort_index()
print(shibor)


# 2.1.3 acquire: quarterly balance sheet, resample to daily data
quarterly = pd.read_excel("Balance Sheet 2007-01-01 to 2023-09-30/FS_Combas.xlsx", dtype={"Stkcd": str})
quarterly["Accper"] = pd.to_datetime(quarterly["Accper"], format='%Y-%m-%d', errors='coerce')
quarterly.rename(columns={"Accper": "Trddt"}, inplace=True)
quarterly = quarterly.iloc[2:]
quarterly = quarterly[quarterly["Typrep"] != "B"]
quarterly.drop(columns=["Typrep", "ShortName"], inplace=True)
quarterly = quarterly.set_index(["Stkcd", "Trddt"]).sort_index()

quarterly_stockcodes = quarterly.index.get_level_values("Stkcd").unique()
quarterly_frames = []
for stock_code in quarterly_stockcodes:
    quarterly_stocks = quarterly.loc[stock_code].resample("D").ffill()
    quarterly_stocks.dropna(inplace=True)
    quarterly_stocks.reset_index(inplace=True)
    quarterly_stocks['Stkcd'] = stock_code
    quarterly_frames.append(quarterly_stocks)

quarterly_new = pd.concat(quarterly_frames)
quarterly_new = quarterly_new.set_index(["Stkcd", "Trddt"]).sort_index()
print(quarterly_new)


# 2.1.4 merge daily, shibor and quarterly
merged_stock = daily.join(shibor, how="inner")
merged_stock.dropna(inplace=True)
merged_stock.reset_index(inplace=True)
merged_stock = merged_stock.set_index(["Stkcd", "Trddt"]).sort_index()
merged = merged_stock.join(quarterly_new, how="inner")
merged.dropna(inplace=True)
print(merged)


# [optional] write into csv file, then read into python
# merged.to_csv("final.csv")
final = pd.read_csv("final.csv", dtype={"Stkcd": str})
final["Trddt"] = pd.to_datetime(final["Trddt"], format='%Y-%m-%d')
final = final.set_index(["Stkcd", "Trddt"]).sort_index()
print(final)


# 2.1.5 descriptive statistics: cross-sectional and time-series
# cross-sectional statistics
columns = ['Yield', 'Volume', 'Value', 'TotalValue', 'InterestRate']
statistics = {}

for column in columns:
    stats = final.groupby('Trddt')[column].agg(['mean', 'std', skew, kurtosis,
                                                'min', lambda x: np.percentile(x, 25),
                                                'median', lambda x: np.percentile(x, 75), 'max'])
    stats.columns = [f'{column}_{stat}' for stat in ['Mean', 'Std', 'Skew',
                                                     'Kurtosis', 'Min', '25thPercentile',
                                                     'Median', '75thPercentile', 'Max']]
    statistics[column] = stats

statistics_df = pd.concat([statistics[column] for column in columns], axis=1)
print(statistics_df)
# statistics_df.to_excel("cross_sectional.xlsx")


# time series statistics
time_series = {}
for column in columns:
    stats = {
        'Mean': final[column].mean(),
        'Std Dev': final[column].std(),
        'Skewness': skew(final[column]),
        'Kurtosis': kurtosis(final[column]),
        'Min': final[column].min(),
        '25th Percentile': final[column].quantile(0.25),
        'Median': final[column].median(),
        '75th Percentile': final[column].quantile(0.75),
        'Max': final[column].max()
    }
    time_series[column] = stats
time_series_df = pd.DataFrame(time_series)
print(time_series_df)
# time_series_df.to_excel("time_series.xlsx")


# 2.2.1 construct market portfolio, calculate RMKT
total_market_cap = final.groupby(level='Trddt')['TotalValue'].sum()
final['Weight'] = final['TotalValue'] / total_market_cap[final.index.get_level_values('Trddt')].values
daily_portfolio_return = final.groupby(level='Trddt')\
    .apply(lambda group: np.sum(group['Yield'] * group['Weight']))
final['MarketReturn'] = daily_portfolio_return - (final['InterestRate'] / 100)
print(final[["Yield", "InterestRate", "MarketReturn"]])


# 2.2.2 calculate RI
final["IndividualReturn"] = final["Yield"] - (final["InterestRate"] / 100)
print(final["IndividualReturn"])


# 2.2.3 create CAPM model for each individual stock in the sample
individual_results = {}
for stock_code in final.index.get_level_values('Stkcd').unique():
    stock_data = final.loc[final.index.get_level_values('Stkcd') == stock_code].copy()
    X = sm.add_constant(stock_data["MarketReturn"])
    y = stock_data["IndividualReturn"]
    try:
        model = sm.OLS(y, X)
        results = model.fit()
        individual_results[stock_code] = results

        # Print results for each stock
        print(f"\nResults for Stock {stock_code}:")
        print(results.summary())

    except Exception as e:
        print(f"Error for Stock {stock_code}: {e}")


# 2.2.4 plot one individual stock with its regression line
chosen_stock = '600507'

chosen_stock_data = final.loc[final.index.get_level_values('Stkcd') == chosen_stock].copy()
chosen_stock_data['IndividualReturn'] = (chosen_stock_data['Yield']) - (chosen_stock_data['InterestRate'] / 100)
chosen_stock_data['MarketReturn'] = daily_portfolio_return - (chosen_stock_data['InterestRate'] / 100)

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.scatterplot(x='MarketReturn', y='IndividualReturn', data=chosen_stock_data, alpha=0.5, color='blue')

X = sm.add_constant(chosen_stock_data['MarketReturn'])
model = sm.OLS(chosen_stock_data['IndividualReturn'], X).fit()

plt.plot(chosen_stock_data['MarketReturn'], model.predict(X), color='red', linewidth=2)

plt.title(f'Scatter Plot and Regression Line for {chosen_stock} Daily Excess Return vs Market Daily Excess Return')
plt.xlabel('Market Daily Excess Return (RMKT)')
plt.ylabel(f'{chosen_stock} Daily Excess Return (Ri)')

plt.show()
