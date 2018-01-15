import csv
import numpy as np
import matplotlib as mp
mp.use('Agg')
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import tushare as ts

PORTFOLIOS = ['601939', '600036', '600016','601398','601988','000001']

#PORTFOLIOS = ['601939']

def get_data(code):
	dates = []
	prices = []
	stock_close_prices = ts.get_hist_data(stock)[0:31].loc[:, ['close']].sort_index(axis=0,ascending=True)
	#print(stock_close_prices)
	i = 0
	for row in stock_close_prices.values:
		#print(row)
		#dates.append(int(row[0].split('-')[2]))
		dates.append(i)
		prices.append(float(row[0]))
		i += 1
		#print(dates, prices)
	return dates, prices

def predict_price(dates, prices, x,stock):
	
	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1
		
	svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
	svr_rbf.fit(dates, prices) # fitting the data points in the models
	
	svr_poly.fit(dates, prices)
	
	plt.scatter(dates, prices, color= 'black') # plotting the initial datapoints 
	plt.plot(dates, svr_rbf.predict(dates), color= 'red') # plotting the line made by the RBF kernel
	
	plt.plot(dates,svr_poly.predict(dates), color= 'blue') # plotting the line made by polynomial kernel
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Recent 30 days of SVR model')
	plt.legend()
	plt.savefig(stock + '.png')

	return svr_rbf.predict(x)[0], svr_poly.predict(x)[0]

def write_predict_to_google_sheet():
	pass

if __name__ == '__main__':

	for stock in PORTFOLIOS:
		dates, prices = get_data(stock)
		predict_price(dates,prices,31,stock)
