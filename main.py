#!python3.5

import pandas, re, datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import arima, train_rbfn, numpy as np
from genetic_alg import GeneticAlgorithm

parser  = lambda date : pandas.datetime.strptime(re.compile('^(\d*)').match(date).group(), '%Y')
columns = ['YEAR', 'INPUT', 'OUTPUT']
file    = 'data.csv'

data = pandas.read_csv(file, sep=',', header=None, names=columns, usecols=columns, parse_dates=[0], date_parser=parser, index_col=[0])
training_set = data[dt.date(1700,1,1) : dt.date(1920,1,1)]
test_set = data[dt.date(1921,1,1) : dt.date(1987,1,1)]

# model = arima.auto(data, 3)

ga = GeneticAlgorithm(training_set, test_set)
ga.init_pop(60)
rbfn = ga.evolve(2)

# input = np.asarray([[x.tolist()] for x in training_set["INPUT"].values])
# output = np.asarray([[x.tolist()] for x in training_set["OUTPUT"].values])
# rbfn = train_rbfn.train(input, output, input)
rbfn.aic(training_set, True)
