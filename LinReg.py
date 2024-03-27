import numpy as np
import pandas as pd

filepath = 'C:\DataAnalytics\dataframe.xlsx'
dataframe = pd.read_excel(filepath)
dataframe['z1'] = dataframe['x1'] / 1000
dataframe['z2'] = dataframe['z1'] * dataframe['x2']

# Observed z-values and y values
Z = np.array(dataframe[['z1', 'z2']])
Y = np.array(dataframe['y'])

# Create matrix X for least squares
n = Z.shape[0]
k = Z.shape[1] + 1
X = np.c_[np.ones(n), Z]

# Compute beta using least squares
stdErrMatrix = np.linalg.inv(X.T @ X)
beta = stdErrMatrix @ X.T @ Y

dataframe['mu'] = X @ beta

yBar = dataframe['y'].mean()
muBar = dataframe['mu'].mean()

SStot = 0
for value in dataframe['y']:
    SStot += (value - yBar) ** 2

SSr = 0
for value in dataframe['mu']:
    SSr += (value - yBar) ** 2

SSe = 0
for y, mu in zip(dataframe['y'], dataframe['mu']):
    SSe += (y - mu) ** 2

# Calculate R2 value
R2 = SSr / SStot

# Estimate variance
s2 = SSe / (n - k - 1)

print("-----  Raw Data  -----")
print(dataframe)
print()

print("-----  Result  -----")
print("Beta: ", beta)
print("R2: ", R2)
print("Residual standard error: ", np.sqrt(s2))
print("SStot: ", SStot)
print("SSr: ", SSr)
print("SSe: ", SSe)

print("\n-----  Variable results  -----")
for i in range(0, k):
    stdErr = np.sqrt(s2 * stdErrMatrix[i, i])
    TS = beta[i] / stdErr
    print("Beta{} = {}; Standard error: {}, TS: {}".format(i, beta[i], stdErr, TS))


with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
    dataframe.to_excel(writer, sheet_name='Result', index=False)
