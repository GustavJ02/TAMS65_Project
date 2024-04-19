import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt

filename = 'dataframe'
filepath = filename + '.xlsx'
dataframe = pd.read_excel(filepath)
dataframe['z1'] = dataframe['x1'] / 1000
dataframe['z2'] = dataframe['z1'] * dataframe['x2']

explanatoryVariables = dataframe[['z1', 'z2']]
resultVariable = dataframe['y']

# Creating 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(explanatoryVariables['z1'], explanatoryVariables['z2'], resultVariable, c='r', marker='o')

# Setting labels and title
ax.set_xlabel('z1')
ax.set_ylabel('z2')
ax.set_zlabel('y')
ax.set_title('3D Scatter Plot')

# Show plot
plt.show()

correlation_matrix = dataframe.corr()
plt.figure(figsize=(10, 8))
cmap = sns.diverging_palette(10, 133, as_cmap=True)
sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=".2f", annot_kws={"size": 10})
plt.title("Correlation Matrix Heatmap")

for column_title in explanatoryVariables.columns:
    plt.figure()
    plt.scatter(dataframe[column_title], resultVariable, color='blue', label=f'# working hours lost vs {column_title}')
    plt.title(f'Scatter Plot of {column_title}')
    plt.ylabel(resultVariable.name)
    plt.xlabel(column_title)
    plt.legend()

plt.show()

# Observed z-values and y values
Z = np.array(explanatoryVariables)
Y = np.array(resultVariable)

# Create matrix X for least squares
n = Z.shape[0]
try:
    k = Z.shape[1] + 1
except:
    k = 2

X = np.c_[np.ones(n), Z]

# Compute beta using least squares
stdErrMatrix = np.linalg.inv(X.T @ X)
beta = stdErrMatrix @ X.T @ Y

dataframe['mu'] = X @ beta

yBar = resultVariable.mean()
muBar = dataframe['mu'].mean()

SStot = 0
for value in resultVariable:
    SStot += (value - yBar) ** 2

SSr = 0
for value in dataframe['mu']:
    SSr += (value - yBar) ** 2

SSe = 0
for y, mu in zip(resultVariable, dataframe['mu']):
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

# Calculate a 95% 2-sided confidence interval


def confidance(var, error, degreesOfFreedom):
    t_value = stats.t.ppf(0.975, degreesOfFreedom)
    lower = var - t_value * error
    upper = var + t_value * error
    return f"[{lower}, {upper}]"


print("\n-----  Variable results  -----")
for i in range(0, k):
    stdErr = np.sqrt(s2 * stdErrMatrix[i, i])
    TS = beta[i] / stdErr
    print("Beta{} = {}; Standard error: {}, TS: {}, 95% confidence intervall: {}".format(i, beta[i], stdErr, TS, confidance(beta[i], stdErr, n - k - 1)))


dataframe['dev'] = dataframe['y'] - dataframe["mu"]

n = dataframe['dev'].size - 1

s = 0
for value in dataframe['dev'].to_list():
    s += value**2

sampleVarience = s/n

sampleVarience = sqrt(sampleVarience)
print(f"sampleVarience: {sampleVarience}")

plt.figure()
plt.scatter(dataframe['z1'], dataframe['dev'], color='blue', label=f'Deviation vs z1')
plt.ylabel("Deviation")
plt.xlabel("$z_1$")
plt.legend()

plt.figure()
plt.hist(dataframe['dev'], 10, (-25, 25), density=True)
# x-axis ranges from -3 and 3 with .001 steps
x = np.linspace(-25, 25, 1000)

# plot normal distribution with mean 0 and standard deviation 1
plt.plot(x, norm.pdf(x, 0, sampleVarience))

plt.figure()

# Create histogram
n, bins, patches = plt.hist(dataframe['dev'], 10, (-25, 25))

# Annotate each bin with the number of elements
for i in range(len(n)):
    plt.text(bins[i], n[i], str(int(n[i])), ha='center', va='bottom')


plt.show()

with pd.ExcelWriter(filename + 'Result.xlsx', engine='xlsxwriter') as writer:
    dataframe.to_excel(writer, sheet_name='Result', index=False)
