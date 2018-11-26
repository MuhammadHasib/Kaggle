import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import norm
from collections import Counter
from time import time
from matplotlib import collections  as mc

sns.set_style('whitegrid')

df = pd.read_csv("cities.csv")
df.head()

plt.figure(figsize=(15, 10))
plt.scatter(df.X, df.Y, s=1)
plt.scatter(df.iloc[0: 1, 1], df.iloc[0: 1, 2], s=10, c="red")
plt.grid(False)
plt.show()

nb_cities = max(df.CityId)
print("Number of cities to visit : ", nb_cities)

def sieve_eratosthenes(n):
    primes = [False, False] + [True for i in range(n-1)]
    p = 2
    while (p * p <= n):
        if (primes[p] == True):
            for i in range(p * 2, n + 1, p):
                primes[i] = False
        p += 1
    return primes

primes = np.array(sieve_eratosthenes(nb_cities)).astype(int)
df['Prime'] = primes
