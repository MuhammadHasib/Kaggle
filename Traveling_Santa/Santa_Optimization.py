import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import os
print(os.listdir("../input"))

pd.options.mode.chained_assignment = None  # default='warn'

# calculate total distance of the path
def total_distance(dfcity,path):
    prev_city = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        next_city = city_num
        total_distance = total_distance + \
            np.sqrt(pow((dfcity.X[city_num] - dfcity.X[prev_city]),2) + pow((dfcity.Y[city_num] - dfcity.Y[prev_city]),2)) * \
            (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[prev_city]))))
        prev_city = next_city
        step_num = step_num + 1
    return total_distance

santa_cities = pd.read_csv('../input/cities.csv')
santa_cities.head()
santa_cities.describe()

def sieve_of_eratosthenes(n):
    n = int(n)
    primes = [True for i in range(n+1)] # Start assuming all numbers are primes
    primes[0] = False # 0 is not a prime
    primes[1] = False # 1 is not a prime
    for i in range(2,int(np.sqrt(n)) + 1):
        if primes[i]:
            k = 2
            while i*k <= n:
                primes[i*k] = False
                k += 1
    return(primes)

# find cities that are prime numbers
prime_cities = sieve_of_eratosthenes(max(santa_cities.CityId))
santa_cities['prime'] = prime_cities

# add a few columns
santa_cities = pd.concat([santa_cities,pd.DataFrame(columns=['next_city',
                                                             'next_city_distance'
                                                            ]
                                                   )],sort=False)
santa_cities[['next_city','next_city_distance']] = santa_cities[['next_city','next_city_distance']].astype(float)

santa_cities.head(10
