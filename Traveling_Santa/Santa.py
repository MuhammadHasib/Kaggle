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
