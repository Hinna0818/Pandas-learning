## preparation for exp4 in biostats
import pandas as pd
from scipy import stats
from scipy.stats import t
import math

data = pd.read_csv("D:/大三/生物统计/实验04-20241117/Exp04_Data.csv")
data.head(10)

Burley = data[['Burley']]
Vermon = data[['Vermon']].dropna()


## t test in python
## 正态性检验(Q7)
stats.shapiro(Burley) ## p<0.05，不满足正态性假设
stats.shapiro(Vermon) ## p<0.05，不满足正态性假设

## Q1-Q2
stats.ttest_1samp(Burley, 3250)
stats.ttest_1samp(Vermon, 3250)

## Q3-Q4
stats.ttest_1samp(Burley, 3500)
stats.ttest_1samp(Vermon, 3500)

## Q5-Q6
stats.ttest_1samp(Burley, 3700)
stats.ttest_1samp(Vermon, 3700) ## 接受原假设

## Q8
stats.ttest_ind(Burley, Vermon)


## AQ3
n1 = 36
x1_mean = 40
x1_sd = 9

n2 = 49
x2_mean = 35
x2_sd = 10

sp = math.sqrt(((n1-1) * x1_sd + (n2-1) * x2_sd) / (n1+n2-2))
t = (x1_mean - x2_mean) / (sp * (math.sqrt(1/n1 + 1/n2)))
a = t.ppf(1-0.05/2, n1+n2-2)