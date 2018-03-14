import pandas as pd
import numpy as np
import random as rnd
# visualization
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('./experiment_data.csv')
data.columns.values

f1 = pd.DataFrame(data) 
f1

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
plt.figure(figsize=(8,6))
acc=plt.plot(data["Mixing-Error Ratios"], data.Accuracy, sns.xkcd_rgb[colors[0]], lw=3, marker='o');
rec=plt.plot(data["Mixing-Error Ratios"], data.Recall, sns.xkcd_rgb[colors[1]], lw=3, marker='o');
pre=plt.plot(data["Mixing-Error Ratios"], data.Precision, sns.xkcd_rgb[colors[2]], lw=3, marker='o');
f1=plt.plot(data["Mixing-Error Ratios"], data["F1-Score"], sns.xkcd_rgb[colors[3]], lw=3, marker='o');
cor=plt.plot(data["Mixing-Error Ratios"], data["Correction Rate"], sns.xkcd_rgb[colors[4]], lw=3, marker='o');
cor=plt.plot(data["Mixing-Error Ratios"], data["Specificity"], sns.xkcd_rgb['red'], lw=3, marker='o');
plt.xlabel('Mixing-Error Ratios', fontsize=12)
plt.ylabel('value', fontsize=12)
plt.legend(loc='lower right')
plt.title('Mesurements aginst diffrent ratios', color='k', fontsize=12)
#plt.grid(True, linestyle = "-.", color = "k", linewidth = "1")
#grid = sns.pointplot(data["Mixing-Error Ratios"], data.Accuracy, alpha=0.8, color="r");

plt.show()