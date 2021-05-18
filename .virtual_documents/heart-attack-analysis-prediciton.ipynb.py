import os
import numpy as np
import pandas as pd

# for visualizations 
import matplotlib.pyplot as plt
import seaborn as sns

# For modeling




df = pd.read_csv("heart.csv")
df.head()


df.info()


df.describe()


outcome_counts = df["output"].value_counts()
heart_attack = outcome_counts[1]
no_heart_attack = outcome_counts[0]

print("heart attack: {:.2f}get_ipython().run_line_magic("".format(heart_attack/len(df)", " * 100))")
print("no heart attack: {:.2f}get_ipython().run_line_magic("".format(no_heart_attack/len(df)", " * 100))")

sns.barplot(x = ['heart attack', 'no heart attack'], y = [heart_attack, no_heart_attack])
plt.title("Heart attack vs no heart attack totals")
plt.show()



sns.countplot(data=df, x='sex', hue='output')
plt.title("Heart attack vs no heart attack totals wrt sex")
plt.show()


counts_gender = df.groupby('sex')['output'].value_counts()
male_heart_attack_num = counts_gender[1][1]
female_heart_attack_num = counts_gender[0][1]

male_heart_attack_percentage = male_heart_attack_num/(counts_gender[1][1] + counts_gender[1][0])*100
female_heart_attack_percentage = female_heart_attack_num/(counts_gender[0][1] + counts_gender[0][0])*100

print("male heart attack percentage: {:.2f}get_ipython().run_line_magic("".format(male_heart_attack_percentage))", "")
print("female heart attack percentage: {:.2f}get_ipython().run_line_magic("".format(female_heart_attack_percentage))", "")



sns.histplot(data=df, x='age', kde=True)
plt.title("Age distribution")
plt.show()

# view sex differences 
sns.histplot(data=df, x='age', hue='sex', multiple="stack")
plt.title("Age distribution wrt sex")
plt.show()



# exercise induced angina distribution 
sns.countplot(data=df, x = 'exng')
plt.title("Exercise induced Angina")
plt.show()



# chest pain type distribution 
sns.countplot(data=df, x = 'cp', hue = 'output')
plt.title("Chest pain wrt to heart attack")
plt.show()


# resting blood pressure wrt Heart attack
sns.displot(data=df, x = 'trtbps', hue='output', kind='kde')
plt.title("Resting blood pressure wrt heart attack")
plt.show()


# cholesterol wrt heart attack
sns.displot(data=df, x = 'chol', hue='output', kind='kde')
plt.title("Cholesterol wrt heart attack")
plt.show()


# max heart rate reached wrt to heart attack
sns.displot(data=df, x = 'thalachh', hue='output', kind='kde')
plt.title("Maximum HR wrt heart attack")
plt.show()


# fasting blood sugar wrt Heart Attack
sns.countplot(data=df, x = 'fbs', hue='output')
plt.title("Fasting blood sugar wrt heart attack")
plt.show()
print("Reminder:\nfbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)")


# correlation matrix
plt.figure(figsize=(15,5))
sns.heatmap(df.corr(), annot=True)
plt.title("Atttribute Correlation Heatmap")
plt.show()


# load libraries for model building








# split data
from sklearn.model_selection import train_test_split




# Logistic Regression


# kNN


# Decision Tree


# SVC






