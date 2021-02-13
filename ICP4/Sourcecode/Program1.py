# Importing pandas package as pd
import pandas as pd
# Loading train.csv file using pandas
train_df = pd.read_csv('train.csv')
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
# Finding the correlation between the Sex Feature and Survived target variable
train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
# Printing the correlation
print(train_df['Survived'].corr(train_df['Sex']))
# Printing how many females and males survived (1=female : 0=male)
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))