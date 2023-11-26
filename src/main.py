import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess the data
df = pd.read_csv('/Users/JessFort/Documents/My_Coding_folder/Survey/data/survey.csv')
df.drop('Timestamp', axis=1, inplace=True)
df['sleep'] = [4 if answer[0] != '3' else 10 for answer in df['sleep']]
df['work'] = df['work'].apply(lambda x: 0 if x.startswith('1') else 4 if x.startswith('2') else 6 if x.startswith('3') else 10)
df['socialMedia'] = df['socialMedia'].apply(lambda x: 0 if x <= 1 else 4 if x == 2 else 6 if x == 3 else 10)
df = df.astype(float)

# Identify outliers
condition_1 = (df['sleep'] == 10) & ((df['work'] == 0) | (df['work'] == 4)) & ((df['socialMedia'] == 0) | (df['socialMedia'] == 3)) & (df['major'] > df['major'].mean())
condition_2 = (df['sleep'] == 4) & ((df['work'] == 6) | (df['work'] == 10)) & ((df['socialMedia'] == 6) | (df['socialMedia'] == 10)) & (df['major'] < df['major'].mean())
outliers = df[(condition_1) | (condition_2)].index

df.drop(outliers, inplace=True)
y = df['grades'].values
X = df.drop('grades', axis=1).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=40)

# Train and evaluate the model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test).round()
mse = mean_squared_error(y_test, y_pred)
std_dev = np.std(y_test)
print(f"Sample size after preprocessing: {len(y)}")
print(f"Mean Squared Error: {mse}")
print(f"Standard Deviation: {std_dev}")

# Analyze feature importance
coef_df = pd.DataFrame({'Feature': df.columns.drop('grades'), 'Coefficient': model.coef_})
coef_df['Absolute_Coefficient'] = np.abs(coef_df['Coefficient'])
ranked_coef_df = coef_df.sort_values(by='Absolute_Coefficient', ascending=False)
print("Feature Impact Ranking:")
for idx, row in ranked_coef_df.iterrows():
    print(f"{row['Feature']} is ranked {idx + 1} in terms of impact on academic performance.")

# Visualize the results
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([0, 10], [0, 10], linestyle='--', color='red', label='Ideal Line (y=x)')
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Grades")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.legend()
plt.show()
