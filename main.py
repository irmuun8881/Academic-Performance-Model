import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('/Users/JessFort/Documents/My_Coding_folder/Survey/survey.csv')
df = df.drop('Timestamp', axis=1)

# Working with the sleep column (have a good sleeping routine or not)
df['sleep'] = [4 if answer[0] != '3' else 10 for answer in df['sleep']]

# Working with the work column
df['work'] = df['work'].apply(lambda x: 0 if x.startswith('1') else 
                                   4 if x.startswith('2') else 
                                   6 if x.startswith('3') else 
                                   10)
# Working with the social media column
df['socialMedia'] = df['socialMedia'].apply(lambda x: 0 if x<=1 else 
                                   4 if x==2 else 
                                   6 if x==3 else 
                                   10)

df=df.astype(float)

# Preparing a label
y = np.array(df['grades'])

# Preparing features
df = df.drop('grades', axis=1)

# Calculate the average score for the label
avg_label_score = np.mean(y)

# Threshold 1 conditions to identify outliers
condition_1 = (df['sleep'] == 10) & ((df['work'] == 0) | (df['work'] == 4)) & ((df['socialMedia'] == 0) | (df['socialMedia'] == 3)) & (df['major'] > df['major'].mean()) & (y < avg_label_score)

# Threshold 2 conditions to identify outliers
condition_2 = (df['sleep'] == 4) & ((df['work'] == 6) | (df['work'] == 10)) & ((df['socialMedia'] == 6) | (df['socialMedia'] == 10)) & (df['major'] < df['major'].mean()) & (y > avg_label_score)

# Combine the conditions to get the final set of outliers
outliers = df[(condition_1) | (condition_2)].index

# Remove outliers from the dataframe
df= df.drop(outliers)
x = np.array(df)
y = y[df.index.difference(outliers)]

# Splitting for training set and testing set on cleaned data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)

# Train the model on cleaned data
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions on the cleaned test set
y_pred = model.predict(x_test).round()

# Evaluate the model on cleaned data
mse = mean_squared_error(y_test, y_pred)
# Calculate standard deviation of y_test on cleaned data
std_dev= np.std(y_test)
# Print results
print(f"Sample size after preprocessing: {len(y)}")
print(f"Mean Squared Error: {mse}")
print(f"Standard Deviation: {std_dev}")

coefficients = model.coef_

# Create a DataFrame to associate features with their coefficients
coef_df = pd.DataFrame({'Feature': df.columns, 'Coefficient': coefficients})

# Rank the features based on the absolute value of coefficients
coef_df['Absolute_Coefficient'] = np.abs(coef_df['Coefficient'])
ranked_coef_df = coef_df.sort_values(by='Absolute_Coefficient', ascending=False)

# Print the ranked coefficients
print("Feature Impact Ranking:")
for idx, row in ranked_coef_df.iterrows():
    print(f"{row['Feature']} is ranked {idx + 1} in terms of impact on academic performance.")

# Set up the scatter plot for cleaned data
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred)

# Plot the ideal y=x line
plt.plot([0, 10], [0, 10], linestyle='--', color='red', label='Ideal Line (y=x)')

# Set axis labels and plot title
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Grades")

# Set axis limits to 0-10 range
plt.xlim(0, 10)
plt.ylim(0, 10)

# Display legend
plt.legend()

# Show the plot
plt.show()
