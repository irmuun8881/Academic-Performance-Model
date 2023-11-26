import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess the data
df = pd.read_csv('/Users/JessFort/Documents/My_Coding_folder/Survey/data/survey.csv')
df.drop('Timestamp', axis=1, inplace=True)
df['sleep'] = [4 if str(answer)[0] != '3' else 10 for answer in df['sleep']]
df['work'] = df['work'].apply(lambda x: 0 if str(x).startswith('1') else 4 if str(x).startswith('2') else 6 if str(x).startswith('3') else 10)
df['socialMedia'] = df['socialMedia'].apply(lambda x: 0 if x <= 1 else 4 if x == 2 else 6 if x == 3 else 10)
df = df.astype(float)

# Generate histograms for the numerical variables
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['socialMedia'], kde=False, bins=5)
plt.title('Distribution of Social Media Usage')

plt.subplot(1, 3, 2)
sns.histplot(df['major'], kde=False, bins=8)
plt.title('Distribution of Major Satisfaction')

plt.subplot(1, 3, 3)
sns.histplot(df['grades'], kde=False, bins=8)
plt.title('Distribution of Grades')

plt.tight_layout()
plt.show()

# Generate boxplots for the same variables
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(y=df['socialMedia'])
plt.title('Boxplot of Social Media Usage')

plt.subplot(1, 3, 2)
sns.boxplot(y=df['major'])
plt.title('Boxplot of Major Satisfaction')

plt.subplot(1, 3, 3)
sns.boxplot(y=df['grades'])
plt.title('Boxplot of Grades')

plt.tight_layout()
plt.show()

# Create a correlation matrix and visualize it using a heatmap
correlation_matrix = df[['socialMedia', 'major', 'grades', 'sleep', 'work']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()
