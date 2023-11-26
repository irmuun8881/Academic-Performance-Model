import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
survey_data = pd.read_csv('/Users/JessFort/Documents/My_Coding_folder/Survey/data/survey.csv')

# Convert categorical variables to numerical codes for correlation analysis
survey_data['sleep_code'] = survey_data['sleep'].astype('category').cat.codes
survey_data['work_code'] = survey_data['work'].astype('category').cat.codes

# Generate histograms for the numerical variables
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(survey_data['socialMedia'], kde=False, bins=5)
plt.title('Distribution of Social Media Usage')

plt.subplot(1, 3, 2)
sns.histplot(survey_data['major'], kde=False, bins=8)
plt.title('Distribution of Major Satisfaction')

plt.subplot(1, 3, 3)
sns.histplot(survey_data['grades'], kde=False, bins=8)
plt.title('Distribution of Grades')

plt.tight_layout()
plt.show()

# Generate boxplots for the same variables
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(y=survey_data['socialMedia'])
plt.title('Boxplot of Social Media Usage')

plt.subplot(1, 3, 2)
sns.boxplot(y=survey_data['major'])
plt.title('Boxplot of Major Satisfaction')

plt.subplot(1, 3, 3)
sns.boxplot(y=survey_data['grades'])
plt.title('Boxplot of Grades')

plt.tight_layout()
plt.show()

# Create a correlation matrix and visualize it using a heatmap
correlation_matrix = survey_data[['socialMedia', 'major', 'grades', 'sleep_code', 'work_code']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()
