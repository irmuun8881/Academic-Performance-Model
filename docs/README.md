
# Academic Performance Satisfaction Analysis

## Overview
This project aims to identify factors that impact students' satisfaction with their academic performance. Recognizing that direct queries about GPA could affect the candidness of responses, we used a satisfaction score as a proxy measure for academic performance.

## Data Exploration
Initial exploration with histograms and boxplots provided insights into the distribution and spread of variables like 'Social Media Usage', 'Major Satisfaction', and 'Grades'. A correlation matrix heatmap was used to discern potential relationships between variables.

## Outlier Detection
An initial attempt at automatic outlier detection removed a significant portion of the data, which was deemed too aggressive. A manual approach was taken, leading to the removal of 2 outliers for a more balanced dataset.

## Model Selection and Evaluation
Linear Regression emerged as the most accurate model, providing a mean squared error of approximately 1.4. The model's effectiveness suggests that it can reliably predict academic performance satisfaction from the given features.

## Feature Impact
The analysis demonstrated that the 'sleep' variable was the most impactful on grades, which is supported by the model's weights and confirmed by the heatmap illustration.

![Feature Impact](plot0.35.png)

## Conclusion
Despite the small sample size, the analysis successfully extracted meaningful insights, highlighting the significant influence of sleep patterns on academic satisfaction. This underscores the need to consider lifestyle factors in educational outcomes.

---
### Requirements

- Python 3.6 or higher
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   pip install -r requirements.txt