# Correlation between Habits and Academic Satisfaction Survey

## Overview

This repository contains an analysis script focusing on exploring the potential correlation between daily habits and the academic satisfaction of college students. The analysis uses Python with libraries like Pandas, NumPy, Scikit-learn, Seaborn, and Matplotlib to investigate relationships within the provided survey data.

## Purpose

The primary objective of this analysis is to uncover potential connections between various daily habits, routines, and the overall satisfaction levels of college students concerning their academic performance. By examining different factors like wake-up times, part-time employment, social media usage, major satisfaction, and perceived academic performance satisfaction, we aim to identify any correlations that might exist.

## Analysis Workflow

The analysis script performs the following tasks:

- Cleans the survey data by categorizing and processing responses related to sleep patterns, work schedules, social media usage, and academic performance.
- Identifies outliers based on predefined thresholds and removes them from the dataset.
- Utilizes linear regression to model the relationship between various habits and academic satisfaction.
- Evaluates the model's performance using Mean Squared Error (MSE) and compares predictions against actual academic performance.

## Getting Started

### Requirements

- Python 3.6 or higher
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   pip install -r requirements.txt
