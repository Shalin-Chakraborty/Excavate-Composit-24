# Predicting Glass Forming Ability of Metallic Glass

## Overview
This project aims to design and implement a machine learning model to predict the glass forming ability (GFA) of metallic glass, expressed as Dmax (in mm), using various intrinsic parameters. By leveraging AI and data science, we aim to reduce the need for extensive experimentation, saving time and resources.

## Scenario
Predicting the glass forming ability of metallic alloys has traditionally been a slow, trial-and-error process. This project uses machine learning models to analyze datasets of existing metallic glasses to predict the GFA of new alloys.

## Problem Statement
Design a robust and accurate predictive model to predict the Dmax of a metallic glass sample, given the input features:
- Total Electronegativity (TEN)
- Atomic Size Difference (d)
- Average Atomic Volume (VA)
- Mixing Entropy (Sm)
- Glass-Transition Temperature (Tg)
- Onset Crystallisation Temperature (Tx)
- Liquidus Temperature (Tl)

## Approach
To address the issue of a small dataset and single inputs for some Dmax values, we employed various data augmentation methodologies. The approach includes:
1. Exploratory Data Analysis (EDA)
2. Principal Component Analysis (PCA)
3. Data Preprocessing
4. Regression Model Implementation
5. Model Testing and Evaluation

## Steps

### 1. Exploratory Data Analysis (EDA)
- Analyzed the correlation between input features using a heatmap to identify and drop highly correlated features.
- Plotted the relationship of Dmax with different features to identify significant impacts.

### 2. Principal Component Analysis (PCA)
- Reduced the dimensionality of the dataset while preserving essential patterns.
- Transformed correlated variables into uncorrelated principal components.

### 3. Data Preprocessing
- Normalized the data using min-max scaling.
- Handled missing values and ensured data consistency.

### 4. Data Augmentation
- Applied Gaussian noise to create new augmented samples.
- Used SMOTE (Synthetic Minority Over-sampling Technique) to balance the class distribution.

### 5. Model: Regression
Explored multiple regression models including:
- Linear Regression
- Polynomial Regression
- Ridge Regression
- Support Vector Regression (SVR)
- Artificial Neural Network (ANN)
- XGBoost (Extreme Gradient Boosting)

### 6. Testing
- Kept 20% of the dataset for testing.
- Evaluated models using Mean Squared Error (MSE), R² Score, Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE).

## Results
- XGBoost emerged as the best model for predicting Dmax.
- Achieved an average MSE of 0.978274 across 10 training iterations.
- Other metrics included a high R² Score of 0.989941, indicating excellent model performance.

## Sustainability @ Metallic Glass
The project also highlights the sustainability measures in the use and manufacturing of metallic glass, emphasizing energy efficiency and resource optimization.

## Appendix
Includes detailed explanations and plots for:
- XGBoost
- PCA
- EDA
- Data Augmentation
- Regression Models
- Dataset Analysis
- Heatmaps and Normalization Techniques

## Conclusion
The project successfully developed a predictive model for the GFA of metallic glass, reducing experimental costs and paving the way for the discovery of new alloy compositions with superior properties.

## How to Use
1. **Install dependencies:** 
    ```bash
    pip install -r requirements.txt
    ```
2. **Run the notebook:** 
    Open and run the provided Jupyter notebook for detailed step-by-step execution.

3. **Data Preparation:** 
    Ensure the dataset is in the correct format as expected by the preprocessing script.

4. **Training the Model:** 
    Use the provided scripts to train the model and perform data augmentation as described.

5. **Evaluation:** 
    Evaluate the model using the provided test dataset and metrics.

6. **Visualization:** 
    Use the provided plotting scripts to visualize the results and understand the model performance.

## Contributors
- Data_Riders Team

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
