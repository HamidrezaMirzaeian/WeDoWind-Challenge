# WeDoWind - Wind Turbine Predictive Maintenance

A comprehensive machine learning pipeline for predicting Time-to-Failure (TTF) of wind turbines using SCADA data.

## Features
- Data preprocessing and feature engineering
- Unsupervised learning with K-means clustering
- Time-to-Failure target calculation
- Bayesian hyperparameter optimization
- Leave-One-Turbine-Out cross-validation
- XGBoost and Random Forest models
- SHAP explainability analysis
- Model persistence and resume capabilities

## Project Structure
\\\
WeDoWind/
├── src/                    # Source code
├── data/                   # Dataset files
├── notebooks/              # Jupyter notebooks 
├── results/                # Output results
├── requirements.txt        # Python dependencies
└── README.md              
\\\


++ DetectionUsingSMOTE
│Fault Detection using RFClassifier and SMOTE Technique for Mitigating Imbalance Data Challenge
├── 1. Data Preparation for Classification
│   ├── Create a binary 'failure' target from the 'errorcode' feature
│   ├── Define features (X_class) and target (y_class) for classification
│   └── Split data into training and test sets while maintaining class distribution
│
├── 2. Imbalance Handling with SMOTE
│   ├── Check and print the original imbalanced class distribution
│   ├── Apply SMOTE to the training data to balance the classes
│   ├── Print the new balanced class distribution
│   └── Visualize the class distribution before and after SMOTE using bar plots
│
├── 3. Model Training and Evaluation (with and without SMOTE)
│   ├── Train a Random Forest Classifier on the SMOTE-enhanced training data
│   ├── Evaluate the model on the original test data
│   ├── Print the classification report and plot a confusion matrix for the SMOTE model
│   ├── Train a baseline Random Forest Classifier without SMOTE for comparison
│   └── Print the classification report and plot a confusion matrix for the baseline model
│
├── 4. Performance Comparison
│   ├── Calculate predicted probabilities for both models
│   ├── Plot and compare the ROC curves for both the SMOTE and non-SMOTE models
│   └── Display the AUC score for each model to quantify performance
│
├── 5. Per-Turbine Failure Analysis and Visualization
│   ├── Define `plot_failure_comparison` and `plot_failure_comparison_nosmote` functions
│   │   ├── Plot actual failures as scatter points
│   │   ├── Plot predicted failure probability as a line
│   │   └── Plot binary predicted failures as 'x' markers
│   ├── For each turbine:
│   │   ├── Plot the failure comparison for the model trained **with** SMOTE
│   │   └── Plot the failure comparison for the model trained **without** SMOTE
│   └── Print the classification report and confusion matrix for each individual turbine
│
├── 6. Time-Windowed Failure Visualization
│   ├── Define `plot_two_month_comparison` to visualize predictions over a specific period
│   ├── Find a 2-month period for each turbine that contains a failure
│   └── For each of the first few turbines:
│       ├── Plot the 2-month failure prediction comparison for the SMOTE-enhanced model
│       └── Plot the 2-month failure prediction comparison for the standard model (no SMOTE)
│




++PrognosisTTFSimple
│Fault Prognosis using Feature Engineering, Clustering, and SHAP Analysis
├── 1. Data Loading & Initial Preprocessing
│   ├── Import all required libraries (pandas, numpy, scikit-learn, etc.)
│   ├── Load dataset from 'wedowind.csv'
│   ├── Drop 'Unnamed: 0' and filter for operational data
│   └── Handle missing values by dropping rows
│
├── 2. Feature Engineering
│   ├── Create 'hour' and 'day_of_week' features
│   ├── Create rolling average and standard deviation features
│   └── Create 'power_per_rpm' stress indicator
│
├── 3. Unsupervised Learning (Clustering)
│   ├── Prepare features for clustering
│   ├── Scale features using StandardScaler
│   ├── Perform PCA for dimensionality reduction
│   ├── Plot PCA explained variance
│   ├── Use Elbow method to determine optimal cluster number
│   ├── Fit K-means with the optimal number of clusters (e.g., 5)
│   └── Assign cluster labels to the dataset
│
├── 4. Target Engineering (Create Time-to-Failure Dataset)
│   ├── Create binary 'failure' indicator
│   ├── For each turbine group, apply TTF calculation function
│   └── Filter the final TTF dataset to include only relevant records
│
├── 5. Supervised Learning Setup
│   ├── Define features (X) and target (y = 'ttf')
│   ├── Define groups for the cross-validation
│   └── Initialize LeaveOneGroupOut
│
├── 6. Baseline Model Training & Prediction
│   └── For each unique turbine (using Leave-One-Turbine-Out CV):
│       ├── Split data into train and test sets
│       ├── Initialize a baseline XGBoost Regressor with fixed parameters
│       ├── Fit the model on the training data
│       ├── Predict on the test data for the held-out turbine
│       └── Calculate and store MAE and predictions
│
├── 7. Results Analysis & Model Explainability
│   ├── Print a formatted table of MAE results for each turbine
│   ├── Plot the TTF comparison for each turbine (Actual vs. Predicted)
│   └── Perform SHAP analysis on the last trained model:
│       ├── Initialize a SHAP explainer
│       ├── Generate SHAP values for the test set
│       └── Create a summary plot to visualize feature importance and impact
│

++ PrognosisTTFOptimized
│XGB and RF Regressor + Bayes Hyperparameters Optimization
├── 1. Initial Data Loading and Preprocessing
│   ├── Load dataset from 'wedowind.csv' and perform basic cleaning
│   ├── Filter out non-operational data points
│   └── Check for and handle missing values by dropping rows
│
├── 2. Feature Engineering and Scaling
│   ├── Extract temporal features ('hour', 'day_of_week')
│   ├── Create rolling statistics for key sensor readings
│   ├── Engineer equipment stress indicators like 'power_per_rpm'
│   └── Scale all numerical features using `StandardScaler`
│
├── 3. Unsupervised Learning for Pattern Discovery
│   ├── Use the Elbow Method to determine the optimal number of clusters
│   ├── Fit K-means clustering to the scaled features (using 5 clusters)
│   ├── Perform PCA for dimensionality reduction and visualization
│   └── Visualize the clusters in the PCA-reduced space
│
├── 4. Target Engineering (Time-to-Failure Calculation)
│   ├── Create a binary 'failure' indicator
│   ├── Define and apply a function to calculate Time-to-Failure (TTF) for each record
│   └── Filter the final TTF dataset to include only the last 15 hours before failure
│
├── 5. Supervised Learning: Baseline Model Training
│   ├── Define features (X), target (y), and turbine groups for cross-validation
│   ├── Initialize a `LeaveOneGroupOut` (LOGO) cross-validator
│   ├── For each turbine:
│   │   ├── Train a baseline XGBoost Regressor model
│   │   ├── Train a baseline Random Forest Regressor model
│   │   ├── Predict on the held-out turbine data
│   │   └── Calculate and store the MAE for both models
│
├── 6. Hyperparameter Optimization (Bayesian)
│   ├── Define hyperparameter search spaces for both XGBoost and Random Forest
│   └── For each turbine (within the same LOGO loop):
│       ├── Run `BayesSearchCV` to find the best parameters for each model
│       ├── Train the optimized models and make predictions
│       └── Store the optimized models, their parameters, and MAEs
│
├── 7. Performance Analysis and Model Explainability
│   ├── Print a comparison table of original vs. optimized MAE for both models
│   ├── Plot TTF comparisons for each turbine, showing actual and all four predicted outputs (original & optimized)
│   ├── Plot traditional feature importance for both models
│   ├── Perform SHAP analysis on the optimized models for explainability
│   │   ├── Use `shap.TreeExplainer` for XGBoost
│   │   └── Use `shap.KernelExplainer` for Random Forest
│   └── Visualize SHAP summary plots to show the impact of each feature on predictions
│



++ PrognosisTTFFeatureSelection
│Feature Selection by Classification, and Regression TTF
├── 1. Data Loading and Initial Preparation
│   ├── Load dataset and drop unnecessary columns
│   ├── Convert 'timestamp' to datetime format
│   ├── Filter for operational data (power > 0, rotor_speed > 0)
│   └── Create a binary 'failure' target
│
├── 2. Enhanced Feature Engineering
│   ├── Create rolling mean and standard deviation features for multiple windows (3, 6, 12 hours)
│   ├── Engineer equipment stress indicators ('power_per_rpm', 'temp_power_ratio')
│   └── Explicitly exclude all time-based features
│
├── 3. Failure Prediction (Classification)
│   ├── Split data into training and test sets
│   ├── Initialize and train an XGBoost Classifier with `scale_pos_weight` to handle imbalance
│   ├── Find the optimal classification threshold using the F1 score
│   └── Evaluate and print the classification report
│
├── 4. Feature Importance Analysis
│   ├── Plot and print XGBoost's built-in feature importance using 'weight', 'gain', and 'cover'
│   ├── Use SHAP to calculate feature importance
│   ├── Combine importance scores from multiple methods (XGBoost and SHAP)
│   └── Select a final list of important non-temporal features for the regression task
│
├── 5. Time-to-Failure (TTF) Regression Setup
│   ├── Define and apply a function to calculate the TTF target variable
│   ├── Filter the TTF dataset to include only the last 15 hours before failure
│   └── Prepare the features (X), target (y), and groups using the selected features
│
├── 6. TTF Prediction (Regression)
│   ├── Initialize a `LeaveOneGroupOut` (LOGO) cross-validator
│   ├── For each turbine in the LOGO loop:
│   │   ├── Train a pre-configured XGBoost Regressor
│   │   └── Make predictions and calculate the Mean Absolute Error (MAE)
│
├── 7. Results and Visualization
│   ├── Print a formatted table of MAE results for each turbine
│   ├── Plot TTF comparisons for specific turbines, visualizing actual vs. predicted values over a time window
│   ├── Plot the SHAP summary plot for the final regression model to explain its predictions
│   └── Display the final MAE results table with visual styling
│


++ PrognosisTTFSMOTE
│ Feature Selection by Classifier with SMOTE + Regressor for TTF
│── 1. Data Loading and Preprocessing
│   ├── Load dataset from 'wedowind.csv'
│   ├── Convert 'timestamp' to datetime and extract temporal features
│   ├── Filter data to remove records with non-positive rotor speeds
│   └── Create a binary 'failure' target
│
├── 2. Advanced Feature Engineering
│   ├── Generate rolling statistics (mean and standard deviation) for multiple sensors
│   ├── Create equipment stress indicators ('power_per_wind', 'temp_power_ratio')
│   └── Fill missing values in the features with the median
│
├── 3. Classification for Feature Selection
│   ├── Split data into training and test sets
│   ├── Address class imbalance using SMOTE on the training data
│   ├── Train a specialized XGBoost Classifier
│   ├── Determine the optimal classification threshold using the F1 score
│   └── Evaluate and visualize the model's performance with a classification report and confusion matrix
│
├── 4. Feature Importance and Selection
│   ├── Conduct SHAP analysis to understand feature contributions
│   ├── Plot global and aggregate SHAP importance
│   ├── Visualize XGBoost's built-in feature importance using 'weight', 'gain', and 'cover'
│   ├── Create SHAP dependence plots for top features
│   ├── Combine importance scores from multiple methods to select a final list of key features
│   └── Print a summary table of the top features
│
├── 5. Time-to-Failure (TTF) Regression Setup
│   ├── Define and apply a function to calculate the TTF for each turbine
│   ├── Filter the TTF dataset to focus on the period just before failure
│   └── Prepare features (X), target (y), and groups for the final regression task
│
├── 6. TTF Prediction with Random Forest
│   ├── Initialize a `LeaveOneGroupOut` (LOGO) cross-validator
│   ├── For each turbine in the LOGO loop:
│   │   ├── Train a `RandomForestRegressor`
│   │   ├── Predict on the held-out turbine data
│   │   └── Calculate and store the Mean Absolute Error (MAE)
│
├── 7. Results and Visualization
│   ├── Print a formatted table of MAE results for all turbines
│   ├── Plot the TTF comparison (actual vs. predicted) for specific example turbines
│   └── Display the final performance table with visual formatting
│



++ DetectionNaiveBays
│
├── 1. Data Loading and Initial Exploration
│   ├── Load dataset from 'wedowind.csv' and perform basic data checks
│   ├── Display data shape, statistics, and missing values
│   └── Visualize time series for key sensor readings
│
├── 2. Preprocessing and Feature Engineering
│   ├── Drop unnecessary columns and convert `timestamp`
│   ├── Create temporal features (`hour`, `day_of_week`, `month`)
│   ├── Create the binary `failure` target variable
│   ├── Encode the categorical `plant_name`
│   ├── Handle missing values with median imputation
│   └── Standardize features using `StandardScaler`
│
├── 3. Exploratory Data Analysis & Feature Selection
│   ├── Visualize feature correlations with a heatmap
│   ├── Use **Isolation Forest** to detect anomalies
│   ├── Apply **Random Forest** (unsupervised) for feature importance
│   ├── Use **`SelectKBest`** for feature selection with mutual information
│   └── Perform and plot **PCA** for dimensionality reduction
│
├── 4. Supervised Learning and Evaluation
│   ├── Split data into training and test sets
│   ├── Train a **Random Forest Classifier** as a baseline
│   ├── Evaluate the baseline model with a classification report and confusion matrix
│   ├── Train an **XGBoost Classifier** with `GridSearchCV` for hyperparameter tuning
│   └── Address class imbalance with **SMOTE**
│
├── 5. Model Explainability and Interpretation
│   ├── Use **SHAP** to analyze global feature importance
│   ├── Use **`eli5`** to perform and show **Permutation Importance**
│   └── Define a function to explain individual predictions with SHAP force plots
│
├── 6. Comprehensive Performance Analysis and Visualization
│   ├── Plot **Precision-Recall and ROC curves**
│   ├── Analyze and plot performance metrics at different probability thresholds
│   ├── Examine misclassified cases and visualize feature distributions for them
│   └── Save the final model and scaler for deployment
│



++ PrognosisFIbayesReg
│ 
├── 1. Data Loading and Initial Preparation
│   ├── Load dataset from `wedowind.csv`
│   └── Remove any unnecessary index columns
│
├── 2. Outlier Filtering
│   ├── Apply a **Mahalanobis distance** filter to remove multivariate outliers
│   └── Visualize the filtered data on a scatter plot
│
├── 3. Healthy Baseline Definition
│   ├── Filter the data to create a "healthy" learning set (e.g., records with no errors and sufficient wind speed)
│   ├── Create a test set from the remaining data
│   └── Define a list of candidate SCADA variables for analysis
│
├── 4. Preliminary Feature Importance (FI)
│   ├── Standardize and rank-transform the candidate features
│   ├── Fit a **Gaussian Mixture Model (GMM)** to the transformed data
│   ├── Calculate a preliminary **Functional Impairment (FI)** score based on the GMM's log-likelihood
│   ├── Calculate and visualize **Spearman correlation** between the FI score and each variable
│   └── Select the top `K` most correlated features for the final model
│
├── 5. Final FI Calculation and Aggressive Smoothing
│   ├── Standardize and rank-transform the selected features for both learn and test sets
│   ├── Fit a new GMM on the learn set and calculate the raw FI scores for both sets
│   ├── Apply an aggressive smoothing technique (Spline or Savitzky-Golay filter) to the raw FI scores
│   └── Plot the raw, smoothed, and thresholded FI scores to visualize the effect of smoothing
│
├── 6. Final Visualization
│   ├── Plot the smoothed FI score against the raw values of the top SCADA variables in the learn set
│   └── Plot the same comparison for the test set to show how FI tracks potential anomalies
│


++ PrognosisFIGMM
│
├── 1. Data Loading and Preprocessing
│   ├── Load the SCADA dataset from `wedowind.csv`
│   ├── Drop unnecessary columns and set `timestamp` as the index
│   └── Identify unique plants and their last known failure dates
│
├── 2. Data Splitting and Cleaning
│   ├── For each plant with a known failure:
│   │   ├── Define the "test period" as the last 20 days leading up to the failure
│   │   ├── Create the "learn set" with data preceding the test period
│   │   └── Clean both sets by filtering out non-operational data and missing values
│
├── 3. Feature Selection
│   ├── Identify available SCADA variables with sufficient variance
│   ├── Use a predefined list of **candidate features**
│   └── Automatically select features that are relevant and not constant
│
├── 4. Functional Impairment (FI) Calculation
│   ├── Standardize and rank-transform the selected features
│   ├── Fit a **Gaussian Mixture Model (GMM)** to the healthy learn set
│   ├── Use the trained GMM to calculate a raw **Functional Impairment (FI)** score for the test set
│   └── Apply **spline smoothing** to the raw FI scores to reveal underlying trends
│
├── 5. Visualization and Anomaly Analysis
│   ├── Generate a multi-plot figure for each plant's test period
│   ├── Plot the **smoothed FI score** alongside a predefined failure threshold
│   ├── For key **failure indicator features**, plot their raw values and a **rolling Z-score**
│   │   (this identifies periods where feature values deviate significantly from the baseline)
│   ├── Highlight points that are flagged as anomalies by the Z-score
│   └── Add a vertical line to indicate the precise time of the failure event for context
│
├── 6. Diagnostic Output
│   ├── Print descriptive statistics for the failure-indicator features in the period immediately preceding failure
│   └── Report the timestamps and scores of the top three highest FI points, providing key insights for predictive maintenance
│

