# Fraudbuster-project
Fraud Buster - Fraud Detection in Financial Transactions , Summer Project, FAC, IITK.
# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using a highly imbalanced dataset. The notebook covers a complete machine learning workflow, including exploratory data analysis, data preprocessing, handling class imbalance with SMOTE, training various classification models, and evaluating their performance. Additionally, it explores the use of Principal Component Analysis (PCA) for dimensionality reduction.

## 📋 Table of Contents

- [Dataset](#-dataset)
- [Project Workflow](#️-project-workflow)
- [Models Implemented](#-models-implemented)
- [Results](#-results)
- [How to Run](#-how-to-run)
- [Dependencies](#-dependencies)

## 📊 Dataset

The project utilizes the `creditcard.csv` dataset, which contains anonymized credit card transaction data. The features V1 through V28 are the result of a PCA transformation on the original data. The only features that have not been transformed with PCA are 'Time' and 'Amount'. The 'Class' feature is the response variable, where:

- **1** indicates a fraudulent transaction
- **0** indicates a valid transaction

The dataset is highly imbalanced, with the vast majority of transactions being non-fraudulent.

## ⚙️ Project Workflow

### 1. Exploratory Data Analysis (EDA)
- The dataset's shape, structure, and statistical summary were examined
- Visualizations, including histograms and distribution plots for 'Time' and 'Amount' features, were created to understand the data's characteristics

### 2. Data Preprocessing
- **Handling Missing Values**: Missing values were identified and imputed using the median value of their respective columns
- **Handling Duplicates**: Duplicate entries were identified and removed from the dataset
- **Feature Scaling**: The Amount feature was scaled using StandardScaler to normalize its range
- **Feature Selection**: The Time feature was dropped as it was deemed not critical for the fraud detection model

### 3. Handling Class Imbalance
- A count plot of the 'Class' variable confirmed the severe class imbalance
- To address this, the Synthetic Minority Over-sampling Technique (SMOTE) was applied. This technique generates synthetic samples for the minority class (fraudulent transactions) to create a balanced dataset for training
- **Note**: For computational efficiency, SMOTE was applied to a subset of 10,000 data points

### 4. Model Training and Evaluation
- The balanced dataset was split into training (80%) and testing (20%) sets
- Multiple classification algorithms were trained on the data
- Each model's performance was evaluated using accuracy scores and detailed classification reports (including precision, recall, and F1-score)

### 5. Dimensionality Reduction with PCA
- Principal Component Analysis (PCA) was performed to reduce the dataset's dimensionality
- A scree plot was generated to determine the optimal number of components to retain while preserving most of the variance
- The performance of the K-Nearest Neighbors (KNN) model was compared with and without PCA to assess the impact of dimensionality reduction

## 🤖 Models Implemented

The following machine learning models were trained and evaluated:

- Random Forest Classifier
- Decision Tree Classifier
- Logistic Regression
- Support Vector Machine (LinearSVC)
- K-Nearest Neighbors (KNN)

## 📈 Results

After balancing the dataset with SMOTE, the models achieved the following accuracy scores on the test set:

| Model | Accuracy Score |
|-------|----------------|
| Random Forest Classifier | 100% |
| Decision Tree Classifier | 99.94% |
| Logistic Regression | 99.94% |
| K-Nearest Neighbors | 99.94% |
| Support Vector Machine (SVC) | 98.94% |

### PCA Impact

The accuracy of the KNN classifier was tested with and without PCA to observe the effects of dimensionality reduction:

- **Accuracy without PCA**: 99.94%
- **Accuracy with PCA (2 components)**: 99.75%

This shows a minor trade-off in accuracy for a significant reduction in feature complexity.

## 🚀 How to Run

To run this project, follow these steps:

1. Clone this repository to your local machine
2. Ensure you have Jupyter Notebook or a compatible environment installed
3. Install all the required libraries listed in the Dependencies section
4. Place the `creditcard.csv` dataset in the same directory as the notebook
5. Open and run the `Fraudbuster(1).ipynb` notebook

## 📦 Dependencies

This project requires the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

## 📝 Notes

- The dataset is highly imbalanced, making it a challenging but realistic machine learning problem
- SMOTE is used to address class imbalance, but care should be taken when interpreting results on synthetic data
- The excellent performance metrics should be validated on a separate holdout dataset to ensure the model generalizes well to unseen data
- Consider implementing additional evaluation metrics like AUC-ROC and precision-recall curves for a more comprehensive assessment

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements or additional features.
