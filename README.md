# House Price Prediction using Linear Regression

A machine learning project that predicts house prices using **Multiple Linear Regression** based on selected features from the King County house sales dataset.

## 📌 Project Overview

The goal of this project is to build a baseline regression model that predicts house prices using features that have a meaningful relationship with the target variable.

The project covers the complete basic machine learning workflow:

* Data loading
* Exploratory Data Analysis (EDA)
* Correlation analysis
* Feature selection
* Train-test splitting
* Feature scaling
* Model training
* Prediction
* Model evaluation
* Residual analysis

## 📊 Dataset

The project uses the **King County House Sales Dataset**, which contains information about houses and their sale prices.

The target variable is:

```text
price
```

### Features Used

The final model uses the following five features:

| Feature         | Description                             |
| --------------- | --------------------------------------- |
| `sqft_living`   | Living area of the house in square feet |
| `grade`         | Overall grade given to the house        |
| `bathrooms`     | Number of bathrooms                     |
| `lat`           | Latitude of the house                   |
| `sqft_basement` | Basement area in square feet            |

## 🔎 Exploratory Data Analysis

The dataset was analyzed using:

* Correlation matrix
* Correlation heatmap
* Scatter plots between selected features and house price

These analyses were used to understand relationships between the input features and the target variable and to select relevant features for the model.

## ⚙️ Machine Learning Workflow

### 1. Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 2. Feature Scaling

`StandardScaler` is used to standardize the input features.

The scaler is fitted only on the training data and then used to transform the test data.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

This prevents information from the test set from leaking into the training process.

### 3. Model Training

The model used is **Linear Regression** from Scikit-learn.

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train_scaled, y_train)
```

The model learns the coefficients for the selected features by minimizing prediction error on the training data.

### 4. Prediction

After training, predictions are generated using the unseen test data:

```python
y_pred = lr.predict(X_test_scaled)
```

## 📈 Model Evaluation

The model is evaluated using:

* **MAE — Mean Absolute Error**
* **MSE — Mean Squared Error**
* **RMSE — Root Mean Squared Error**
* **R² Score**

### Metrics

Add your actual results here after running the notebook:

| Metric   |       Result |
| -------- | -----------: |
| MAE      | `147122.887` |
| MSE      | `60348.4574` |
| RMSE     | `245658.942` |
| R² Score | ` 0.6008094` |

### What the metrics represent

**MAE:** Average absolute difference between actual and predicted prices.

**MSE:** Mean squared prediction error, which gives greater weight to larger errors.

**RMSE:** Square root of MSE, expressed in the same unit as the target variable.

**R²:** Measures how much of the variation in house prices is explained by the model.

## 📉 Model Diagnostics

The project also uses visual analysis to understand model performance:

### Actual vs Predicted

Compares actual house prices against the prices predicted by the model.

A prediction closer to the diagonal reference line indicates better performance.

### Residual Analysis

Residuals are calculated as:

```text
Residual = Actual Price - Predicted Price
```

Residual plots and residual distribution are used to identify patterns in the model's errors and evaluate whether the linear model is capturing the relationships adequately.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* Git & GitHub

## 📁 Project Structure

```text
house-price-prediction/
│
├── house_price_prediction_linear_regression.ipynb
├── kc_house_data.csv
├── README.md
└── .gitignore
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate into the project

```bash
cd house-price-prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 6. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
house_price_prediction_linear_regression.ipynb
```

## 🔮 Future Improvements

This project intentionally focuses on building and understanding a **Linear Regression baseline**.

Future versions can explore:

* Ridge Regression
* Lasso Regression
* Random Forest Regression
* Gradient Boosting
* Feature engineering
* Hyperparameter tuning
* Cross-validation
* Model comparison
* Deployment as an API

## 👨‍💻 Author

**Sree Vardhan**

This project was created as part of my machine learning learning journey, with a focus on understanding the complete workflow from data analysis to model evaluation.
