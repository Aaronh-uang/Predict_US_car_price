# Predict_US_car_price
# 🚗 US Used Car Price Predictor

## 📌 Project Overview
This project builds a robust machine learning pipeline to predict the market price of used cars in the United States. Leveraging a dataset of over **3 million listings**, the study benchmarks a linear baseline (**Ridge Regression**) against a non-linear ensemble method (**Histogram Gradient Boosting**).

The final model achieves a Mean Absolute Error (MAE) of approximately **$3,280**, effectively capturing complex non-linear relationships such as depreciation curves and mileage penalties. The project is deployed as an interactive web application using **Streamlit**.

## 🚀 Key Features
* **Data Pipeline:** rigorous cleaning process to handle noise (VINs, descriptions), impute missing values, and filter realistic price/mileage outliers.
* **Feature Engineering:**
    * *Temporal Analysis:* Converted model years to "Vehicle Age" relative to the current year.
    * *Text Extraction:* Parsed cylinder counts and simplified transmission types from unstructured text.
* **Model Architecture:**
    * *Baseline:* Ridge Regression with One-Hot Encoding for categorical features.
    * *Champion:* Histogram Gradient Boosting Regressor (HGBR) utilizing native categorical support and Ordinal Encoding.
* **Interpretability:** Permutation Feature Importance analysis identifies **Mileage**, **Horsepower**, and **Vehicle Age** as the primary drivers of price.

## 🛠️ Tech Stack
* **Python** (Pandas, NumPy)
* **Scikit-Learn** (Pipelines, HGBR, Ridge, Preprocessing)
* **Streamlit** (Web App Interface)
* **Matplotlib / Seaborn** (Data Visualization)

## 📊 Results Summary
| Model | RMSE (Log Scale) | MAE (USD) | RMSE (USD) |
| :--- | :--- | :--- | :--- |
| **Ridge Regression** | 0.284 | ~$6,147 | ~$13,258 |
| **Gradient Boosting** | **0.167** | **~$3,280** | **~$9,914** |

## 💻 How to Run Locally
1.  Clone the repository:
    ```bash
    git clone [https://github.com/Aaronh-uang/Predict_US_car_price.git](https://github.com/Aaronh-uang/Predict_US_car_price.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

---
*Created by Aaron Huang*
