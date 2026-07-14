# GPU-Accelerated Groundwater Quality Prediction using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange">
  <img src="https://img.shields.io/badge/GPU-PyTorch%20%2B%20CUDA-green">
  <img src="https://img.shields.io/badge/Status-Completed-success">
</p>

## Overview

Groundwater quality assessment plays a vital role in environmental monitoring and sustainable water resource management. Traditional laboratory-based assessment methods are often time-consuming and difficult to scale for large datasets.

This project presents a Machine Learning-based framework for predicting groundwater quality using physicochemical parameters such as **pH, temperature, and electrical conductivity**. The workflow combines data preprocessing, feature engineering, Water Quality Index (WQI) generation, and multiple classification models to evaluate predictive performance. GPU-accelerated Logistic Regression was also implemented using **PyTorch with CUDA** to explore hardware-accelerated machine learning.

This project was developed during my **Summer Internship at CSIR – National Geophysical Research Institute (CSIR-NGRI), Hyderabad**.

---

## Objectives

- Develop an intelligent groundwater quality prediction system.
- Perform comprehensive data preprocessing and feature engineering.
- Generate Water Quality Index (WQI) based target labels.
- Compare multiple Machine Learning algorithms.
- Explore GPU-accelerated model training using CUDA.
- Evaluate model performance using standard classification metrics.

---

## Dataset

The dataset contains groundwater quality measurements collected from multiple locations.

### Input Features

- pH
- Temperature
- Electrical Conductivity
- Engineered groundwater quality features

### Data Preprocessing

- Missing value handling
- Data cleaning
- Feature engineering
- Log transformation for conductivity
- Histogram analysis
- Skewness evaluation
- Water Quality Index (WQI) generation
- Stratified train-test split

---

# Machine Learning Pipeline

```text
Groundwater Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Water Quality Index (WQI)
        │
        ▼
Train-Test Split
        │
        ▼
Model Training
        │
        ▼
Prediction
        │
        ▼
Performance Evaluation
```

---

## Machine Learning Models

The following models were implemented and evaluated:

- Logistic Regression
- GPU-Accelerated Logistic Regression (PyTorch + CUDA)
- Decision Tree
- Random Forest
- XGBoost

---

## GPU Acceleration

A GPU-enabled implementation of Logistic Regression was developed using **PyTorch** on an **NVIDIA GeForce RTX 3050** with CUDA support.

The objective was to explore GPU-assisted machine learning and compare its performance with the conventional CPU-based implementation.

---

## Results

| Model | Accuracy |
|--------|----------|
| Logistic Regression | **66.28%** |
| GPU Logistic Regression | **67.35%** |
| Decision Tree | **66.02%** |
| Random Forest | **67.81%** |
| XGBoost | **69.21%** |

Among all evaluated models, **XGBoost achieved the highest prediction accuracy**, demonstrating the effectiveness of ensemble learning techniques for groundwater quality prediction.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch
- CUDA
- Jupyter Notebook
- XGBoost

---

## Repository Structure

```text
GroundWater-Quality-Prediction
│
├── data/
│   ├── groundwater_raw.csv
│   └── groundwater_processed.csv
│
├── images/
│
├── notebooks/
│   └── Groundwater_Quality_Prediction.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/Nanditha3105/GroundWater-Quality-Prediction.git
```

Move into the project directory:

```bash
cd GroundWater-Quality-Prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
notebooks/Groundwater_Quality_Prediction.ipynb
```

Run all cells sequentially.

---

## Future Improvements

- Deploy the model using Streamlit.
- Integrate real-time groundwater monitoring data.
- Evaluate additional boosting algorithms such as LightGBM and CatBoost.
- Apply optimization techniques such as Differential Evolution for hyperparameter tuning.
- Develop an end-to-end web application for groundwater quality prediction.

---

## Acknowledgements

This project was carried out during my Summer Internship at **CSIR – National Geophysical Research Institute (CSIR-NGRI), Hyderabad** under the guidance of **Dr. Kalyan Netti (Scientist-G, Head, IT Division)**.

I sincerely thank the institute and the IT Division for their guidance, support, and the opportunity to work on this project.

---

## Author

**Sornapudi Nanditha**

B.Tech – Artificial Intelligence and Machine Learning  
Anurag University

GitHub: https://github.com/Nanditha3105