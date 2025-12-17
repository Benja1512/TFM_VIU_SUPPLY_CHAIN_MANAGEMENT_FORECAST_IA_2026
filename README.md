# Intelligent Inventory Management Using Demand Forecasting and Exploratory Quantum Optimization

This repository contains the complete codebase developed for the **Master’s Thesis (TFM)** entitled:

**“Intelligent Inventory Management in a Supply Chain through Scenario Simulation Using Artificial Intelligence: Application to Weekly Demand of 1-Liter Whole Milk in an Urban Retail Environment in Utrecht.”**

The project analyzes how demand forecasting accuracy affects inventory performance and explores the potential of **exploratory quantum optimization** as an emerging approach for inventory decision-making in retail supply chains.

---

## Academic Context

- **Degree:** Master’s Degree in Supply Chain Management and Logistics  
- **University:** Universidad Internacional de Valencia (VIU)  
- **Type:** Master’s Thesis (TFM)  
- **Author:** Benjamin Ontiveros  
- **Year:** 2025  

This repository supports the computational and empirical results presented in the Master’s Thesis.

---

## Repository Structure

Pronostico/
    ├── data/ # Simulated demand data (52 weeks)
    ├── traditional_models/ # MA-3 and SES forecasting models
    ├── prophet_model/ # Prophet demand forecasting
    ├── deep_learning/ # ANN (Keras) forecasting models
    ├── quantum_models/ # Classical and exploratory quantum optimization (QAOA)
    ├── notebooks/ # Exploratory and validation notebooks
    ├── results/ # Forecasts, metrics, plots, and CSV outputs
    ├── utils/ # Utility and helper functions
    ├── main.py # Main execution script
    ├── requirements.txt # General dependencies
    ├── requirements_tfm.txt # Dependencies used for the TFM analysis
    ├── requirements_qaoa.txt # Dependencies for quantum optimization experiments
    ├── README.md

---

## Case Study Description

- **Product:** Whole milk (1 liter)  
- **Sector:** Retail food supply chain  
- **Environment:** Urban retail store 
- **Time Horizon:** 52 weekly periods  
- **Data:** Fully simulated data incorporating trend, seasonality, and stochastic variability  

All datasets used in this project are **synthetic** and generated exclusively for academic purposes.

---

## Methodological Overview

### Demand Forecasting

The project compares traditional forecasting techniques with advanced Artificial Intelligence models:

- Moving Average (MA-3)  
- Simple Exponential Smoothing (SES)  
- Prophet  
- Random Forest  
- Artificial Neural Networks (ANN) implemented using Keras  

Forecast accuracy is evaluated using **MAE** and **RMSE**, supported by visual analysis against the simulated demand series.

---

### Inventory Simulation

Forecast outputs are used as inputs for inventory simulation under different replenishment policies, assessing:

- Stockouts  
- Overstock  
- Inventory accumulation  
- Service level implications  

This approach allows evaluation of how forecasting improvements translate into operational and logistical performance.

---

### Optimization Approaches

Inventory planning decisions are addressed through multiple optimization strategies:

- **Classical optimization methods:**  
  - Brute-force optimization for reduced-scale problems  
  - Continuous optimization using SciPy  

- **Exploratory quantum optimization:**  
  - Quantum Approximate Optimization Algorithm (QAOA)  
  - Binary QUBO formulation  
  - Simulation under NISQ-era constraints  

The quantum optimization component is **exploratory in nature** and intended for research and methodological comparison rather than operational deployment.

---

## Key Contributions

- Demonstrates that higher forecasting accuracy does not necessarily lead to optimal inventory performance.  
- Quantifies trade-offs between service level, overstock, and stockout risk.  
- Provides an applied comparison of traditional, AI-based, and exploratory quantum approaches.  
- Contributes to academic research on intelligent inventory management in retail environments.

---

## Disclaimer

This repository is intended **solely for academic and research purposes**.

- All data are simulated.  
- The quantum optimization module is exploratory and constrained by current NISQ hardware limitations.  
- Results should **not** be interpreted as real-world operational recommendations.

---

## How to Run (Optional)

1. Install the required dependencies:
```bash
pip install -r requirements.txt
Run the main execution script:

bash
Copiar código
python main.py
(Individual modules may also be executed independently.)

License
This project is shared for academic reference and educational use only.
No commercial use is intended or authorized.

Author
Benjamin Ontiveros
Master’s Degree in Supply Chain Management and Logistics
Universidad Internacional de Valencia (VIU) - Spain.
email: bontiveroso@student.universidadviu.com