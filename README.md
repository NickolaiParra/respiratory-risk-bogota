# Respiratory Risk Bogotá

Predictive analysis of respiratory disease propensity in Bogotá D.C. using air quality data from the RMCAB monitoring network, machine learning models (Random Forest, XGBoost), and SHAP interpretability.

## Live Demo

🗺️ [Interactive Map](https://nickolaiparra.github.io/respiratory-risk-bogota)

## Overview

This project implements a full CRISP-DM pipeline to predict the Air Quality Index (ICA) category per monitoring station using hourly pollutant data (PM₂.₅, PM₁₀, CO, O₃, NO, NO₂, SO₂) collected from 19 RMCAB stations across Bogotá (2024–2025).

## Results

| Model | Accuracy | F1-weighted |
|-------|----------|-------------|
| Random Forest | 0.7827 | 0.7693 |
| XGBoost | 0.7770 | 0.7658 |

PM₂.₅ was identified as the dominant predictor (SHAP mean |value| = 0.58).

## Project Structure
├── download_data.py       # RMCAB web scraper

├── preprocessing.py       # Outlier removal, pivot, Z-score, sine-cosine encoding

├── model_training.py      # ICA calculation, Random Forest & XGBoost + GridSearch CV

├── evaluation.py          # Metrics and confusion matrices

├── shap_analysis.py       # SHAP feature importance

├── map_generation.py      # Leaflet.js data export

├── data/

│   ├── raw/               # Raw scraped data (.csv.gz)

│   └── processed/         # Processed dataset (.csv.gz)

├── models/                # Trained models (.pkl)

├── docs/                  # Web app (GitHub Pages)

│   ├── index.html

│   └── data.js

└── data/figures/          # Generated plots

## Data Source

RMCAB – Red de Monitoreo de Calidad del Aire de Bogotá  
http://rmcab.ambientebogota.gov.co

## ICA Categories (Resolución 2254 de 2017)

| ICA | Category |
|-----|----------|
| 0–50 | Buena |
| 51–100 | Aceptable |
| 101–150 | Dañina a grupos sensibles |
| 151–200 | Dañina |
| >200 | Muy dañina |

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn joblib beautifulsoup4 lxml
```

## Authors

David Nickolai Parra Ariza  
Universidad Nacional de Colombia
