# CO₂ Storage Surrogate Modeling using SPE11C - A Full 3D Field-Scale Simulation Benchmark.

### **Abstract**
Geological carbon storage plays a critical role in global decarbonization, yet full 3D reservoir simulations remain computationally intensive for evaluating CO₂ injection performance across multiple scenarios. This work develops a machine-learning surrogate model for the SPE CSP 11C field-scale benchmark, aiming to predict CO₂ storage performance with significantly reduced computation time. A complete workflow is implemented — from preprocessing and physics-aware feature engineering to model benchmarking — and multiple regression methods are evaluated, including Linear Regression, Random Forest, Gradient Boosting, and a Multilayer Perceptron (MLP). Results show that the MLP surrogate achieves the highest predictive accuracy while preserving key reservoir dynamics represented in the simulation outputs. The final surrogate provides rapid prediction capability suitable for scenario screening and sensitivity analysis, demonstrating the practical potential of data-driven models to complement classical numerical reservoir simulation in carbon capture and storage applications.

---

## 1. Research Objectives
- Convert high-fidelity SPE11C reservoir simulation outputs into ML-ready inputs using physics-aware preprocessing (spatial binning, interpolation, and engineered features such as normalized depth, density ratio, and CO₂ fraction).
- Construct and compare regression baselines (Linear Regression, Random Forest, Gradient Boosting) and neural network surrogates (MLP) to quantify trade-offs in accuracy, interpretability, and runtime.
- Develop an optimized MLP surrogate that reproduces simulator pressure fields with high fidelity while reducing compute time by several orders of magnitude.
- Rigorously evaluate surrogate generalization using hold-out testing, residual analysis, and cross-scenario checks (different timesteps / grid cells / injection conditions).
- Provide physically interpretable diagnostics (SHAP and Partial Dependence Plots) to demonstrate that the surrogate captures known reservoir physics and to identify the most important control variables for pressure evolution.
- Deliver a reproducible, publication-ready analysis and figure set to support rapid scenario screening, uncertainty quantification, and decision support for CCS planning.

---

## 2. Dataset Overview: SPE CSP 11C (2023–2025)
The SPE CSP 11C (2023–2025) benchmark provides a realistic 3D field-scale environment with geological layering, heterogeneity, and multiphase CO₂ behavior—ideal conditions for surrogate model development.

### **2.1 Spatial Maps**
Each timestep snapshot includes:
- x, y, z coordinates  
- Pressure (Pa)  
- CO₂ saturation  
- CO₂ mass components (mobile, immobile, dissolved)  
- Water mass and densities  
- Gas/liquid densities  
- Temperature  

These represent the full 3D spatial distribution of CO₂-related properties.

### **2.2 Time Series**
Includes:
- Injection pressures (p1, p2)  
- Mass balance components (mobile, immobile, dissolved, sealed)  
- Boundary flux  
- Containment metrics  
- Time evolution of total CO₂ mass  

---

## 3. Methodology

### **3.1 Data Preparation**
- Convert large CSV files into efficient formats (Parquet/HDF5).  
- Remove inactive cells.  
- Normalize and align spatial maps with time-series data.  
- Standardize features and construct train/test partitions.

### **3.2 Feature Engineering**
Physics-inspired features:
- Depth and structural zones  
- Distance to injection well  
- Saturation gradients & Pressure derivatives 
- PCO₂ mobility type & storage efficiency ratio

### **3.3 Machine Learning Models**
Baseline regression: 
- LinearRegression
- RandomForest 
- GradientBoosting
  
Neural networks:
- Dense NN surrogate (Keras)
  
Advanced surrogate models:
- 3D Convolutional Neural Network (3D-CNN)  
- 3D U-Net for spatial prediction  
- Fourier Neural Operator (FNO)  
- Physics-Informed Neural Networks (PINNs)  

Targets:
- Pressure distribution  
- CO₂ saturation  
- Storage efficiency  
- Mass evolution over time  

### **3.4 Evaluation Metrics**
- Root Mean Squared Error (RMSE)
- Coefficient of determination (R²)
- Parity and residual analysis
- Time evolution and plume compatibility

---

## 4. Tools & Libaries
- Python  
- NumPy, Pandas, SciPy  
- Scikit-learn, XGBoost  
- TensorFlow / PyTorch  
- PyVista, Plotly  
- dask, pyarrow  

---

## 5. Results Summary
| Model                     | RMSE       | R²          |
| ------------------------- | ---------- | ----------- |
| LinearRegression_baseline | **4.24**   | **1.0000**  |
| RandomForest_baseline     | 17,577     | 0.99997     |
| GradientBoosting_baseline | 23,000     | 0.99995     |
| Surrogate_NN_Final        | **36,938** | **0.99987** |
| NN_baseline               | 37,215     | 0.99987     |
| MLPRegressor_baseline     | 120,858    | 0.99863     |

The final surrogate neural network generalizes well across the SPE11C dataset with fast inference (< 0.1 sec per prediction)

---

## 6. Scientific Impact
The surrogate model developed in this work delivers rapid, physically coherent approximations of reservoir pressure that enable:
- Massive speed-up: evaluate scenarios in milliseconds versus hours for full numerical simulators — enabling real-time scenario exploration and optimization.
- Physics-consistent predictions: feature importance (SHAP) and PDP analyses show the model respects core physics drivers (pressure, temperature, normalized depth, fluid density ratios and CO₂ fraction), increasing trust for engineering use.
- Operational utility: supports rapid site screening, injection schedule optimization, sensitivity studies, and multi-scenario uncertainty evaluation without requiring full simulator re-runs.

Applications:
- Rapid site screening
- Sensitivity and risk analysis
- Real-time injection control prototypes

---

## 7. Future Work
- 3D U-Net / 3D CNN — to enable full 3D reconstruction of the CO₂ plume across the reservoir grid.
- Fourier Neural Operator (FNO) — to improve temporal generalization for long-term forecast of injection and post-injection periods.
- PINN-enhanced surrogate — to enforce physical consistency by incorporating pressure–mass-balance constraints during training.
- Transfer learning — to adapt the surrogate model to real industrial CCS projects such as the Sleipner field.
- Open-source CCS surrogate benchmark — to release a reproducible baseline framework supporting research and collaboration in CO₂ storage modeling.
   
---

## Repository Structure
```
spe11c_surrogate_model/
│
├── data/
│   ├── raw/                         # Original unprocessed SPE 11C data
│   ├── interim/cleaned/             # Cleaned spatial & time-series data before feature engineering
│   │   ├── spatial_all.parquet
│   │   ├── timeseries.parquet
│   │   └── README.md
│   └── processed/                   # Final ML-ready datasets
│       ├── ml_ready_4D_dataset.npy
│       ├── X.npy
│       ├── y.npy
│       ├── X_train_scaled.npy
│       ├── X_test_scaled.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       └── README.md
│
├── models/                          # Trained baseline & surrogate models
│   ├── LinearRegression_baseline.pkl
│   ├── RandomForest_baseline.pkl
│   ├── GradientBoosting_baseline.pkl
│   ├── MLP_baseline.pkl
│   ├── MLP_baseline.h5
│   ├── Surrogate_MLP.keras
│   ├── feature_scaler.pkl
│   └── y_scaler.pkl
│
├── reports/
│   └── model_performance.csv        # Evaluation metrics comparison table
│
├── src/                             # Reproducible ML pipeline source code
│   ├── data_utils.py                # Loading & preprocessing utilities
│   ├── feature_utils.py             # Feature engineering utilities
│   ├── train_models.py              # Training script for baseline & surrogate models
│   ├── evaluate.py                  # Evaluation & metrics computation
│   ├── 01_exploration.ipynb         # Data exploration notebook
│   ├── 02_preprocessing.ipynb       # Cleaning & transformation
│   ├── 03_feature_engineering.ipynb # Physics-informed feature design
│   ├── 04_ML_Baseline_Models.ipynb  # Baseline model training & results
│   ├── 05_surrogate_3D_model.ipynb  # Surrogate model development
│   ├── 06_Visualization_Results.ipynb # Spatial & temporal prediction visualization
│   └── feature.ipynb                # Sandbox/experimentation
│
├── requirements.txt
├── environment.yml                  # Replicable conda environment
└── README.md


```

To install dependencies
pip install -r requirements.txt

---

## Author
**Anastasya Lesnussa**
Petroleum Engineer | Data Science for Energy Transition  
Portfolio: databyanna.dev

---
### Keywords
`CO2 Storage`, `Machine Learning`, `Energy Transition`, `Surrogate Modeling`, `Reservoir Simulation`, `Decarbonization`
