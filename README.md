# Machine Learning Prediction of CO₂ Storage Efficiency
### **Abstract**
This project applies machine learning and data-driven modeling to predict the efficiency of CO₂ geological storage. Using the SPE10 synthetic reservoir dataset and later integrating the Sleipner CO₂ injection field data, this work develops a surrogate model that predicts CO₂ plume evolution, storage efficiency, and risk zones. The goal is to bridge high-fidelity reservoir simulations with machine learning to reduce computational cost and accelerate decision-making in carbon capture and storage (CCS) — supporting the broader mission of energy transition and decarbonization.

---

## 1. Research Objectives
- Develop a **data-driven surrogate model** for CO₂ storage simulation.  
- Predict **CO₂ plume behavior**, storage efficiency, and heterogeneity-driven risk.  
- Integrate **reservoir simulation data** with ML workflows to enable faster screening of geological storage sites.  
- Support **decarbonization and carbon management** through scalable, reproducible modeling.  

---

## 2. Datasets
### **SPE10 Synthetic Model**
- **Files:** `spe_phi.dat` (porosity) and `spe_perm.dat` (permeabilty)
- **Grid size:** 60 x 220 x 85
- **Variables:**
  - Porosity field (ϕ)
  - Permeabilty tensor (Kx, Ky, Kz)
- Source: [SPE Comparative Solution Project](https://www.spe.org/web/csp/datasets/set02.htm)

### **Sleipner CO2 Storage**
- Real field CO2 injection time series (pressure, plume extent, seismic depth maps)
- Source: [Sleipner Benchmark Dataset - SINTEF / IEAGHG](https://co2datashare.org)

---

## 3. Methodology

This workflow combines **reservoir data preprocessing**, **synthetic injection scenario generation**, and **ML-based prediction modeling**.

### **3.1 Data Preparation**
- Load and clean porosity/permeability fields  
- Normalize and reshape 3D grid structure  
- Visualize reservoir heterogeneity distribution  

### **3.2 Simulation Proxy**
- Generate synthetic CO₂ saturation fronts  
- Label data for ML training (storage efficiency, saturation ratio)  

### **3.3 Machine Learning Model**
- Algorithms tested: `RandomForestRegressor`, `XGBoost`, `3D CNN`  
- Predict target variables: CO₂ efficiency, plume radius, or saturation  
- Evaluate using **R²**, **RMSE**, and uncertainty intervals  

### **3.4 Sensitivity & Risk Assessment**
- Perform feature importance and parameter sensitivity  
- Identify zones of high leakage risk or low storage performance  

---

## 4. Tools & Libaries
- **Languages**: Python  
- **Libraries**: NumPy, Pandas, Matplotlib, Scikit-learn, XGBoost, TensorFlow / PyTorch  
- **Visualization**: PyVista, Plotly  
- **Supporting Tools**: SciPy, pytorch-lightning  

---

## 5. Expected Results
- 3D visualization of predicted CO₂ plume evolution  
- Model validation against simulated results  
- Sensitivity plots showing the influence of porosity/permeability on storage performance  
- Comparison between ML-predicted and simulation-based storage efficiency
  
---

## 🔬 6. Future Work
- Integrate **real Sleipner CO₂ injection data** for transfer learning  
- Extend the model to **multi-well injection optimization**  
- Publish results as an open-source **CO₂ storage ML benchmark**
   
---

## Repository Structure
```
CO2_Storage_Efficiency_ML/
│
├── data/
│ ├── spe_phi.dat
│ ├── spe_perm.dat
│ └── sleipner_data/ (optional, planned integration)
│
├── notebooks/
│ ├── 1_Data_Preparation.ipynb
│ ├── 2_Feature_Engineering.ipynb
│ ├── 3_ML_Modeling.ipynb
│ ├── 4_Model_Interpretation.ipynb
│ ├── 5_Visualization_and_Risk.ipynb
│ └── 6_Report_Summary.ipynb
│
├── models/
│ ├── trained_model.pkl
│ └── metrics.json
│
├── results/
│ ├── plots/
│ └── 3D_visualizations/
│
├── README.md
└── requirements.txt
```
---

## Author
**Anastasya Lesnussa**
Petroleum Engineer | Data Science for Energy Transition  
Portfolio: [yourwebsite.com] (replace later when ready)

---
### Keywords
`CO2 Storage`, `Machine Learning`, `Energy Transition`, `Surrogate Modeling`, `Reservoir Simulation`, `Decarbonization`
