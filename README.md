# 💨 Wind Turbine Power Forecasting with LSTM

This project predicts the power output of wind turbines using a **Long Short-Term Memory (LSTM)** deep learning model.  
The dataset is obtained from Kaggle and contains SCADA measurements from wind turbines.  
The model is implemented and trained in a **Jupyter Notebook (`.ipynb`)** format and python format.

---

## 📂 Project Structure

Wind-Turbine-Power-Forecasting-with-LSTM/  
│  
├── LSTM.ipynb                 # Main Jupyter Notebook with all code and outputs  
├── LSTM.py                    # Python script version of the notebook   
├── LICENCE                    # Project license file  
└── README.md                  # Project documentation  

---
## 📊 Datase

- Source: Kaggle - Wind Turbine SCADA Dataset  
- License: CC0: Public Domain  
- You are free to copy, modify, distribute, and use the dataset without permission.  

⚠️ Important: The dataset file must be placed in the same directory as LSTM.ipynb before running the notebook

---
## 📦 Required Libraries

Before running the notebook, make sure you have the following Python libraries installed:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
```
You can install them manually

---
## 🛠️ Installation & Requirements

1.  Clone this repository
2. Download the dataset from Kaggle and place it next to LSTM.ipynb or LSTM.py.

---
## 📈 Model

- Architecture: LSTM → Dense layers  
- Features: Time-series preprocessing, feature scaling, and R²/MAE/RMSE evaluation metrics  
- Output: Predicted vs Actual power output graphs  

---
## 📜 License

- This project is under the MIT License.  
- The dataset is licensed under CC0: Public Domain.  

---
## 🙌 Acknowledgements

- Kaggle for hosting the dataset  
- TensorFlow & Keras for deep learning tools  
- Pandas, NumPy, and Matplotlib for data processing & visualization  
