# **Project Submission: Rainfall Time Series Prediction**

## **Project Details**

- **Project Code:** PRAICP-1004-RainfallTS
- **Project Team ID:** PTID-AI-NOV-24-1060
- **Project Name:** Rainfall Time Series

---

## **Problem Statement**

The objective of this project is to develop a time-series prediction model for forecasting total rainfall over the next six months. The dataset consists of rainfall records from **Changi Climate Station (Singapore)** spanning **January 1982 to June 2020**.

The dataset includes:

1. **Monthly highest daily rainfall**
2. **Monthly number of rainy days**
3. **Total monthly rainfall**

---

## **Project Workflow**

### **1. Data Preparation**

- Load the dataset from Google Drive.
- Perform exploratory data analysis (EDA).
- Handle missing values and outliers.
- Normalize the dataset using **MinMaxScaler**.

### **2. Model Selection**

- Utilize **LSTM (Long Short-Term Memory)** neural networks for time-series forecasting.
- Implement **Dropout layers** for regularization.
- Use **Dense layers** for output prediction.

### **3. Model Training**

- Train the model using the **Adam optimizer**.
- Use **Huber loss** to handle outliers effectively.
- Implement callbacks such as **EarlyStopping** and **ReduceLROnPlateau**.

### **4. Hyperparameter Tuning**

- Use **Keras Tuner (Hyperband)** to optimize LSTM parameters.

### **5. Model Evaluation**

- Evaluate performance using:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
- Compare predictions with actual rainfall data.

### **6. Results & Visualization**

- Generate visualizations for:
  - Time-series trends.
  - Model loss and accuracy over epochs.
  - Predicted vs. actual rainfall values.

---

## **Model Summary**

| Layer (type)        | Output Shape    | Param # |
| ------------------- | --------------- | ------: |
| lstm_3 (LSTM)       | (None, 12, 96)  |  48,384 |
| dropout_3 (Dropout) | (None, 12, 96)  |       0 |
| lstm_4 (LSTM)       | (None, 12, 192) | 221,952 |
| dropout_4 (Dropout) | (None, 12, 192) |       0 |
| lstm_5 (LSTM)       | (None, 224)     | 373,632 |
| dropout_5 (Dropout) | (None, 224)     |       0 |
| dense_1 (Dense)     | (None, 3)       |     675 |

**Total Parameters:** 644,643 (2.46 MB)
**Trainable Parameters:** 644,643 (2.46 MB)
**Non-trainable Parameters:** 0

---

## **Conclusion**

- A deep learning model using **LSTM networks** was successfully trained for rainfall prediction.
- **Performance metrics** indicate the model's ability to forecast rainfall trends effectively.
- Further improvements can be made using **additional features, ensemble models, or alternative architectures**.

---

## **Future Scope**

- Incorporate **external climate variables** (temperature, humidity, wind speed) to enhance predictions.
- Use **transformer-based models** for improved time-series forecasting.
- Deploy the model as a **web-based application** for real-time rainfall prediction.

---

## **References**

- Dataset: Changi Climate Station, Singapore (1982-2020)
- TensorFlow, Keras, Scikit-Learn for deep learning & preprocessing
- Hyperparameter tuning via **Keras Tuner**
