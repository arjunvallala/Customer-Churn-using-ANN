# Customer Churn Prediction using ANN

## 🚀 Live Demo

**Streamlit App:** [Click here to try the live application](https://customer-churn-using-ann-7tjfdkim4wbaejed4dqbb2.streamlit.app/)

---


## Overview
This project predicts whether a customer is likely to leave a company using an Artificial Neural Network (ANN). It uses customer information such as tenure, monthly charges, support calls, payment method, and contract type.

A Streamlit web application allows users to enter customer details and view the churn prediction.

## Features
- Customer churn prediction using ANN.
- Data preprocessing using One-Hot Encoding and StandardScaler.
- Model training and evaluation.
- Training visualization using TensorBoard.
- Interactive web interface built with Streamlit.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- TensorBoard

## Project Structure
```text
Customer-Churn-using-ANN/
├── data.csv
├── nb.ipynb
├── st.py
├── ann.keras
├── ohe.pkl
├── scaler.pkl
├── training_log.csv
├── logs/
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
cd Customer-Churn-using-ANN
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run st.py
```

## How It Works
1. Customer data is loaded and explored.
2. Categorical features are converted using One-Hot Encoding.
3. Numerical features are scaled using StandardScaler.
4. An ANN is trained to classify customer churn.
5. The trained model and preprocessing objects are saved.
6. The Streamlit application accepts customer details and predicts the likelihood of churn.

## Model Output
The model returns a churn probability.

- Probability greater than or equal to 0.5: Customer is predicted to churn.
- Probability below 0.5: Customer is predicted not to churn.

## Model Monitoring
TensorBoard is used to visualize training and validation metrics, including accuracy and loss.

## Future Improvements
- Improve model performance through hyperparameter tuning.
- Compare ANN with other classification algorithms.
- Add more customer features.
- Deploy the application online.

## Author
Arjun
