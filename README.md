# 🔮 Customer Churn Prediction App

A smart and interactive customer churn prediction system built with Machine Learning and Streamlit. This app helps businesses identify customers who are likely to churn and provides AI-based suggestions to retain them using a Large Language Model (Groq).

---
**Live Link**
```bash
https://saurabh-customer-churn.streamlit.app/
```

## 📌 Features

- ✅ Predict whether a customer will **churn or not**
- 🤖 Get AI-powered retention suggestions using **Groq LLM API**
- 📊 Clean and responsive UI built with Streamlit
- 🤖 FastApi backend
- 🧠 Uses a trained ML model (e.g. Logistic Regression, XGBoost)
- 🔐 Secure API key handling using `.env` and `python-dotenv`

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- Streamlit
- Joblib (for model saving)
- Groq API (for LLM-based recommendations)
- `python-dotenv` for environment variable management
- FastAPI (for backend)

---

## 🚀 How it Works

1. User fills in customer information (like gender, contract type, tenure, etc.)
2. The model predicts whether the customer will **churn** or **stay**
3. If the customer is likely to churn, the app sends details to a **Groq LLM** for tailored retention suggestions
4. Streamlit displays both the **prediction** and the **recommendation**
5.The backend of this project is powered by FastAPI, a modern and high-performance web framework for building APIs with Python.
6.The ML model is wrapped inside a FastAPI endpoint (/predict)
7.Calls the trained model using a custom predict_pipeline to return predictions

---

## 📦 Installation (Manual)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-churn-app.git
cd customer-churn-app
```
2. Create a virtual environment
```bash
python3 -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Create a .env file
```bash
GROQ_API_KEY=your_groq_api_key_here
```
5. Run the app
```bash
streamlit run app.py
```

🌐 Hosting Options
You can deploy this app on:

🔵 Streamlit Cloud — Free and easy

🟣 Render — For backend API if separating UI and model

🟢 Replit — For quick testing (use UptimeRobot to avoid sleep)

🙌 Acknowledgements
Groq API

Streamlit

Scikit-learn

Python Dotenv

👨‍💻 Author
Saurabh
---
