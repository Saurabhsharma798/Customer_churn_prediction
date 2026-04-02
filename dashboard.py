import streamlit as st
import requests
import os

API_URL = os.getenv('API_URL')

st.sidebar.title('Navigation')
if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.sidebar.button("Home"):
    st.session_state.page = "Home"

if st.sidebar.button("Prediction"):
    st.session_state.page = "Prediction"

if st.session_state.page == "Home":
    st.write("Home Page")

elif st.session_state.page == "Prediction":
    st.write("Predict Page")

if st.session_state.page == "Home":
    st.title("Welcome to the Customer Churn Prediction App")
    st.write("""
    👋 This app predicts whether a customer will churn based on their usage and account details.

    ### How to Use:
    - Go to the 'Predict' section using the sidebar.
    - Fill in the customer information.
    - Click on 'Predict' to see the result.

    Built using **FastAPI** + **Streamlit**.
    """)



elif st.session_state.page == "Prediction":


    st.title('Customer Churn Prediction')

    age=st.number_input("Age",min_value=0,max_value=120)
    gender=st.selectbox("Gender",["Male","Female"])
    tenure=st.number_input("Tenure")
    usage_frequency=st.number_input("usage frequency")
    support_calls=st.number_input("support calls")
    payment_delay=st.number_input("payment delays")
    subscription_type=st.selectbox("Subscription Type",["Basic","Standard","Premium"])
    contract_length=st.selectbox("Contract Length",["Monthly", "Quarterly", "Annual"])
    total_spend = st.number_input("Total Spend")
    last_interaction = st.number_input("Last Interaction")


    if st.button('Predict'):
        data = {
            "Age": age,
            "Gender": gender,
            "Tenure": tenure,
            "Usage_Frequency": usage_frequency,
            "Support_Calls": support_calls,
            "Payment_Delay": payment_delay,
            "Subscription_Type": subscription_type,
            "Contract_Length": contract_length,
            "Total_Spend": total_spend,
            "Last_Interaction": last_interaction
        }

        try:
            response=requests.post(API_URL,json=data)
            result=response.json()

            
            pred=result['prediction']
            recommendation=result['recommendation']

            if "I recommend" in recommendation:
                recommendation = recommendation.split("I recommend")[-1]
                
            

            if pred == 1.0:
                st.error("⚠️ High risk of churn")
                st.markdown("### 💡 Recommendation to Reduce Churn")
                st.markdown(recommendation.replace("\n", "<br>"), unsafe_allow_html=True)
            else:
                st.success("✅ Low risk of churn. No recommendation needed.")

        except Exception as e:
            st.error(f'error contacting FastAPI{e}')



