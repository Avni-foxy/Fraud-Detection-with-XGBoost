import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model('xgboost_model.json')
    return model

st.set_page_config(layout="wide")
st.title('Fraud Detection Dashboard')
st.markdown('**XGBoost Model - AUC 0.91**')

model = load_model()

# Upload your test_results.csv
uploaded = st.file_uploader('Upload test_results.csv')
if uploaded:
    df = pd.read_csv(uploaded)
    
    # Metrics
    col1, col2 = st.columns(2)
    col1.metric('Total Transactions', len(df))
    col2.metric('High Risk (>10%)', (df['fraud_probability'] > 0.1).sum())
    
    # Charts
    fig = px.histogram(df, x='fraud_probability', nbins=50, title='Fraud Distribution')
    st.plotly_chart(fig)
    
    # Top 10
    st.subheader('Top 10 Highest Risk')
    top10 = df.nlargest(10, 'fraud_probability')
    st.dataframe(top10)
