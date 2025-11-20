import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Credit Analyzer", page_icon="💳", layout="wide")

# Beautiful styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main h1, .main h2, .main h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: white !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        color: white;
        font-weight: bold;
        padding: 15px 30px;
        font-size: 18px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(255, 105, 135, .4);
        border: none;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 20px rgba(255, 105, 135, .6);
    }
    .stDownloadButton>button {
        background: linear-gradient(45deg, #11998e 30%, #38ef7d 90%);
        color: white;
        font-weight: bold;
        padding: 12px 24px;
        font-size: 16px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(17, 153, 142, .4);
        border: none;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(17, 153, 142, .6);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 4rem; font-family: Arial, sans-serif;'>💳 Credit Risk Analyzer</h1>", 
            unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-family: Arial, sans-serif;'>AI-Powered Loan Assessment</h3>", 
            unsafe_allow_html=True)
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    rf = joblib.load('rf_credit_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    defaults = joblib.load('default_values.pkl')
    return rf, preprocessor, defaults

rf, preprocessor, defaults = load_models()

def predict(inputs):
    data = defaults.copy()
    data.update(inputs)
    df = pd.DataFrame([data])
    df_prep = preprocessor.transform(df)
    pred = rf.predict(df_prep.to_numpy(dtype=np.float32))[0]
    prob = rf.predict_proba(df_prep.to_numpy(dtype=np.float32))[0]
    return pred, prob

# Sidebar
with st.sidebar:
    st.markdown("## 📝 Application Form")
    st.markdown("---")
    
    st.markdown("### 🏦 Banking")
    checking = st.selectbox("Checking Balance", 
        ['no checking account', 'Rs. < 0', '0 <= Rs. < 2000', 'Rs. >=2000'])
    
    st.markdown("---")
    st.markdown("### 💰 Loan Details")
    amount = st.number_input("Amount (Rs.)", 250, 20000, 5000, 500)
    duration = st.slider("Duration (months)", 4, 72, 24)
    
    st.markdown("---")
    st.markdown("### 👤 Personal")
    age = st.number_input("Age", 18, 100, 35)
    
    st.markdown("---")
    st.markdown("### 💳 Financial")
    savings = st.selectbox("Savings Balance",
        ['no savings account', 'Rs. < 1000', '1000 <= Rs. < 5000', 
         '5000 <= Rs. < 10,000', 'Rs. >= 10,000'])
    rate = st.slider("Installment Rate (%)", 1, 4, 2)
    
    st.markdown("---")
    st.markdown("### 📋 Additional")
    plans = st.selectbox("Other Plans", ['none', 'bank', 'stores'])
    marital = st.selectbox("Marital Status", 
        ['single male', 'married', 'single female', 'divorced'])
    
    st.markdown("---")
    st.markdown("### 🎓 Background")
    purpose = st.selectbox("Loan Purpose", 
        ['electronics', 'new vehicle', 'furniture', 'domestic needs', 
         'education', 'business', 'renovation', 'second hand vehicle'])
    employment = st.selectbox("Employment Duration", 
        ['less than 1 year', '1-2 years', '2-4 years', '4-7 years', 'more than 7 years'])
    
    st.markdown("---")
    btn = st.button("🔮 ANALYZE", use_container_width=True)

# Main area - Simple info cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style='background: rgba(255,255,255,0.95); padding: 30px; border-radius: 15px; box-shadow: 0 8px 20px rgba(0,0,0,0.2);'>
        <h3 style='color: #667eea; margin-top: 0;'>🎯 Quick & Smart</h3>
        <p style='color: #555; font-size: 1.1rem;'>
            Get instant loan decisions powered by AI. Fast, accurate, and secure.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: rgba(255,255,255,0.95); padding: 30px; border-radius: 15px; box-shadow: 0 8px 20px rgba(0,0,0,0.2);'>
        <h3 style='color: #667eea; margin-top: 0;'>⚡ Easy Process</h3>
        <p style='color: #555; font-size: 1.1rem;'>
            Fill the form, click analyze, and get your result instantly!
        </p>
    </div>
    """, unsafe_allow_html=True)

# Prediction
if btn:
    inputs = {
        'Cbal': checking, 'Camt': amount, 'Cdur': duration,
        'age': age, 'Sbal': savings, 'InRate': rate, 'inPlans': plans,
        'MSG': marital, 'Cpur': purpose, 'Edur': employment
    }
    
    with st.spinner("🔄 Analyzing..."):
        pred, prob = predict(inputs)
    
    score = prob[1] * 100
    
    st.markdown("---")
    
    # Big result card
    if pred == 1:
        st.balloons()
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 50px; border-radius: 20px; text-align: center; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin: 30px 0;'>
            <h1 style='color: white; font-size: 5rem; margin: 0;'>✅</h1>
            <h1 style='color: white; font-size: 3rem; margin: 10px 0;'>APPROVED!</h1>
            <h2 style='color: white; font-size: 2.5rem; margin: 0;'>{score:.1f}%</h2>
            <p style='color: white; font-size: 1.3rem; margin: 10px 0;'>Creditworthiness Score</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 50px; border-radius: 20px; text-align: center; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin: 30px 0;'>
            <h1 style='color: white; font-size: 5rem; margin: 0;'>⚠️</h1>
            <h1 style='color: white; font-size: 3rem; margin: 10px 0;'>REVIEW NEEDED</h1>
            <h2 style='color: white; font-size: 2.5rem; margin: 0;'>{score:.1f}%</h2>
            <p style='color: white; font-size: 1.3rem; margin: 10px 0;'>Creditworthiness Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Credit Score", 'font': {'size': 28}},
        number={'font': {'size': 60}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue", 'thickness': 0.8},
            'steps': [
                {'range': [0, 40], 'color': "#ffcccc"},
                {'range': [40, 70], 'color': "#ffffcc"},
                {'range': [70, 100], 'color': "#ccffcc"}
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    # 3 cards
    col1, col2, col3 = st.columns(3)
    
    risk = "🟢 LOW" if score > 70 else "🟡 MEDIUM" if score > 40 else "🔴 HIGH"
    action = "✅ APPROVE" if score > 60 else "📋 REVIEW" if score > 40 else "❌ REJECT"
    
    with col1:
        st.markdown(f"""
        <div style='background: white; padding: 30px; border-radius: 15px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.2);'>
            <h2 style='color: #667eea; margin: 0; font-size: 2rem;'>{risk}</h2>
            <p style='color: #888; margin: 10px 0 0 0;'>Risk Level</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: white; padding: 30px; border-radius: 15px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.2);'>
            <h2 style='color: #667eea; margin: 0; font-size: 2rem;'>{action}</h2>
            <p style='color: #888; margin: 10px 0 0 0;'>Action</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background: white; padding: 30px; border-radius: 15px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.2);'>
            <h2 style='color: #667eea; margin: 0; font-size: 2rem;'>Rs. {amount:,}</h2>
            <p style='color: #888; margin: 10px 0 0 0;'>Amount</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Recommendations
    if score > 70:
        st.success("""
        **✅ STRONG CANDIDATE**
        - Approve immediately
        - Standard terms apply
        - No extra documents needed
        """)
    elif score > 50:
        st.info("""
        **📋 GOOD CANDIDATE**
        - Approve with verification
        - Check recent bank statements
        - Standard monitoring
        """)
    elif score > 40:
        st.warning("""
        **⚠️ RISKY CANDIDATE**
        - Needs guarantor
        - Consider lower amount
        - Extra documentation required
        """)
    else:
        st.error("""
        **❌ HIGH RISK**
        - Reject or need collateral
        - Too many risk factors
        - Alternative products only
        """)
    
    # Summary with download option
    st.markdown("<h3 style='color: white;'>📄 Application Summary</h3>", unsafe_allow_html=True)
    
    # Prepare data for download
    summary_data = {
        'Field': ['Checking Balance', 'Credit Amount', 'Duration', 'Age', 'Savings Balance', 
                  'Installment Rate', 'Other Plans', 'Marital Status', 'Loan Purpose', 'Employment Duration'],
        'Value': [checking, f'Rs. {amount:,}', f'{duration} months', f'{age} years', savings,
                  f'{rate}%', plans, marital, purpose, employment],
        'Credit Score': [f'{score:.1f}%'] * 10,
        'Decision': ['APPROVED' if pred == 1 else 'REVIEW NEEDED'] * 10,
        'Risk Level': [risk.replace('🟢 ', '').replace('🟡 ', '').replace('🔴 ', '')] * 10
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **💰 Financial Details**
        - Amount: Rs. {amount:,}
        - Duration: {duration} months
        - Rate: {rate}%
        - Purpose: {purpose}
        """)
    with col2:
        st.info(f"""
        **👤 Applicant Profile**
        - Age: {age} years
        - Marital Status: {marital}
        - Checking: {checking}
        - Savings: {savings}
        """)
    
    # Download button
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Application Report",
        data=csv,
        file_name=f"credit_report_{amount}_{age}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h3 style='color: white;'>🤖 Powered by AI</h3>
    <p style='font-size: 1rem;'>🔒 Secure • ⚡ Fast • 🎯 Accurate</p>
    <p style='font-size: 0.85rem; opacity: 0.7;'>© 2025 Credit Analyzer</p>
</div>
""", unsafe_allow_html=True)