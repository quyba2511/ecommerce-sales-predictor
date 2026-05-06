# app.py
# E-commerce Sales Predictor - Streamlit App
# Author: Pô | Linear Regression Project

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="E-commerce Sales Predictor",
    page_icon="🛍️",
    layout="wide"
)

# ============================================================
# LOAD MODEL VÀ DATA
# ============================================================
@st.cache_resource
def load_model():
    """Cache model để không load lại mỗi lần user tương tác"""
    return joblib.load('models/linear_regression_model.pkl')

@st.cache_data
def load_data():
    """Cache dataset"""
    df = pd.read_csv('data/Ecommerce_Customers.csv')
    return df.drop(columns=['Email', 'Address', 'Avatar'])

model = load_model()
df = load_data()

# ============================================================
# HEADER
# ============================================================
st.title("🛍️ E-commerce Customer Sales Predictor")
st.markdown("""
**Dự án Data Analyst | Linear Regression Model**

Ứng dụng dự đoán **doanh thu hàng năm** của khách hàng dựa trên hành vi sử dụng app/website.
""")
st.divider()

# ============================================================
# SIDEBAR — INPUT TỪ USER (luôn hiển thị ở mọi tab)
# ============================================================
st.sidebar.header("📝 Nhập thông tin khách hàng")

avg_session = st.sidebar.slider(
    "Avg. Session Length (phút)",
    min_value=29.0, max_value=37.0, value=33.0, step=0.1,
    help="Thời gian trung bình mỗi session tư vấn tại store"
)

time_on_app = st.sidebar.slider(
    "Time on App (phút)",
    min_value=8.0, max_value=15.0, value=12.0, step=0.1,
    help="Thời gian dùng app mỗi lần"
)

time_on_web = st.sidebar.slider(
    "Time on Website (phút)",
    min_value=33.0, max_value=41.0, value=37.0, step=0.1,
    help="Thời gian dùng website mỗi lần"
)

length_membership = st.sidebar.slider(
    "Length of Membership (năm)",
    min_value=0.0, max_value=7.0, value=3.5, step=0.1,
    help="Số năm là thành viên"
)

# Tính prediction (dùng cho tab1)
input_data = np.array([[avg_session, time_on_app, time_on_web, length_membership]])
prediction = model.predict(input_data)[0]
avg_spending = df['Yearly Amount Spent'].mean()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📊 Explore Data", "ℹ️ About"])

# ------------------------------------------------------------
# TAB 1: PREDICT
# ------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Kết quả dự đoán")
        
        # Hiển thị kết quả lớn
        st.metric(
            label="Doanh thu hàng năm dự đoán",
            value=f"${prediction:,.2f}",
            delta=f"{(prediction - avg_spending):,.2f} so với TB"
        )
        
        # Đánh giá khách hàng
        if prediction > avg_spending * 1.2:
            st.success("⭐ Khách hàng VIP — chi tiêu cao hơn trung bình 20%+")
        elif prediction > avg_spending:
            st.info("✅ Khách hàng tốt — chi tiêu trên trung bình")
        else:
            st.warning("📊 Khách hàng thường — chi tiêu dưới trung bình")
        
        # Hiển thị input đã nhập
        st.markdown("**📋 Thông tin khách hàng:**")
        info_df = pd.DataFrame({
            'Feature': ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership'],
            'Value': [f"{avg_session:.1f} phút", f"{time_on_app:.1f} phút", 
                     f"{time_on_web:.1f} phút", f"{length_membership:.1f} năm"]
        })
        st.dataframe(info_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("📊 So sánh với toàn bộ khách hàng")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['Yearly Amount Spent'],
            nbinsx=30,
            name='Toàn bộ KH',
            marker_color='lightblue',
            opacity=0.7
        ))
        fig.add_vline(
            x=prediction,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"KH này: ${prediction:.0f}",
            annotation_position="top"
        )
        fig.update_layout(
            xaxis_title="Yearly Amount Spent (USD)",
            yaxis_title="Số lượng KH",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================
    # COEFFICIENTS — INSIGHT KINH DOANH
    # ============================================================
    st.divider()
    st.subheader("💡 Business Insights từ Model")
    
    features = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
    coefficients = model.coef_
    
    col1, col2, col3, col4 = st.columns(4)
    for col, feat, coef in zip([col1, col2, col3, col4], features, coefficients):
        with col:
            st.metric(
                label=feat,
                value=f"+${coef:.2f}",
                help=f"Mỗi đơn vị tăng → doanh thu tăng ${coef:.2f}"
            )
    
    st.info("""
    **📌 Key Insights:**
    - **Length of Membership** là yếu tố quan trọng nhất → đầu tư chương trình loyalty
    - **Time on App** ảnh hưởng lớn hơn **Time on Website** rất nhiều → ưu tiên phát triển App
    - **Time on Website** gần như không tác động → cân nhắc giảm ngân sách Web
    
    *Phân tích bởi: Pô | Linear Regression | R² = 0.98*
    """)

# ------------------------------------------------------------
# TAB 2: EXPLORE DATA
# ------------------------------------------------------------
with tab2:
    st.subheader("📊 Khám phá dữ liệu")
    
    # Stats tổng quan
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng KH", f"{len(df):,}")
    col2.metric("Doanh thu TB", f"${df['Yearly Amount Spent'].mean():.0f}")
    col3.metric("Doanh thu Max", f"${df['Yearly Amount Spent'].max():.0f}")
    col4.metric("Doanh thu Min", f"${df['Yearly Amount Spent'].min():.0f}")
    
    st.divider()
    
    # Correlation heatmap
    st.subheader("🔥 Correlation Matrix")
    corr = df.corr(numeric_only=True).round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1, aspect='auto')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Scatter plot
    st.subheader("📈 Tương quan với Doanh thu")
    feature_choice = st.selectbox(
        "Chọn feature để phân tích:",
        ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
    )
    fig = px.scatter(df, x=feature_choice, y='Yearly Amount Spent',
                    trendline='ols', opacity=0.6,
                    trendline_color_override='red')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Hiển thị 1 phần dữ liệu raw
    st.divider()
    st.subheader("📋 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

# ------------------------------------------------------------
# TAB 3: ABOUT
# ------------------------------------------------------------
with tab3:
    st.subheader("ℹ️ Về dự án này")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Mục tiêu
        Dự đoán **doanh thu hàng năm** của khách hàng TMĐT dựa trên hành vi sử dụng app/website,
        giúp công ty đưa ra quyết định kinh doanh dựa trên dữ liệu.
        
        ### 🛠️ Công nghệ sử dụng
        - **Python 3.12**: ngôn ngữ chính
        - **Pandas & NumPy**: xử lý dữ liệu
        - **Scikit-learn**: Linear Regression model
        - **Plotly**: visualization tương tác
        - **Streamlit**: deployment
        - **Statsmodels**: thống kê & VIF analysis
        
        ### 📊 Pipeline
        1. **EDA & Visualization**: phân tích dữ liệu, tìm insights
        2. **Feature Engineering**: tạo features mới từ data gốc
        3. **VIF Analysis**: kiểm tra multicollinearity
        4. **Train/Test Split & Scaling**: chuẩn bị data
        5. **Linear Regression**: train model
        6. **Evaluation**: R² = 0.98, MAE ≈ $8, RMSE ≈ $10
        7. **Deployment**: Streamlit Cloud
        
        ### 💡 Key Insights
        - Length of Membership là yếu tố mạnh nhất (corr = 0.81)
        - Time on App tác động lớn hơn Time on Website 200 lần
        - Khuyến nghị: đầu tư App + chương trình loyalty
        """)
    
    with col2:
        st.markdown("""
        ### 👤 Liên hệ
        **Tác giả:** Pô  
        **Email:** your.email@gmail.com  
        **GitHub:** [your-username](https://github.com/your-username)  
        **LinkedIn:** [your-profile](https://linkedin.com/in/your-profile)
        
        ---
        
        ### 📈 Model Performance
        """)
        st.metric("R² Score", "0.98")
        st.metric("MAE", "$8")
        st.metric("RMSE", "$10")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    Made with ❤️ using Streamlit | Linear Regression Model<br>
    <a href='https://github.com/your-username/ecommerce-sales-predictor'>GitHub</a> | 
    <a href='https://linkedin.com/in/your-profile'>LinkedIn</a>
</div>
""", unsafe_allow_html=True)
