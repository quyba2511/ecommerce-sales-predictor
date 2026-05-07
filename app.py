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
# 頁面配置
# ============================================================
st.set_page_config(
    page_title="E-commerce Sales Predictor",
    page_icon="🛍️",
    layout="wide"
)

# ============================================================
# 載入模型與資料
# ============================================================
@st.cache_resource
def load_model():
    """快取模型,避免使用者每次互動時重新載入"""
    return joblib.load('models/linear_regression_model.pkl')

@st.cache_data
def load_data():
    """快取資料集"""
    df = pd.read_csv('data/Ecommerce_Customers.csv')
    return df.drop(columns=['Email', 'Address', 'Avatar'])

model = load_model()
df = load_data()

# ============================================================
# 標題
# ============================================================
st.title("🛍️ 電子商務客戶銷售預測器")
st.markdown("""
**資料分析師專案 | 線性迴歸模型**

本應用程式根據客戶使用 App / 網站的行為,預測其**年度消費金額**。
""")
st.divider()

# ============================================================
# 側邊欄 — 使用者輸入(在所有分頁皆顯示)
# ============================================================
st.sidebar.header("📝 輸入客戶資訊")

avg_session = st.sidebar.slider(
    "平均單次諮詢時長(分鐘)",
    min_value=29.0, max_value=37.0, value=33.0, step=0.1,
    help="客戶於門市每次諮詢的平均時間"
)

time_on_app = st.sidebar.slider(
    "App 使用時長(分鐘)",
    min_value=8.0, max_value=15.0, value=12.0, step=0.1,
    help="每次使用 App 的時間"
)

time_on_web = st.sidebar.slider(
    "網站使用時長(分鐘)",
    min_value=33.0, max_value=41.0, value=37.0, step=0.1,
    help="每次使用網站的時間"
)

length_membership = st.sidebar.slider(
    "會員年資(年)",
    min_value=0.0, max_value=7.0, value=3.5, step=0.1,
    help="加入會員的年數"
)

# 計算預測值(用於 tab1)
input_data = np.array([[avg_session, time_on_app, time_on_web, length_membership]])
prediction = model.predict(input_data)[0]
avg_spending = df['Yearly Amount Spent'].mean()

# ============================================================
# 分頁
# ============================================================
tab1, tab2, tab3 = st.tabs(["🎯 預測", "📊 資料探索", "ℹ️ 關於"])

# ------------------------------------------------------------
# TAB 1: 預測
# ------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 預測結果")

        # 顯示主要結果
        st.metric(
            label="預測年度消費金額",
            value=f"${prediction:,.2f}",
            delta=f"{(prediction - avg_spending):,.2f} 與平均值差距"
        )

        # 客戶評估
        if prediction > avg_spending * 1.2:
            st.success("⭐ VIP 客戶 — 消費高於平均 20% 以上")
        elif prediction > avg_spending:
            st.info("✅ 優質客戶 — 消費高於平均")
        else:
            st.warning("📊 一般客戶 — 消費低於平均")

        # 顯示已輸入的資料
        st.markdown("**📋 客戶資訊:**")
        info_df = pd.DataFrame({
            '特徵': ['平均單次諮詢時長', 'App 使用時長', '網站使用時長', '會員年資'],
            '數值': [f"{avg_session:.1f} 分鐘", f"{time_on_app:.1f} 分鐘",
                     f"{time_on_web:.1f} 分鐘", f"{length_membership:.1f} 年"]
        })
        st.dataframe(info_df, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("📊 與全體客戶比較")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['Yearly Amount Spent'],
            nbinsx=30,
            name='全體客戶',
            marker_color='lightblue',
            opacity=0.7
        ))
        fig.add_vline(
            x=prediction,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"此客戶: ${prediction:.0f}",
            annotation_position="top"
        )
        fig.update_layout(
            xaxis_title="年度消費金額(美元)",
            yaxis_title="客戶數量",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # 係數 — 商業洞察
    # ============================================================
    st.divider()
    st.subheader("💡 模型商業洞察")

    features = ['平均單次諮詢時長', 'App 使用時長', '網站使用時長', '會員年資']
    coefficients = model.coef_

    col1, col2, col3, col4 = st.columns(4)
    for col, feat, coef in zip([col1, col2, col3, col4], features, coefficients):
        with col:
            st.metric(
                label=feat,
                value=f"+${coef:.2f}",
                help=f"每增加一單位 → 營收增加 ${coef:.2f}"
            )

    st.info("""
    **📌 關鍵洞察:**
    - **會員年資**為最重要的因素 → 應投資忠誠度計畫
    - **App 使用時長**的影響遠大於**網站使用時長** → 優先發展 App
    - **網站使用時長**幾乎沒有影響 → 可考慮削減網站預算

    *分析者:Pô | 線性迴歸 | R² = 0.98*
    """)

# ------------------------------------------------------------
# TAB 2: 資料探索
# ------------------------------------------------------------
with tab2:
    st.subheader("📊 資料探索")

    # 總覽統計
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("客戶總數", f"{len(df):,}")
    col2.metric("平均消費", f"${df['Yearly Amount Spent'].mean():.0f}")
    col3.metric("最高消費", f"${df['Yearly Amount Spent'].max():.0f}")
    col4.metric("最低消費", f"${df['Yearly Amount Spent'].min():.0f}")

    st.divider()

    # 相關性熱力圖
    st.subheader("🔥 相關性矩陣")
    corr = df.corr(numeric_only=True).round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1, aspect='auto')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 散點圖
    st.subheader("📈 與營收的相關性")
    feature_choice = st.selectbox(
        "選擇要分析的特徵:",
        ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
    )
    fig = px.scatter(df, x=feature_choice, y='Yearly Amount Spent',
                    trendline='ols', opacity=0.6,
                    trendline_color_override='red')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 顯示部分原始資料
    st.divider()
    st.subheader("📋 資料樣本")
    st.dataframe(df.head(10), use_container_width=True)

# ------------------------------------------------------------
# TAB 3: 關於
# ------------------------------------------------------------
with tab3:
    st.subheader("ℹ️ 關於本專案")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🎯 專案目標
        根據電商客戶使用 App / 網站的行為,預測其**年度消費金額**,
        協助公司基於資料做出商業決策。

        ### 🛠️ 使用技術
        - **Python 3.12**:主要程式語言
        - **Pandas & NumPy**:資料處理
        - **Scikit-learn**:線性迴歸模型
        - **Plotly**:互動式視覺化
        - **Streamlit**:佈署
        - **Statsmodels**:統計分析與 VIF 分析

        ### 📊 流程
        1. **EDA 與視覺化**:資料分析、發掘洞察
        2. **特徵工程**:從原始資料建立新特徵
        3. **VIF 分析**:檢查多重共線性
        4. **訓練/測試集切分與標準化**:準備資料
        5. **線性迴歸**:訓練模型
        6. **評估**:R² = 0.98、MAE ≈ $8、RMSE ≈ $10
        7. **佈署**:Streamlit Cloud

        ### 💡 關鍵洞察
        - 會員年資是最強的影響因素(corr = 0.81)
        - App 使用時長的影響比網站高出 200 倍
        - 建議:投資 App 與忠誠度計畫
        """)

    with col2:
        st.markdown("""
        ### 👤 聯絡方式
        **作者:** Pô
        **Email:** your.email@gmail.com
        **GitHub:** [your-username](https://github.com/your-username)
        **LinkedIn:** [your-profile](https://linkedin.com/in/your-profile)

        ---

        ### 📈 模型表現
        """)
        st.metric("R² Score", "0.98")
        st.metric("MAE", "$8")
        st.metric("RMSE", "$10")

# ============================================================
# 頁尾
# ============================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    Made with ❤️ using Streamlit | Linear Regression Model<br>
    <a href='https://github.com/your-username/ecommerce-sales-predictor'>GitHub</a> |
    <a href='https://linkedin.com/in/your-profile'>LinkedIn</a>
</div>
""", unsafe_allow_html=True)
