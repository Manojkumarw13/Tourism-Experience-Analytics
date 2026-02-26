"""
app.py
------
Tourism Experience Analytics - Streamlit Application
Multi-page app with:
  1. Home        - Project overview & key stats
  2. Rating      - Predict attraction rating (Regression)
  3. Visit Mode  - Predict visit mode (Classification)
  4. Recommender - Content-based attraction recommendations
  5. Dashboard   - EDA analytics charts
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from model_utils import (
    VISIT_MODE_MAP,
    get_recommendations,
    load_and_preprocess,
    month_to_season_code,
    predict_rating,
    predict_visit_mode,
    train_all_models,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e0f0 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #c8c8e8 !important; }

    /* Main background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2e 100%);
        color: #e0e0f0;
    }
    
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a3e 0%, #2d2d60 100%);
        border: 1px solid rgba(147, 112, 219, 0.3);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-4px); }
    .metric-card h2 { color: #a78bfa; font-size: 2rem; margin: 0; font-weight: 700; }
    .metric-card p  { color: #c4b5fd; margin: 4px 0 0 0; font-size: 0.9rem; }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 8px;
    }

    .section-title {
        color: #a78bfa;
        font-size: 1.5rem;
        font-weight: 700;
        border-left: 4px solid #a78bfa;
        padding-left: 12px;
        margin-bottom: 20px;
    }

    .result-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #1a2f4a 100%);
        border: 1px solid rgba(96, 165, 250, 0.4);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    }
    .result-box h1 { color: #60a5fa; font-size: 3.5rem; margin: 0; }
    .result-box h3 { color: #93c5fd; margin-top: 8px; }
    .result-box p  { color: #94a3b8; font-size: 0.9rem; margin-top: 4px; }

    .prediction-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 6px;
    }

    /* Input form styling */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #c4b5fd !important;
        font-weight: 500;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 32px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: all 0.3s !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6) !important;
    }

    .info-card {
        background: rgba(30, 30, 60, 0.7);
        border: 1px solid rgba(147, 112, 219, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .info-card h4 { color: #a78bfa; margin: 0 0 6px 0; }
    .info-card p  { color: #94a3b8; margin: 0; font-size: 0.9rem; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Tables */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA & MODELS (cached)
# ─────────────────────────────────────────────────────────────

df, df_model = load_and_preprocess()
models = train_all_models(df_model)

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 20px 0 10px 0;'>
            <div style='font-size:3rem;'>🌍</div>
            <div style='font-size:1.1rem; font-weight:700; color:#a78bfa; letter-spacing:1px;'>TOURISM ANALYTICS</div>
            <div style='font-size:0.75rem; color:#64748b; margin-top:4px;'>ML-Powered Platform</div>
        </div>
        <hr style='border-color: rgba(147,112,219,0.2); margin: 10px 0 20px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        options=[
            "🏠  Home",
            "⭐  Rating Predictor",
            "🧳  Visit Mode Classifier",
            "🗺️  Attraction Recommender",
            "📊  Analytics Dashboard",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
        <hr style='border-color: rgba(147,112,219,0.2); margin: 20px 0 16px 0;'>
        <div style='font-size:0.85rem; color:#94a3b8; text-align:center; padding-bottom:10px; line-height: 1.6;'>
            📦 Dataset: 52,922 records<br>
            🤖 3 ML Models Active<br>
            📅 Data: 2013–2022
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE 1: HOME
# ─────────────────────────────────────────────────────────────

if "Home" in page:
    st.markdown('<div class="hero-title">Tourism Experience Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">AI-powered platform for rating prediction, traveler segmentation & personalized attraction recommendations</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h2>{len(df):,}</h2><p>Total Transactions</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h2>{df['UserId'].nunique():,}</h2><p>Unique Users</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h2>{df['AttractionId'].nunique():,}</h2><p>Attractions</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><h2>{df['CountryId'].nunique():,}</h2><p>Countries Evaluated</p></div>", unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color: rgba(147,112,219,0.2);'><br>", unsafe_allow_html=True)

    # ML Capabilities overview
    st.markdown('<div class="section-title">Machine Learning Capabilities</div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="info-card" style="height: 170px;">
            <h4>⭐ Rating Predictor</h4>
            <p>Predicts the user satisfaction out of 5 stars based on demographics, visit time, and attraction characteristics using a Random Forest Regressor.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-card" style="height: 170px;">
            <h4>🧳 Visit Mode Classifier</h4>
            <p>Classifies tourists into specific segments (Family, Couples, Business, Friends) using a balanced Random Forest Classifier to personalize travel marketing.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="info-card" style="height: 170px;">
            <h4>🗺️ Attraction Recommender</h4>
            <p>Suggests new, highly-relevant attractions using a content-based filtering approach with Cosine Similarity across complex attraction profiles.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Raw Data Snapshot</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 2: RATING PREDICTOR (REGRESSION)
# ─────────────────────────────────────────────────────────────

elif "Rating" in page:
    st.markdown('<div class="section-title">⭐ Rating Predictor</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;'>Input visit attributes to predict the star rating (1-5) a user will give an attraction.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.form("rating_form"):
            r1c1, r1c2 = st.columns(2)
            visit_year = r1c1.slider("Visit Year", 2013, 2022, 2022)
            visit_month = r1c2.slider("Visit Month", 1, 12, 6)
            
            r2c1, r2c2 = st.columns(2)
            visit_mode = r2c1.selectbox("Visit Mode", options=list(VISIT_MODE_MAP.keys()), format_func=lambda x: f"{x} - {VISIT_MODE_MAP[x]}")
            # BUG 4 FIX: Cast to int() to avoid numpy.int64 type mismatch
            continent_id = r2c2.number_input("Continent ID", min_value=1, max_value=int(df['ContinentId'].max()), value=1)

            r3c1, r3c2, r3c3 = st.columns(3)
            # BUG 5 FIX: Added min_value=1 to all ID inputs to prevent negative/zero IDs
            region_id = r3c1.number_input("Region ID", min_value=1, value=2)
            country_id = r3c2.number_input("Country ID", min_value=1, value=5)
            city_id = r3c3.number_input("City ID", min_value=1, value=10)

            r4c1, r4c2, r4c3 = st.columns(3)
            attraction_id = r4c1.number_input("Attraction ID", min_value=1, value=1)
            attr_city_id = r4c2.number_input("Attraction City ID", min_value=1, value=1)
            attr_type_id = r4c3.number_input("Attraction Type ID", min_value=1, value=2)
            
            submit = st.form_submit_button("Predict Attraction Rating ⭐")

    with col2:
        if submit:
            user_in = {
                "VisitYear": visit_year,
                "VisitMonth": visit_month,
                "VisitMode": visit_mode,
                "ContinentId": continent_id,
                "RegionId": region_id,
                "CountryId": country_id,
                "CityId": city_id,
                "AttractionId": attraction_id,
                "AttractionCityId": attr_city_id,
                "AttractionTypeId": attr_type_id,
                "Season": month_to_season_code(visit_month)
            }
            
            with st.spinner("Analyzing patterns..."):
                pred_rating = predict_rating(models, user_in)
            
            st.markdown(f"""
            <div class="result-box">
                <div class="prediction-label">Predicted Rating</div>
                <h1>{pred_rating:.2f}</h1>
                <h3>{'⭐' * round(pred_rating)}</h3>
                <p>Based on Random Forest Regressor</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-card">
                <h4>Model Metrics</h4>
                <p><b>R² Score:</b> {r2}</p>
                <p><b>RMSE:</b> {rmse}</p>
                <p><b>MAE:</b> {mae}</p>
                <br>
                <p><i>The model explains variance in reviews well, predicting within ~{rmse} stars on average.</i></p>
            </div>
            """.format(
                r2=models['reg_metrics']['R2'],
                rmse=models['reg_metrics']['RMSE'],
                mae=models['reg_metrics']['MAE']
            ), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 3: VISIT MODE CLASSIFIER
# ─────────────────────────────────────────────────────────────

elif "Visit Mode" in page:
    st.markdown('<div class="section-title">🧳 Visit Mode Classifier</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;'>Predict the traveler segment (e.g. Family, Business) based on demographics and visit details.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.form("clf_form"):
            r1c1, r1c2, r1c3 = st.columns(3)
            visit_year = r1c1.slider("Visit Year", 2013, 2022, 2022)
            visit_month = r1c2.slider("Visit Month", 1, 12, 6)
            rating = r1c3.slider("Attraction Rating", 1, 5, 4)
            
            r2c1, r2c2 = st.columns(2)
            continent_id = r2c1.number_input("Continent ID", min_value=1, value=1)
            # BUG 5 FIX: Added min_value=1 to all ID inputs
            region_id = r2c2.number_input("Region ID", min_value=1, value=2)

            r3c1, r3c2 = st.columns(2)
            country_id = r3c1.number_input("Country ID", min_value=1, value=5)
            city_id = r3c2.number_input("City ID", min_value=1, value=10)

            r4c1, r4c2, r4c3 = st.columns(3)
            attraction_id = r4c1.number_input("Attraction ID", min_value=1, value=1)
            attr_city_id = r4c2.number_input("Attraction City ID", min_value=1, value=1)
            attr_type_id = r4c3.number_input("Attraction Type ID", min_value=1, value=2)
            
            submit = st.form_submit_button("Classify Visit Mode 🧳")

    with col2:
        if submit:
            user_in = {
                "VisitYear": visit_year,
                "VisitMonth": visit_month,
                "ContinentId": continent_id,
                "RegionId": region_id,
                "CountryId": country_id,
                "CityId": city_id,
                "AttractionId": attraction_id,
                "AttractionCityId": attr_city_id,
                "AttractionTypeId": attr_type_id,
                "Season": month_to_season_code(visit_month),
                "Rating": rating
            }
            
            with st.spinner("Classifying user profile..."):
                code, label = predict_visit_mode(models, user_in)
            
            icons = {"Business": "💼", "Couples": "💑", "Family": "👨‍👩‍👧‍👦", "Friends": "🍻", "Solo": "🎒"}
            icon = icons.get(label, "✈️")
            
            st.markdown(f"""
            <div class="result-box">
                <div class="prediction-label">Predicted Segment</div>
                <h1>{icon}</h1>
                <h3 style="color: #34d399; font-size: 2rem;">{label}</h3>
                <p>Class Mode ID: {code}</p>
                <p>Based on Random Forest Classifier</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            acc = models['clf_metrics']['Accuracy']
            st.markdown(f"""
            <div class="info-card">
                <h4>Model Metrics</h4>
                <p><b>Test Accuracy:</b> {acc * 100:.2f}%</p>
                <br>
                <p><i>Trained with balanced class weights to prevent majority class bias.</i></p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 4: ATTRACTION RECOMMENDER
# ─────────────────────────────────────────────────────────────

elif "Recommender" in page:
    st.markdown('<div class="section-title">🗺️ Attraction Recommender</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;'>Content-based filtering using cosine similarity. Select an attraction you liked to get similar recommendations.</p>", unsafe_allow_html=True)

    avail_ids = sorted(models["attraction_profile"]["AttractionId"].unique())

    # BUG 2 FIX: Initialize recs before columns so col2 can always reference it
    recs = pd.DataFrame()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("<br>", unsafe_allow_html=True)
        chosen_id = st.selectbox(
            "Select reference Attraction ID:",
            options=avail_ids,
            index=0
        )
        top_n = st.slider("Number of recommendations:", 3, 10, 5)

        btn = st.button("Generate Recommendations ✨")

        # BUG 1 FIX: Removed `or True` — button now properly controls regeneration
        if btn:
            recs = get_recommendations(models, chosen_id, top_n)

            # Show the reference attraction profile
            ref_row = models["attraction_profile"][models["attraction_profile"]["AttractionId"] == chosen_id].iloc[0]
            st.markdown("""
            <div class="info-card" style="margin-top:20px;">
                <h4 style="color:#60a5fa;">Target Attraction Profile</h4>
                <p><b>ID:</b> {id}</p>
                <p><b>Type ID:</b> {type_id}</p>
                <p><b>City ID:</b> {city_id}</p>
                <p><b>Avg Rating:</b> {rating:.2f} ⭐</p>
                <p><b>Total Visits:</b> {visits:,}</p>
            </div>
            """.format(
                id=ref_row["AttractionId"],
                type_id=ref_row["AttractionTypeId"],
                city_id=ref_row["AttractionCityId"],
                rating=ref_row["AvgRating"],
                visits=ref_row["VisitCount"]
            ), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-card" style="margin-top:20px;">
                <h4 style="color:#60a5fa;">How to Use</h4>
                <p>Select an Attraction ID from the dropdown, choose how many recommendations you want, then click <b>Generate Recommendations</b>.</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if len(recs) > 0:
            st.markdown(f"**Top {top_n} Similar Attractions to ID {chosen_id}**")
            st.dataframe(
                recs.style.background_gradient(cmap='viridis_r', subset=['Similarity Score']),
                use_container_width=True,
                height=350
            )
        elif btn:
            st.warning("No recommendations found for the selected attraction.")
        else:
            st.info("Click **Generate Recommendations** to see similar attractions here.")


# ─────────────────────────────────────────────────────────────
# PAGE 5: ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────

elif "Dashboard" in page:
    st.markdown('<div class="section-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Custom plotly layout
    layout_tr = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        xaxis=dict(gridcolor='rgba(147,112,219,0.1)'),
        yaxis=dict(gridcolor='rgba(147,112,219,0.1)')
    )
    
    tab1, tab2, tab3 = st.tabs(["Distributions", "Trends", "Correlations"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.histogram(df, x="Rating", nbins=5, title="Rating Distribution", 
                               color_discrete_sequence=['#a78bfa'])
            fig1.update_layout(**layout_tr, bargap=0.2)
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            mode_counts = df["VisitMode"].map(VISIT_MODE_MAP).value_counts().reset_index()
            mode_counts.columns = ['Visit Mode', 'Count']
            fig2 = px.pie(mode_counts, values='Count', names='Visit Mode', 
                         title="Visit Mode Distribution", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'))
            st.plotly_chart(fig2, use_container_width=True)
            
        c3, c4 = st.columns(2)
        with c3:
            fig_cont = px.histogram(df, x="ContinentId", title="Continent Distribution",
                                   color_discrete_sequence=['#60a5fa'])
            fig_cont.update_layout(**layout_tr, bargap=0.2)
            st.plotly_chart(fig_cont, use_container_width=True)
            
        with c4:
            top_attr = df["AttractionId"].value_counts().head(10).reset_index()
            top_attr.columns = ["AttractionId", "Visits"]
            top_attr["AttractionId"] = top_attr["AttractionId"].astype(str)
            fig_attr = px.bar(top_attr, x="Visits", y="AttractionId", orientation='h',
                             title="Top 10 Attractions",
                             color_discrete_sequence=['#34d399'])
            fig_attr.update_layout(**layout_tr, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_attr, use_container_width=True)
            
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            year_trend = df["VisitYear"].value_counts().sort_index().reset_index()
            year_trend.columns = ["Year", "Visits"]
            fig3 = px.line(year_trend, x="Year", y="Visits", markers=True,
                          title="Visits by Year", color_discrete_sequence=['#f87171'])
            fig3.update_layout(**layout_tr)
            st.plotly_chart(fig3, use_container_width=True)
            
        with c2:
            month_trend = df["VisitMonth"].value_counts().sort_index().reset_index()
            month_trend.columns = ["Month", "Visits"]
            fig4 = px.bar(month_trend, x="Month", y="Visits", 
                         title="Visits by Month (Seasonality)", color_discrete_sequence=['#fbbf24'])
            fig4.update_layout(**layout_tr, xaxis=dict(tickmode='linear'))
            st.plotly_chart(fig4, use_container_width=True)
            
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        # Heatmap
        # BUG 3 FIX: numeric_only=True required for pandas >= 2.0
        corr = df_model.corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Feature Correlation Heatmap")
        fig_corr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0')
        )
        st.plotly_chart(fig_corr, use_container_width=True)
