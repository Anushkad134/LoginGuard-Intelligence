import random
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Cyberpunk Configuration
st.set_page_config(
    page_title="🤖 AI BODYGUARD - Cyberpunk Security",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_cyberpunk_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto+Mono:wght@300;400;700&display=swap');
    
    /* Global Cyberpunk Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Main Header - Neon Glow Effect */
    .main-header {
        background: linear-gradient(135deg, #0a0a0a 0%, #16213e 100%);
        border: 2px solid #39FF14;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 0 30px #39FF14, inset 0 0 30px rgba(57, 255, 20, 0.1);
        animation: pulse-green 3s ease-in-out infinite alternate;
    }
    
    .main-header h1 {
        font-family: 'Orbitron', monospace;
        font-weight: 900;
        text-shadow: 0 0 20px #39FF14;
        color: #39FF14;
        margin: 0;
    }
    
    /* Neon Animations */
    @keyframes pulse-green {
        0% { box-shadow: 0 0 30px #39FF14, inset 0 0 30px rgba(57, 255, 20, 0.1); }
        100% { box-shadow: 0 0 50px #39FF14, inset 0 0 50px rgba(57, 255, 20, 0.2); }
    }
    
    @keyframes pulse-orange {
        0% { box-shadow: 0 0 20px #FF7B00, inset 0 0 20px rgba(255, 123, 0, 0.1); }
        100% { box-shadow: 0 0 40px #FF7B00, inset 0 0 40px rgba(255, 123, 0, 0.2); }
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 20px #ff073a, inset 0 0 20px rgba(255, 7, 58, 0.1); }
        100% { box-shadow: 0 0 40px #ff073a, inset 0 0 40px rgba(255, 7, 58, 0.2); }
    }
    
    /* 3D Metric Cards */
    .metric-card-safe {
        background: linear-gradient(135deg, #0a2e0a 0%, #1a4d1a 100%);
        border: 1px solid #39FF14;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: #39FF14;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(57, 255, 20, 0.3);
        transform: translateZ(0);
        transition: all 0.3s ease;
    }
    
    .metric-card-warning {
        background: linear-gradient(135deg, #2e1a0a 0%, #4d2d1a 100%);
        border: 1px solid #FF7B00;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: #FF7B00;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(255, 123, 0, 0.3);
        animation: pulse-orange 2s ease-in-out infinite alternate;
    }
    
    .metric-card-danger {
        background: linear-gradient(135deg, #2e0a0a 0%, #4d1a1a 100%);
        border: 1px solid #ff073a;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: #ff073a;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(255, 7, 58, 0.3);
        animation: pulse-red 1.5s ease-in-out infinite alternate;
    }
    
    .metric-card-safe:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 15px 45px rgba(57, 255, 20, 0.5);
    }
    
    /* Anomaly Alerts */
    .anomaly-alert {
        background: linear-gradient(135deg, #2e0a0a 0%, #4d1a1a 100%);
        border: 2px solid #ff073a;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 8px solid #ff073a;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(255, 7, 58, 0.4);
        animation: pulse-red 2s ease-in-out infinite alternate;
    }
    
    .safe-alert {
        background: linear-gradient(135deg, #0a2e0a 0%, #1a4d1a 100%);
        border: 2px solid #39FF14;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 8px solid #39FF14;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(57, 255, 20, 0.4);
    }
    
    /* Sidebar Styling */
    .sidebar-content {
        background: linear-gradient(135deg, #16213e 0%, #0f172a 100%);
        border: 1px solid #39FF14;
        padding: 1.5rem;
        border-radius: 15px;
        color: #39FF14;
        margin-bottom: 1rem;
        box-shadow: 0 0 20px rgba(57, 255, 20, 0.3);
    }
    
    /* Welcome Message */
    .welcome-message {
        background: linear-gradient(45deg, #16213e, #0f172a, #1e293b);
        border: 2px solid #39FF14;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: #39FF14;
        margin: 2rem 0;
        box-shadow: 0 0 40px rgba(57, 255, 20, 0.4);
        animation: pulse-green 4s ease-in-out infinite alternate;
    }
    
    .welcome-message h2 {
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 10px #39FF14;
    }
    
    /* Activity Feed */
    .activity-feed {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #FF7B00;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 0 15px rgba(255, 123, 0, 0.2);
    }
    
    /* Risk Profile Cards */
    .risk-profile-low {
        background: linear-gradient(135deg, #0a2e0a 0%, #1a4d1a 100%);
        border: 2px solid #39FF14;
        padding: 1rem;
        border-radius: 12px;
        color: #39FF14;
        box-shadow: 0 0 20px rgba(57, 255, 20, 0.3);
    }
    
    .risk-profile-medium {
        background: linear-gradient(135deg, #2e1a0a 0%, #4d2d1a 100%);
        border: 2px solid #FF7B00;
        padding: 1rem;
        border-radius: 12px;
        color: #FF7B00;
        box-shadow: 0 0 20px rgba(255, 123, 0, 0.3);
    }
    
    .risk-profile-high {
        background: linear-gradient(135deg, #2e0a0a 0%, #4d1a1a 100%);
        border: 2px solid #ff073a;
        padding: 1rem;
        border-radius: 12px;
        color: #ff073a;
        box-shadow: 0 0 20px rgba(255, 7, 58, 0.3);
    }
    
    /* Custom Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #39FF14 0%, #32cd32 100%);
        color: #000000;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(57, 255, 20, 0.4);
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(57, 255, 20, 0.8);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #39FF14, #FF7B00);
        border-radius: 10px;
    }
    
    /* Data Tables */
    .dataframe {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid #39FF14;
        border-radius: 10px;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #0a0a0a 0%, #16213e 100%);
        border-top: 2px solid #39FF14;
        padding: 2rem;
        text-align: center;
        color: #39FF14;
        margin-top: 3rem;
        box-shadow: 0 -10px 30px rgba(57, 255, 20, 0.2);
    }
    
    .footer h3 {
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 10px #39FF14;
    }
    </style>
    """, unsafe_allow_html=True)

def time_to_minutes(time_str):
    """Convert HH:MM to minutes since midnight"""
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    except:
        return 0

def calculate_distance_feature(locations):
    """Simple distance calculation based on location changes"""
    location_map = {
        'India': (20.5937, 78.9629),
        'USA': (39.8283, -98.5795),
        'UK': (55.3781, -3.4360),
        'Russia': (61.5240, 105.3188),
        'Canada': (56.1304, -106.3468)
    }
    
    distances = []
    prev_loc = None
    
    for loc in locations:
        if prev_loc is None or loc not in location_map or prev_loc not in location_map:
            distances.append(0)
        else:
            lat1, lon1 = location_map[prev_loc]
            lat2, lon2 = location_map[loc]
            distance = np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2)
            distances.append(distance * 1000)
        prev_loc = loc
    
    return distances

def calculate_risk_score(anomaly_scores):
    """Convert anomaly scores to 0-100 risk scores"""
    min_score = anomaly_scores.min()
    max_score = anomaly_scores.max()
    
    if max_score == min_score:
        return np.zeros(len(anomaly_scores))
    
    normalized = (anomaly_scores - min_score) / (max_score - min_score)
    risk_scores = (1 - normalized) * 100
    return risk_scores

def create_enhanced_features(df):
    """Create advanced features for anomaly detection"""
    df_enhanced = df.copy()
    
    df_enhanced['login_minutes'] = df_enhanced['login_time'].apply(time_to_minutes)
    
    user_avg_times = df_enhanced.groupby('user_id')['login_minutes'].mean()
    df_enhanced['avg_login_time'] = df_enhanced['user_id'].map(user_avg_times)
    df_enhanced['time_deviation'] = np.abs(df_enhanced['login_minutes'] - df_enhanced['avg_login_time'])
    
    df_enhanced = df_enhanced.sort_values(['user_id', 'login_time']).reset_index(drop=True)
    
    distance_features = []
    for user_id in df_enhanced['user_id'].unique():
        user_data = df_enhanced[df_enhanced['user_id'] == user_id]
        distances = calculate_distance_feature(user_data['location'].tolist())
        distance_features.extend(distances)
    
    df_enhanced['location_distance'] = distance_features
    
    le_location = LabelEncoder()
    le_device = LabelEncoder()
    
    df_enhanced['location_encoded'] = le_location.fit_transform(df_enhanced['location'])
    df_enhanced['device_encoded'] = le_device.fit_transform(df_enhanced['device'])
    
    return df_enhanced, le_location, le_device

def create_risk_gauge(risk_score):
    """Create a cyberpunk risk gauge"""
    if risk_score <= 30:
        color = '#39FF14'
        status = 'SAFE'
        bg_color = '#0a2e0a'
    elif risk_score <= 70:
        color = '#FF7B00'
        status = 'WATCH'
        bg_color = '#2e1a0a'
    else:
        color = '#ff073a'
        status = 'THREAT'
        bg_color = '#2e0a0a'
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"RISK STATUS: {status}", 'font': {'family': 'Orbitron', 'size': 20, 'color': color}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': color, 'tickfont': {'color': color}},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': bg_color,
            'borderwidth': 3,
            'bordercolor': color,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(57, 255, 20, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(255, 123, 0, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(255, 7, 58, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=color,
        height=300
    )
    
    return fig

def create_location_map(df_enhanced):
    """Create a cyberpunk world map of login locations"""
    location_coords = {
        'India': {'lat': 20.5937, 'lon': 78.9629},
        'USA': {'lat': 39.8283, 'lon': -98.5795},
        'UK': {'lat': 55.3781, 'lon': -3.4360},
        'Russia': {'lat': 61.5240, 'lon': 105.3188},
        'Canada': {'lat': 56.1304, 'lon': -106.3468}
    }
    
    map_data = []
    for location in df_enhanced['location'].unique():
        if location in location_coords:
            location_data = df_enhanced[df_enhanced['location'] == location]
            anomaly_count = (location_data['is_anomaly'] == -1).sum()
            total_count = len(location_data)
            avg_risk = location_data['risk_score'].mean()
            
            map_data.append({
                'location': location,
                'lat': location_coords[location]['lat'],
                'lon': location_coords[location]['lon'],
                'total_logins': total_count,
                'anomalies': anomaly_count,
                'avg_risk': avg_risk,
                'color': '#ff073a' if avg_risk > 70 else '#FF7B00' if avg_risk > 30 else '#39FF14'
            })
    
    map_df = pd.DataFrame(map_data)
    
    fig = px.scatter_mapbox(
        map_df,
        lat='lat',
        lon='lon',
        size='total_logins',
        color='avg_risk',
        hover_data=['location', 'total_logins', 'anomalies'],
        color_continuous_scale=[[0, '#39FF14'], [0.5, '#FF7B00'], [1, '#ff073a']],
        mapbox_style="carto-darkmatter",
        title="GLOBAL THREAT MAP - LOGIN LOCATIONS",
        height=600
    )
    
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=30, lon=0),
            zoom=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font=dict(family='Orbitron', size=20, color='#39FF14')
    )
    
    return fig

def explain_anomaly(row):
    """Generate explanation for why login was flagged as anomaly"""
    reasons = []
    
    if row['time_deviation'] > 120:  # More than 2 hours deviation
        reasons.append(f"⏰ Unusual login time - {row['time_deviation']:.0f} minutes from normal pattern")
    
    if row['location_distance'] > 1000:
        reasons.append(f"🌍 Suspicious location jump - {row['location_distance']:.0f} units traveled")
    
    # Check if login is at unusual hours (late night/early morning)
    login_hour = row['login_minutes'] // 60
    if login_hour < 6 or login_hour > 22:
        reasons.append(f"🌙 Login at unusual hour: {login_hour:02d}:00")
    
    if not reasons:
        reasons.append("🤖 AI detected unusual behavioral pattern")
    
    return " | ".join(reasons)

def run_cyberpunk_dashboard():
    load_cyberpunk_css()
    
    # Cyberpunk Main Header with Animation
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI BODYGUARD</h1>
        <h2>CYBERPUNK SECURITY PROTOCOL ACTIVATED</h2>
        <p style="font-family: 'Roboto Mono', monospace; color: #FF7B00;">
            "Your digital guardian never sleeps. Advanced neural networks protecting your identity 24/7."
        </p>
        <p style="font-size: 0.9rem; opacity: 0.8;">
            ⚡ Real-time Threat Detection | 🧠 Machine Learning | 🔐 Zero-Trust Security
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Personalized Welcome Feature
    st.markdown("### 👤 IDENTITY VERIFICATION")
    user_name = st.text_input("🔹 Enter your codename for personalized dashboard:", placeholder="Agent007")
    
    if user_name:
        st.markdown(f"""
        <div class="welcome-message">
            <h2>🚨 WELCOME, AGENT {user_name.upper()} 🚨</h2>
            <p style="font-size: 1.2rem;">Your AI Bodyguard is now ACTIVE and monitoring all login activities.</p>
            <p style="color: #FF7B00;">⚡ Neural networks initialized | 🛡️ Protection protocols engaged</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cyberpunk Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h2>🔧 CONTROL MATRIX</h2>
            <p style="color: #FF7B00;">Configure your security parameters</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ THREAT SENSITIVITY")
        contamination = st.slider("AI Detection Sensitivity", 0.05, 0.3, 0.15, 0.01, 
                                help="Higher values = More aggressive threat detection")
        
        st.markdown("### 🎯 ANALYSIS MODULES")
        feature_selection = st.multiselect(
            "Select Active Security Modules",
            ["Location Tracking", "Device Fingerprinting", "Temporal Analysis", "Geographic Intelligence"],
            default=["Location Tracking", "Device Fingerprinting", "Temporal Analysis", "Geographic Intelligence"]
        )
        
        st.markdown("### 🚨 ALERT SETTINGS")
        alert_threshold = st.slider("High Risk Alert Threshold", 50, 95, 75, 5)
        
        st.markdown("""
        <div style="margin-top: 2rem; padding: 1rem; background: rgba(57, 255, 20, 0.1); border: 1px solid #39FF14; border-radius: 8px;">
            <h4 style="color: #39FF14;">🛡️ SYSTEM STATUS</h4>
            <p style="color: #39FF14; margin: 0;">AI BODYGUARD: ONLINE ✅</p>
            <p style="color: #39FF14; margin: 0;">THREAT DETECTION: ACTIVE 🔍</p>
            <p style="color: #39FF14; margin: 0;">NEURAL NETWORKS: LEARNING 🧠</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load and process data
    file_path = 'login_data.csv'
    try:
        original_df = pd.read_csv(file_path)
        
    except FileNotFoundError:
        st.error("🚨 Critical Error: Data source not found. Contact your system administrator.")
        return

    if "original_df" not in st.session_state:
        st.session_state["original_df"] = original_df.copy()

    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = original_df.copy()

    df = st.session_state["raw_df"]
    df_enhanced, le_location, le_device = create_enhanced_features(df)
    
    # Feature mapping
    feature_map = {
        "Location Tracking": 'location_encoded',
        "Device Fingerprinting": 'device_encoded', 
        "Temporal Analysis": ['login_minutes', 'time_deviation'],
        "Geographic Intelligence": 'location_distance'
    }
    
    feature_cols = []
    for feature in feature_selection:
        if feature in feature_map:
            if isinstance(feature_map[feature], list):
                feature_cols.extend(feature_map[feature])
            else:
                feature_cols.append(feature_map[feature])
    
    if not feature_cols:
        st.error("⚠️ Error: Select at least one security module!")
        return
    
    # AI Processing
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_enhanced[feature_cols])
    
    model = IsolationForest(contamination=contamination, random_state=42)
    anomaly_predictions = model.fit_predict(features_scaled)
    anomaly_scores = model.decision_function(features_scaled)
    
    risk_scores = calculate_risk_score(anomaly_scores)
    
    df_enhanced['is_anomaly'] = anomaly_predictions
    df_enhanced['anomaly_score'] = anomaly_scores
    df_enhanced['risk_score'] = risk_scores.round(1)
    
    # Cyberpunk Dashboard Metrics
    st.markdown("## 🎛️ SECURITY COMMAND CENTER")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_logins = len(df_enhanced)
    anomalous_logins = len(df_enhanced[df_enhanced['is_anomaly'] == -1])
    high_risk_logins = len(df_enhanced[df_enhanced['risk_score'] > alert_threshold])
    unique_users = df_enhanced['user_id'].nunique()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card-safe">
            <h3 style="margin: 0; font-family: 'Orbitron';">🔐 TOTAL SCANS</h3>
            <h1 style="margin: 0.5rem 0; color: #39FF14;">{total_logins}</h1>
            <p style="margin: 0;">Login attempts analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_color = "danger" if anomalous_logins > 5 else "warning" if anomalous_logins > 2 else "safe"
        st.markdown(f"""
        <div class="metric-card-{risk_color}">
            <h3 style="margin: 0; font-family: 'Orbitron';">⚠️ THREATS DETECTED</h3>
            <h1 style="margin: 0.5rem 0;">{anomalous_logins}</h1>
            <p style="margin: 0;">{(anomalous_logins/total_logins*100):.1f}% anomaly rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        alert_color = "danger" if high_risk_logins > 3 else "warning" if high_risk_logins > 1 else "safe"
        st.markdown(f"""
        <div class="metric-card-{alert_color}">
            <h3 style="margin: 0; font-family: 'Orbitron';">🚨 HIGH ALERTS</h3>
            <h1 style="margin: 0.5rem 0;">{high_risk_logins}</h1>
            <p style="margin: 0;">Risk > {alert_threshold}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card-safe">
            <h3 style="margin: 0; font-family: 'Orbitron';">👥 AGENTS TRACKED</h3>
            <h1 style="margin: 0.5rem 0; color: #39FF14;">{unique_users}</h1>
            <p style="margin: 0;">Under AI surveillance</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced Cyberpunk Tabs
    st.markdown("## 🔍 SECURITY INTELLIGENCE CENTER")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🤖 AI Risk Analyzer", 
        "👤 Agent Profiles", 
        "🌍 Global Threat Map", 
        "⏰ Temporal Patterns", 
        "🚨 Live Threat Feed",
        "🧠 Explainable AI"
    ])
    
    with tab1:
        st.markdown("### 🎯 AI RISK ASSESSMENT")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            avg_risk = df_enhanced['risk_score'].mean()
            risk_gauge = create_risk_gauge(avg_risk)
            st.plotly_chart(risk_gauge, use_container_width=True)
        
        with col2:
            # Risk distribution with cyberpunk colors
            fig_risk = px.histogram(
                df_enhanced, 
                x='risk_score', 
                nbins=20,
                title="🔥 THREAT DISTRIBUTION ANALYSIS",
                labels={'risk_score': 'Risk Score (%)', 'count': 'Number of Logins'},
                color_discrete_sequence=['#39FF14']
            )
            fig_risk.add_vline(x=alert_threshold, line_dash="dash", line_color="#ff073a", 
                              annotation_text="🚨 Alert Threshold")
            fig_risk.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font=dict(family='Orbitron', color='#39FF14')
            )
            st.plotly_chart(fig_risk, use_container_width=True)
    
    with tab2:
        st.markdown("### 👤 AGENT SURVEILLANCE")
        
        # Agent selector with search
        selected_agent = st.selectbox(
            "🔍 Select Agent for Deep Analysis", 
            sorted(df_enhanced['username'].unique()),
            key="agent_selector"
        )
        
        agent_data = df_enhanced[df_enhanced['username'] == selected_agent].copy()
        agent_data = agent_data.sort_values('login_time').reset_index(drop=True)
        
        if not agent_data.empty:
            col1, col2, col3 = st.columns(3)
            
            avg_risk = agent_data['risk_score'].mean()
            max_risk = agent_data['risk_score'].max()
            anomaly_count = (agent_data['is_anomaly'] == -1).sum()
            
            # Agent profile cards
            risk_level = "high" if avg_risk > 70 else "medium" if avg_risk > 30 else "low"
            
            with col1:
                st.markdown(f"""
                <div class="risk-profile-{risk_level}">
                    <h4>📊 RISK PROFILE</h4>
                    <h2>{avg_risk:.1f}%</h2>
                    <p>Average Risk Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                max_level = "high" if max_risk > 70 else "medium" if max_risk > 30 else "low"
                st.markdown(f"""
                <div class="risk-profile-{max_level}">
                    <h4>⚡ PEAK THREAT</h4>
                    <h2>{max_risk:.1f}%</h2>
                    <p>Maximum Risk Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                anomaly_level = "high" if anomaly_count > 2 else "medium" if anomaly_count > 0 else "low"
                st.markdown(f"""
                <div class="risk-profile-{anomaly_level}">
                    <h4>🚨 ALERTS</h4>
                    <h2>{anomaly_count}</h2>
                    <p>Anomalous Logins</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Agent behavior analysis
            st.markdown("#### 🧠 BEHAVIORAL INTELLIGENCE")
            
            # Most common patterns
            most_common_location = agent_data['location'].mode().iloc[0] if not agent_data['location'].mode().empty else "Unknown"
            most_common_device = agent_data['device'].mode().iloc[0] if not agent_data['device'].mode().empty else "Unknown"
            avg_login_time = agent_data['login_minutes'].mean()
            avg_hour = int(avg_login_time // 60)
            avg_minute = int(avg_login_time % 60)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="activity-feed">
                    <h4 style="color: #39FF14;">🏠 HOME BASE</h4>
                    <p>{most_common_location}</p>
                    <h4 style="color: #39FF14;">💻 PREFERRED DEVICE</h4>
                    <p>{most_common_device}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="activity-feed">
                    <h4 style="color: #39FF14;">⏰ TYPICAL LOGIN TIME</h4>
                    <p>{avg_hour:02d}:{avg_minute:02d}</p>
                    <h4 style="color: #39FF14;">📈 ANOMALY RATE</h4>
                    <p>{(anomaly_count/len(agent_data)*100):.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Agent timeline
            fig_timeline = px.line(
                agent_data.reset_index(), 
                x='index', 
                y='risk_score',
                title=f"🕒 AGENT {selected_agent.upper()} - THREAT TIMELINE",
                labels={'index': 'Login Sequence', 'risk_score': 'Risk Score (%)'},
                color_discrete_sequence=['#39FF14']
            )
            fig_timeline.add_hline(y=alert_threshold, line_dash="dash", line_color="#ff073a", 
                                 annotation_text="🚨 Alert Threshold")
            fig_timeline.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font=dict(family='Orbitron', color='#39FF14')
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Agent login history
            st.markdown("#### 📋 LOGIN HISTORY")
            display_data = agent_data[['login_time', 'location', 'device', 'risk_score']].copy()
            display_data['THREAT_LEVEL'] = display_data['risk_score'].apply(
                lambda x: '🔴 CRITICAL' if x > 90 else '🟡 HIGH' if x > 70 else '🟠 MEDIUM' if x > 30 else '🟢 SAFE'
            )
            
            # Color code the dataframe
            def highlight_risk(row):
                if row['risk_score'] > 90:
                    return ['background-color: #2e0a0a; color: #ff073a'] * len(row)
                elif row['risk_score'] > 70:
                    return ['background-color: #2e1a0a; color: #FF7B00'] * len(row)
                elif row['risk_score'] > 30:
                    return ['background-color: #1a1a2e; color: #FF7B00'] * len(row)
                else:
                    return ['background-color: #0a2e0a; color: #39FF14'] * len(row)
            
            st.dataframe(
                display_data.style.apply(highlight_risk, axis=1),
                use_container_width=True,
                hide_index=True
            )
    
    with tab3:
        st.markdown("### 🌍 GLOBAL THREAT INTELLIGENCE")
        
        # World map
        location_map = create_location_map(df_enhanced)
        st.plotly_chart(location_map, use_container_width=True)
        
        # Location statistics
        st.markdown("#### 🏴‍☠️ THREAT ZONES")
        location_stats = df_enhanced.groupby('location').agg({
            'risk_score': ['mean', 'max', 'count'],
            'is_anomaly': lambda x: (x == -1).sum()
        }).round(2)
        
        location_stats.columns = ['Avg Risk', 'Max Risk', 'Total Logins', 'Threats']
        location_stats = location_stats.reset_index()
        location_stats['Threat Level'] = location_stats['Avg Risk'].apply(
            lambda x: '🔴 HIGH' if x > 70 else '🟡 MEDIUM' if x > 30 else '🟢 SAFE'
        )
        
        # Sort by average risk
        location_stats = location_stats.sort_values('Avg Risk', ascending=False)
        
        st.dataframe(
            location_stats,
            use_container_width=True,
            hide_index=True
        )
    
    with tab4:
        st.markdown("### ⏰ TEMPORAL THREAT ANALYSIS")
        
        # Time pattern analysis
        df_enhanced['login_hour'] = df_enhanced['login_minutes'] // 60
        time_stats = df_enhanced.groupby('login_hour').agg({
            'risk_score': 'mean',
            'is_anomaly': lambda x: (x == -1).sum(),
            'user_id': 'count'
        }).round(2)
        time_stats.columns = ['Avg Risk Score', 'Threats Detected', 'Total Logins']
        time_stats = time_stats.reset_index()
        
        # 24-hour threat pattern
        fig_time = px.line(
            time_stats, 
            x='login_hour', 
            y='Avg Risk Score',
            title="🕐 24-HOUR THREAT PATTERN ANALYSIS",
            labels={'login_hour': 'Hour of Day', 'Avg Risk Score': 'Average Risk Score (%)'},
            color_discrete_sequence=['#39FF14']
        )
        fig_time.add_hline(y=alert_threshold, line_dash="dash", line_color="#ff073a")
        fig_time.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font=dict(family='Orbitron', color='#39FF14'),
            xaxis=dict(tickmode='linear', dtick=2)
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Heatmap of user patterns
        st.markdown("#### 🔥 AGENT ACTIVITY HEATMAP")
        heatmap_data = df_enhanced.pivot_table(
            values='risk_score', 
            index='username', 
            columns='login_hour', 
            aggfunc='mean'
        ).fillna(0)
        
        fig_heatmap = px.imshow(
            heatmap_data,
            title="🔥 HOURLY RISK PATTERNS BY AGENT",
            labels={'x': 'Hour of Day', 'y': 'Agent ID', 'color': 'Risk Score (%)'},
            color_continuous_scale=[[0, '#39FF14'], [0.5, '#FF7B00'], [1, '#ff073a']],
            aspect='auto'
        )
        fig_heatmap.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font=dict(family='Orbitron', color='#39FF14')
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab5:
     st.markdown("### 🚨 LIVE THREAT MONITORING")
    st.markdown("#### ⚡ SECURITY EVENT STREAM")
    import io

    csv = df_enhanced[df_enhanced['is_anomaly'] == -1].to_csv(index=False)
    st.download_button(
    label="📥 Download Anomaly Report",
    data=csv,
    file_name="anomaly_report.csv",
    mime="text/csv"
)


    col_sim, col_clear = st.columns([1, 1])

    with col_sim:
        if st.button("➕ Simulate New Login", key="simulate_login"):
            sample_user = random.choice(st.session_state["original_df"]["username"].unique())
            if "user_id" in st.session_state["original_df"].columns:
                sample_user_id = st.session_state["original_df"].loc[
                    st.session_state["original_df"]["username"] == sample_user, "user_id"
                ].iloc[0]
            else:
                sample_user_id = sample_user

            new_entry = {
                "user_id": sample_user_id,
                "username": sample_user,
                "login_time": f"{random.randint(0,23):02d}:{random.randint(0,59):02d}",
                "location": random.choice(st.session_state["original_df"]["location"].unique()),
                "device": random.choice(st.session_state["original_df"]["device"].unique())
            }

            st.session_state["raw_df"] = pd.concat(
                [st.session_state["raw_df"], pd.DataFrame([new_entry])],
                ignore_index=True
            )

            st.success("Simulated a new login — recomputing AI...")
            st.rerun()

    with col_clear:
        if st.button("🧹 Reset Simulated Logins", key="clear_sim"):
            st.session_state["raw_df"] = st.session_state["original_df"].copy()
            st.success("Simulated logins cleared.")
            st.experimental_rerun()

    st.markdown("#### ⚡ Recent Security Events")

    recent_logins = df_enhanced.sort_values("risk_score", ascending=False).head(10)

    if recent_logins.empty:
        st.info("No login events available.")
    else:
        for _, row in recent_logins.iterrows():
            username = row.get("username", "Unknown")
            login_time = row.get("login_time", "00:00")
            location = row.get("location", "Unknown")
            device = row.get("device", "Unknown")
            risk = row.get("risk_score", 0.0)
            status = "⚠️ ANOMALY DETECTED" if row.get("is_anomaly", 1) == -1 else "✅ VERIFIED LOGIN"

            if risk > 75:
                alert_class = "anomaly-alert"
                alert_label = "🔴 CRITICAL THREAT"
            elif risk > 30:
                alert_class = "anomaly-alert"
                alert_label = "🟡 ELEVATED RISK"
            else:
                alert_class = "safe-alert"
                alert_label = "🟢 NORMAL ACTIVITY"

            st.markdown(f"""
            <div class="{alert_class}">
                <h4>{alert_label} — Risk: {risk:.1f}%</h4>
                <p><strong>Agent:</strong> {username} | <strong>Time:</strong> {login_time} |
                   <strong>Location:</strong> {location} | <strong>Device:</strong> {device}</p>
                <p><strong>Status:</strong> {status}</p>
            </div>
            """, unsafe_allow_html=True)

    if st.button("🔄 Refresh Feed (no simulation)", key="manual_refresh"):
        st.experimental_rerun()

    
    with tab6:
        st.markdown("### 🧠 EXPLAINABLE AI INTELLIGENCE")
        
        anomalies = df_enhanced[df_enhanced['is_anomaly'] == -1].copy()
        
        if not anomalies.empty:
            st.markdown("#### 🔍 WHY THESE LOGINS ARE SUSPICIOUS")
            
            # Add explanations to anomalies
            anomalies['explanation'] = anomalies.apply(explain_anomaly, axis=1)
            anomalies = anomalies.sort_values('risk_score', ascending=False)
            
            for idx, row in anomalies.iterrows():
                risk_emoji = "🔴" if row['risk_score'] > 90 else "🟡" if row['risk_score'] > 70 else "🟠"
                
                st.markdown(f"""
                <div class="anomaly-alert">
                    <h4>{risk_emoji} THREAT ANALYSIS - Agent: {row['username']}</h4>
                    <p><strong>Risk Score:</strong> {row['risk_score']:.1f}% | 
                    <strong>Time:</strong> {row['login_time']} | 
                    <strong>Location:</strong> {row['location']} | 
                    <strong>Device:</strong> {row['device']}</p>
                    <h5 style="color: #FF7B00;">🤖 AI EXPLANATION:</h5>
                    <p>{row['explanation']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # AI Model Insights
            st.markdown("#### 🔬 MODEL INTELLIGENCE METRICS")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                precision = len(anomalies[anomalies['risk_score'] > 70]) / len(anomalies) * 100 if len(anomalies) > 0 else 0
                st.metric("🎯 Precision", f"{precision:.1f}%", help="Percentage of high-risk anomalies")
            
            with col2:
                model_confidence = np.mean(np.abs(anomaly_scores)) * 100
                st.metric("🧠 Model Confidence", f"{model_confidence:.1f}%", help="AI certainty in predictions")
            
            with col3:
                feature_importance = len(feature_cols)
                st.metric("📊 Features Used", feature_importance, help="Number of security modules active")
        
        else:
            st.markdown("""
            <div class="safe-alert">
                <h3>✅ SYSTEM SECURE</h3>
                <p>🤖 AI Analysis: No suspicious activities detected. All login patterns appear normal.</p>
                <p>🛡️ Your security perimeter is intact. Continue monitoring...</p>
            </div>
            """, unsafe_allow_html=True)
        if high_risk_logins > 0:
           st.error(f"🚨 ALERT: {high_risk_logins} high-risk logins detected!")
        else:
           st.success("✅ System Secure - No threats detected")

    # Cyberpunk Footer
    st.markdown("""
    <div class="footer">
        <h3>🤖 AI BODYGUARD - NEURAL SECURITY MATRIX</h3>
        <p>⚡ Quantum-encrypted • 🧠 Self-learning algorithms • 🔐 Zero-trust architecture</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">
            "In cyberspace, threats evolve at light speed. Your AI guardian evolves faster."
        </p>
        <p style="font-family: 'Roboto Mono'; color: #FF7B00;">
            STATUS: ONLINE | UPTIME: 99.99% | THREATS BLOCKED: ∞
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    run_cyberpunk_dashboard()