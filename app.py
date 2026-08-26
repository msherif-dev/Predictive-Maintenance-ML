import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AEROGUARD - Aircraft Engine Health Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium glassmorphism design
st.markdown("""
<style>
    /* Main container and background */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0d1229 100%);
        color: #e0e6ed;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, rgba(0, 255, 255, 0.1) 0%, rgba(0, 100, 255, 0.05) 100%);
        border: 2px solid rgba(0, 255, 255, 0.3);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 0 40px rgba(0, 255, 255, 0.2);
    }
    
    /* KPI cards */
    .kpi-card {
        background: linear-gradient(145deg, rgba(0, 255, 255, 0.08) 0%, rgba(0, 100, 255, 0.03) 100%);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 255, 255, 0.2);
    }
    
    /* Status badges */
    .status-healthy {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        color: #000;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ff9500 0%, #ff7700 100%);
        color: #000;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 20px rgba(255, 149, 0, 0.4);
    }
    
    .status-critical {
        background: linear-gradient(135deg, #ff3366 0%, #ff0044 100%);
        color: #fff;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 20px rgba(255, 51, 102, 0.4);
    }
    
    /* AI indicator */
    .ai-indicator {
        background: linear-gradient(135deg, #00ffff 0%, #0088ff 100%);
        color: #000;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin-top: 10px;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
    }
    
    /* Section headers */
    .section-header {
        color: #00ffff;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 2px solid rgba(0, 255, 255, 0.3);
        padding-bottom: 10px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    /* Metric values */
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #00ffff;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }
    
    .metric-label {
        font-size: 14px;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 10px;
    }
    
    /* Sensor table */
    .sensor-table {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 12px;
        overflow: hidden;
    }
    
    .sensor-table table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .sensor-table th {
        background: rgba(0, 255, 255, 0.1);
        color: #00ffff;
        padding: 12px;
        text-align: left;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .sensor-table td {
        padding: 12px;
        border-bottom: 1px solid rgba(0, 255, 255, 0.1);
        color: #e0e6ed;
    }
    
    .sensor-table tr:hover {
        background: rgba(0, 255, 255, 0.05);
    }
    
    /* Info cards */
    .info-card {
        background: rgba(0, 100, 255, 0.05);
        border-left: 4px solid #00ffff;
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 0 12px 12px 0;
    }
    
    /* Streamlit native elements override */
    div.stSelectbox > div > div > select {
        background: rgba(0, 0, 0, 0.3);
        color: #e0e6ed;
        border: 1px solid rgba(0, 255, 255, 0.3);
        border-radius: 8px;
    }
    
    div.stButton > button {
        background: linear-gradient(135deg, #00ffff 0%, #0088ff 100%);
        color: #000;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0, 255, 255, 0.3);
    }
    
    div.stButton > button:hover {
        box-shadow: 0 6px 20px rgba(0, 255, 255, 0.5);
        transform: translateY(-2px);
    }
    
    /* Error message styling */
    .error-message {
        background: rgba(255, 51, 102, 0.1);
        border: 2px solid #ff3366;
        border-radius: 12px;
        padding: 20px;
        color: #ff3366;
    }
    
    /* Success message styling */
    .success-message {
        background: rgba(0, 255, 136, 0.1);
        border: 2px solid #00ff88;
        border-radius: 12px;
        padding: 20px;
        color: #00ff88;
    }
    
    /* Chart containers */
    .chart-container {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(0, 255, 255, 0.1);
    }
    
    /* Timeline section */
    .timeline-card {
        background: linear-gradient(145deg, rgba(255, 149, 0, 0.05) 0%, rgba(255, 51, 102, 0.02) 100%);
        border: 1px solid rgba(255, 149, 0, 0.2);
        border-radius: 16px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Load model and data
@st.cache_resource
def load_model_and_data():
    """Load the trained model and test data"""
    # Load the model
    model_path = "bagging_rul_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None, None, None
    
    artifact = joblib.load(model_path)
    model_data = artifact
    
    # Extract model components
    model = model_data['model']
    feature_columns = model_data['feature_columns']
    constant_columns = model_data['constant_columns']
    columns = model_data['columns']
    rolling_window = model_data['rolling_window']
    
    # Load test data
    test_path = "test_FD001.txt"
    if not os.path.exists(test_path):
        st.error(f"❌ Test file not found: {test_path}")
        return None, None, None
    
    # Read test data (space-separated)
    raw_test_df = pd.read_csv(test_path, sep=' ', header=None, engine='python')
    
    # Remove empty columns
    raw_test_df = raw_test_df.dropna(axis=1, how='all')
    
    # Set column names from artifact
    raw_test_df.columns = columns
    
    # Sort by unit_id and cycle
    raw_test_df = raw_test_df.sort_values(['unit_id', 'cycle'])
    
    # Keep both unit_id and cycle as normal columns
    assert "cycle" in raw_test_df.columns, "cycle column not found in raw_test_df"
    assert "unit_id" in raw_test_df.columns, "unit_id column not found in raw_test_df"
    
    return model, raw_test_df, {
        'feature_columns': feature_columns,
        'constant_columns': constant_columns,
        'columns': columns,
        'rolling_window': rolling_window
    }


def prepare_features(engine_data, model_info):
    """Prepare features for prediction"""
    feature_columns = model_info['feature_columns']
    constant_columns = model_info['constant_columns']
    
    # Get the raw sensor columns (exclude unit_id and cycle)
    raw_sensor_cols = [col for col in engine_data.columns if col not in ['unit_id', 'cycle']]
    
    # Start with raw features - drop only unit_id and constant columns
    features = engine_data.drop(columns=['unit_id'] + constant_columns, errors='ignore').copy()
    
    # Add cycle_squared feature
    features['cycle_squared'] = engine_data['cycle'] ** 2
    
    # Create rolling features with window=5, grouped by unit_id
    rolling_window = 5
    for col in raw_sensor_cols:
        if col not in constant_columns:
            features[f'{col}_rolling_mean'] = engine_data.groupby('unit_id')[col].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1).mean()
            )
            features[f'{col}_rolling_std'] = engine_data.groupby('unit_id')[col].transform(
                lambda x: x.rolling(window=rolling_window, min_periods=1).std().fillna(0)
            )
    
    # Fill any missing values
    features = features.fillna(features.mean())
    
    # Reorder features using artifact feature_columns
    features = features[feature_columns]
    
    # Assertion for debugging
    assert features.shape[1] == len(feature_columns), f"Expected {len(feature_columns)} features, got {features.shape[1]}"
    
    return features


def predict_rul(model, features):
    """Make RUL prediction"""
    try:
        # Get the last row (most recent cycle)
        latest_features = features.iloc[[-1]]
        
        # Make prediction
        prediction = model.predict(latest_features)[0]
        
        return max(0, int(prediction))
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None


def get_condition_status(rul, max_rul=125):
    """Determine engine condition based on RUL"""
    if rul > max_rul * 0.5:
        return "Healthy", "healthy"
    elif rul > max_rul * 0.2:
        return "Warning", "warning"
    else:
        return "Critical", "critical"


def get_risk_level(rul, max_rul=125):
    """Determine risk level based on RUL"""
    if rul > max_rul * 0.6:
        return "Low", "#00ff88"
    elif rul > max_rul * 0.3:
        return "Medium", "#ff9500"
    else:
        return "High", "#ff3366"


def create_rul_gauge(rul, max_rul=125):
    """Create an interactive RUL gauge using Plotly"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = rul,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Remaining Useful Life (Cycles)", 'font': {'size': 20, 'color': '#00ffff'}},
        delta = {'reference': max_rul, 'increasing': {'color': "#00ff88"}, 'relative': False, 'position': "bottom"},
        gauge = {
            'axis': {'range': [None, max_rul], 'tickwidth': 1, 'tickcolor': "#00ffff"},
            'bar': {'color': "#00ffff"},
            'bgcolor': "rgba(0, 0, 0, 0.5)",
            'borderwidth': 2,
            'bordercolor': "#00ffff",
            'steps': [
                {'range': [0, max_rul * 0.2], 'color': 'rgba(255, 51, 102, 0.3)'},
                {'range': [max_rul * 0.2, max_rul * 0.5], 'color': 'rgba(255, 149, 0, 0.3)'},
                {'range': [max_rul * 0.5, max_rul], 'color': 'rgba(0, 255, 136, 0.3)'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_rul * 0.2
            }
        }
    ))
    
    # Add annotation to clarify reference is training RUL cap
    fig.add_annotation(
        text=f"Reference: {max_rul} (Training RUL Cap)",
        xref="paper", yref="paper",
        x=0.5, y=0.85,
        showarrow=False,
        font=dict(size=12, color="#8899aa"),
        xanchor="center"
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font={'color': '#e0e6ed'},
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_sensor_chart(engine_data, sensor_columns, selected_sensor):
    """Create interactive sensor analytics chart"""
    fig = go.Figure()
    
    # Add sensor data line
    fig.add_trace(go.Scatter(
        x=engine_data['cycle'],
        y=engine_data[selected_sensor],
        mode='lines+markers',
        name=f'Observed {selected_sensor} value',
        line=dict(color='#00ffff', width=2),
        marker=dict(size=4, color='#00ffff'),
        hovertemplate='<b>Cycle</b>: %{x}<br>Observed value: %{y:.2f}<extra></extra>'
    ))
    
    # Add rolling trend
    rolling_mean = engine_data[selected_sensor].rolling(window=10, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=engine_data['cycle'],
        y=rolling_mean,
        mode='lines',
        name='Rolling mean (10 cycles)',
        line=dict(color='#ff9500', width=2, dash='dash'),
        hovertemplate='Rolling mean: %{y:.2f}<extra></extra>'
    ))
    
    # Add vertical line for latest observed cycle
    latest_cycle = engine_data['cycle'].max()
    fig.add_vline(
        x=latest_cycle,
        line_dash="dash",
        line_color="#ff3366",
        line_width=2,
        annotation_text="Latest observed cycle",
        annotation_position="top",
        annotation_font=dict(size=11, color="#ff3366")
    )
    
    fig.update_layout(
        title=f'Sensor Analytics: {selected_sensor}',
        xaxis_title='Cycle',
        yaxis_title='Sensor Value',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0.3)',
        font={'color': '#e0e6ed'},
        hovermode='x',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis=dict(
            gridcolor='rgba(0, 255, 255, 0.1)',
            zerolinecolor='rgba(0, 255, 255, 0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(0, 255, 255, 0.1)',
            zerolinecolor='rgba(0, 255, 255, 0.1)'
        )
    )
    
    return fig


def create_degradation_chart(engine_data, rul):
    """Create engine degradation timeline chart"""
    fig = go.Figure()
    
    # Calculate degradation metric (inverse of normalized sensor values)
    sensor_cols = [col for col in engine_data.columns if col not in ['unit_id', 'cycle']]
    if len(sensor_cols) > 0:
        # Normalize and aggregate
        normalized = (engine_data[sensor_cols] - engine_data[sensor_cols].min()) / (
            engine_data[sensor_cols].max() - engine_data[sensor_cols].min()
        )
        degradation = 1 - normalized.mean(axis=1)
    else:
        degradation = engine_data['cycle'] / engine_data['cycle'].max()
    
    fig.add_trace(go.Scatter(
        x=engine_data['cycle'],
        y=degradation,
        mode='lines',
        name='Degradation Level',
        line=dict(color='#ff3366', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 51, 102, 0.2)'
    ))
    
    # Add predicted end of life
    current_cycle = engine_data['cycle'].max()
    predicted_eol = current_cycle + rul
    
    fig.add_vline(
        x=predicted_eol,
        line_dash="dash",
        line_color="#00ffff",
        annotation_text="Predicted EOL",
        annotation_position="top"
    )
    
    fig.update_layout(
        title='Engine Degradation Overview',
        xaxis_title='Cycle',
        yaxis_title='Degradation Level',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0.3)',
        font={'color': '#e0e6ed'},
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis=dict(
            gridcolor='rgba(0, 255, 255, 0.1)',
            zerolinecolor='rgba(0, 255, 255, 0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(0, 255, 255, 0.1)',
            zerolinecolor='rgba(0, 255, 255, 0.1)',
            range=[0, 1]
        )
    )
    
    return fig


# Main application
def main():
    # Load model and data
    model, test_data, model_info = load_model_and_data()
    
    if model is None or test_data is None:
        st.markdown("""
        <div class="error-message">
            <h2>⚠️ System Initialization Failed</h2>
            <p>Please ensure the following files exist in the project directory:</p>
            <ul>
                <li>bagging_rul_model.pkl</li>
                <li>test_FD001.txt</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Get unique engine IDs
    engine_ids = sorted(test_data['unit_id'].unique())
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding: 20px; text-align: center;">
            <h1 style="color: #00ffff; font-size: 28px; margin-bottom: 5px;">AEROGUARD</h1>
            <p style="color: #8899aa; font-size: 12px; margin-top: 0;">Aircraft Engine Health Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System information
        st.markdown("### 📊 System Information")
        st.markdown(f"""
        <div class="info-card">
            <strong>Dataset:</strong> NASA C-MAPSS FD001<br>
            <strong>Model:</strong> Bagging Regressor<br>
            <strong>Status:</strong> <span style="color: #00ff88;">● Online</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Technical specs
        st.markdown("### ⚙️ Technical Specifications")
        st.markdown(f"""
        <div class="info-card">
            <strong>Features:</strong> {len(model_info['feature_columns'])}<br>
            <strong>Rolling Window:</strong> {model_info['rolling_window']}<br>
            <strong>Preprocessing:</strong> Complete
        </div>
        """, unsafe_allow_html=True)
    
    # Engine selector (at the top, before data processing)
    selected_engine = st.selectbox(
        "🔧 Select Engine ID",
        engine_ids,
        index=0,
        key="engine_selector",
        help="Choose an engine to analyze"
    )
    
    # Get engine data
    engine_data = test_data[test_data['unit_id'] == selected_engine].copy()
    
    # Prepare features and make prediction
    features = prepare_features(engine_data, model_info)
    if features is None:
        return
    
    rul_prediction = predict_rul(model, features)
    if rul_prediction is None:
        return
    
    current_cycle = engine_data['cycle'].max()
    condition_status, condition_class = get_condition_status(rul_prediction)
    risk_level, risk_color = get_risk_level(rul_prediction)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 style="color: #00ffff; font-size: 48px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 3px;">
            AEROGUARD
        </h1>
        <h2 style="color: #e0e6ed; font-size: 24px; margin-bottom: 15px; font-weight: 300;">
            Aircraft Engine Health Intelligence
        </h2>
        <p style="color: #8899aa; font-size: 16px; margin-bottom: 20px; max-width: 800px;">
            Advanced AI-powered predictive maintenance system for aircraft engines. 
            Monitor engine health in real-time, predict remaining useful life, 
            and optimize maintenance schedules with cutting-edge machine learning.
        </p>
        <span class="ai-indicator">🤖 AI-POWERED PREDICTIVE ANALYTICS</span>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards Row - 5 cards as shown in screenshot
    st.markdown('<div class="section-header">📊 ENGINE STATUS - KEY METRICS</div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="metric-value">#{selected_engine}</div>
            <div class="metric-label">ENGINE ID<br><small>selected engine</small></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="metric-value">{current_cycle}</div>
            <div class="metric-label">CURRENT CYCLE<br><small>latest observed cycle</small></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="metric-value">{rul_prediction}</div>
            <div class="metric-label">PREDICTED RUL<br><small>cycles remaining</small></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="status-{condition_class}">{condition_status}</div>
            <div class="metric-label">ENGINE CONDITION<br><small>health classification</small></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div style="color: {risk_color}; font-size: 36px; font-weight: bold; text-shadow: 0 0 10px {risk_color};">
                {risk_level.split()[0]}
            </div>
            <div class="metric-label">RISK LEVEL<br><small>maintenance priority</small></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metric Explanations Section
    st.markdown('<div class="section-header">📋 METRIC EXPLANATIONS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #00ffff; margin-bottom: 15px;">Understanding Your Engine Metrics</h4>
        <ul style="color: #e0e6ed; line-height: 1.8;">
            <li><strong>Engine ID:</strong> Unique identifier for the selected aircraft engine in the fleet</li>
            <li><strong>Current Cycle:</strong> The latest observed operating cycle from sensor data</li>
            <li><strong>Predicted RUL:</strong> Remaining Useful Life - estimated cycles before failure</li>
            <li><strong>Engine Condition:</strong> Health classification based on RUL thresholds (Healthy/Warning/Critical)</li>
            <li><strong>Risk Level:</strong> Maintenance priority assessment (Low/Medium/High Risk)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # RUL Gauge Section
    st.markdown('<div class="section-header">📈 RUL PREDICTION GAUGE</div>', unsafe_allow_html=True)
    fig_gauge = create_rul_gauge(rul_prediction)
    st.plotly_chart(fig_gauge, use_container_width=True, key="rul_gauge")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Condition Status Section
    st.markdown('<div class="section-header">⚡ CONDITION STATUS</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <h3 style="color: #00ffff; margin-bottom: 10px;">Current Status</h3>
        <span class="status-{condition_class}">{condition_status}</span>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <h3 style="color: #00ffff; margin-bottom: 10px;">Risk Assessment</h3>
        <div style="color: {risk_color}; font-size: 24px; font-weight: bold;">{risk_level}</div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sensor Analytics Section
    st.markdown('<div class="section-header">� SENSOR ANALYTICS</div>', unsafe_allow_html=True)
    
    # Get sensor columns (exclude unit_id and cycle)
    sensor_columns = [col for col in engine_data.columns if col not in ['unit_id', 'cycle']]
    
    if len(sensor_columns) > 0:
        selected_sensor = st.selectbox(
            "SELECT SENSOR",
            sensor_columns,
            index=0,
            key="sensor_selector",
            help="Choose a sensor to analyze its historical data"
        )
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_sensor = create_sensor_chart(engine_data, sensor_columns, selected_sensor)
        st.plotly_chart(fig_sensor, use_container_width=True, key="sensor_chart")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Latest Sensor Readings Section
    st.markdown('<div class="section-header">🔬 LATEST SENSOR READINGS</div>', unsafe_allow_html=True)
    
    # Get latest cycle data
    latest_data = engine_data.iloc[-1]
    sensor_cols = [col for col in engine_data.columns if col not in ['unit_id', 'cycle']]
    latest_sensors = latest_data[sensor_cols]
    
    # Create sensor table
    sensor_df = pd.DataFrame({
        'Sensor': sensor_cols,
        'Value': latest_sensors.values
    })
    
    st.markdown('<div class="sensor-table">', unsafe_allow_html=True)
    st.dataframe(
        sensor_df,
        hide_index=True,
        use_container_width=True,
        height=300
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # AI Prediction Summary Section
    st.markdown('<div class="section-header">🤖 AI PREDICTION SUMMARY</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    st.markdown(f"### Engine Analysis Report - Engine #{selected_engine}")
    
    st.metric("Current Operating Cycle", f"{current_cycle}", help="Latest observed cycle in test trajectory")
    
    st.metric("Predicted Remaining Useful Life", f"{rul_prediction} cycles", help="Predicted remaining operating cycles")
    
    st.markdown(f"**Health Condition:**")
    st.markdown(f'<span class="status-{condition_class}">{condition_status}</span>', unsafe_allow_html=True)
    
    st.markdown("**Risk Interpretation:**")
    st.info(get_risk_interpretation(rul_prediction, condition_status))
    
    st.markdown("**Recommendation:**")
    st.warning(get_recommendation(condition_status))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Engine Degradation Timeline Section
    st.markdown('<div class="section-header">📉 ENGINE DEGRADATION TIMELINE</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8899aa; font-size: 14px; margin-bottom: 15px;">Sensor-based degradation trend over operating cycles</p>', unsafe_allow_html=True)
    st.markdown('<div class="chart-container timeline-card">', unsafe_allow_html=True)
    fig_degradation = create_degradation_chart(engine_data, rul_prediction)
    st.plotly_chart(fig_degradation, use_container_width=True, key="degradation_chart")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Technical Information Section
    st.markdown('<div class="section-header">⚙️ TECHNICAL INFORMATION</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <strong>Model Type:</strong><br>
            Bagging Regressor
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <strong>Number of Features:</strong><br>
            {len(model_info['feature_columns'])}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="info-card">
            <strong>Dataset:</strong><br>
            NASA C-MAPSS FD001
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="info-card">
            <strong>Preprocessing:</strong><br>
            <span style="color: #00ff88;">✓ Complete</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 30px; color: #8899aa; border-top: 1px solid rgba(0, 255, 255, 0.1); margin-top: 30px;">
        <p>AEROGUARD - Aircraft Engine Health Intelligence</p>
        <p style="font-size: 12px;">Powered by Advanced Machine Learning | NASA C-MAPSS Dataset</p>
    </div>
    """, unsafe_allow_html=True)


def get_risk_interpretation(rul, condition):
    """Generate risk interpretation text"""
    if condition == "Healthy":
        return f"The engine is operating within normal parameters. With {rul} predicted remaining cycles, maintenance can be scheduled according to standard intervals."
    elif condition == "Warning":
        return f"The engine shows early signs of degradation. With {rul} predicted remaining cycles, increased monitoring is recommended. Plan maintenance within the next 30-50 cycles."
    else:
        return f"The engine is in critical condition. With only {rul} predicted remaining cycles, immediate maintenance attention is required. Schedule inspection as soon as possible."


def get_recommendation(condition):
    """Generate maintenance recommendation"""
    if condition == "Healthy":
        return "Continue normal operation. Schedule routine maintenance at next planned interval."
    elif condition == "Warning":
        return "Increase monitoring frequency. Prepare maintenance team for upcoming service window."
    else:
        return "URGENT: Immediate maintenance required. Do not delay service beyond next available opportunity."


if __name__ == "__main__":
    main()
