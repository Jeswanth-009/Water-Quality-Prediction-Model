import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
import io
import base64
from datetime import datetime

warnings.filterwarnings('ignore')
from datetime import datetime

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Water Quality Prediction Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E86AB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">🌊 Water Quality Prediction Dashboard</div>', unsafe_allow_html=True)

# Define target columns (based on your notebook)
TARGET_COLS = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# Sidebar
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Home", "📊 Data Analysis", "🤖 Model Prediction", "📈 Visualizations", "📋 Model Info"]
)

# Helper functions
def preprocess_data(df):
    """Preprocess data similar to the notebook"""
    # Make a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Handle non-numeric columns
    non_numeric_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    def is_date_column(col_name, sample_values):
        date_patterns = ['/', '.', '-']
        sample_str = str(sample_values.dropna().iloc[0]) if len(sample_values.dropna()) > 0 else ""
        return any(pattern in sample_str for pattern in date_patterns) and len(sample_str) > 5
    
    # Process non-numeric columns
    for col in non_numeric_cols:
        if col in TARGET_COLS:
            continue
        
        sample_values = df_processed[col].head(10)
        
        if is_date_column(col, sample_values):
            try:
                df_processed[col + '_datetime'] = pd.to_datetime(df_processed[col], errors='coerce')
                df_processed[col + '_year'] = df_processed[col + '_datetime'].dt.year
                df_processed[col + '_month'] = df_processed[col + '_datetime'].dt.month
                df_processed[col + '_day'] = df_processed[col + '_datetime'].dt.day
                df_processed[col + '_dayofweek'] = df_processed[col + '_datetime'].dt.dayofweek
                df_processed = df_processed.drop(columns=[col, col + '_datetime'])
            except:
                df_processed = df_processed.drop(columns=[col])
        else:
            try:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            except:
                df_processed = df_processed.drop(columns=[col])
    
    # Convert target columns to numeric
    for col in TARGET_COLS:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Handle categorical encoding for 'id' column if it exists
    if 'id' in df_processed.columns:
        # Get unique values for id to create dummy variables
        id_dummies = pd.get_dummies(df_processed['id'], prefix='id')
        df_processed = pd.concat([df_processed, id_dummies], axis=1)
        df_processed = df_processed.drop(columns=['id'])
    
    # Replace infinite values with NaN
    df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
    
    return df_processed

def load_model():
    """Load the trained model and feature columns"""
    try:
        model = joblib.load("best_rf_multioutput.pkl")
        # Try to load the feature columns if available
        try:
            feature_columns = joblib.load("model_columns.pkl")
            st.session_state.feature_names = feature_columns
        except:
            # If model_columns.pkl doesn't exist, we'll need to infer from the original data
            st.session_state.feature_names = None
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'best_rf_multioutput.pkl' not found. Please ensure the model is trained and saved.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def get_expected_features():
    """Get the expected feature names based on the notebook preprocessing"""
    # Based on your notebook, these are the likely features after preprocessing
    # You may need to adjust this based on your actual data
    expected_features = [
        'id',  # Original id column
        'date_year',  # From date column processing
        'date_month',  # From date column processing  
        'date_day',  # From date column processing
        'date_dayofweek'  # From date column processing
    ]
    return expected_features

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample features based on the expected model features
    data = {
        'id': range(1, n_samples + 1),
        'date_year': np.random.choice([2020, 2021, 2022, 2023], n_samples),
        'date_month': np.random.choice(range(1, 13), n_samples),
        'date_day': np.random.choice(range(1, 29), n_samples),
        'date_dayofweek': np.random.choice(range(0, 7), n_samples),
    }
    
    # Add target variables
    for target in TARGET_COLS:
        if target == 'O2':
            data[target] = np.random.normal(8, 2, n_samples)
        elif target == 'NH4':
            data[target] = np.random.exponential(2, n_samples)
        elif target == 'BSK5':
            data[target] = np.random.exponential(5, n_samples)
        elif target == 'Suspended':
            data[target] = np.random.exponential(10, n_samples)
        elif target in ['NO3', 'NO2', 'SO4', 'PO4', 'CL']:
            data[target] = np.random.exponential(3, n_samples)
        else:
            data[target] = np.random.exponential(5, n_samples)
    
    return pd.DataFrame(data)

# Page content
if page == "🏠 Home":
    st.markdown("""
    ## Welcome to the Water Quality Prediction Dashboard! 🌊
    
    This application is built based on your water quality analysis notebook and provides the following features:
    
    ### 🔧 Features:
    - **Data Upload & Analysis**: Upload your water quality data (CSV format with ';' separator)
    - **Model Predictions**: Use the trained Random Forest model to predict water quality parameters
    - **Interactive Visualizations**: Explore data patterns and relationships
    - **Model Performance**: View model metrics and feature importance
    
    ### 📊 Target Parameters:
    The model predicts the following water quality parameters:
    """)
    
    # Display target parameters in a nice format
    cols = st.columns(3)
    targets_info = {
        'NH4': 'Ammonium', 'BSK5': 'BOD5', 'Suspended': 'Suspended Solids',
        'O2': 'Dissolved Oxygen', 'NO3': 'Nitrate', 'NO2': 'Nitrite',
        'SO4': 'Sulfate', 'PO4': 'Phosphate', 'CL': 'Chloride'
    }
    
    for i, (code, name) in enumerate(targets_info.items()):
        with cols[i % 3]:
            st.info(f"**{code}**: {name}")
    
    st.markdown("""
    ### 🚀 Getting Started:
    1. Navigate to **📊 Data Analysis** to upload and explore your data
    2. Use **🤖 Model Prediction** to make predictions with the trained model
    3. Explore **📈 Visualizations** for detailed data insights
    4. Check **📋 Model Info** for model performance details
    
    ### 📁 Data Format:
    - CSV file with semicolon (;) separator
    - Should contain water quality measurement features
    - Date columns will be automatically processed
    """)

elif page == "📊 Data Analysis":
    st.header("📊 Data Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload your water quality data (CSV format with ';' separator)"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file, sep=';')
            st.session_state.data = df
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            df = None
    
    # Option to use sample data
    if st.button("🎲 Use Sample Data"):
        df = create_sample_data()
        st.session_state.data = df
        st.info("📝 Using sample data for demonstration")
    
    # Display data analysis if data is available
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Rows", len(df))
        with col2:
            st.metric("📋 Total Columns", len(df.columns))
        with col3:
            st.metric("🔢 Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("📝 Object Columns", len(df.select_dtypes(include=['object']).columns))
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe().T, use_container_width=True)
        
        # Missing values
        st.subheader("❓ Missing Values Analysis")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            fig_missing = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                title="Missing Values by Column"
            )
            fig_missing.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
        
        # Data types
        st.subheader("🏷️ Data Types")
        dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
        st.dataframe(dtype_df, use_container_width=True)

elif page == "🤖 Model Prediction":
    st.header("🤖 Model Prediction")
    
    # Load model
    if st.session_state.model is None:
        with st.spinner("Loading model..."):
            st.session_state.model = load_model()
    
    if st.session_state.model is None:
        st.warning("⚠️ No model available. Please ensure 'best_rf_multioutput.pkl' is in the same directory.")
        st.markdown("""
        ### 📝 To use this feature:
        1. Run your notebook to train the model
        2. Ensure 'best_rf_multioutput.pkl' is saved in the same directory as this script
        3. Refresh this page
        """)
    else:
        st.success("✅ Model loaded successfully!")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["📝 Manual Input", "📁 Upload Data for Batch Prediction"]
        )
        
        if input_method == "📝 Manual Input":
            st.subheader("🔧 Manual Feature Input")
            
            # Show expected features if available
            if st.session_state.feature_names is not None:
                st.info(f"📋 Model expects these features: {', '.join(st.session_state.feature_names)}")
            
            # Create input fields based on what the model actually expects
            col1, col2 = st.columns(2)
            
            with col1:
                # Core features that are likely in your data
                id_val = st.number_input("� ID", value=1, min_value=1, step=1)
                
            with col2:
                year = st.selectbox("📅 Year", [2020, 2021, 2022, 2023, 2024], index=3)
                month = st.selectbox("📅 Month", list(range(1, 13)), index=5)
                day = st.selectbox("📅 Day", list(range(1, 29)), index=14)
                dayofweek = st.selectbox("📅 Day of Week", list(range(0, 7)), index=2)
            
            # Show a note about feature requirements
            st.info("💡 The model was trained with specific features from your dataset. Make sure your input data matches the training features.")
            
            if st.button("🔮 Make Prediction"):
                # Create input dataframe with the expected feature names
                input_data = pd.DataFrame({
                    'id': [id_val],
                    'date_year': [year],
                    'date_month': [month],
                    'date_day': [day],
                    'date_dayofweek': [dayofweek]
                })
                
                # If we know the expected features, use only those
                if st.session_state.feature_names is not None:
                    # Create a dataframe with all expected features, filling missing ones with 0
                    full_input = pd.DataFrame(columns=st.session_state.feature_names)
                    full_input.loc[0] = 0  # Initialize with zeros
                    
                    # Update with the values we have
                    for col in input_data.columns:
                        if col in full_input.columns:
                            full_input[col] = input_data[col].values
                    
                    input_data = full_input
                
                try:
                    # Make prediction
                    prediction = st.session_state.model.predict(input_data)
                    
                    # Display results
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.subheader("🎯 Prediction Results")
                    
                    # Create columns for results
                    cols = st.columns(3)
                    for i, (target, value) in enumerate(zip(TARGET_COLS, prediction[0])):
                        with cols[i % 3]:
                            st.metric(f"{target}", f"{value:.3f}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("💡 This might be due to feature mismatch. The model expects specific features from your training data.")
                    if st.session_state.feature_names is not None:
                        st.write("Expected features:", st.session_state.feature_names)
        
        else:  # Batch prediction
            st.subheader("📁 Batch Prediction")
            
            batch_file = st.file_uploader(
                "Upload CSV for batch prediction",
                type="csv",
                help="Upload a CSV file with the same format as your training data"
            )
            
            if batch_file is not None:
                try:
                    batch_df = pd.read_csv(batch_file, sep=';')
                    st.info(f"📊 Loaded {len(batch_df)} rows for prediction")
                    
                    # Show data preview
                    st.write("Data preview:")
                    st.dataframe(batch_df.head())
                    
                    if st.button("🚀 Run Batch Prediction"):
                        try:
                            # Preprocess the data (same as training)
                            batch_processed = preprocess_data(batch_df)
                            
                            # Remove target columns if they exist to avoid duplicates
                            feature_cols = [col for col in batch_processed.columns if col not in TARGET_COLS]
                            X_batch = batch_processed[feature_cols].copy()
                            
                            # Display what features we have vs what we need
                            if st.session_state.feature_names is not None:
                                expected_features = st.session_state.feature_names
                                available_features = list(X_batch.columns)
                                missing_features = set(expected_features) - set(available_features)
                                extra_features = set(available_features) - set(expected_features)
                                
                                if missing_features:
                                    st.warning(f"⚠️ Missing features: {list(missing_features)}")
                                    st.info("These will be filled with zeros.")
                                
                                if extra_features:
                                    st.info(f"ℹ️ Extra features that will be ignored: {list(extra_features)}")
                                
                                # Create a dataframe with exactly the features the model expects
                                X_batch_aligned = pd.DataFrame(columns=expected_features)
                                for idx in range(len(X_batch)):
                                    X_batch_aligned.loc[idx] = 0  # Initialize with zeros
                                
                                # Fill in the available features
                                for col in available_features:
                                    if col in expected_features:
                                        X_batch_aligned[col] = X_batch[col].values
                                
                                X_batch = X_batch_aligned
                            
                            # Fill any remaining NaN values with 0
                            X_batch = X_batch.fillna(0)
                            
                            # Make predictions
                            predictions = st.session_state.model.predict(X_batch)
                            
                            # Create results dataframe - avoid duplicate columns
                            predictions_df = pd.DataFrame(predictions, columns=[f"predicted_{col}" for col in TARGET_COLS])
                            
                            # Only include original data columns that are not target columns
                            original_cols = [col for col in batch_df.columns if col not in TARGET_COLS]
                            original_data = batch_df[original_cols].reset_index(drop=True)
                            
                            # Combine original data with predictions
                            results_df = pd.concat([original_data, predictions_df], axis=1)
                            
                            st.success(f"✅ Predictions completed for {len(results_df)} samples!")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Batch prediction error: {str(e)}")
                            
                            # Show debug information
                            st.subheader("🔍 Debug Information")
                            if st.session_state.feature_names is not None:
                                st.write("Expected features:", st.session_state.feature_names)
                            st.write("Available columns after preprocessing:", list(batch_processed.columns))
                            
                            # Show feature mismatch details
                            if hasattr(st.session_state.model, 'feature_names_in_'):
                                st.write("Model was trained with features:", list(st.session_state.model.feature_names_in_))
                            
                except Exception as e:
                    st.error(f"❌ Error loading batch file: {str(e)}")

elif page == "📈 Visualizations":
    st.header("📈 Data Visualizations")
    
    if st.session_state.data is None:
        st.warning("⚠️ No data available. Please upload data in the Data Analysis section first.")
    else:
        df = st.session_state.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            st.warning("⚠️ No numeric columns found for visualization.")
        else:
            # Correlation heatmap
            st.subheader("🔥 Correlation Heatmap")
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Distribution plots
            st.subheader("📊 Distribution Analysis")
            selected_col = st.selectbox("Select column for distribution:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    df,
                    x=selected_col,
                    title=f"Distribution of {selected_col}",
                    nbins=30
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_box = px.box(
                    df,
                    y=selected_col,
                    title=f"Box Plot of {selected_col}"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Scatter plot
            if len(numeric_cols) >= 2:
                st.subheader("🎯 Scatter Plot Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    x_axis = st.selectbox("Select X-axis:", numeric_cols, key="x_scatter")
                with col2:
                    y_axis = st.selectbox("Select Y-axis:", numeric_cols, key="y_scatter", index=1)
                
                fig_scatter = px.scatter(
                    df,
                    x=x_axis,
                    y=y_axis,
                    title=f"{x_axis} vs {y_axis}",
                    trendline="ols"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Target variables analysis
            available_targets = [col for col in TARGET_COLS if col in df.columns]
            if available_targets:
                st.subheader("🎯 Target Variables Analysis")
                
                # Target correlations
                if len(available_targets) > 1:
                    target_corr = df[available_targets].corr()
                    fig_target_corr = px.imshow(
                        target_corr,
                        text_auto=True,
                        title="Target Variables Correlation",
                        color_continuous_scale="viridis"
                    )
                    st.plotly_chart(fig_target_corr, use_container_width=True)
                
                # Target distributions
                selected_target = st.selectbox("Select target for analysis:", available_targets)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_target_dist = px.histogram(
                        df,
                        x=selected_target,
                        title=f"Distribution of {selected_target}",
                        nbins=30
                    )
                    st.plotly_chart(fig_target_dist, use_container_width=True)
                
                with col2:
                    fig_target_box = px.box(
                        df,
                        y=selected_target,
                        title=f"Box Plot of {selected_target}"
                    )
                    st.plotly_chart(fig_target_box, use_container_width=True)

elif page == "📋 Model Info":
    st.header("📋 Model Information")
    
    # Model details
    st.subheader("🤖 Model Architecture")
    st.markdown("""
    - **Model Type**: Multi-Output Random Forest Regressor
    - **Base Estimator**: Random Forest Regressor
    - **Hyperparameter Tuning**: RandomizedSearchCV with 3-fold cross-validation
    - **Target Variables**: 9 water quality parameters
    - **Optimization Metric**: Negative Mean Squared Error
    """)
    
    # Load and display model if available
    if st.session_state.model is None:
        st.session_state.model = load_model()
    
    if st.session_state.model is not None:
        st.success("✅ Model loaded successfully!")
        
        # Try to get model parameters
        try:
            model_params = st.session_state.model.get_params()
            st.subheader("⚙️ Model Parameters")
            
            # Display key parameters
            key_params = {
                'estimator__n_estimators': 'Number of Trees',
                'estimator__max_depth': 'Maximum Depth',
                'estimator__min_samples_split': 'Min Samples Split',
                'estimator__min_samples_leaf': 'Min Samples Leaf',
                'estimator__max_features': 'Max Features'
            }
            
            for param, description in key_params.items():
                if param in model_params:
                    st.write(f"**{description}**: {model_params[param]}")
            
        except Exception as e:
            st.warning(f"Could not retrieve model parameters: {str(e)}")
    
    # Performance metrics (if you have them)
    st.subheader("📊 Expected Performance")
    st.markdown("""
    Based on your notebook training:
    - The model uses RandomizedSearchCV for hyperparameter optimization
    - Performance is evaluated using R² Score and RMSE
    - Feature importance is calculated across all target variables
    """)
    
    # Target variables info
    st.subheader("🎯 Target Variables")
    target_info = {
        'NH4': {'name': 'Ammonium', 'unit': 'mg/L', 'description': 'Ammonium nitrogen concentration'},
        'BSK5': {'name': 'BOD5', 'unit': 'mg/L', 'description': 'Biochemical Oxygen Demand (5-day)'},
        'Suspended': {'name': 'Suspended Solids', 'unit': 'mg/L', 'description': 'Total suspended solids'},
        'O2': {'name': 'Dissolved Oxygen', 'unit': 'mg/L', 'description': 'Dissolved oxygen concentration'},
        'NO3': {'name': 'Nitrate', 'unit': 'mg/L', 'description': 'Nitrate nitrogen concentration'},
        'NO2': {'name': 'Nitrite', 'unit': 'mg/L', 'description': 'Nitrite nitrogen concentration'},
        'SO4': {'name': 'Sulfate', 'unit': 'mg/L', 'description': 'Sulfate concentration'},
        'PO4': {'name': 'Phosphate', 'unit': 'mg/L', 'description': 'Phosphate concentration'},
        'CL': {'name': 'Chloride', 'unit': 'mg/L', 'description': 'Chloride concentration'}
    }
    
    for code, info in target_info.items():
        with st.expander(f"{code} - {info['name']}"):
            st.write(f"**Unit**: {info['unit']}")
            st.write(f"**Description**: {info['description']}")
    
    # Usage instructions
    st.subheader("🚀 Usage Instructions")
    st.markdown("""
    ### 📝 To use this application:
    
    1. **Data Analysis**: Upload your water quality data (CSV with ';' separator)
    2. **Model Prediction**: Use the trained model for predictions
       - Manual input for single predictions
       - Batch upload for multiple predictions
    3. **Visualizations**: Explore data patterns and relationships
    
    ### 📊 Data Requirements:
    - CSV format with semicolon (;) separator
    - Numeric water quality measurement features
    - Date columns (will be automatically processed)
    - Missing values are handled automatically
    
    ### 🔧 Model Features:
    - Multi-output prediction (9 water quality parameters simultaneously)
    - Robust preprocessing pipeline
    - Feature importance analysis
    - Cross-validated hyperparameter tuning
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🌊 Water Quality Prediction Dashboard | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)