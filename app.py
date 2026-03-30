import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 Student Exam Score Predictor")
st.markdown("""
This app predicts a student's exam score based on their study habits and lifestyle factors.
The model uses **Linear Regression** trained on student performance data.
""")

# Sidebar for model info
with st.sidebar:
    st.header("ℹ️ About the Model")
    st.markdown("""
    - **Algorithm**: Linear Regression
    - **Features Used**:
        - Hours Studied
        - Attendance (%)
        - Sleep Hours
    - **Target**: Exam Score (55-101)
    """)
    
    st.divider()
    
    st.header("📊 Model Performance")
    
    # Cache the dataset loading
    @st.cache_data
    def load_data():
        df = pd.read_csv('StudentPerformanceFactors.csv')
        return df
    
    # Cache model training
    @st.cache_resource
    def train_model():
        # Load data
        df = load_data()
        
        # Handle missing values
        df = df.dropna()
        
        # Select features and target
        features = ['Hours_Studied', 'Attendance', 'Sleep_Hours']
        X = df[features]
        y = df['Exam_Score']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return model, rmse, r2, X_test, y_test, y_pred
    
    # Load model and metrics
    model, rmse, r2, X_test, y_test, y_pred = train_model()
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RMSE", f"{rmse:.2f}")
    with col2:
        st.metric("R² Score", f"{r2:.3f}")
    
    st.caption("Lower RMSE = Better | R² closer to 1 = Better")

# Main content area - Input section
st.header("🎯 Predict Your Exam Score")

col1, col2, col3 = st.columns(3)

with col1:
    hours_studied = st.slider(
        "📖 Hours Studied per Week",
        min_value=1,
        max_value=44,
        value=20,
        help="Average hours spent studying per week"
    )

with col2:
    attendance = st.slider(
        "🏫 Attendance Percentage",
        min_value=60,
        max_value=100,
        value=80,
        help="Class attendance percentage"
    )

with col3:
    sleep_hours = st.slider(
        "😴 Average Sleep Hours",
        min_value=4,
        max_value=10,
        value=7,
        help="Average hours of sleep per night"
    )

# Create input dataframe
input_data = pd.DataFrame({
    'Hours_Studied': [hours_studied],
    'Attendance': [attendance],
    'Sleep_Hours': [sleep_hours]
})

# Prediction button
if st.button("🔮 Predict Score", type="primary"):
    prediction = model.predict(input_data)[0]
    
    # Display prediction with styling
    st.markdown("---")
    st.subheader("📈 Prediction Result")
    
    # Create a nice display for the prediction
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
            <h2 style='color: #0066cc; margin-bottom: 10px;'>Predicted Exam Score</h2>
            <h1 style='color: #ff4b4b; font-size: 72px; margin: 0;'>{prediction:.1f}</h1>
            <p style='color: #666; margin-top: 10px;'>out of 100</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Interpretation
    st.subheader("📝 Interpretation")
    if prediction >= 85:
        st.success("🎉 Excellent performance! Keep up the great work!")
    elif prediction >= 70:
        st.info("👍 Good performance! You're on the right track.")
    elif prediction >= 55:
        st.warning("📚 Average performance. Consider increasing study hours or improving attendance.")
    else:
        st.error("⚠️ Below average. Need significant improvement in study habits.")

# Visualizations section
st.header("📊 Data Insights")
tab1, tab2, tab3 = st.tabs(["Feature Analysis", "Model Performance", "Dataset Overview"])

with tab1:
    st.subheader("Relationship Between Features and Exam Scores")
    
    df = load_data()
    df = df.dropna()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Hours Studied vs Exam Score
    axes[0].scatter(df['Hours_Studied'], df['Exam_Score'], alpha=0.5, c='blue')
    axes[0].set_xlabel('Hours Studied')
    axes[0].set_ylabel('Exam Score')
    axes[0].set_title('Study Hours vs Score')
    axes[0].grid(True, alpha=0.3)
    
    # Attendance vs Exam Score
    axes[1].scatter(df['Attendance'], df['Exam_Score'], alpha=0.5, c='green')
    axes[1].set_xlabel('Attendance (%)')
    axes[1].set_ylabel('Exam Score')
    axes[1].set_title('Attendance vs Score')
    axes[1].grid(True, alpha=0.3)
    
    # Sleep Hours vs Exam Score
    axes[2].scatter(df['Sleep_Hours'], df['Exam_Score'], alpha=0.5, c='orange')
    axes[2].set_xlabel('Sleep Hours')
    axes[2].set_ylabel('Exam Score')
    axes[2].set_title('Sleep Hours vs Score')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Actual vs Predicted Scores")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Score')
    ax.set_ylabel('Predicted Score')
    ax.set_title('Actual vs Predicted Exam Scores')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Residual plot
    st.subheader("Residual Plot")
    residuals = y_test - y_pred
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Score')
    ax.set_ylabel('Residuals (Error)')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab3:
    st.subheader("Dataset Statistics")
    
    df = load_data()
    st.write(f"**Total Records:** {len(df)}")
    st.write(f"**Features:** {len(df.columns)}")
    
    # Show numerical columns statistics
    st.write("**Numerical Features Statistics:**")
    st.dataframe(df.describe())
    
    # Show sample data
    st.write("**Sample Data (First 10 rows):**")
    st.dataframe(df.head(10))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with ❤️ using Streamlit | Model trained on Student Performance Factors Dataset</p>
    <p>Disclaimer: This is a predictive model. Actual results may vary based on multiple factors.</p>
</div>
""", unsafe_allow_html=True)