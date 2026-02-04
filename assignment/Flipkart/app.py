"""
Flipkart Review Sentiment Analysis - Streamlit Web Application
Real-time sentiment prediction for product reviews
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import (
    load_model_and_vectorizer, 
    predict_sentiment, 
    load_model_metadata,
    get_sentiment_color,
    get_confidence_level,
    preprocess_text
)
import os

# Page configuration
st.set_page_config(
    page_title="Flipkart Sentiment Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2874f0;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .positive {
        background-color: #d4edda;
        border-left: 5px solid #2ecc71;
    }
    .negative {
        background-color: #f8d7da;
        border-left: 5px solid #e74c3c;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #2874f0;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #1557bf;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load model and vectorizer (cached)"""
    try:
        model, vectorizer = load_model_and_vectorizer()
        metadata = load_model_metadata()
        return model, vectorizer, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def main():
    # Header
    st.markdown('<div class="main-header">🛒 Flipkart Review Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyze customer reviews and predict sentiment in real-time</div>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        model, vectorizer, metadata = load_models()
    
    if model is None or vectorizer is None:
        st.error(" Failed to load model. Please ensure model files exist in the 'models' directory.")
        st.info("Run the Jupyter notebooks (1-4) to train and save the model first.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header(" Model Information")
        
        if metadata:
            st.success(f"**Model:** {metadata.get('model_name', 'Unknown')}")
            st.metric("Test F1-Score", f"{metadata.get('test_f1_score', 0):.4f}")
            st.metric("Test Accuracy", f"{metadata.get('test_accuracy', 0):.4f}")
            st.metric("Features", f"{metadata.get('feature_count', 0):,}")
            
            with st.expander("Best Parameters"):
                st.json(metadata.get('best_params', {}))
        else:
            st.info("Model metadata not available")
        
        st.divider()
        
        st.header("ℹ About")
        st.write("""
        This application uses machine learning to analyze 
        product reviews from Flipkart and classify them as 
        **Positive** or **Negative**.
        
        **Features:**
        - Real-time sentiment prediction
        - Confidence scores
        - Text preprocessing visualization
        - Batch processing support
        """)
        
        st.divider()
        st.caption("Built with Streamlit & Scikit-learn")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs([" Single Review", " Batch Analysis", " Model Performance"])
    
    # Tab 1: Single Review Analysis
    with tab1:
        st.subheader("Analyze a Single Review")
        
        # Sample reviews
        sample_reviews = [
            "This product is absolutely amazing! Best quality and highly recommended.",
            "Terrible experience. Product is defective and customer service is horrible.",
            "Good product for the price. Quick delivery and well packaged.",
            "Not worth the money. Very disappointed with the quality.",
            "Excellent! Exceeded my expectations in every way."
        ]
        
        # Initialize session state for review text
        if 'review_text' not in st.session_state:
            st.session_state.review_text = ""
        
        col1, col2 = st.columns([3, 1])
        with col1:
            review_text = st.text_area(
                "Enter product review:",
                value=st.session_state.review_text,
                height=150,
                placeholder="Type or paste a product review here...",
                key="review_input"
            )
            # Update session state with current input
            st.session_state.review_text = review_text
        
        with col2:
            st.write("**Quick Samples:**")
            for i, sample in enumerate(sample_reviews[:3], 1):
                if st.button(f"Sample {i}", key=f"sample_{i}"):
                    st.session_state.review_text = sample
                    st.rerun()
        
        col_predict, col_clear = st.columns([1, 1])
        with col_predict:
            predict_btn = st.button(" Predict Sentiment", type="primary", use_container_width=True)
        with col_clear:
            if st.button(" Clear", use_container_width=True):
                st.session_state.review_text = ""
                st.rerun()
        
        if predict_btn and review_text:
            with st.spinner("Analyzing..."):
                result = predict_sentiment(review_text, model, vectorizer)
            
            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                # Display results
                sentiment = result['sentiment']
                confidence = result['confidence']
                
                # Prediction box
                box_class = 'positive' if sentiment == 'Positive' else 'negative'
                st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h2 style="margin:0;">Sentiment: {sentiment}</h2>
                    <p style="font-size:1.2rem; margin:0.5rem 0;">
                        Confidence: <b>{confidence:.1f}%</b> ({get_confidence_level(confidence)})
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Positive Probability",
                        f"{result['positive_prob']:.1f}%"
                    )
                with col2:
                    st.metric(
                        "Negative Probability",
                        f"{result['negative_prob']:.1f}%"
                    )
                with col3:
                    st.metric(
                        "Confidence Level",
                        get_confidence_level(confidence)
                    )
                
                # Probability chart
                st.subheader("Probability Distribution")
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Negative', 'Positive'],
                        y=[result['negative_prob'], result['positive_prob']],
                        marker_color=['#e74c3c', '#2ecc71'],
                        text=[f"{result['negative_prob']:.1f}%", f"{result['positive_prob']:.1f}%"],
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    yaxis_title="Probability (%)",
                    yaxis_range=[0, 100],
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show preprocessing
                with st.expander("🔧 View Text Preprocessing"):
                    st.write("**Original Text:**")
                    st.info(review_text)
                    st.write("**Processed Text:**")
                    st.success(result['processed_text'])
        
        elif predict_btn:
            st.warning(" Please enter a review text to analyze.")
    
    # Tab 2: Batch Analysis
    with tab2:
        st.subheader("Batch Review Analysis")
        
        st.write("Upload a CSV file with reviews to analyze multiple reviews at once.")
        st.info("CSV file should have a column named 'review' or 'Review' containing the review text.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Find review column
                review_col = None
                for col in df.columns:
                    if col.lower() in ['review', 'reviews', 'text', 'comment']:
                        review_col = col
                        break
                
                if review_col is None:
                    st.error("No review column found. Please ensure your CSV has a 'review' column.")
                else:
                    st.success(f" Loaded {len(df)} reviews from column '{review_col}'")
                    
                    if st.button(" Analyze All Reviews", type="primary"):
                        with st.spinner(f"Analyzing {len(df)} reviews..."):
                            # Process all reviews
                            predictions = []
                            for review in df[review_col]:
                                result = predict_sentiment(str(review), model, vectorizer)
                                predictions.append({
                                    'Sentiment': result['sentiment'],
                                    'Confidence': f"{result['confidence']:.1f}%",
                                    'Positive Prob': f"{result['positive_prob']:.1f}%",
                                    'Negative Prob': f"{result['negative_prob']:.1f}%"
                                })
                            
                            # Add predictions to dataframe
                            results_df = pd.concat([df, pd.DataFrame(predictions)], axis=1)
                            
                            # Display results
                            st.subheader("Results")
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Summary statistics
                            st.subheader("Summary Statistics")
                            col1, col2, col3 = st.columns(3)
                            
                            positive_count = (results_df['Sentiment'] == 'Positive').sum()
                            negative_count = (results_df['Sentiment'] == 'Negative').sum()
                            
                            with col1:
                                st.metric("Total Reviews", len(results_df))
                            with col2:
                                st.metric("Positive Reviews", positive_count, 
                                         f"{positive_count/len(results_df)*100:.1f}%")
                            with col3:
                                st.metric("Negative Reviews", negative_count,
                                         f"{negative_count/len(results_df)*100:.1f}%")
                            
                            # Pie chart
                            fig = px.pie(
                                values=[positive_count, negative_count],
                                names=['Positive', 'Negative'],
                                title="Sentiment Distribution",
                                color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label=" Download Results as CSV",
                                data=csv,
                                file_name="sentiment_analysis_results.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Tab 3: Model Performance
    with tab3:
        st.subheader("Model Performance Metrics")
        
        if metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("###  Test Set Performance")
                metrics_data = {
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    'Score': [
                        metadata.get('test_accuracy', 0),
                        metadata.get('test_precision', 0),
                        metadata.get('test_recall', 0),
                        metadata.get('test_f1_score', 0)
                    ]
                }
                metrics_df = pd.DataFrame(metrics_data)
                
                # Bar chart
                fig = px.bar(
                    metrics_df,
                    x='Metric',
                    y='Score',
                    title="Model Performance Metrics",
                    color='Score',
                    color_continuous_scale='Greens',
                    text='Score'
                )
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig.update_layout(yaxis_range=[0, 1.1], showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("###  Model Configuration")
                st.json({
                    "Model Type": metadata.get('model_name', 'Unknown'),
                    "Vectorizer": metadata.get('vectorizer', 'Unknown'),
                    "Feature Count": metadata.get('feature_count', 0),
                    "Best Parameters": metadata.get('best_params', {})
                })
                
                st.markdown("###  Performance Insights")
                f1 = metadata.get('test_f1_score', 0)
                if f1 >= 0.90:
                    st.success(" Excellent performance! F1-Score >= 0.90")
                elif f1 >= 0.80:
                    st.success(" Good performance! F1-Score >= 0.80")
                elif f1 >= 0.70:
                    st.warning(" Moderate performance. F1-Score >= 0.70")
                else:
                    st.error(" Low performance. Consider retraining.")
        else:
            st.info("No metadata available. Run the training notebook to generate model metadata.")


if __name__ == "__main__":
    main()
