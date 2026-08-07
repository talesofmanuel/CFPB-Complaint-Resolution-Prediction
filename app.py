import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CFPB Complaint Resolution Predictor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2e5496 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { color: white; font-size: 1.8rem; margin-bottom: 0.5rem; }
    .main-header p { color: #cdd9f0; font-size: 0.95rem; margin: 0; }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .favourable {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
    }
    .unfavourable {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-card h3 { font-size: 1.8rem; margin: 0; color: #2e5496; }
    .metric-card p { font-size: 0.85rem; color: #666; margin: 0; }
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #2e5496;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .section-header {
        color: #2e5496;
        border-bottom: 2px solid #2e5496;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND ENCODERS
# ============================================================
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load('xgboost_fusion_model.pkl')
        feature_names = joblib.load('feature_names.pkl')
        company_freq = joblib.load('company_freq_encoding.pkl')
        encoders = joblib.load('label_encoders.pkl')
        return model, feature_names, company_freq, encoders
    except FileNotFoundError as e:
        st.error("Model files not found. Please ensure all .pkl files are in the same directory as app.py")
        st.stop()

model, feature_names, company_freq, encoders = load_model_and_encoders()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>⚖️ CFPB Complaint Resolution Predictor</h1>
    <p>Predicting Consumer Complaint Resolution Outcomes Using Fused NLP-Derived
    Sentiment and Structured Metadata</p>
    <p style="margin-top:0.5rem; font-size:0.85rem; color:#aac4e8;">
    MSc Data Science Dissertation | Coventry University | Onyekachukwu Emmanuel Ogbodo
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Seal_of_the_United_States_Consumer_Financial_Protection_Bureau.svg/200px-Seal_of_the_United_States_Consumer_Financial_Protection_Bureau.svg.png",
                 width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to",
    ["Complaint Predictor", "Model Performance", "SHAP Analysis", "About"])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model:** XGBoost Fusion
**AUC-ROC:** 0.7968
**Dataset:** CFPB Complaints
**Records:** 1,000,000
**Features:** 17 (8 structured + 9 sentiment)
""")

# ============================================================
# COMPANIES
# ============================================================
COMPANIES = [
    'EQUIFAX, INC.',
    'TRANSUNION INTERMEDIATE HOLDINGS, INC.',
    'Experian Information Solutions Inc.'
]

# ============================================================
# PAGE 1 — COMPLAINT PREDICTOR
# ============================================================
if page == "Complaint Predictor":

    st.markdown('<h2 class="section-header">Submit a Complaint for Prediction</h2>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Enter the details of a CFPB credit bureau complaint below to predict whether
    it is likely to receive a <strong>favourable</strong> or
    <strong>unfavourable</strong> resolution. Optionally include the consumer
    narrative for enhanced sentiment-based prediction.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Complaint Details")

        company = st.selectbox("Credit Bureau", COMPANIES)

        product_options = encoders['product_classes']
        product = st.selectbox("Product", sorted(product_options))

        subproduct_options = encoders['subproduct_classes']
        subproduct = st.selectbox("Sub-product", sorted(subproduct_options))

        issue_options = encoders['top_20_issues'] + ['Other']
        issue = st.selectbox("Issue", issue_options)

        subissue_options = encoders['top_20_subissues'] + ['Other', 'Unknown']
        subissue = st.selectbox("Sub-issue", subissue_options)

        state_options = encoders['state_classes']
        state = st.selectbox("State", sorted(state_options))

    with col2:
        st.subheader("Consumer Narrative (Optional)")
        narrative = st.text_area(
            "Enter the consumer complaint narrative",
            height=200,
            placeholder="Type the consumer narrative here. Leave blank to predict using structured features only..."
        )

        st.markdown("---")
        st.subheader("Narrative Statistics")
        if narrative.strip():
            word_count = len(narrative.split())
            st.info("Word count: " + str(word_count) + " words")
            if word_count < 20:
                st.warning("Narrative below 20 words. Sentiment features will not be extracted.")
        else:
            st.info("No narrative provided. Structured features only.")

    st.markdown("---")
    predict_btn = st.button("Predict Resolution Outcome", type="primary",
                            use_container_width=True)

    if predict_btn:
        with st.spinner("Processing complaint and generating prediction..."):

            # Encode structured features
            try:
                product_enc = encoders['product'].transform([product])[0]
            except ValueError:
                product_enc = 0

            try:
                subproduct_enc = encoders['sub-product'].transform([subproduct])[0]
            except ValueError:
                subproduct_enc = 0

            try:
                issue_val = issue if issue in encoders['top_20_issues'] else 'Other'
                issue_enc = encoders['issue'].transform([issue_val])[0]
            except ValueError:
                issue_enc = 0

            try:
                subissue_val = subissue if subissue in encoders['top_20_subissues'] else 'Other'
                subissue_enc = encoders['sub-issue'].transform([subissue_val])[0]
            except ValueError:
                subissue_enc = 0

            try:
                state_enc = encoders['state'].transform([state])[0]
            except ValueError:
                state_enc = 0

            comp_freq = company_freq.get(company, 0.001)

            # Narrative features
            narrative_present = 0
            narrative_word_count = 0
            vader_compound = 0.0
            vader_positive = 0.0
            vader_negative = 0.0
            vader_neutral = 0.0
            finbert_label_encoded = 1
            finbert_positive = 0.0
            finbert_negative = 0.0
            finbert_neutral = 1.0
            finbert_confidence = 0.7

            if narrative.strip() and len(narrative.split()) >= 20:
                narrative_present = 1
                narrative_word_count = len(narrative.split())

                try:
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    analyzer = SentimentIntensityAnalyzer()
                    scores = analyzer.polarity_scores(narrative)
                    vader_compound = scores['compound']
                    vader_positive = scores['pos']
                    vader_negative = scores['neg']
                    vader_neutral = scores['neu']
                except Exception:
                    pass

                try:
                    from transformers import BertTokenizer, BertForSequenceClassification
                    import torch
                    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
                    fb_model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
                    fb_model.eval()
                    inputs = tokenizer(narrative, return_tensors='pt',
                                       truncation=True, max_length=512, padding=True)
                    with torch.no_grad():
                        outputs = fb_model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
                    label_map = {0: 2, 1: 0, 2: 1}
                    pred_class = int(np.argmax(probs))
                    finbert_label_encoded = label_map[pred_class]
                    finbert_positive = float(probs[0])
                    finbert_negative = float(probs[1])
                    finbert_neutral = float(probs[2])
                    finbert_confidence = float(probs.max())
                except Exception:
                    pass

            # Build feature vector
            features = np.array([[
                product_enc, subproduct_enc, issue_enc, subissue_enc,
                state_enc, narrative_word_count, narrative_present, comp_freq,
                vader_compound, vader_positive, vader_negative, vader_neutral,
                finbert_label_encoded, finbert_positive, finbert_negative,
                finbert_neutral, finbert_confidence
            ]])

            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            fav_prob = round(float(probability[1]) * 100, 1)
            unfav_prob = round(float(probability[0]) * 100, 1)

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.markdown("""
            <div class="prediction-box favourable">
                <h2 style="color:#155724; margin:0;">✅ FAVOURABLE RESOLUTION</h2>
                <p style="color:#155724; font-size:1.1rem; margin:0.5rem 0;">
                This complaint is predicted to receive a favourable resolution.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-box unfavourable">
                <h2 style="color:#721c24; margin:0;">❌ UNFAVOURABLE RESOLUTION</h2>
                <p style="color:#721c24; font-size:1.1rem; margin:0.5rem 0;">
                This complaint is predicted to receive an unfavourable resolution.
                </p>
            </div>
            """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>""" + str(fav_prob) + """%</h3>
                <p>Probability of Favourable Resolution</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>""" + str(unfav_prob) + """%</h3>
                <p>Probability of Unfavourable Resolution</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>""" + ("✅ Yes" if narrative_present else "❌ No") + """</h3>
                <p>Narrative Used in Prediction</p>
            </div>
            """, unsafe_allow_html=True)

        # SHAP explanation
        st.markdown("---")
        st.subheader("Feature Contributions for This Prediction")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)
            fig, ax = plt.subplots(figsize=(10, 5))
            feature_labels = [
                'Product', 'Sub-product', 'Issue', 'Sub-issue', 'State',
                'Narrative Word Count', 'Narrative Present', 'Company Freq',
                'VADER Compound', 'VADER Positive', 'VADER Negative',
                'VADER Neutral', 'FinBERT Label', 'FinBERT Positive',
                'FinBERT Negative', 'FinBERT Neutral', 'FinBERT Confidence'
            ]
            sv = shap_values[0]
            colors = ['#28a745' if v > 0 else '#dc3545' for v in sv]
            sorted_idx = np.argsort(np.abs(sv))
            ax.barh([feature_labels[i] for i in sorted_idx],
                    [sv[i] for i in sorted_idx],
                    color=[colors[i] for i in sorted_idx])
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.set_xlabel('SHAP Value (impact on prediction)')
            ax.set_title('Feature Contributions: Green = Favourable, Red = Unfavourable')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.info("SHAP explanation unavailable for this prediction.")

# ============================================================
# PAGE 2 — MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":

    st.markdown('<h2 class="section-header">Model Performance Results</h2>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    All nine model configurations were evaluated under stratified 5-fold
    cross-validation. XGBoost Fusion achieved the highest AUC-ROC of 0.7968.
    </div>
    """, unsafe_allow_html=True)

    results_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost',
                  'Logistic Regression', 'Random Forest', 'XGBoost',
                  'Logistic Regression', 'Random Forest', 'XGBoost'],
        'Configuration': ['Structured Only', 'Structured Only', 'Structured Only',
                          'Sentiment Only', 'Sentiment Only', 'Sentiment Only',
                          'Fusion', 'Fusion', 'Fusion'],
        'AUC-ROC': [0.7177, 0.7830, 0.7941,
                    0.5069, 0.5258, 0.5253,
                    0.5461, 0.7955, 0.7968],
        'Std': [0.0011, 0.0013, 0.0011,
                0.0013, 0.0009, 0.0007,
                0.0010, 0.0012, 0.0009],
        'F1': [0.6982, 0.7249, 0.7372,
               0.2182, 0.1925, 0.2016,
               0.4618, 0.7333, 0.7377],
        'Precision': [0.5661, 0.6312, 0.6315,
                      0.4570, 0.5134, 0.5063,
                      0.4688, 0.6382, 0.6327],
        'Recall': [0.9105, 0.8511, 0.8856,
                   0.1434, 0.1184, 0.1259,
                   0.4550, 0.8616, 0.8844]
    }

    df_results = pd.DataFrame(results_data)

    def highlight_best(row):
        if row['AUC-ROC'] == 0.7968:
            return ['background-color: #d4edda'] * len(row)
        elif row['AUC-ROC'] < 0.55:
            return ['background-color: #f8d7da'] * len(row)
        return [''] * len(row)

    st.dataframe(df_results.style.apply(highlight_best, axis=1),
                 use_container_width=True, hide_index=True)
    st.caption("Green row: best performing model. Red rows: near-random performance.")

    st.markdown("---")
    st.subheader("AUC-ROC Comparison Chart")

    models = ['LR', 'RF', 'XGBoost']
    structured = [0.7177, 0.7830, 0.7941]
    sentiment = [0.5069, 0.5258, 0.5253]
    fusion = [0.5461, 0.7955, 0.7968]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, structured, width, label='Structured Only', color='#2E5496')
    ax.bar(x, sentiment, width, label='Sentiment Only', color='#70AD47')
    ax.bar(x + width, fusion, width, label='Fusion', color='#ED7D31')

    for i, v in enumerate(structured):
        ax.text(i - width, v + 0.005, str(v), ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(sentiment):
        ax.text(i, v + 0.005, str(v), ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(fusion):
        ax.text(i + width, v + 0.005, str(v), ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('AUC-ROC Performance Across All Nine Model Configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(['Logistic Regression', 'Random Forest', 'XGBoost'])
    ax.set_ylim(0.4, 0.85)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1,
               label='Random Chance (0.5)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>0.7968</h3>
            <p>Best AUC-ROC (XGBoost Fusion)</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>1M</h3>
            <p>Training Records</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>9</h3>
            <p>Model Configurations</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>5-Fold</h3>
            <p>Cross-Validation</p>
        </div>""", unsafe_allow_html=True)

# ============================================================
# PAGE 3 — SHAP ANALYSIS
# ============================================================
elif page == "SHAP Analysis":

    st.markdown('<h2 class="section-header">Global SHAP Feature Importance</h2>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    SHAP (SHapley Additive exPlanations) analysis was applied to the best-performing
    XGBoost Fusion model to identify which features most strongly predict favourable
    complaint resolution outcomes.
    </div>
    """, unsafe_allow_html=True)

    shap_data = {
        'Rank': list(range(1, 18)),
        'Feature': [
            'Company Frequency', 'Product', 'Sub-issue', 'State', 'Issue',
            'Narrative Word Count', 'VADER Compound', 'VADER Negative',
            'Sub-product', 'FinBERT Label', 'VADER Positive', 'FinBERT Positive',
            'VADER Neutral', 'FinBERT Confidence', 'FinBERT Negative',
            'FinBERT Neutral', 'Narrative Present'
        ],
        'Mean Absolute SHAP': [
            1.0851, 0.2759, 0.1118, 0.0470, 0.0451,
            0.0444, 0.0296, 0.0284, 0.0261, 0.0251,
            0.0237, 0.0232, 0.0201, 0.0201, 0.0183,
            0.0160, 0.0103
        ],
        'Modality': [
            'Structured', 'Structured', 'Structured', 'Structured', 'Structured',
            'Structured', 'Sentiment', 'Sentiment', 'Structured', 'Sentiment',
            'Sentiment', 'Sentiment', 'Sentiment', 'Sentiment', 'Sentiment',
            'Sentiment', 'Structured'
        ]
    }

    df_shap = pd.DataFrame(shap_data)

    def highlight_modality(row):
        if row['Modality'] == 'Structured':
            return ['background-color: #dce8f5'] * len(row)
        return ['background-color: #fef9e7'] * len(row)

    st.dataframe(df_shap.style.apply(highlight_modality, axis=1),
                 use_container_width=True, hide_index=True)
    st.caption("Blue rows: structured features. Yellow rows: sentiment features.")

    st.markdown("---")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#2E5496' if m == 'Structured' else '#ED7D31'
              for m in df_shap['Modality']]
    ax.barh(df_shap['Feature'][::-1], df_shap['Mean Absolute SHAP'][::-1],
            color=colors[::-1])
    ax.set_xlabel('Mean Absolute SHAP Value')
    ax.set_title('Global SHAP Feature Importance: XGBoost Fusion Model')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2E5496', label='Structured Feature'),
                       Patch(facecolor='#ED7D31', label='Sentiment Feature')]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Dominant Feature**
        Company frequency has a mean absolute SHAP value of 1.0851, nearly
        four times greater than the next most important feature (product at 0.2759).
        Which bureau a consumer complains against is the strongest predictor of outcome.

        **Structured Features Dominate**
        The top six features are all structured metadata, confirming that complaint
        structural characteristics drive predictive power more than sentiment.
        """)
    with col2:
        st.markdown("""
        **Narrative Word Count Insight**
        Narrative word count ranks sixth at 0.0444, outranking all sentiment
        score features. How much a consumer writes is more predictive than
        the sentiment they express.

        **Sentiment Contribution**
        VADER compound score (0.0296) and FinBERT label (0.0251) both contribute
        meaningful but modest signal, confirming that fusion adds value over
        structured-only modelling.
        """)

# ============================================================
# PAGE 4 — ABOUT
# ============================================================
elif page == "About":

    st.markdown('<h2 class="section-header">About This Project</h2>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Research Overview

        This application is the interactive artefact of an MSc Data Science
        dissertation at Coventry University. It implements a multi-modal machine
        learning framework that predicts whether a consumer complaint filed with
        the Consumer Financial Protection Bureau (CFPB) will receive a favourable
        or unfavourable resolution.

        ### Research Questions

        **Main RQ:** Can the fusion of NLP-derived sentiment features from consumer
        complaint narratives with structured metadata improve the prediction of
        complaint resolution outcomes in CFPB credit bureau complaints?

        **SRQ1:** Does a fusion model outperform single-modality models as measured
        by AUC-ROC?

        **SRQ2:** Which features most strongly predict favourable outcomes via
        SHAP analysis?

        **SRQ3:** Does FinBERT outperform VADER as a predictive feature?

        ### Methodology

        - **Dataset:** CFPB Consumer Complaints Database (1,000,000 records)
        - **NLP Tools:** FinBERT (ProsusAI) and VADER for sentiment extraction
        - **Models:** Logistic Regression, Random Forest, XGBoost
        - **Configurations:** 9 comparative runs (3 models x 3 feature sets)
        - **Evaluation:** Stratified 5-fold cross-validation, AUC-ROC primary metric
        - **Explainability:** SHAP TreeExplainer

        ### Key Findings

        - XGBoost Fusion achieved the best AUC-ROC of 0.7968
        - Company identity is the dominant predictor (SHAP value 1.0851)
        - Narrative word count outranks all sentiment scores
        - FinBERT and VADER contribute comparable predictive signal
        """)

    with col2:
        st.markdown("""
        ### Project Details

        **Student:**
        Onyekachukwu Emmanuel Ogbodo

        **Student ID:** 16480931

        **Programme:**
        MSc Data Science

        **University:**
        Coventry University

        **Supervisor:**
        Dr. Stenford Ruvinga

        **Module:** 7005SCN

        **Academic Year:** 2025/2026

        **Ethics Approval:** P194996

        ---

        ### Technologies Used

        Python 3.12
        XGBoost
        HuggingFace Transformers
        VADER Sentiment
        SHAP
        Scikit-learn
        Imbalanced-learn
        Streamlit
        Pandas
        NumPy
        Matplotlib
        """)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>Data Source:</strong> Consumer Financial Protection Bureau (CFPB) Consumer
    Complaint Database. Publicly available at
    <a href="https://www.consumerfinance.gov/data-research/consumer-complaints/"
    target="_blank">consumerfinance.gov</a>.
    All data is anonymised and publicly available with no personal identifiers.
    Ethics approval reference: P194996.
    </div>
    """, unsafe_allow_html=True)

