# ============================================================
# FREIGHT PREDICTION SYSTEM — STREAMLIT FRONTEND
# Run: streamlit run app.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap, joblib, json, os, warnings
warnings.filterwarnings("ignore")
from config import (TARGET_COL, DISTANCE_COL, APP_TITLE,
                    APP_CURRENCY, CORRELATION_THRESHOLD,
                    BLACKLIST_FEATURES, DATASET_FILENAME)

st.set_page_config(page_title=APP_TITLE, page_icon="🚚",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.metric-card {
    background: white; padding: 1.2rem; border-radius: 10px;
    border-left: 4px solid #1F4E79;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1rem;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #1F4E79; }
.metric-label { font-size: 0.85rem; color: #666;
                text-transform: uppercase; letter-spacing: 0.04em; }
.section-header {
    font-size: 1.2rem; font-weight: 600; color: #1F4E79;
    border-bottom: 2px solid #1F4E79;
    padding-bottom: 0.3rem; margin-bottom: 1rem;
}
.prediction-box {
    background: linear-gradient(135deg, #1F4E79, #2E86AB);
    color: white; padding: 2rem; border-radius: 15px;
    text-align: center; margin: 1rem 0;
}
.prediction-price { font-size: 3rem; font-weight: 800; }
.input-section {
    background: #f8f9fa; padding: 1rem;
    border-radius: 8px; margin-bottom: 0.8rem;
    border: 1px solid #e9ecef;
}
.input-section-title {
    font-size: 0.8rem; font-weight: 700;
    color: #1F4E79; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 🚚 {APP_TITLE}")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 EDA",
        "🔬 Feature Selection",
        "📊 Model Comparison",
        "🤖 Model Results",
        "🔍 SHAP Analysis",
        "💡 Predict Price",
    ])
    st.markdown("---")
    st.markdown(f"📁 `{DATASET_FILENAME}`")
    st.markdown(f"🎯 Target: `{TARGET_COL}`")
    st.markdown(f"💱 Currency: `{APP_CURRENCY}`")

# ── Helpers ────────────────────────────────────────────────
@st.cache_data
def load_processed():
    return pd.read_csv("data/processed_data.csv", low_memory=False)

@st.cache_data
def load_raw():
    return pd.read_csv(f"data/{DATASET_FILENAME}", low_memory=False)

@st.cache_resource
def load_model():
    return joblib.load("models/freight_model.pkl")

def load_json(path):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {}

def get_base_value(explainer):
    raw = explainer.expected_value
    return float(raw[0]) if hasattr(raw, '__len__') else float(raw)

# Columns that are binary — should be Yes/No not sliders
BINARY_COLS = {
    "IS_WEEKEND", "IS_FESTIVAL_MONTH", "IS_MONTH_END",
    "SAME_REGION", "SAME_STATE", "IS_EXPRESS",
    "RETURN_LOAD_AVAILABLE", "GPS_TRACKING",
}

# Mean-encoded columns and their source categorical column
MEAN_ENCODED_SOURCES = {
    "VEHICLE_TYPE_MEAN_PRICE_PER_KM"  : "VEHICLE_TYPE",
    "SERVICE_TYPE_MEAN_PRICE_PER_KM"  : "SERVICE_TYPE",
    "LOAD_TYPE_MEAN_PRICE_PER_KM"     : "LOAD_TYPE",
    "ORIGIN_CITY_MEAN_PRICE_PER_KM"   : "ORIGIN_CITY",
    "ORIGIN_STATE_MEAN_PRICE_PER_KM"  : "ORIGIN_STATE",
    "ORIGIN_REGION_MEAN_PRICE_PER_KM" : "ORIGIN_REGION",
    "DEST_CITY_MEAN_PRICE_PER_KM"     : "DEST_CITY",
    "DEST_STATE_MEAN_PRICE_PER_KM"    : "DEST_STATE",
    "DEST_REGION_MEAN_PRICE_PER_KM"   : "DEST_REGION",
    "CARRIER_ID_MEAN_PRICE_PER_KM"    : "CARRIER_ID",
    "RELATION_MEAN_PRICE_PER_KM"      : None,  # computed from combination
}

# ══════════════════════════════════════════════════════════
# PAGE 1 — EDA
# ══════════════════════════════════════════════════════════
if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown(f"Dataset: **{DATASET_FILENAME}** | Target: **{TARGET_COL}**")

    df_raw = load_raw()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{len(df_raw):,}",                          "Total Records"),
        (c2, str(df_raw.shape[1]),                        "Raw Columns"),
        (c3, f"{APP_CURRENCY}{df_raw[TARGET_COL].mean():,.0f}", f"Mean Price"),
        (c4, f"{df_raw[TARGET_COL].skew():.2f}",          "Price Skewness"),
    ]:
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Price Distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(df_raw[TARGET_COL].min(),
                           df_raw[TARGET_COL].max(), 60)
        ax.hist(df_raw[TARGET_COL].dropna(), bins=bins,
                color='#2E86AB', alpha=0.85, edgecolor='white')
        ax.axvline(df_raw[TARGET_COL].mean(),   color='#E53935',
                   linestyle='--', linewidth=2,
                   label=f"Mean {APP_CURRENCY}{df_raw[TARGET_COL].mean():,.0f}")
        ax.axvline(df_raw[TARGET_COL].median(), color='#43A047',
                   linestyle='--', linewidth=2,
                   label=f"Median {APP_CURRENCY}{df_raw[TARGET_COL].median():,.0f}")
        ax.axvline(df_raw[TARGET_COL].quantile(.95), color='#7B1FA2',
                   linestyle='--', linewidth=2,
                   label=f"95th pct {APP_CURRENCY}{df_raw[TARGET_COL].quantile(.95):,.0f}")
        ax.set_xlabel(TARGET_COL); ax.set_ylabel("Frequency")
        ax.legend(fontsize=9); ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.markdown('<div class="section-header">Price Statistics</div>',
                    unsafe_allow_html=True)
        s = df_raw[TARGET_COL].describe()
        st.dataframe(pd.DataFrame({
            "Statistic": ["Count","Mean","Std Dev","Min","25th pct",
                          "Median","75th pct","Max","Skewness"],
            f"Value": [
                f"{int(s['count']):,}",
                f"{APP_CURRENCY}{s['mean']:,.2f}",
                f"{APP_CURRENCY}{s['std']:,.2f}",
                f"{APP_CURRENCY}{s['min']:,.2f}",
                f"{APP_CURRENCY}{s['25%']:,.2f}",
                f"{APP_CURRENCY}{s['50%']:,.2f}",
                f"{APP_CURRENCY}{s['75%']:,.2f}",
                f"{APP_CURRENCY}{s['max']:,.2f}",
                f"{df_raw[TARGET_COL].skew():.3f}",
            ]
        }), use_container_width=True, hide_index=True)

        st.markdown("---")
        missing = df_raw.isnull().sum()
        if missing.sum() == 0:
            st.success("✅ No missing values in this dataset.")
        else:
            m = missing[missing>0].reset_index()
            m.columns = ["Column","Missing Count"]
            m["Missing %"] = (m["Missing Count"]/len(df_raw)*100).round(2)
            st.dataframe(m, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Top Feature Correlations with Target</div>',
                unsafe_allow_html=True)
    df_proc = load_processed()
    num = df_proc.select_dtypes(include=[np.number])
    corr = num.corr()[TARGET_COL].abs().drop(TARGET_COL, errors='ignore'
           ).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#1F4E79' if v>0.3 else '#2E86AB' if v>0.1 else '#90CAF9'
              for v in corr.values]
    ax.bar(corr.index, corr.values, color=colors, edgecolor='white')
    ax.axhline(CORRELATION_THRESHOLD, color='red', linestyle='--',
               alpha=0.6, label=f'Threshold {CORRELATION_THRESHOLD}')
    ax.set_ylabel(f"Abs Correlation with {TARGET_COL}")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">Raw Data Preview</div>',
                unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True)


# ══════════════════════════════════════════════════════════
# PAGE 2 — FEATURE SELECTION
# ══════════════════════════════════════════════════════════
elif page == "🔬 Feature Selection":
    st.title("🔬 Feature Selection")
    st.markdown("Two-stage pipeline: correlation filter → Genetic Algorithm.")

    df_proc = load_processed()
    sel_path = "data/selected_features.csv"
    freq_path = "data/feature_selection_frequency.csv"
    feat_path = "data/best_features.json"

    num_after_eng  = df_proc.shape[1]
    num_after_corr = len(pd.read_csv(sel_path)) if os.path.exists(sel_path) else "?"
    final_feats    = json.load(open(feat_path)) if os.path.exists(feat_path) else []
    num_final      = len(final_feats) if final_feats else "Run Step 7"

    c1, c2, c3 = st.columns(3)
    for col, val, label in [
        (c1, num_after_eng,  "After Feature Engineering"),
        (c2, num_after_corr, f"After Correlation Filter (>{CORRELATION_THRESHOLD})"),
        (c3, num_final,      "Final GA-Selected Features"),
    ]:
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(f'<div class="section-header">Stage 1 — Correlation Filter</div>',
                    unsafe_allow_html=True)
        num = df_proc.select_dtypes(include=[np.number])
        corr = num.corr()[TARGET_COL].abs().drop(TARGET_COL, errors='ignore'
               ).sort_values(ascending=False)
        filtered = corr[corr > CORRELATION_THRESHOLD].head(20)
        fig, ax = plt.subplots(figsize=(6, 8))
        colors = ['#1F4E79' if v>0.3 else '#2E86AB' if v>0.1 else '#64B5F6'
                  for v in filtered.values]
        ax.barh(filtered.index[::-1], filtered.values[::-1], color=colors[::-1])
        ax.axvline(CORRELATION_THRESHOLD, color='red', linestyle='--',
                   alpha=0.6, label=f'Threshold {CORRELATION_THRESHOLD}')
        ax.set_xlabel(f"Abs Correlation with {TARGET_COL}")
        ax.legend(); ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        if os.path.exists(freq_path):
            st.markdown('<div class="section-header">Stage 2 — GA Feature Frequency</div>',
                        unsafe_allow_html=True)
            freq_df = pd.read_csv(freq_path).sort_values(
                'Times_Selected', ascending=False)
            max_f = freq_df['Times_Selected'].max()
            fig, ax = plt.subplots(figsize=(6, 8))
            colors = ['#1F4E79' if v==max_f else '#2E86AB' if v>=max_f-1
                      else '#90CAF9' for v in freq_df['Times_Selected']]
            ax.barh(freq_df['Feature'][::-1],
                    freq_df['Times_Selected'][::-1], color=colors[::-1])
            ax.axvline(max_f, color='#E53935', linestyle='--',
                       alpha=0.6, label='All folds')
            ax.set_xlabel("Times Selected (out of 5 folds)")
            ax.legend(); ax.grid(axis='x', linestyle='--', alpha=0.4)
            ax.spines[['top','right']].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("Run step6_feature_selection_ga.py to see GA results.")

    if final_feats:
        st.markdown("---")
        st.markdown('<div class="section-header">Final Selected Features</div>',
                    unsafe_allow_html=True)
        corr_all = num.corr()[TARGET_COL].abs()
        cols = st.columns(min(len(final_feats), 4))
        for i, feat in enumerate(final_feats):
            corr_val = corr_all.get(feat, 0)
            with cols[i % 4]:
                st.markdown(f"""<div class="metric-card" style="min-height:90px;">
                    <div style="font-weight:700;color:#1F4E79;font-size:0.75rem;
                                margin-bottom:0.4rem;">{feat}</div>
                    <div class="metric-value" style="font-size:1.2rem;">
                        r={corr_val:.3f}</div>
                </div>""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("Baseline (distance only) vs XGBoost (GA-tuned features + hyperparameters).")

    comp = load_json("data/model_comparison_results.json")

    if not comp:
        st.info("Run step8_model_comparison.py first.")
    else:
        base  = comp["baseline"]
        xgb   = comp["xgboost"]

        # ── Metric cards ──────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""<div class="metric-card">
            <div class="metric-value">{comp["mape_improvement"]}%</div>
            <div class="metric-label">MAPE Improvement over Baseline</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card">
            <div class="metric-value">{xgb["mape_mean"]:.2f}%</div>
            <div class="metric-label">XGBoost MAPE</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card">
            <div class="metric-value">{APP_CURRENCY}{xgb["mae_mean"]:,.0f}</div>
            <div class="metric-label">XGBoost Mean Absolute Error</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Summary table ─────────────────────────────────
        st.markdown('<div class="section-header">Metrics Comparison Table</div>',
                    unsafe_allow_html=True)
        table_df = pd.DataFrame({
            "Model"    : ["Baseline (Linear Regression)", "XGBoost (GA tuned)"],
            "MAE"      : [f"{APP_CURRENCY}{base['mae_mean']:,.0f}",
                           f"{APP_CURRENCY}{xgb['mae_mean']:,.0f}"],
            "RMSE"     : [f"{APP_CURRENCY}{base['rmse_mean']:,.0f}",
                           f"{APP_CURRENCY}{xgb['rmse_mean']:,.0f}"],
            "MAPE (%)" : [f"{base['mape_mean']:.2f}%",
                           f"{xgb['mape_mean']:.2f}%"],
            "Winner"   : ["❌", "✅"],
        })
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="section-header">Side-by-Side Metrics</div>',
                        unsafe_allow_html=True)
            metrics = ["MAE", "RMSE", "MAPE (%)"]
            base_v  = [base["mae_mean"], base["rmse_mean"], base["mape_mean"]]
            xgb_v   = [xgb["mae_mean"],  xgb["rmse_mean"],  xgb["mape_mean"]]
            x = np.arange(len(metrics))
            fig, ax = plt.subplots(figsize=(6, 4))
            b1 = ax.bar(x - 0.2, base_v, 0.35, label="Baseline",
                        color="#90CAF9", edgecolor="white")
            b2 = ax.bar(x + 0.2, xgb_v,  0.35, label="XGBoost",
                        color="#1F4E79", edgecolor="white")
            ax.set_xticks(x); ax.set_xticklabels(metrics)
            ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines[["top","right"]].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with col_r:
            st.markdown('<div class="section-header">Per-Fold MAPE</div>',
                        unsafe_allow_html=True)
            fold_labels = [f"Fold {i+1}" for i in range(len(base["fold_mape"]))]
            x2 = np.arange(len(fold_labels))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(x2 - 0.2, base["fold_mape"], 0.35, label="Baseline",
                   color="#90CAF9", edgecolor="white")
            ax.bar(x2 + 0.2, xgb["fold_mape"],  0.35, label="XGBoost",
                   color="#1F4E79", edgecolor="white")
            ax.set_xticks(x2); ax.set_xticklabels(fold_labels)
            ax.set_ylabel("MAPE (%)")
            ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines[["top","right"]].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.success(f"✅ **XGBoost selected as final model** — {comp['mape_improvement']}% lower MAPE and {comp['mae_improvement']}% lower MAE than baseline.")

# ══════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ══════════════════════════════════════════════════════════
elif page == "🤖 Model Results":
    st.title("🤖 Model Results")
    st.markdown("Ablation study — baseline (distance only) vs full tuned model.")

    ablation    = load_json("data/model_comparison_results.json")
    best_params = load_json("data/best_hyperparameters.json")

    if ablation:
        c1, c2, c3 = st.columns(3)
        for col, val, label in [
            (c1, f"{ablation.get('baseline',{}).get('mape_mean','?')}%", "Baseline MAPE"),
            (c2, f"{ablation.get('xgboost',{}).get('mape_mean','?')}%","Full Model MAPE"),
            (c3, f"{ablation.get('mape_improvement','?')}%",     "Error Reduction"),
        ]:
            col.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="section-header">Baseline vs XGBoost — MAPE</div>',
                        unsafe_allow_html=True)
            base_mape    = ablation.get('baseline', {}).get('mape_mean', 0)
            xgb_mape     = ablation.get('xgboost',  {}).get('mape_mean', 0)
            baseline_col = ablation.get('baseline', {}).get('baseline_col', DISTANCE_COL)
            labels = [f"Baseline\n({baseline_col})", "XGBoost\n(GA + tuned)"]
            values = [base_mape, xgb_mape]
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, values,
                          color=['#90CAF9','#1F4E79'],
                          width=0.5, edgecolor='white')
            for bar, val in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.3,
                        f"{val:.2f}%", ha='center',
                        fontsize=13, fontweight='bold')
            ax.set_ylabel("MAPE (%)"); ax.set_ylim(0, max(values)*1.25)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.spines[['top','right']].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with col_r:
            base_folds = ablation.get('baseline', {}).get('fold_mape', [])
            xgb_folds  = ablation.get('xgboost',  {}).get('fold_mape', [])
            if base_folds and xgb_folds:
                st.markdown('<div class="section-header">Per-Fold MAPE</div>',
                            unsafe_allow_html=True)
                fold_labels = [f"Fold {i+1}" for i in range(len(base_folds))]
                x = np.arange(len(fold_labels))
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(x-0.2, base_folds, 0.35,
                       label='Baseline', color='#90CAF9', edgecolor='white')
                ax.bar(x+0.2, xgb_folds,  0.35,
                       label='XGBoost',  color='#1F4E79', edgecolor='white')
                ax.set_xticks(x); ax.set_xticklabels(fold_labels)
                ax.set_ylabel("MAPE (%)")
                ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.4)
                ax.spines[['top','right']].set_visible(False)
                fig.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("Run step8_model_comparison.py first.")

    if best_params:
        st.markdown("---")
        st.markdown('<div class="section-header">Best Hyperparameters (GA Tuned)</div>',
                    unsafe_allow_html=True)
        hp_info = {
            "n_estimators"     : "Number of decision trees",
            "learning_rate"    : "Step size per tree (smaller = more careful)",
            "max_depth"        : "Maximum depth each tree can grow",
            "subsample"        : "Fraction of data each tree sees",
            "min_samples_split": "Minimum rows needed to split a node",
        }
        st.dataframe(pd.DataFrame([
            {"Parameter": k, "Value": v,
             "What it controls": hp_info.get(k,"")}
            for k, v in best_params.items() if k != "random_state"
        ]), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# PAGE 4 — SHAP ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "🔍 SHAP Analysis":
    st.title("🔍 SHAP Analysis")
    st.markdown("Understanding what the model learned — globally and per shipment.")

    feat_path = "data/best_features.json"
    if not os.path.exists(feat_path):
        st.warning("Run step7_hyperparameter_tuning.py first to generate best_features.json")
        st.stop()

    FEATURES = json.load(open(feat_path))
    df_proc  = load_processed()
    model    = load_model()
    X_sample = df_proc[FEATURES].sample(300, random_state=42)

    with st.spinner("Calculating SHAP values..."):
        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer.shap_values(X_sample)
        base_value  = get_base_value(explainer)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Global Feature Importance</div>',
                    unsafe_allow_html=True)
        mean_shap = np.abs(shap_vals).mean(axis=0)
        order = np.argsort(mean_shap)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh([FEATURES[i] for i in order], mean_shap[order],
                color='#1F4E79')
        ax.set_xlabel(f"Mean |SHAP| (avg {APP_CURRENCY} impact on price)")
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.caption("Higher bar = more influence on predicted price across all shipments.")

    with col2:
        st.markdown('<div class="section-header">SHAP Beeswarm</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        shap.summary_plot(shap_vals, X_sample, show=False, plot_size=None)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.caption("Red = high feature value. Right = pushed price UP. Left = pushed price DOWN.")

    st.markdown("---")
    st.markdown('<div class="section-header">Single Shipment Explanation</div>',
                unsafe_allow_html=True)

    idx  = st.slider("Pick a shipment to explain", 0, len(X_sample)-1, 0)
    row  = X_sample.iloc[[idx]]
    pred = model.predict(row)[0]
    sv   = shap_vals[idx]

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown(f"""<div class="prediction-box">
            <div style="font-size:0.9rem;opacity:0.85">Predicted Price</div>
            <div class="prediction-price">{APP_CURRENCY}{pred:,.0f}</div>
            <div style="font-size:0.8rem;opacity:0.75;margin-top:0.5rem">
                Base avg: {APP_CURRENCY}{base_value:,.0f}
            </div>
        </div>""", unsafe_allow_html=True)
        for feat in FEATURES:
            st.markdown(f"- **{feat}**: `{row[feat].values[0]:.2f}`")

    with col_b:
        sorted_idx = np.argsort(np.abs(sv))[::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        feats  = [FEATURES[i] for i in sorted_idx]
        values = [sv[i] for i in sorted_idx]
        ax.barh(feats[::-1], values[::-1],
                color=['#E53935' if v>0 else '#1E88E5' for v in values[::-1]])
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel(f"SHAP Value ({APP_CURRENCY} contribution)")
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        running = base_value
        st.markdown(f"**Base average:** {APP_CURRENCY}{base_value:,.0f}")
        for i in sorted_idx:
            running += sv[i]
            st.markdown(
                f"{'🔴 ↑' if sv[i]>0 else '🔵 ↓'} **{FEATURES[i]}**: "
                f"`{APP_CURRENCY}{sv[i]:+,.0f}` → `{APP_CURRENCY}{running:,.0f}`"
            )
        st.markdown(f"**→ Final predicted: {APP_CURRENCY}{pred:,.0f}**")


# ══════════════════════════════════════════════════════════
# PAGE 5 — PREDICT
# ══════════════════════════════════════════════════════════
elif page == "💡 Predict Price":
    st.title("💡 Predict Freight Price")
    st.markdown("Fill in shipment details and get an instant price prediction with explanation.")

    feat_path = "data/best_features.json"
    if not os.path.exists(feat_path):
        st.warning("Run the full pipeline first (steps 1-10) before predicting.")
        st.stop()

    FEATURES = json.load(open(feat_path))
    model    = load_model()
    df_proc  = load_processed()
    df_raw   = load_raw()

    # Build lookup: category → mean price per km (from processed data)
    # This lets us convert user's dropdown choice to the encoded value
    def get_mean_lookup(source_col, mean_col):
        if source_col in df_raw.columns and mean_col in df_proc.columns:
            tmp = df_raw[[source_col]].copy()
            tmp[mean_col] = df_proc[mean_col].values
            return tmp.groupby(source_col)[mean_col].mean().to_dict()
        return {}

    col_means = df_proc[FEATURES].mean().to_dict()

    # ── Input Form ─────────────────────────────────────────
    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        st.markdown('<div class="section-header">Shipment Details</div>',
                    unsafe_allow_html=True)

        user_values = {}   # stores final numeric values to feed the model
        category_choices = {}  # stores user's dropdown selections

        for feat in FEATURES:

            # ── BINARY — Yes / No ───────────────────────────
            if feat in BINARY_COLS:
                label_map = {
                    "IS_FESTIVAL_MONTH"   : "🎉 Festival / Peak Season Month?",
                    "IS_WEEKEND"          : "📅 Is it a Weekend?",
                    "IS_MONTH_END"        : "📆 Is it Month End?",
                    "IS_EXPRESS"          : "⚡ Express Delivery?",
                    "RETURN_LOAD_AVAILABLE": "↩️ Return Load Available?",
                    "GPS_TRACKING"        : "📍 GPS Tracking Enabled?",
                    "SAME_REGION"         : "📍 Same Region (Origin = Destination)?",
                    "SAME_STATE"          : "📍 Same State (Origin = Destination)?",
                }
                display = label_map.get(feat, feat)
                choice = st.selectbox(display, ["No", "Yes"], key=feat)
                user_values[feat] = 1 if choice == "Yes" else 0

            # ── MEAN ENCODED — show dropdown, auto-compute ──
            elif feat in MEAN_ENCODED_SOURCES:
                source_col = MEAN_ENCODED_SOURCES[feat]

                if feat == "RELATION_MEAN_PRICE_PER_KM":
                    # RELATION is built from origin city + dest city + vehicle
                    # We compute it later after all three are known
                    user_values[feat] = col_means[feat]  # fallback default

                elif source_col and source_col in df_raw.columns:
                    label_map = {
                        "VEHICLE_TYPE" : "🚛 Vehicle Type",
                        "SERVICE_TYPE" : "📦 Service Type",
                        "LOAD_TYPE"    : "🏷️ Load / Cargo Type",
                        "ORIGIN_CITY"  : "📍 Origin City",
                        "ORIGIN_STATE" : "🗺️ Origin State",
                        "ORIGIN_REGION": "🌐 Origin Region",
                        "DEST_CITY"    : "📍 Destination City",
                        "DEST_STATE"   : "🗺️ Destination State",
                        "DEST_REGION"  : "🌐 Destination Region",
                        "CARRIER_ID"   : "🏢 Carrier",
                    }
                    display = label_map.get(source_col, source_col)
                    options = sorted(df_raw[source_col].dropna().unique().tolist())
                    choice  = st.selectbox(display, options, key=feat)
                    category_choices[source_col] = choice
                    # Look up the mean price per km for this choice
                    lookup = get_mean_lookup(source_col, feat)
                    user_values[feat] = lookup.get(choice, col_means[feat])
                else:
                    user_values[feat] = col_means[feat]

            # ── NUMERIC — number input ──────────────────────
            else:
                label_map = {
                    "ROAD_DISTANCE_KM"    : "🛣️ Road Distance (km)",
                    "STRAIGHT_DISTANCE_KM": "📏 Straight-line Distance (km)",
                    "WEIGHT_KG"           : "⚖️ Cargo Weight (kg)",
                    "VOLUME_CBM"          : "📦 Volume (cubic metres)",
                    "NUM_PALLETS"         : "🪵 Number of Pallets",
                    "VEHICLE_CAPACITY_KG" : "🏋️ Vehicle Capacity (kg)",
                    "UTILIZATION_PCT"     : "📊 Load Utilization (%)",
                    "CARRIER_RATING"      : "⭐ Carrier Rating (1-5)",
                    "DEST_LON"            : "🌐 Destination Longitude",
                    "ORIGIN_LON"          : "🌐 Origin Longitude",
                    "TRANSIT_DAYS"        : "📅 Transit Days",
                    "LOAD_DURATION_HOURS" : "⏱️ Load Duration (hours)",
                    "DIESEL_PRICE_PER_LITRE": "⛽ Diesel Price (₹/litre)",
                }
                display  = label_map.get(feat, feat)
                min_val  = float(df_proc[feat].min())
                max_val  = float(df_proc[feat].max())
                mean_val = float(df_proc[feat].mean())

                # Use step that makes sense for the range
                rng = max_val - min_val
                step = max(rng / 200, 0.01)

                user_values[feat] = st.number_input(
                    display,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=step,
                    format="%.2f" if step < 1 else "%.0f",
                    help=f"Range: {min_val:.1f} – {max_val:.1f} | Dataset avg: {mean_val:.1f}",
                    key=feat,
                )

        # Auto-compute RELATION_MEAN if we have the components
        if "RELATION_MEAN_PRICE_PER_KM" in FEATURES:
            origin_city = category_choices.get("ORIGIN_CITY", "")
            dest_city   = category_choices.get("DEST_CITY", "")
            vehicle     = category_choices.get("VEHICLE_TYPE", "")
            if origin_city and dest_city and vehicle:
                relation_key = f"{origin_city}_{dest_city}_{vehicle}"
                lookup = get_mean_lookup("RELATION", "RELATION_MEAN_PRICE_PER_KM")
                if not lookup:
                    # Build it from processed data
                    tmp = df_raw[["ORIGIN_CITY","DEST_CITY","VEHICLE_TYPE"]].copy()
                    tmp["RELATION"] = (tmp["ORIGIN_CITY"] + "_" +
                                       tmp["DEST_CITY"] + "_" +
                                       tmp["VEHICLE_TYPE"])
                    tmp["RELATION_MEAN_PRICE_PER_KM"] = df_proc["RELATION_MEAN_PRICE_PER_KM"].values
                    lookup = tmp.groupby("RELATION")["RELATION_MEAN_PRICE_PER_KM"].mean().to_dict()
                user_values["RELATION_MEAN_PRICE_PER_KM"] = lookup.get(
                    relation_key, col_means.get("RELATION_MEAN_PRICE_PER_KM", 0))

        predict_btn = st.button("🚀 Predict Price",
                                use_container_width=True, type="primary")

    # ── Result Panel ───────────────────────────────────────
    with col_result:
        st.markdown('<div class="section-header">Prediction Result</div>',
                    unsafe_allow_html=True)

        if predict_btn:
            input_df   = pd.DataFrame([user_values])[FEATURES]
            prediction = model.predict(input_df)[0]

            explainer      = shap.TreeExplainer(model)
            shap_values    = explainer.shap_values(input_df)
            base_value     = get_base_value(explainer)
            shap_row       = shap_values[0]

            # Price display
            st.markdown(f"""<div class="prediction-box">
                <div style="font-size:0.9rem;opacity:0.85;margin-bottom:0.3rem">
                    Estimated Freight Price
                </div>
                <div class="prediction-price">{APP_CURRENCY}{prediction:,.0f}</div>
                <div style="font-size:0.85rem;opacity:0.75;margin-top:0.5rem">
                    Dataset average: {APP_CURRENCY}{base_value:,.0f}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Why this price? (SHAP Explanation)**")

            sorted_idx = np.argsort(np.abs(shap_row))[::-1]

            # SHAP bar chart
            feats  = [FEATURES[i] for i in sorted_idx]
            values = [shap_row[i] for i in sorted_idx]

            fig, ax = plt.subplots(figsize=(6, max(3.5, len(FEATURES)*0.4)))
            ax.barh(feats[::-1], values[::-1],
                    color=['#E53935' if v>0 else '#1E88E5'
                           for v in values[::-1]])
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_xlabel(f"SHAP Value ({APP_CURRENCY} contribution to price)")
            ax.set_title("🔴 Red = pushed price UP  |  🔵 Blue = pushed price DOWN",
                         fontsize=9, pad=8)
            ax.grid(axis='x', linestyle='--', alpha=0.4)
            ax.spines[['top','right']].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

            # Text breakdown
            st.markdown(f"**Base average price:** {APP_CURRENCY}{base_value:,.0f}")
            for i in sorted_idx:
                feat = FEATURES[i]; sv = shap_row[i]
                icon  = "🔴" if sv > 0 else "🔵"
                arrow = "↑" if sv > 0 else "↓"
                st.markdown(
                    f"{icon} **{feat}** {arrow} `{APP_CURRENCY}{sv:+,.0f}`"
                )
            st.markdown(f"**→ Predicted price: {APP_CURRENCY}{prediction:,.0f}**")

        else:
            st.markdown("""
            <div style="padding:3rem;text-align:center;color:#aaa;
                        border:2px dashed #ddd;border-radius:12px;
                        margin-top:2rem;">
                <div style="font-size:3.5rem">🚚</div>
                <div style="margin-top:1rem;font-size:1rem;color:#666">
                    Fill in the shipment details<br>
                    and click <strong>Predict Price</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)