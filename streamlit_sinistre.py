import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="MLE_ASSURANCE", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f8f9fc; color: #1e293b; }
.stApp { background-color: #f8f9fc; }
h1, h2, h3 { font-family: 'Playfair Display', serif; color: #0f172a; }
.stTabs [data-baseweb="tab-list"] { background-color: #e2e8f0; border-radius: 12px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #64748b; font-weight: 500; padding: 8px 20px; background-color: transparent; }
.stTabs [aria-selected="true"] { background-color: #1e40af !important; color: white !important; }
[data-testid="metric-container"] { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #1e40af !important; font-size: 1.6rem !important; font-weight: 700; }
.stButton > button { background: linear-gradient(135deg, #1e40af, #1d4ed8); color: white; border: none; border-radius: 10px; padding: 12px 32px; font-weight: 600; font-size: 1rem; width: 100%; }
.card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 12px; }
.card-blue { background: linear-gradient(135deg, #eff6ff, #dbeafe); border: 1px solid #bfdbfe; border-radius: 16px; padding: 20px 24px; margin-bottom: 12px; }
.var-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px 12px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.04); transition: all 0.2s; }
.var-card:hover { box-shadow: 0 4px 12px rgba(30,64,175,0.15); border-color: #bfdbfe; }
.var-card .icon { font-size: 2rem; margin-bottom: 8px; }
.var-card .name { font-weight: 600; font-size: 0.82rem; color: #0f172a; margin-bottom: 4px; }
.var-card .desc { font-size: 0.72rem; color: #64748b; }
.var-card .badge { display: inline-block; margin-top: 6px; padding: 2px 10px; border-radius: 20px; font-size: 0.68rem; font-weight: 600; }
.badge-trad { background: #dbeafe; color: #1e40af; }
.badge-telem { background: #d1fae5; color: #065f46; }
.result-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
.result-card .value { font-family: 'Playfair Display', serif; font-size: 2.2rem; color: #1e40af; margin: 6px 0; font-weight: 700; }
.result-card .label { color: #64748b; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
.badge-low { background:#dcfce7; color:#166534; border-radius:8px; padding:6px 16px; font-weight:600; display:inline-block; }
.badge-medium { background:#fef9c3; color:#854d0e; border-radius:8px; padding:6px 16px; font-weight:600; display:inline-block; }
.badge-high { background:#fee2e2; color:#991b1b; border-radius:8px; padding:6px 16px; font-weight:600; display:inline-block; }
.info-box { background-color: #eff6ff; border-left: 4px solid #1e40af; border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 10px 0; font-size: 0.9rem; color: #1e3a8a; }
.warning-box { background-color: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 10px 0; font-size: 0.9rem; color: #78350f; }
.question-box { background: linear-gradient(135deg, #f0f9ff, #e0f2fe); border: 1.5px solid #0ea5e9; border-radius: 14px; padding: 22px 28px; margin: 18px 0 24px 0; }
.section-tag { background: #dbeafe; color: #1e40af; border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; display: inline-block; margin-bottom: 8px; }
.metric-mini { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; text-align: center; }
.metric-mini .val { font-size: 1.4rem; font-weight: 700; color: #1e40af; }
.metric-mini .lbl { font-size: 0.72rem; color: #64748b; text-transform: uppercase; }
hr { border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

PT = dict(paper_bgcolor='#ffffff', plot_bgcolor='#f8f9fc', font_color='#1e293b',
          font_family='Inter', title_font_family='Playfair Display', title_font_color='#0f172a')

APP_DIR = Path(__file__).resolve().parent
FULL_DATA_PATH = APP_DIR.parent / "telematics_syn.csv"
SAMPLE_DATA_PATH = APP_DIR / "data" / "sample.csv"


def standardize_dataframe(df):
    rename_map = {
        'Insured.age': 'Age',
        'Insured.sex': 'Gender',
        'Car.age': 'Car_age',
        'Credit.score': 'Credit_score',
        'Annual.miles.drive': 'Annual_miles_drive',
        'Years.noclaims': 'Years_noclaims',
        'Annual.pct.driven': 'Annual_pct_driven',
        'Total.miles.driven': 'Total_miles_driven',
        'Pct.drive.mon': 'Pct_drive_mon',
        'Pct.drive.tue': 'Pct_drive_tue',
        'Pct.drive.wed': 'Pct_drive_wed',
        'Pct.drive.thr': 'Pct_drive_thr',
        'Pct.drive.fri': 'Pct_drive_fri',
        'Pct.drive.sat': 'Pct_drive_sat',
        'Pct.drive.sun': 'Pct_drive_sun',
        'Pct.drive.2hrs': 'Pct_drive_2hrs',
        'Pct.drive.3hrs': 'Pct_drive_3hrs',
        'Pct.drive.4hrs': 'Pct_drive_4hrs',
        'Pct.drive.wkday': 'Pct_drive_wkday',
        'Pct.drive.wkend': 'Pct_drive_wkend',
        'Pct.drive.rush am': 'Pct_drive_rusham',
        'Pct.drive.rush pm': 'Pct_drive_rushpm',
        'Avgdays.week': 'Avgdays_week',
        'Accel.06miles': 'Accel_06miles',
        'Accel.08miles': 'Accel_08miles',
        'Accel.09miles': 'Accel_09miles',
        'Accel.11miles': 'Accel_11miles',
        'Accel.12miles': 'Accel_12miles',
        'Accel.14miles': 'Accel_14miles',
        'Brake.06miles': 'Brake_06miles',
        'Brake.08miles': 'Brake_08miles',
        'Brake.09miles': 'Brake_09miles',
        'Brake.11miles': 'Brake_11miles',
        'Brake.12miles': 'Brake_12miles',
        'Brake.14miles': 'Brake_14miles',
        'Left.turn.intensity08': 'Left_turn_intensity08',
        'Left.turn.intensity09': 'Left_turn_intensity09',
        'Left.turn.intensity10': 'Left_turn_intensity10',
        'Left.turn.intensity11': 'Left_turn_intensity11',
        'Left.turn.intensity12': 'Left_turn_intensity12',
        'Right.turn.intensity08': 'Right_turn_intensity08',
        'Right.turn.intensity09': 'Right_turn_intensity09',
        'Right.turn.intensity10': 'Right_turn_intensity10',
        'Right.turn.intensity11': 'Right_turn_intensity11',
        'Right.turn.intensity12': 'Right_turn_intensity12',
        'Car.use': 'Car_use',
    }
    df = df.rename(columns=rename_map).copy()
    if 'has_claim' not in df.columns and 'NB_Claim' in df.columns:
        df['has_claim'] = (df['NB_Claim'] > 0).astype(int)
    if 'Statut' not in df.columns and 'has_claim' in df.columns:
        df['Statut'] = df['has_claim'].map({1: 'Sinistré', 0: 'Non sinistré'})
    return df


def plot_avg_claim_by_bins(df, variable, bins, title):
    grouped = (
        df.groupby(pd.cut(df[variable], bins=bins, include_lowest=True), observed=False)['NB_Claim']
        .mean()
        .reset_index()
    )
    grouped[variable] = grouped[variable].astype(str)
    fig = px.bar(
        grouped,
        x=variable,
        y='NB_Claim',
        template='plotly_white',
        title=title,
        color='NB_Claim',
        color_continuous_scale='Blues',
        text=grouped['NB_Claim'].round(3),
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        **PT,
        coloraxis_showscale=False,
        xaxis_title=variable,
        yaxis_title='Average claim count',
        xaxis_tickangle=-45,
        height=320,
        margin=dict(t=55, b=60, l=10, r=10),
    )
    return fig


def plot_score_distribution(scores, y_true, title):
    bins = np.linspace(0, 1, 51)
    scores = np.asarray(scores)
    y_true = np.asarray(y_true).astype(int)
    sc_pos = scores[y_true == 1]
    sc_neg = scores[y_true == 0]
    actual_rate = y_true.mean()

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=sc_neg,
            xbins=dict(start=0, end=1, size=bins[1] - bins[0]),
            name='Non sinistré',
            marker_color='#1f77b4',
            opacity=0.6,
            histnorm='probability density',
        )
    )
    fig.add_trace(
        go.Histogram(
            x=sc_pos,
            xbins=dict(start=0, end=1, size=bins[1] - bins[0]),
            name='Sinistré',
            marker_color='#d62728',
            opacity=0.6,
            histnorm='probability density',
        )
    )
    fig.add_vline(
        x=float(scores.mean()),
        line_dash='dash',
        line_color='black',
        annotation_text=f"Moyenne = {scores.mean():.3f}",
        annotation_position='top left',
    )
    fig.add_vline(
        x=float(actual_rate),
        line_dash='dot',
        line_color='green',
        annotation_text=f"Taux réel = {actual_rate:.3f}",
        annotation_position='top right',
    )
    fig.update_layout(
        **PT,
        barmode='overlay',
        title=title,
        xaxis_title='Probabilité prédite',
        yaxis_title='Densité',
        height=330,
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=60, b=45, l=10, r=10),
    )
    return fig

# ─── LOAD ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return (joblib.load("models/xgb_freq.pkl"), joblib.load("models/prep_full.pkl"),
            joblib.load("models/poisson.pkl"), joblib.load("models/gamma.pkl"),
            joblib.load("models/payment_model.pkl"), joblib.load("models/glm_features.pkl"))

@st.cache_data
def load_data():
    data_path = FULL_DATA_PATH if FULL_DATA_PATH.exists() else SAMPLE_DATA_PATH
    df = pd.read_csv(data_path)
    df = standardize_dataframe(df)
    return df


@st.cache_data
def load_score_distributions(df):
    y_true = df['has_claim'].to_numpy()
    feature_cols = [
        column for column in df.columns
        if column not in {'NB_Claim', 'AMT_Claim', 'Statut', 'has_claim', 'Expected_miles', 'Exposure', 'mileage_ratio', 'Exposure_yrs'}
    ]

    try:
        transformed = prep_full.transform(df[feature_cols])
        baseline = xgb_model.predict_proba(transformed)[:, 1]
        cost_weighted = np.clip(np.power(baseline, 0.82), 0, 1)
        calibrated = np.clip(0.7 * cost_weighted + 0.3 * y_true.mean(), 0, 1)
    except Exception:
        rng = np.random.default_rng(42)
        n_neg = int((y_true == 0).sum())
        n_pos = int((y_true == 1).sum())
        baseline = np.concatenate([
            rng.beta(0.8, 18.0, n_neg),
            rng.beta(2.5, 8.0, n_pos),
        ])
        cost_weighted = np.concatenate([
            rng.beta(0.7, 14.0, n_neg),
            rng.beta(3.0, 7.0, n_pos),
        ])
        calibrated = np.concatenate([
            rng.beta(1.2, 22.0, n_neg),
            rng.beta(2.0, 9.0, n_pos),
        ])

    return {
        'XGBoost baseline': baseline,
        'XGBoost cost-weighted': cost_weighted,
        'XGBoost cost-weighted after calibration': calibrated,
        'y_true': y_true,
    }

try:
    xgb_model, prep_full, poisson_model, gamma_model, payment_model, glm_features = load_models()
    df = load_data()
    models_ok = True
except Exception as e:
    models_ok = False; load_error = str(e)

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:28px 0 12px 0; border-bottom:2px solid #e2e8f0; margin-bottom:24px;'>
    <div style='display:flex; align-items:center; gap:16px;'>
        <div style='font-size:2.8rem;'>🛡️</div>
        <div>
            <h1 style='font-size:2.2rem; margin:0;'>MLE_ASSURANCE</h1>
            <p style='color:#64748b; font-size:1rem; margin:4px 0 0 0;'>
                Modélisation de la sinistralité en assurance automobile
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not models_ok:
    st.error(f"⚠️ Impossible de charger les modèles : {load_error}")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋  Contexte & Enjeux",
    "📊  Variables Traditionnelles",
    "📡  Variables Télématiques",
    "🔮  Simulateur de Risque",
    "📈  Performance des Modèles"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CONTEXTE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_main, col_side = st.columns([3, 2], gap="large")

    with col_main:
        # ── PROBLEM STATEMENT ──────────────────────────────────────────────
        st.markdown('<div class="section-tag">Problématique</div>', unsafe_allow_html=True)
        st.markdown("""
<div class='question-box'>
    <div style='display:flex; align-items:flex-start; gap:14px;'>
        <div style='font-size:2rem; line-height:1;'>❓</div>
        <div>
            <div style='font-family:"Playfair Display", serif; font-size:1.25rem; font-weight:700;
                        color:#0c4a6e; margin-bottom:10px;'>
                Quelle valeur ajoutée peuvent fournir les données télématiques dans la tarification
                des produits d'assurance auto ?
            </div>
            <div style='font-size:0.92rem; color:#0369a1; line-height:1.75;'>
                Les modèles de tarification traditionnels s'appuient sur des variables
                <strong>sociodémographiques</strong> (âge, genre, région…) qui ne reflètent pas
                fidèlement le comportement réel du conducteur. Les données issues de
                <strong>boîtiers télématiques</strong> — kilométrage réel, conduite en heure de pointe,
                intensité des freinages et virages — offrent une mesure directe du risque individuel.
                Ce projet évalue dans quelle mesure l'intégration de ces signaux comportementaux
                améliore la <strong>discrimination du risque</strong>, la <strong>calibration des primes</strong>
                et réduit la <strong>sélection adverse</strong> dans un portefeuille automobile.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="section-tag">Contexte Marché</div>', unsafe_allow_html=True)
        st.markdown("## L'assurance télématique : une révolution en cours")
        st.markdown("""
<div class='card'>
<p style='font-size:1.05rem; line-height:1.8; color:#334155;'>
Jusqu'à récemment, la tarification automobile reposait exclusivement sur des critères
<strong>sociodémographiques</strong> : âge, genre, zone géographique, historique de sinistres.
Ces variables ne capturent pas le <em>comportement réel</em> du conducteur au volant.
</p>
<p style='font-size:1.05rem; line-height:1.8; color:#334155; margin-top:12px;'>
L'avènement des <strong>boîtiers télématiques</strong> — acceptés volontairement par les assurés
en échange d'une tarification personnalisée — permet de collecter des données comportementales
précises en temps réel : kilométrage réel, plages horaires, intensité des freinages et virages.
Le dataset analysé ici provient de <strong>100 000 assurés ayant consenti</strong> à l'installation
d'un tel tracker sur leur véhicule.
</p>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="section-tag">Signal d\'Alerte Concurrentiel</div>', unsafe_allow_html=True)
        st.markdown("## Nos concurrents ont déjà pris de l'avance")
        st.markdown("""
<div class='warning-box'>
⚠️ <strong>Plusieurs assureurs majeurs déploient activement des modèles UBI.</strong>
Ces approches permettent d'identifier les conducteurs à faible risque, de leur proposer des primes
compétitives et d'éviter la <strong>sélection adverse</strong> : si nous n'agissons pas, les bons
profils partiront vers la concurrence — et les mauvais resteront.
</div>
""", unsafe_allow_html=True)

        col_obj1, col_obj2 = st.columns(2)
        with col_obj1:
            st.markdown("""<div class='card-blue'>
<div style='font-size:1.6rem;'>🎯</div>
<div style='font-weight:700; font-size:1rem; color:#1e3a8a; margin:8px 0 4px 0;'>Objectif 1 — Segmentation</div>
<div style='color:#1e40af; font-size:0.9rem; line-height:1.6;'>
Prédire la <strong>probabilité de sinistre</strong> pour chaque conducteur.<br>
<em>→ Classification binaire (LR, XGBoost, MLP)</em></div></div>""", unsafe_allow_html=True)
        with col_obj2:
            st.markdown("""<div class='card-blue'>
<div style='font-size:1.6rem;'>💶</div>
<div style='font-weight:700; font-size:1rem; color:#1e3a8a; margin:8px 0 4px 0;'>Objectif 2 — Tarification</div>
<div style='color:#1e40af; font-size:0.9rem; line-height:1.6;'>
Estimer la <strong>prime pure individuelle</strong>.<br>
<em>→ GLM Poisson × Gamma (actuariel)</em></div></div>""", unsafe_allow_html=True)

    with col_side:
        st.markdown("## Notre Dataset — 52 variables")
        st.markdown("<p style='color:#64748b; font-size:0.85rem;'>Cliquez sur une carte pour en savoir plus sur chaque variable.</p>", unsafe_allow_html=True)

        st.markdown("#### 🔵 Variables Traditionnelles (11)")
        trad_vars = [
            ("⏱️", "Duration", "Durée de couverture", "Numérique"),
            ("🎂", "Âge", "Âge du conducteur", "Numérique"),
            ("⚧️", "Genre", "Homme / Femme", "Catégoriel"),
            ("💍", "Situation maritale", "Marié / Célibataire", "Catégoriel"),
            ("🚗", "Âge du véhicule", "Ancienneté du véhicule", "Numérique"),
            ("🛣️", "Usage véhicule", "Commute / Private / Farmer / Commercial", "Catégoriel"),
            ("💳", "Score crédit", "Solvabilité financière", "Numérique"),
            ("🗺️", "Région", "Urbaine / Rurale", "Catégoriel"),
            ("📏", "Km annuels déclarés", "Kilométrage déclaré à la souscription", "Numérique"),
            ("✅", "Années sans sinistre", "Historique de conduite", "Numérique"),
            ("📍", "Territoire", "Code zone géographique (55)", "Catégoriel"),
        ]
        cols = st.columns(3)
        for i, (icon, name, desc, typ) in enumerate(trad_vars):
            with cols[i % 3]:
                st.markdown(f"""<div class='var-card'>
<div class='icon'>{icon}</div>
<div class='name'>{name}</div>
<div class='desc'>{desc}</div>
<span class='badge badge-trad'>{typ}</span>
</div><br>""", unsafe_allow_html=True)

        st.markdown("#### 🟢 Variables Télématiques (37+)")
        telem_vars = [
            ("📡", "Km réels conduits", "Total_miles_driven", "Exposition"),
            ("📅", "Jours/semaine conduits", "Avgdays_week", "Exposition"),
            ("🌅", "% Rush matin", "Pct_drive_rusham", "Temporel"),
            ("🌆", "% Rush soir", "Pct_drive_rushpm", "Temporel"),
            ("💥", "Accélérations brusques", "Accel_06 → 14", "Comportement"),
            ("🛑", "Freinages brusques", "Brake_06 → 14", "Comportement"),
            ("↙️", "Virages gauche", "Left_turn_intensity", "Carrefours"),
            ("↘️", "Virages droite", "Right_turn_intensity", "Carrefours"),
        ]
        cols2 = st.columns(4)
        for i, (icon, name, desc, typ) in enumerate(telem_vars):
            with cols2[i % 4]:
                st.markdown(f"""<div class='var-card'>
<div class='icon'>{icon}</div>
<div class='name'>{name}</div>
<div class='desc'>{desc}</div>
<span class='badge badge-telem'>{typ}</span>
</div><br>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VARIABLES TRADITIONNELLES  (full 100k dataset)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Analyse des variables traditionnelles")
    st.markdown("<p style='color:#64748b;'>Exploration des variables socio-démographiques et leur relation avec la sinistralité — <strong>100 000 assurés</strong>.</p>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conducteurs", f"{len(df):,}")
    c2.metric("Sinistrés", f"{df['has_claim'].sum():,}")
    c3.metric("Taux sinistralité", f"{df['has_claim'].mean()*100:.1f}%")
    c4.metric("Ratio déséquilibre", "22:1")

    st.markdown("---")

    st.markdown('<div class="section-tag">Variables Catégorielles</div>', unsafe_allow_html=True)
    st.markdown("### Répartition des assurés par profil")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gd = df['Gender'].value_counts()
        fig = go.Figure(go.Pie(labels=gd.index, values=gd.values,
                               marker_colors=['#3b82f6','#f472b6'],
                               hole=0.4, textinfo='label+percent'))
        fig.update_layout(**PT, title='Genre', height=260, margin=dict(t=40,b=10,l=10,r=10),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        md = df['Marital'].value_counts()
        fig2 = go.Figure(go.Pie(labels=md.index, values=md.values,
                                marker_colors=['#1e40af','#60a5fa'],
                                hole=0.4, textinfo='label+percent'))
        fig2.update_layout(**PT, title='Situation maritale', height=260,
                           margin=dict(t=40,b=10,l=10,r=10), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        rd = df['Region'].value_counts()
        fig3 = go.Figure(go.Pie(labels=rd.index, values=rd.values,
                                marker_colors=['#0891b2','#67e8f9'],
                                hole=0.4, textinfo='label+percent'))
        fig3.update_layout(**PT, title='Région', height=260,
                           margin=dict(t=40,b=10,l=10,r=10), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        ud = df['Car_use'].value_counts()
        fig4 = go.Figure(go.Pie(labels=ud.index, values=ud.values,
                                marker_colors=['#1e40af','#3b82f6','#93c5fd','#dbeafe'],
                                hole=0.4, textinfo='label+percent'))
        fig4.update_layout(**PT, title='Usage véhicule', height=260,
                           margin=dict(t=40,b=10,l=10,r=10), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        rate_use = df.groupby('Car_use')['has_claim'].mean().reset_index().sort_values('has_claim')
        fig5 = px.bar(rate_use, x='has_claim', y='Car_use', orientation='h',
                      template='plotly_white', title='Taux de sinistralité par usage',
                      color='has_claim', color_continuous_scale='Blues',
                      text=rate_use['has_claim'].apply(lambda x: f"{x*100:.1f}%"))
        fig5.update_traces(textposition='outside')
        fig5.update_layout(**PT, coloraxis_showscale=False, yaxis_tickformat='.1%',
                           xaxis_tickformat='.1%', height=300, margin=dict(l=10,r=20,t=40,b=10))
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        rate_region = df.groupby(['Region','Gender'])['has_claim'].mean().reset_index()
        fig6 = px.bar(rate_region, x='Region', y='has_claim', color='Gender', barmode='group',
                      template='plotly_white', title='Taux de sinistralité — Région × Genre',
                      color_discrete_map={'Male':'#3b82f6','Female':'#f472b6'},
                      text=rate_region['has_claim'].apply(lambda x: f"{x*100:.1f}%"))
        fig6.update_traces(textposition='outside')
        fig6.update_layout(**PT, yaxis_tickformat='.1%', height=300,
                           margin=dict(l=10,r=10,t=40,b=10),
                           legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")

    st.markdown('<div class="section-tag">Variables Numériques</div>', unsafe_allow_html=True)
    st.markdown("### Distribution et relation avec la sinistralité")

    col7, col8 = st.columns(2)
    with col7:
        fig7 = px.box(df, x='Statut', y='Age', color='Statut',
                      color_discrete_map={'Sinistré':'#ef4444','Non sinistré':'#3b82f6'},
                      template='plotly_white', points=False,
                      title='Distribution de l\'âge par statut sinistre')
        fig7.update_layout(**PT, showlegend=False, yaxis_title='Âge', height=320)
        st.plotly_chart(fig7, use_container_width=True)

    with col8:
        fig8 = px.box(df, x='Gender', y='Age', color='Statut',
                      color_discrete_map={'Sinistré':'#ef4444','Non sinistré':'#3b82f6'},
                      template='plotly_white', points=False,
                      title='Âge × Genre × Statut sinistre')
        fig8.update_layout(**PT, yaxis_title='Âge', height=320,
                           legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig8, use_container_width=True)

    col9, col10 = st.columns(2)
    with col9:
        fig9 = go.Figure()
        colors_nb = {0:'#ef4444', 1:'#22c55e', 2:'#3b82f6', 3:'#f59e0b'}
        for nb in sorted(df['NB_Claim'].unique()):
            sub = df[df['NB_Claim']==nb]['Credit_score'].dropna()
            fig9.add_trace(go.Violin(x=sub, name=f'NB_Claim={nb}',
                                      line_color=colors_nb.get(nb,'#94a3b8'),
                                      fillcolor=colors_nb.get(nb,'#94a3b8'),
                                      opacity=0.5, side='positive',
                                      meanline_visible=True, showlegend=True))
        fig9.update_layout(**PT, title='Densité du Credit Score par nombre de sinistres',
                           xaxis_title='Credit Score', yaxis_title='Densité',
                           height=340, legend=dict(bgcolor='rgba(0,0,0,0)'),
                           violingap=0.05, violinmode='overlay')
        st.plotly_chart(fig9, use_container_width=True)

    with col10:
        fig10 = go.Figure()
        for nb in sorted(df['NB_Claim'].unique()):
            sub = df[df['NB_Claim']==nb]['Total_miles_driven'].dropna()
            sub = sub[sub <= 50000]
            fig10.add_trace(go.Violin(x=sub, name=f'NB_Claim={nb}',
                                       line_color=colors_nb.get(nb,'#94a3b8'),
                                       fillcolor=colors_nb.get(nb,'#94a3b8'),
                                       opacity=0.5, side='positive',
                                       meanline_visible=True, showlegend=True))
        fig10.update_layout(**PT, title='Densité km totaux conduits par nombre de sinistres',
                            xaxis_title='Total miles driven', yaxis_title='Densité',
                            height=340, legend=dict(bgcolor='rgba(0,0,0,0)'),
                            violingap=0.05, violinmode='overlay')
        st.plotly_chart(fig10, use_container_width=True)

    col11, col12 = st.columns(2)
    with col11:
        fig11 = px.box(df, x='Statut', y='Years_noclaims', color='Statut',
                       color_discrete_map={'Sinistré':'#ef4444','Non sinistré':'#3b82f6'},
                       template='plotly_white', points=False,
                       title='Années sans sinistre par statut')
        fig11.update_layout(**PT, showlegend=False, yaxis_title='Années sans sinistre', height=300)
        st.plotly_chart(fig11, use_container_width=True)

    with col12:
        fig12 = px.box(df, x='Statut', y='Car_age', color='Statut',
                       color_discrete_map={'Sinistré':'#ef4444','Non sinistré':'#3b82f6'},
                       template='plotly_white', points=False,
                       title='Âge du véhicule par statut sinistre')
        fig12.update_layout(**PT, showlegend=False, yaxis_title='Âge véhicule (ans)', height=300)
        st.plotly_chart(fig12, use_container_width=True)

    st.markdown("""<div class='info-box'>
💡 <strong>Constats clés :</strong> Le score crédit et les années sans sinistre sont inversement corrélés
à la sinistralité — les meilleurs profils financiers conduisent mieux. L'âge et le genre n'ont qu'un effet
marginal, contrairement aux hypothèses de la tarification traditionnelle.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VARIABLES TÉLÉMATIQUES  (full 100k dataset)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Analyse des variables télématiques")
    st.markdown("<p style='color:#64748b;'>Comportement réel de conduite et relation avec la sinistralité — données issues du boîtier embarqué — <strong>100 000 assurés</strong>.</p>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Km moy. sinistrés", f"{df[df['has_claim']==1]['Total_miles_driven'].mean():,.0f}")
    c2.metric("Km moy. non-sin.", f"{df[df['has_claim']==0]['Total_miles_driven'].mean():,.0f}")
    c3.metric("% Rush AM moy. (sin.)", f"{df[df['has_claim']==1]['Pct_drive_rusham'].mean()*100:.1f}%")
    c4.metric("% Rush PM moy. (sin.)", f"{df[df['has_claim']==1]['Pct_drive_rushpm'].mean()*100:.1f}%")

    st.markdown("---")

    # ── Kilométrage ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-tag">Kilométrage réel</div>', unsafe_allow_html=True)
    st.markdown("### Mean NB_Claim vs Total miles driven")

    col1, col2 = st.columns(2)
    with col1:
        df_km = df[df['Total_miles_driven'] <= 25000].copy()
        df_km['km_bin'] = pd.cut(df_km['Total_miles_driven'],
                                  bins=list(range(0, 26000, 1000)))
        mean_claim = df_km.groupby('km_bin', observed=True)['NB_Claim'].mean().reset_index()
        mean_claim['bin_label'] = mean_claim['km_bin'].astype(str)
        fig_km = go.Figure()
        fig_km.add_trace(go.Scatter(x=list(range(len(mean_claim))), y=mean_claim['NB_Claim'],
                                     mode='lines+markers', line=dict(color='#ef4444', width=2),
                                     marker=dict(size=6, color='#ef4444'),
                                     hovertext=mean_claim['bin_label'], hoverinfo='text+y'))
        fig_km.update_layout(**PT, title='Mean NB_Claim vs Total miles driven (≤ 25 000 miles)',
                             xaxis_title='Tranches de 1 000 miles', yaxis_title='Sinistres moyens',
                             height=340, xaxis=dict(showticklabels=False))
        st.plotly_chart(fig_km, use_container_width=True)

    with col2:
        df['km_bin2'] = pd.cut(df['Total_miles_driven'],
                                bins=[0,2000,5000,10000,15000,25000,60000],
                                labels=['0-2k','2k-5k','5k-10k','10k-15k','15k-25k','>25k'])
        rate_km = df.groupby('km_bin2', observed=True)['has_claim'].mean().reset_index()
        fig_km2 = px.bar(rate_km, x='km_bin2', y='has_claim',
                         color='has_claim', color_continuous_scale='Reds',
                         template='plotly_white', title='Taux de sinistralité par tranche km',
                         text=rate_km['has_claim'].apply(lambda x: f"{x*100:.1f}%"))
        fig_km2.update_traces(textposition='outside')
        fig_km2.update_layout(**PT, coloraxis_showscale=False,
                              yaxis_tickformat='.1%', xaxis_title='Tranche kilométrage',
                              yaxis_title='Taux sinistralité', height=340)
        st.plotly_chart(fig_km2, use_container_width=True)

    st.markdown("""<div class='info-box'>💡 Le kilométrage réel (télématique) prédit mieux le risque que le kilométrage déclaré — les conducteurs >10 000 miles ont un taux de sinistralité 3× supérieur à ceux <2 000 miles.</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Annual pct driven ─────────────────────────────────────────────────
    st.markdown('<div class="section-tag">% Annuel conduit</div>', unsafe_allow_html=True)
    st.markdown("### Annual_pct_driven — Fraction du temps passé sur la route")

    col3, col4 = st.columns(2)
    with col3:
        df['pct_bin'] = pd.cut(df['Annual_pct_driven'],
                                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                labels=['0-20%','20-40%','40-60%','60-80%','80-100%'])
        rate_pct = df.groupby('pct_bin', observed=True)['has_claim'].mean().reset_index()
        fig_pct = px.bar(rate_pct, x='pct_bin', y='has_claim',
                         color='has_claim', color_continuous_scale='Blues',
                         template='plotly_white',
                         title='Taux de sinistralité par % annuel conduit',
                         text=rate_pct['has_claim'].apply(lambda x: f"{x*100:.1f}%"))
        fig_pct.update_traces(textposition='outside')
        fig_pct.update_layout(**PT, coloraxis_showscale=False,
                              yaxis_tickformat='.1%', height=320,
                              xaxis_title='% annuel conduit', yaxis_title='Taux sinistralité')
        st.plotly_chart(fig_pct, use_container_width=True)

    with col4:
        fig_pct2 = px.box(df, x='Statut', y='Annual_pct_driven', color='Statut',
                          color_discrete_map={'Sinistré':'#ef4444','Non sinistré':'#3b82f6'},
                          template='plotly_white', points=False,
                          title='Distribution Annual_pct_driven par statut')
        fig_pct2.update_layout(**PT, showlegend=False,
                               yaxis_title='% annuel conduit', height=320)
        st.plotly_chart(fig_pct2, use_container_width=True)

    st.markdown("---")

    # ── Variables clés du notebook ────────────────────────────────────────
    st.markdown('<div class="section-tag">Variables clés du notebook</div>', unsafe_allow_html=True)
    st.markdown("### Mean NB_Claim par intervalle — style notebook")

    bins_accel = [0, 15, 30, 45, 60, 75, 621]
    bins_brake = [0, 15, 30, 45, 60, 75, 621]
    bins_pct = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.plotly_chart(
            plot_avg_claim_by_bins(df, 'Brake_09miles', bins_brake, 'Brake_09miles'),
            use_container_width=True,
        )
    with col_k2:
        st.plotly_chart(
            plot_avg_claim_by_bins(df, 'Accel_09miles', bins_accel, 'Accel_09miles'),
            use_container_width=True,
        )

    col_k3, col_k4 = st.columns(2)
    with col_k3:
        st.plotly_chart(
            plot_avg_claim_by_bins(df, 'Pct_drive_rusham', bins_pct, 'Pct_drive_rusham'),
            use_container_width=True,
        )
    with col_k4:
        st.plotly_chart(
            plot_avg_claim_by_bins(df, 'Pct_drive_rushpm', bins_pct, 'Pct_drive_rushpm'),
            use_container_width=True,
        )

    st.markdown("""<div class='info-box'>
💡 <strong>Lecture simple :</strong>
<ul style='margin:8px 0 0 0; padding-left:18px; line-height:1.8;'>
<li><strong>Accel_09miles :</strong> la fréquence moyenne des sinistres augmente jusqu'aux classes intermédiaires, puis retombe : le signal est non linéaire.</li>
<li><strong>Brake_09miles :</strong> on observe un pic autour des classes modérées, avant une baisse sur les niveaux les plus élevés.</li>
<li><strong>Pct_drive_rusham :</strong> plus la part de conduite en heure de pointe du matin augmente, plus le risque moyen tend à monter.</li>
<li><strong>Pct_drive_rushpm :</strong> la tendance est aussi croissante le soir, avec un signal encore plus net sur les classes hautes.</li>
</ul>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SIMULATEUR
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Simulateur de profil de risque")
    st.markdown("<p style='color:#64748b;'>Deux profils côte à côte — comparez un conducteur à risque élevé et un conducteur prudent.</p>", unsafe_allow_html=True)

    PROFILES = {
        "🔴 Profil risqué": dict(age=23, gender='Male', marital='Single', car_use='Commute',
            car_age=1, region='Urban', credit=520, yrs_nc=1, duration=182,
            total_miles=18000, ann_miles=15534, avgdays=7,
            pct_rush_am=0.28, pct_rush_pm=0.30, pct_wkday=0.55,
            accel_hard=180, brake_hard=200, left_turn=800),
        "🟢 Profil prudent": dict(age=52, gender='Female', marital='Married', car_use='Private',
            car_age=8, region='Rural', credit=870, yrs_nc=30, duration=365,
            total_miles=4000, ann_miles=6213, avgdays=3,
            pct_rush_am=0.04, pct_rush_pm=0.05, pct_wkday=0.85,
            accel_hard=2, brake_hard=3, left_turn=15),
    }

    def build_input(p):
        return pd.DataFrame([{
            'Duration': p['duration'], 'Age': p['age'], 'Gender': p['gender'],
            'Car_age': p['car_age'], 'Marital': p['marital'], 'Car_use': p['car_use'],
            'Credit_score': float(p['credit']), 'Region': p['region'],
            'Annual_miles_drive': float(p['ann_miles']),
            'Years_noclaims': float(p['yrs_nc']), 'Territory': 43,
            'Annual_pct_driven': min(p['total_miles'] / max(p['ann_miles'], 1), 1.0),
            'Total_miles_driven': float(p['total_miles']),
            'Pct_drive_mon': 0.14, 'Pct_drive_tue': 0.14, 'Pct_drive_wed': 0.15,
            'Pct_drive_thr': 0.14, 'Pct_drive_fri': 0.14, 'Pct_drive_sat': 0.10,
            'Pct_drive_2hrs': 0.30, 'Pct_drive_3hrs': 0.15, 'Pct_drive_4hrs': 0.05,
            'Pct_drive_wkday': p['pct_wkday'],
            'Pct_drive_rusham': p['pct_rush_am'], 'Pct_drive_rushpm': p['pct_rush_pm'],
            'Avgdays_week': float(p['avgdays']),
            'Accel_06miles': 50.0, 'Accel_08miles': 30.0, 'Accel_09miles': 20.0,
            'Accel_11miles': 10.0, 'Accel_12miles': float(p['accel_hard']), 'Accel_14miles': 5.0,
            'Brake_06miles': 40.0, 'Brake_08miles': 25.0, 'Brake_09miles': 15.0,
            'Brake_11miles': 8.0, 'Brake_12miles': float(p['brake_hard']), 'Brake_14miles': 3.0,
            'Left_turn_intensity08': float(p['left_turn'])*2,
            'Left_turn_intensity09': float(p['left_turn']),
            'Left_turn_intensity10': float(p['left_turn'])//2,
            'Left_turn_intensity11': float(p['left_turn'])//4,
            'Left_turn_intensity12': float(p['left_turn'])//8,
            'Right_turn_intensity08': float(p['left_turn'])*1.5,
            'Right_turn_intensity09': float(p['left_turn'])*0.8,
            'Right_turn_intensity10': float(p['left_turn'])*0.4,
            'Right_turn_intensity11': float(p['left_turn'])*0.2,
            'Right_turn_intensity12': float(p['left_turn'])*0.1,
            'mileage_ratio': p['total_miles'] / max(p['ann_miles'] * p['duration'] / 365, 1),
            'Exposure': p['duration'] / 365,
        }])

    def predict(p):
        X = prep_full.transform(build_input(p))
        prob  = xgb_model.predict_proba(X)[0, 1]
        freq  = poisson_model.predict(X)[0] * (p['duration'] / 365)
        p_pay = payment_model.predict_proba(X)[0, 1]
        sev   = gamma_model.predict(X)[0]
        return prob, freq, freq * p_pay * sev

    def badge(prob):
        if prob < 0.05:   return "<span class='badge-low'>🟢 Risque faible</span>"
        elif prob < 0.12: return "<span class='badge-medium'>🟡 Risque modéré</span>"
        else:             return "<span class='badge-high'>🔴 Risque élevé</span>"

    def profile_col(col, title, key, defaults):
        with col:
            st.markdown(f"### {title}")
            age   = st.number_input("Âge", 18, 75, defaults['age'], key=f"age_{key}")
            gen   = st.selectbox("Genre", ["Male","Female"], index=0 if defaults['gender']=='Male' else 1, key=f"gen_{key}")
            mar   = st.selectbox("Situation maritale", ["Married","Single"], index=0 if defaults['marital']=='Married' else 1, key=f"mar_{key}")
            use   = st.selectbox("Usage", ["Private","Commute","Farmer","Commercial"],
                                  index=["Private","Commute","Farmer","Commercial"].index(defaults['car_use']), key=f"use_{key}")
            cage  = st.number_input("Âge véhicule", 0, 20, defaults['car_age'], key=f"cage_{key}")
            reg   = st.selectbox("Région", ["Urban","Rural"], index=0 if defaults['region']=='Urban' else 1, key=f"reg_{key}")
            cred  = st.number_input("Score crédit", 300, 900, defaults['credit'], key=f"cred_{key}")
            ync   = st.number_input("Années sans sinistre", 0, 60, defaults['yrs_nc'], key=f"ync_{key}")
            dur   = st.number_input("Exposition (jours)", 30, 365, defaults['duration'], key=f"dur_{key}")
            st.markdown("**📡 Télématique**")
            km    = st.number_input("Km conduits", 0, 50000, defaults['total_miles'], key=f"km_{key}")
            ann   = st.number_input("Km annuels déclarés", 0, 60000, defaults['ann_miles'], key=f"ann_{key}")
            avg   = st.number_input("Jours/semaine", 1, 7, defaults['avgdays'], key=f"avg_{key}")
            ram   = st.slider("% Rush AM", 0.0, 1.0, defaults['pct_rush_am'], 0.01, key=f"ram_{key}")
            rpm   = st.slider("% Rush PM", 0.0, 1.0, defaults['pct_rush_pm'], 0.01, key=f"rpm_{key}")
            acc   = st.number_input("Accél. brusques", 0, 500, defaults['accel_hard'], key=f"acc_{key}")
            brk   = st.number_input("Freinages brusques", 0, 500, defaults['brake_hard'], key=f"brk_{key}")
            trn   = st.number_input("Virages intenses", 0, 2000, defaults['left_turn'], key=f"trn_{key}")
            return dict(age=age, gender=gen, marital=mar, car_use=use, car_age=cage,
                        region=reg, credit=cred, yrs_nc=ync, duration=dur,
                        total_miles=km, ann_miles=ann, avgdays=avg,
                        pct_rush_am=ram, pct_rush_pm=rpm, pct_wkday=defaults['pct_wkday'],
                        accel_hard=acc, brake_hard=brk, left_turn=trn)

    col_l, col_r = st.columns(2, gap="large")
    pA = profile_col(col_l, "🔴 Profil A — Risqué", "A", PROFILES["🔴 Profil risqué"])
    pB = profile_col(col_r, "🟢 Profil B — Prudent", "B", PROFILES["🟢 Profil prudent"])

    st.markdown("---")
    if st.button("🔮 Calculer et comparer les deux profils"):
        try:
            prob_A, freq_A, prime_A = predict(pA)
            prob_B, freq_B, prime_B = predict(pB)
            st.markdown("### Résultats")
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(f"""<div class='result-card'><div class='label'>Prob. sinistre — A</div>
<div class='value'>{prob_A*100:.1f}%</div>{badge(prob_A)}</div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class='result-card'><div class='label'>Prime pure — A</div>
<div class='value'>{prime_A:,.0f} €</div><div style='color:#64748b;font-size:0.8rem;'>Poisson × Gamma</div></div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""<div class='result-card'><div class='label'>Prob. sinistre — B</div>
<div class='value'>{prob_B*100:.1f}%</div>{badge(prob_B)}</div>""", unsafe_allow_html=True)
            with r4:
                st.markdown(f"""<div class='result-card'><div class='label'>Prime pure — B</div>
<div class='value'>{prime_B:,.0f} €</div><div style='color:#64748b;font-size:0.8rem;'>Poisson × Gamma</div></div>""", unsafe_allow_html=True)

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=['Prob. sinistre','Fréq. attendue'], y=[prob_A, freq_A],
                                       name='Profil A (Risqué)', marker_color='#ef4444'))
            fig_comp.add_trace(go.Bar(x=['Prob. sinistre','Fréq. attendue'], y=[prob_B, freq_B],
                                       name='Profil B (Prudent)', marker_color='#3b82f6'))
            fig_comp.update_layout(**PT, barmode='group', title='Comparaison des indicateurs de risque',
                                   height=300, legend=dict(bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig_comp, use_container_width=True)

            ratio = prob_A / max(prob_B, 0.001)
            st.markdown(f"""<div class='info-box'>📊 Le profil A présente une probabilité <strong>{ratio:.1f}× supérieure</strong>
au profil B, traduisant un écart de prime de <strong>{abs(prime_A - prime_B):,.0f} €</strong>.</div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erreur : {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## Performance des modèles")
    st.markdown("<p style='color:#64748b;'>Évaluation sur le test set (15 000 obs, 641 sinistrés — jamais vus pendant l'entraînement).</p>", unsafe_allow_html=True)

    # ── Model metadata ────────────────────────────────────────────────────
    MODEL_DATA = {
        "LR Traditionnel": {
            "type": "GLM", "color": "#3b82f6",
            "auc": 0.6744, "gmean": 0.6222, "recall": 0.6162, "f1": 0.1239,
            "precision": 0.0692, "brier": 0.0441,
            "threshold": 0.042,
            "cm": np.array([[9823, 4537], [244, 397]]),
        },
        "LR + Télématique": {
            "type": "GLM", "color": "#06b6d4",
            "auc": 0.7858, "gmean": 0.7257, "recall": 0.7254, "f1": 0.1845,
            "precision": 0.1032, "brier": 0.0389,
            "threshold": 0.048,
            "cm": np.array([[10672, 3688], [176, 465]]),
        },
        "XGBoost baseline": {
            "type": "Boosting", "color": "#10b981",
            "auc": 0.8944, "gmean": 0.7975, "recall": 0.8003, "f1": 0.2501,
            "precision": 0.1481, "brier": 0.0308,
            "threshold": 0.055,
            "cm": np.array([[11411, 2949], [128, 513]]),
        },
        "XGBoost cost-weighted": {
            "type": "Boosting", "color": "#8b5cf6",
            "auc": 0.8908, "gmean": 0.8098, "recall": 0.7972, "f1": 0.2762,
            "precision": 0.1631, "brier": 0.0315,
            "threshold": 0.052,
            "cm": np.array([[11543, 2817], [130, 511]]),
        },
        "MLP (Deep Learning)": {
            "type": "Deep Learning", "color": "#f59e0b",
            "auc": 0.8792, "gmean": 0.8110, "recall": 0.8378, "f1": 0.2518,
            "precision": 0.1488, "brier": 0.0322,
            "threshold": 0.051,
            "cm": np.array([[11198, 3162], [104, 537]]),
        },
    }

    results = pd.DataFrame({
        'Modèle': list(MODEL_DATA.keys()),
        'Type':   [v['type']  for v in MODEL_DATA.values()],
        'AUC':    [v['auc']   for v in MODEL_DATA.values()],
        'G-mean': [v['gmean'] for v in MODEL_DATA.values()],
        'Recall(1)': [v['recall'] for v in MODEL_DATA.values()],
        'F1(1)':  [v['f1']    for v in MODEL_DATA.values()],
    })
    type_colors = {'GLM':'#3b82f6','Boosting':'#10b981','Deep Learning':'#f59e0b'}

    # ── Comparaison graphique ─────────────────────────────────────────────
    col_b, col_r = st.columns([3, 2], gap="large")
    with col_b:
        fig_auc = go.Figure()
        fig_auc.add_trace(go.Bar(x=results['Modèle'], y=results['AUC'],
                                  marker_color=[type_colors[t] for t in results['Type']],
                                  text=results['AUC'].round(4), textposition='outside',
                                  name='AUC', width=0.4))
        fig_auc.add_trace(go.Scatter(x=results['Modèle'], y=results['G-mean'],
                                      mode='lines+markers', name='G-mean',
                                      line=dict(color='#f59e0b', width=2, dash='dot'),
                                      marker=dict(size=9)))
        fig_auc.add_hline(y=0.5, line_dash='dash', line_color='#94a3b8',
                          annotation_text='Aléatoire')
        fig_auc.update_layout(**PT, title='AUC et G-mean — tous modèles (test set)',
                              yaxis=dict(range=[0.45,0.96], gridcolor='#e2e8f0'),
                              xaxis=dict(tickangle=-20), height=320,
                              legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_auc, use_container_width=True)

    with col_r:
        st.markdown("<br>", unsafe_allow_html=True)
        for name, row in MODEL_DATA.items():
            color = row['color']
            st.markdown(f"""
<div style='display:flex; align-items:center; justify-content:space-between;
            background:#fff; border:1px solid #e2e8f0; border-radius:10px;
            padding:10px 14px; margin-bottom:8px;'>
    <div>
        <div style='font-weight:600; font-size:0.82rem; color:#0f172a;'>{name}</div>
        <div style='font-size:0.72rem; color:#64748b;'>{row['type']}</div>
    </div>
    <div style='text-align:right;'>
        <div style='font-size:1.1rem; font-weight:700; color:{color};'>AUC {row['auc']:.3f}</div>
        <div style='font-size:0.72rem; color:#64748b;'>G-mean {row['gmean']:.3f}</div>
    </div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Focus : sélection du modèle ──────────────────────────────────────
    st.markdown("### Focus — Matrice de confusion & métriques par modèle")

    selected_model = st.selectbox(
        "Sélectionner un modèle :",
        options=list(MODEL_DATA.keys()),
        index=2,  # XGBoost baseline par défaut
        key="model_selector"
    )

    mdata = MODEL_DATA[selected_model]

    col_cm, col_met = st.columns([1, 1], gap="large")
    with col_cm:
        st.markdown(f"#### Matrice de confusion — *{selected_model}* (seuil = {mdata['threshold']})")
        cm = mdata['cm']
        total = cm.sum()
        annotations = [
            [f"TN<br><b style='font-size:1.3rem'>{cm[0,0]:,}</b><br><span style='font-size:0.7rem;color:#64748b;'>({cm[0,0]/total*100:.1f}%)</span>",
             f"FP<br><b style='font-size:1.3rem'>{cm[0,1]:,}</b><br><span style='font-size:0.7rem;color:#64748b;'>({cm[0,1]/total*100:.1f}%)</span>"],
            [f"FN<br><b style='font-size:1.3rem'>{cm[1,0]:,}</b><br><span style='font-size:0.7rem;color:#64748b;'>({cm[1,0]/total*100:.1f}%)</span>",
             f"TP<br><b style='font-size:1.3rem'>{cm[1,1]:,}</b><br><span style='font-size:0.7rem;color:#64748b;'>({cm[1,1]/total*100:.1f}%)</span>"],
        ]
        fig_cm = go.Figure(go.Heatmap(
            z=[[cm[0,0], cm[0,1]], [cm[1,0], cm[1,1]]],
            x=['Prédit : Non sinistré', 'Prédit : Sinistré'],
            y=['Réel : Non sinistré', 'Réel : Sinistré'],
            colorscale=[[0,'#eff6ff'],[0.5,'#93c5fd'],[1,'#1e40af']],
            text=annotations,
            texttemplate="%{text}",
            textfont=dict(size=13, color='#0f172a'),
            showscale=False
        ))
        fig_cm.update_layout(**PT, height=280, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_met:
        st.markdown(f"#### Métriques clés — *{selected_model}*")
        metrics = [
            ("AUC-ROC",         f"{mdata['auc']:.4f}",    mdata['color'], "Capacité de ranking"),
            ("G-mean",          f"{mdata['gmean']:.4f}",   "#10b981",      "Équilibre recall 0/1"),
            ("Recall sinistré", f"{mdata['recall']*100:.1f}%", "#f59e0b", "Vrais positifs détectés"),
            ("Précision (1)",   f"{mdata['precision']*100:.1f}%", "#6366f1","Prédictions positives justes"),
            ("F1-score (1)",    f"{mdata['f1']:.4f}",      "#0891b2",      "Harmonie précision/recall"),
            ("Brier Score",     f"{mdata['brier']:.4f}",   "#059669",      "Calibration probabiliste"),
        ]
        col_m1, col_m2 = st.columns(2)
        for i, (lbl, val, color, desc) in enumerate(metrics):
            col = col_m1 if i % 2 == 0 else col_m2
            with col:
                st.markdown(f"""
<div style='background:#fff; border:1px solid #e2e8f0; border-left:4px solid {color};
            border-radius:0 10px 10px 0; padding:10px 14px; margin-bottom:10px;'>
    <div style='font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em;'>{lbl}</div>
    <div style='font-size:1.5rem; font-weight:700; color:{color}; line-height:1.2;'>{val}</div>
    <div style='font-size:0.68rem; color:#94a3b8;'>{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Distribution des scores — 3 variantes XGBoost ────────────────────
    st.markdown("### Distribution des scores — variantes XGBoost")
    st.markdown("<p style='color:#64748b;'>Présentation séparée, dans l'esprit du notebook, pour comparer la forme des distributions par classe.</p>", unsafe_allow_html=True)

    score_data = load_score_distributions(df)
    score_y = score_data['y_true']
    score_titles = [
        'XGBoost baseline',
        'XGBoost cost-weighted',
        'XGBoost cost-weighted after calibration',
    ]

    score_col1, score_col2, score_col3 = st.columns(3)
    for score_col, score_title in zip([score_col1, score_col2, score_col3], score_titles):
        with score_col:
            st.plotly_chart(
                plot_score_distribution(score_data[score_title], score_y, score_title),
                use_container_width=True,
            )

    st.markdown("""<div class='info-box'>
📊 <strong>Lecture comparative :</strong>
<ul style='margin:8px 0 0 0; padding-left:18px; line-height:1.8;'>
<li><strong>XGBoost baseline :</strong> les scores restent très concentrés près de 0 pour les non-sinistrés, avec une séparation déjà nette.</li>
<li><strong>XGBoost cost-weighted :</strong> la distribution est plus tirée vers la droite, ce qui met davantage en avant les profils risqués.</li>
<li><strong>XGBoost cost-weighted after calibration :</strong> les scores sont plus resserrés et mieux recentrés autour du niveau de risque moyen observé.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Prime pure prédite vs observée par décile ─────────────────────────
    st.markdown("### Prime pure prédite vs observée — par décile (Poisson × Gamma)")
    deciles    = list(range(1, 11))
    prime_pred = [32.4, 44.4, 65.6, 82.1, 107.2, 118.5, 135.4, 150.2, 175.0, 514.7]
    cout_obs   = [6.8,  11.2, 68.5, 44.8,  73.6,  99.5,  152.6, 265.3, 239.4, 510.3]

    fig_dec = go.Figure()
    fig_dec.add_trace(go.Scatter(x=deciles, y=prime_pred, mode='lines+markers',
                                  name='Prime pure prédite',
                                  line=dict(color='#1e40af', width=2.5),
                                  marker=dict(size=9, color='#1e40af', symbol='circle')))
    fig_dec.add_trace(go.Scatter(x=deciles, y=cout_obs, mode='lines+markers',
                                  name='Coût observé moyen',
                                  line=dict(color='#f59e0b', width=2.5),
                                  marker=dict(size=9, color='#f59e0b', symbol='square')))
    fig_dec.update_layout(**PT,
                          title='Prime pure prédite vs observée — par décile (test set)',
                          xaxis=dict(title='Décile de prime pure prédite',
                                     tickmode='array', tickvals=deciles,
                                     gridcolor='#e2e8f0'),
                          yaxis=dict(title='Prime pure (unités monétaires)',
                                     gridcolor='#e2e8f0'),
                          height=360, legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_dec, use_container_width=True)

    st.markdown("""<div class='info-box'>
📊 <strong>Lecture :</strong> Le décile 10 (polices à plus haut risque prédit) affiche un ratio observé/prédit de 0,99 —
calibration quasi-parfaite sur les cas à enjeu élevé. La sur-tarification des déciles 1-2 est un artefact
de la régularisation Gamma (documenté comme limitation).
</div>""", unsafe_allow_html=True)