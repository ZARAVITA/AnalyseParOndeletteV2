# -*- coding: utf-8 -*-
"""
Analyse vibratoire par transformée en ondelettes - Version Améliorée
Améliorations : Performance, UX, Fonctionnalités avancées, Gestion d'erreurs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt, spectrogram, welch
from scipy.stats import kurtosis, skew
import pywt
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse Vibratoire Avancée",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal avec style
st.markdown("""
# ⚙️ Analyse Vibratoire par Transformée en Ondelettes
### Version Améliorée avec Diagnostics Avancés

Cette application effectue une analyse vibratoire complète en utilisant la transformée en ondelettes. 
**Nouvelles fonctionnalités** :
- 📊 **Statistiques avancées** du signal
- 🔍 **Détection automatique de pics**
- 📈 **Analyse spectrale comparative**
- 🎯 **Diagnostic automatisé**
- 📱 **Interface responsive améliorée**

*Développé par **M. A Angelico** et **ZARAVITA** - Version Améliorée*
""")

# Cache amélioré avec gestion d'erreurs
@st.cache_data(ttl=3600)  # Cache pendant 1 heure
def load_bearing_data():
    """Charge les données de roulements avec gestion d'erreurs améliorée"""
    # Données par défaut étendues
    default_data = {
        'Manufacturer': ['SKF', 'SKF', 'SKF', 'FAG', 'FAG', 'NSK', 'NSK', 'TIMKEN', 'TIMKEN', 'NTN',
                        'AMI', 'AMI', 'DODGE', 'DODGE', 'FAFNIR', 'FAFNIR', 'KOYO', 'KOYO', 
                        'SEALMASTER', 'SNR', 'SNR', 'TORRINGTON', 'TORRINGTON'],
        'Name': ['6205', '6206', '6207', '6305', '6306', '6005', '6006', '30205', '30206', '6305',
                '201', '202', 'P2B5__USAF115TTAH (B)', 'P2B5__USAF115TTAH (C)', '206NPP', 
                '206NPPA1849', '7304B (B)', '7304B (C)', '204', '6316ZZ (B)', 'NU324', '23172B', '23172BW33C08BR'],
        'Number of Rollers': [9, 9, 9, 8, 8, 9, 9, 13, 13, 8,
                             8, 8, 18, 17, 9, 9, 9, 9, 21, 8, 13, 22, 22],
        'FTF': [0.4, 0.4, 0.4, 0.38, 0.38, 0.42, 0.42, 0.39, 0.39, 0.38,
               0.383, 0.383, 0.42, 0.57, 0.39, 0.39, 0.38, 0.38, 0.4404, 0.38, 0.4, 0.44, 0.44],
        'BSF': [2.37, 2.37, 2.37, 2.08, 2.08, 2.54, 2.54, 2.69, 2.69, 2.08,
               2.025, 2.025, 3.22, 6.49, 2.31, 2.31, 1.79, 1.79, 7.296, 2.07, 2.42, 4.16, 4.16],
        'BPFO': [3.6, 3.6, 3.6, 3.05, 3.05, 3.81, 3.81, 4.94, 4.94, 3.05,
                3.066, 3.066, 7.65, 7.24, 3.56, 3.56, 3.47, 3.46, 9.2496, 3.08, 5.21, 9.71, 9.71],
        'BPFI': [5.4, 5.4, 5.4, 4.95, 4.95, 5.19, 5.19, 8.06, 8.06, 4.95,
                4.934, 4.934, 10.34, 9.75, 5.43, 5.43, 5.53, 5.53, 11.7504, 4.91, 7.78, 12.28, 12.28]
    }
    
    try:
        # Tentative de chargement depuis GitHub (URL corrigée)
        url = "https://raw.githubusercontent.com/ZARAVITA/Analyse_vibratoire_par_ondelettes/main/Bearing%20data%20Base.csv"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Essayer différents formats
        try:
            bearing_data = pd.read_csv(BytesIO(response.content))
        except:
            bearing_data = pd.read_excel(BytesIO(response.content))
        
        # Nettoyage des données
        bearing_data = bearing_data.dropna(subset=['Manufacturer'])
        bearing_data['Manufacturer'] = bearing_data['Manufacturer'].astype(str).str.strip()
        
        for col in ['FTF', 'BSF', 'BPFO', 'BPFI']:
            if col in bearing_data.columns:
                bearing_data[col] = pd.to_numeric(bearing_data[col], errors='coerce')
        
        bearing_data = bearing_data.dropna(subset=['FTF','BSF','BPFO', 'BPFI'])
        
        st.success("✅ Données de roulements chargées depuis GitHub")
        return bearing_data
        
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger depuis GitHub ({str(e)}). Utilisation des données par défaut.")
        return pd.DataFrame(default_data)

# Fonctions de traitement du signal améliorées
def advanced_signal_stats(signal):
    """Calcule des statistiques avancées du signal"""
    return {
        'RMS': np.sqrt(np.mean(signal**2)),
        'Peak': np.max(np.abs(signal)),
        'Crest Factor': np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
        'Kurtosis': kurtosis(signal),
        'Skewness': skew(signal),
        'Energy': np.sum(signal**2)
    }

def detect_peaks_auto(signal, time, prominence=None):
    """Détection automatique de pics avec prominence adaptative"""
    from scipy.signal import find_peaks
    
    if prominence is None:
        prominence = np.std(signal) * 2
    
    peaks, properties = find_peaks(signal, prominence=prominence, distance=len(signal)//100)
    return peaks, properties

def create_enhanced_figure(x, y, title, x_title, y_title, stats=None):
    """Crée un graphique amélioré avec statistiques"""
    fig = go.Figure()
    
    # Signal principal
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name='Signal',
        line=dict(width=1),
        hovertemplate='%{x:.3f} s<br>%{y:.3f}<extra></extra>'
    ))
    
    # Détection automatique de pics
    peaks, _ = detect_peaks_auto(y, x)
    if len(peaks) > 0:
        fig.add_trace(go.Scatter(
            x=x[peaks], y=y[peaks],
            mode='markers',
            name='Pics détectés',
            marker=dict(size=8, color='red', symbol='triangle-up'),
            hovertemplate='Pic: %{x:.3f} s<br>%{y:.3f}<extra></extra>'
        ))
    
    # Ligne de moyenne et écart-type
    mean_val = np.mean(y)
    std_val = np.std(y)
    
    fig.add_hline(y=mean_val, line_dash="dash", line_color="green", 
                  annotation_text=f"Moyenne: {mean_val:.3f}")
    fig.add_hline(y=mean_val + 2*std_val, line_dash="dot", line_color="orange", 
                  annotation_text=f"+2σ: {mean_val + 2*std_val:.3f}")
    fig.add_hline(y=mean_val - 2*std_val, line_dash="dot", line_color="orange", 
                  annotation_text=f"-2σ: {mean_val - 2*std_val:.3f}")
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    return fig

# Interface utilisateur améliorée
def create_sidebar():
    """Crée une sidebar améliorée avec validation"""
    st.sidebar.header("🔧 Paramètres de Configuration")
    
    # Chargement des données
    with st.sidebar.expander("📊 Base de Données", expanded=True):
        bearing_data = load_bearing_data()
        st.info(f"**{len(bearing_data)}** roulements disponibles")
    
    # Sélection du roulement avec validation
    st.sidebar.subheader("⚙️ Sélection du Roulement")
    
    manufacturers = sorted(bearing_data['Manufacturer'].unique())
    selected_manufacturer = st.sidebar.selectbox("🏭 Fabricant", manufacturers)
    
    models = bearing_data[bearing_data['Manufacturer'] == selected_manufacturer]['Name'].unique()
    selected_model = st.sidebar.selectbox("🔩 Modèle", models)
    
    selected_bearing = bearing_data[
        (bearing_data['Manufacturer'] == selected_manufacturer) & 
        (bearing_data['Name'] == selected_model)
    ].iloc[0]
    
    roller_count = selected_bearing['Number of Rollers']
    st.sidebar.success(f"**Rouleaux:** {roller_count}")
    
    # Vitesse avec validation
    st.sidebar.subheader("🔄 Conditions de Fonctionnement")
    rotation_speed_rpm = st.sidebar.number_input(
        "Vitesse (RPM)", 
        min_value=1, max_value=10000, value=1000, step=10
    )
    rotation_speed_hz = rotation_speed_rpm / 60
    
    st.sidebar.info(f"**Fréquence:** {rotation_speed_hz:.2f} Hz")
    
    # Calcul des fréquences caractéristiques
    frequencies = {
        'FTF': selected_bearing['FTF'] * rotation_speed_hz,
        'BSF': selected_bearing['BSF'] * rotation_speed_hz,
        'BPFO': selected_bearing['BPFO'] * rotation_speed_hz,
        'BPFI': selected_bearing['BPFI'] * rotation_speed_hz
    }
    
    # Affichage des fréquences avec codes couleur
    st.sidebar.subheader("📊 Fréquences Caractéristiques")
    colors = {'FTF': '🟣', 'BSF': '🟢', 'BPFO': '🔵', 'BPFI': '🔴'}
    
    freq_display = {}
    for freq_type, freq_val in frequencies.items():
        st.sidebar.markdown(f"{colors[freq_type]} **{freq_type}:** {freq_val:.2f} Hz")
        freq_display[freq_type] = freq_val
    
    # Options d'affichage améliorées
    st.sidebar.subheader("🎨 Options d'Affichage")
    display_options = {}
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        display_options['FTF'] = st.checkbox("FTF", False)
        display_options['BPFO'] = st.checkbox("BPFO", True)
    
    with col2:
        display_options['BSF'] = st.checkbox("BSF", False)
        display_options['BPFI'] = st.checkbox("BPFI", True)
    
    display_options['harmonics'] = st.checkbox("Harmoniques", False)
    if display_options['harmonics']:
        display_options['harmonics_count'] = st.slider("Nombre d'harmoniques", 1, 10, 3)
    
    # Paramètres des filtres avec validation
    st.sidebar.subheader("🔧 Paramètres de Filtrage")
    filter_params = {
        'highpass_freq': st.sidebar.slider("Passe-haut (Hz)", 10, 2000, 100),
        'lowpass_freq': st.sidebar.slider("Passe-bas (Hz)", 10, 1000, 200),
        'filter_order': st.sidebar.selectbox("Ordre du filtre", [2, 4, 6, 8], index=1)
    }
    
    # Paramètres des ondelettes
    st.sidebar.subheader("🌊 Paramètres des Ondelettes")
    wavelet_params = {
        'type': st.sidebar.selectbox(
            "Type d'ondelette",
            ['morl', 'cmor', 'cgau', 'gaus', 'mexh'],
            index=0
        ),
        'scale_min': st.sidebar.number_input("Échelle min", 1, 50, 1),
        'scale_max': st.sidebar.number_input("Échelle max", 51, 500, 128),
        'scale_step': st.sidebar.number_input("Pas", 1, 10, 2)
    }
    
    return selected_bearing, freq_display, display_options, filter_params, wavelet_params

# Interface principale
def main():
    # Création de la sidebar
    bearing_info, frequencies, display_opts, filter_params, wavelet_params = create_sidebar()
    
    # Zone principale
    uploaded_file = st.file_uploader(
        "📁 **Importez votre fichier CSV**", 
        type=["csv"],
        help="Le fichier doit contenir les colonnes 'time' et 'amplitude'"
    )
    
    if uploaded_file is not None:
        try:
            # Lecture améliorée du fichier
            data = pd.read_csv(uploaded_file, sep=";", skiprows=1)
            
            if data.shape[1] < 2:
                st.error("❌ Le fichier doit contenir au moins 2 colonnes (time, amplitude)")
                return
            
            time = data.iloc[:, 0].values / 1000  # Conversion en secondes
            amplitude = data.iloc[:, 1].values
            
            # Validation des données
            if len(time) < 100:
                st.warning("⚠️ Signal très court, les résultats peuvent être imprécis")
            
            # Calcul de la fréquence d'échantillonnage
            dt = np.diff(time)
            fs = 1 / np.mean(dt)
            
            # Interface à onglets pour une meilleure organisation
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Données", "🔍 Signal Original", "⚙️ Traitement", "🌊 Ondelettes", "📈 Diagnostic"
            ])
            
            with tab1:
                st.subheader("📊 Informations sur les Données")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📏 Longueur", f"{len(time):,} points")
                with col2:
                    st.metric("🕐 Durée", f"{time[-1]:.2f} s")
                with col3:
                    st.metric("📊 Fréq. Éch.", f"{fs:.0f} Hz")
                with col4:
                    st.metric("🔄 Nyquist", f"{fs/2:.0f} Hz")
                
                if st.checkbox("Afficher les premières lignes"):
                    st.dataframe(data.head(10))
                
                # Statistiques de base
                stats = advanced_signal_stats(amplitude)
                st.subheader("📈 Statistiques du Signal")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMS", f"{stats['RMS']:.3f}")
                    st.metric("Peak", f"{stats['Peak']:.3f}")
                
                with col2:
                    st.metric("Crest Factor", f"{stats['Crest Factor']:.2f}")
                    st.metric("Kurtosis", f"{stats['Kurtosis']:.2f}")
                
                with col3:
                    st.metric("Skewness", f"{stats['Skewness']:.2f}")
                    st.metric("Energy", f"{stats['Energy']:.1e}")
            
            with tab2:
                st.subheader("🔍 Signal Original")
                
                fig_orig = create_enhanced_figure(
                    time, amplitude, 
                    "Signal Original avec Détection de Pics",
                    "Temps (s)", "Amplitude"
                )
                
                st.plotly_chart(fig_orig, use_container_width=True)
                
                # Analyse spectrale du signal original
                if st.checkbox("Afficher l'analyse spectrale"):
                    freqs_fft, psd = welch(amplitude, fs, nperseg=min(2048, len(amplitude)//4))
                    
                    fig_fft = go.Figure()
                    fig_fft.add_trace(go.Scatter(
                        x=freqs_fft, y=10*np.log10(psd),
                        mode='lines',
                        name='PSD'
                    ))
                    
                    fig_fft.update_layout(
                        title="Densité Spectrale de Puissance",
                        xaxis_title="Fréquence (Hz)",
                        yaxis_title="PSD (dB/Hz)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_fft, use_container_width=True)
            
            with tab3:
                st.subheader("⚙️ Traitement BLSD")
                
                # Application des filtres avec gestion d'erreurs
                try:
                    # Filtre passe-haut
                    nyquist = 0.5 * fs
                    high_freq_norm = filter_params['highpass_freq'] / nyquist
                    
                    if high_freq_norm >= 1:
                        st.error("❌ Fréquence passe-haut trop élevée")
                        return
                    
                    b_high, a_high = butter(
                        filter_params['filter_order'], 
                        high_freq_norm, 
                        btype='high'
                    )
                    signal_highpass = filtfilt(b_high, a_high, amplitude)
                    
                    # Redressement
                    signal_rectified = np.abs(signal_highpass)
                    
                    # Filtre passe-bas
                    low_freq_norm = filter_params['lowpass_freq'] / nyquist
                    
                    if low_freq_norm >= 1:
                        st.error("❌ Fréquence passe-bas trop élevée")
                        return
                    
                    b_low, a_low = butter(
                        filter_params['filter_order'], 
                        low_freq_norm, 
                        btype='low'
                    )
                    signal_processed = filtfilt(b_low, a_low, signal_rectified)
                    
                    # Affichage des signaux avec sous-graphiques
                    fig_processing = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Signal Original', 'Après Passe-Haut', 
                                       'Après Redressement', 'Signal Final'),
                        vertical_spacing=0.1
                    )
                    
                    signals = [amplitude, signal_highpass, signal_rectified, signal_processed]
                    titles = ['Original', 'Passe-Haut', 'Redressé', 'Final']
                    
                    for i, (signal, title) in enumerate(zip(signals, titles)):
                        row = (i // 2) + 1
                        col = (i % 2) + 1
                        
                        fig_processing.add_trace(
                            go.Scatter(x=time, y=signal, name=title, line=dict(width=1)),
                            row=row, col=col
                        )
                    
                    fig_processing.update_layout(
                        height=600,
                        title_text="Étapes du Traitement BLSD",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_processing, use_container_width=True)
                    
                    # Comparaison des statistiques
                    st.subheader("📊 Comparaison Avant/Après Traitement")
                    
                    stats_orig = advanced_signal_stats(amplitude)
                    stats_proc = advanced_signal_stats(signal_processed)
                    
                    comparison_df = pd.DataFrame({
                        'Original': [f"{v:.3f}" for v in stats_orig.values()],
                        'Traité': [f"{v:.3f}" for v in stats_proc.values()],
                        'Amélioration': [f"{((v2-v1)/v1*100):+.1f}%" 
                                       for v1, v2 in zip(stats_orig.values(), stats_proc.values())]
                    }, index=stats_orig.keys())
                    
                    st.dataframe(comparison_df)
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement: {str(e)}")
                    return
            
            with tab4:
                st.subheader("🌊 Analyse par Ondelettes")
                
                if st.button("🚀 Lancer l'Analyse CWT", type="primary"):
                    with st.spinner("Calcul en cours..."):
                        try:
                            # Calcul de la CWT
                            scales = np.arange(
                                wavelet_params['scale_min'], 
                                wavelet_params['scale_max'], 
                                wavelet_params['scale_step']
                            )
                            
                            coeffs, freqs_cwt = pywt.cwt(
                                signal_processed, 
                                scales, 
                                wavelet_params['type'], 
                                sampling_period=1/fs
                            )
                            
                            # Création du scalogramme amélioré
                            fig_cwt = go.Figure()
                            
                            # Scalogramme principal
                            fig_cwt.add_trace(go.Heatmap(
                                z=20*np.log10(np.abs(coeffs) + 1e-12),  # En dB
                                x=time,
                                y=freqs_cwt,
                                colorscale='Jet',
                                colorbar=dict(title="Amplitude (dB)"),
                                hoverongaps=False
                            ))
                            
                            # Ajout des fréquences caractéristiques
                            freq_colors = {
                                'FTF': 'violet', 'BSF': 'green', 
                                'BPFO': 'blue', 'BPFI': 'red'
                            }
                            
                            for freq_type, show in display_opts.items():
                                if freq_type in frequencies and show:
                                    freq_val = frequencies[freq_type]
                                    
                                    # Ligne principale
                                    fig_cwt.add_hline(
                                        y=freq_val,
                                        line=dict(color=freq_colors[freq_type], width=2, dash='dot'),
                                        annotation_text=freq_type,
                                        annotation_position="right"
                                    )
                                    
                                    # Harmoniques
                                    if display_opts.get('harmonics', False):
                                        for h in range(2, display_opts.get('harmonics_count', 3) + 1):
                                            fig_cwt.add_hline(
                                                y=freq_val * h,
                                                line=dict(color=freq_colors[freq_type], width=1, dash='dot'),
                                                annotation_text=f"{h}×{freq_type}",
                                                annotation_position="right"
                                            )
                            
                            fig_cwt.update_layout(
                                title="Scalogramme - Transformée en Ondelettes Continue",
                                xaxis_title="Temps (s)",
                                yaxis_title="Fréquence (Hz)",
                                height=600,
                                yaxis_type='log' if st.checkbox("Échelle log") else 'linear'
                            )
                            
                            st.plotly_chart(fig_cwt, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'analyse CWT: {str(e)}")
            
            with tab5:
                st.subheader("📈 Diagnostic Automatisé")
                
                # Calcul des indicateurs de santé
                health_indicators = {
                    'RMS Global': stats['RMS'],
                    'Facteur de Crête': stats['Crest Factor'],
                    'Kurtosis': stats['Kurtosis'],
                    'Énergie Total': stats['Energy']
                }
                
                # Seuils d'alerte (à adapter selon l'application)
                thresholds = {
                    'RMS Global': {'warning': 2.0, 'critical': 5.0},
                    'Facteur de Crête': {'warning': 4.0, 'critical': 8.0},
                    'Kurtosis': {'warning': 4.0, 'critical': 6.0}
                }
                
                # Évaluation de l'état
                st.subheader("🚦 État de Santé du Roulement")
                
                overall_status = "🟢 BON"
                alerts = []
                
                for indicator, value in health_indicators.items():
                    if indicator in thresholds:
                        if value > thresholds[indicator]['critical']:
                            overall_status = "🔴 CRITIQUE"
                            alerts.append(f"⚠️ {indicator}: {value:.2f} (Critique)")
                        elif value > thresholds[indicator]['warning']:
                            if overall_status == "🟢 BON":
                                overall_status = "🟡 ATTENTION"
                            alerts.append(f"⚠️ {indicator}: {value:.2f} (Attention)")
                
                st.markdown(f"### État Global: {overall_status}")
                
                if alerts:
                    st.error("Alertes détectées:")
                    for alert in alerts:
                        st.write(alert)
                else:
                    st.success("✅ Tous les indicateurs sont dans les limites normales")
                
                # Recommandations
                st.subheader("💡 Recommandations")
                
                if overall_status == "🔴 CRITIQUE":
                    st.error("""
                    🚨 **INTERVENTION URGENTE REQUISE**
                    - Arrêter la machine dès que possible
                    - Inspecter visuellement le roulement
