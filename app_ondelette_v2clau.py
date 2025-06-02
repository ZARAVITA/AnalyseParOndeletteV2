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
from scipy.signal import butter, filtfilt, welch, get_window
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
- ⚙️ **Traitement BLSD**
- 📱 **Interface responsive améliorée**
- 🪟 **Fenêtrage de Hanning** pour les analyses spectrales

*Développé par **M. A Angelico** et **ZARAVITA** - Version Améliorée*
""")


#-------------------------------------------------------------------------claude------------------------------------------------------------------------------------
def load_bearing_data():
    """Charge les données des roulements depuis GitHub avec gestion d'erreurs robuste"""
    urls = [
        "https://raw.githubusercontent.com/ZARAVITA/AnalyseParOndeletteV2/main/Bearing%20data%20Base.csv"
    ]
    
    for url in urls:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            content = response.content.decode('utf-8')
            separators = [',', ';', '\t']
            
            for sep in separators:
                try:
                    bearing_data = pd.read_csv(
                        BytesIO(response.content), 
                        sep=sep,
                        encoding='utf-8',
                        on_bad_lines='skip'
                    )
                    
                    if len(bearing_data.columns) >= 5 and len(bearing_data) > 0:
                        # Nettoyage des données
                        bearing_data = bearing_data.dropna(subset=[bearing_data.columns[0]])
                        
                        # Standardisation des noms de colonnes
                        expected_cols = ['Manufacturer', 'Name', 'Number of Rollers', 'FTF', 'BSF', 'BPFO', 'BPFI']
                        if len(bearing_data.columns) >= len(expected_cols):
                            bearing_data.columns = expected_cols[:len(bearing_data.columns)]
                        
                        # Conversion des colonnes numériques
                        numeric_cols = ['Number of Rollers', 'FTF', 'BSF', 'BPFO', 'BPFI']
                        for col in numeric_cols:
                            if col in bearing_data.columns:
                                bearing_data[col] = pd.to_numeric(bearing_data[col], errors='coerce')
                        
                        # Nettoyage final
                        bearing_data = bearing_data.dropna(subset=['FTF', 'BSF', 'BPFO', 'BPFI'])
                        
                        if len(bearing_data) > 0:
                            return bearing_data
                
                except Exception:
                    continue
            
        except Exception:
            continue
    
    # Données par défaut si échec
    default_data = {
        'Manufacturer': ['SKF', 'FAG', 'TIMKEN', 'NSK', 'NTN'],
        'Name': ['6206', '6208', '6210', '6306', '6308'],
        'Number of Rollers': [9, 8, 9, 8, 8],
        'FTF': [0.398, 0.383, 0.404, 0.382, 0.382],
        'BSF': [2.357, 2.027, 2.384, 2.032, 2.032],
        'BPFO': [3.581, 3.052, 3.634, 3.053, 3.053],
        'BPFI': [5.419, 4.948, 5.366, 4.947, 4.947]
    }
    
    return pd.DataFrame(default_data)
    
    # Si toutes les tentatives échouent, utiliser les données par défaut
    st.error("❌ Impossible de charger les données depuis GitHub")
    st.info("🔄 Utilisation des données par défaut intégrées")
    
    # Données par défaut étendues et corrigées
    default_data = {
        'Manufacturer': [
            'SKF', 'SKF', 'SKF', 'SKF', 'SKF',
            'FAG', 'FAG', 'FAG', 'FAG', 'FAG',
            'TIMKEN', 'TIMKEN', 'TIMKEN', 'TIMKEN', 'TIMKEN',
            'NSK', 'NSK', 'NSK', 'NSK', 'NSK',
            'NTN', 'NTN', 'NTN', 'NTN', 'NTN'
        ],
        'Name': [
            '6206', '6208', '6210', '6306', '6308',
            '6206', '6208', '6210', '6306', '6308',
            '6206', '6208', '6210', '6306', '6308',
            '6206', '6208', '6210', '6306', '6308',
            '6206', '6208', '6210', '6306', '6308'
        ],
        'Number of Rollers': [
            9, 8, 9, 8, 8,
            9, 8, 9, 8, 8,
            9, 8, 9, 8, 8,
            9, 8, 9, 8, 8,
            9, 8, 9, 8, 8
        ],
        'FTF': [
            0.398, 0.383, 0.404, 0.382, 0.382,
            0.398, 0.383, 0.404, 0.382, 0.382,
            0.398, 0.383, 0.404, 0.382, 0.382,
            0.398, 0.383, 0.404, 0.382, 0.382,
            0.398, 0.383, 0.404, 0.382, 0.382
        ],
        'BSF': [
            2.357, 2.027, 2.384, 2.032, 2.032,
            2.357, 2.027, 2.384, 2.032, 2.032,
            2.357, 2.027, 2.384, 2.032, 2.032,
            2.357, 2.027, 2.384, 2.032, 2.032,
            2.357, 2.027, 2.384, 2.032, 2.032
        ],
        'BPFO': [
            3.581, 3.052, 3.634, 3.053, 3.053,
            3.581, 3.052, 3.634, 3.053, 3.053,
            3.581, 3.052, 3.634, 3.053, 3.053,
            3.581, 3.052, 3.634, 3.053, 3.053,
            3.581, 3.052, 3.634, 3.053, 3.053
        ],
        'BPFI': [
            5.419, 4.948, 5.366, 4.947, 4.947,
            5.419, 4.948, 5.366, 4.947, 4.947,
            5.419, 4.948, 5.366, 4.947, 4.947,
            5.419, 4.948, 5.366, 4.947, 4.947,
            5.419, 4.948, 5.366, 4.947, 4.947
        ]
    }
    
    return pd.DataFrame(default_data)

# Test de connexion initial
def test_github_connection():
    """Test la connexion à GitHub"""
    try:
        response = requests.get("https://github.com", timeout=5)
        return response.status_code == 200
    except:
        return False

# Affichage du statut de connexion
with st.sidebar:
    if test_github_connection():
        st.success("🌐 Connexion GitHub OK")
    else:
        st.warning("⚠️ Connexion GitHub limitée")

# Charger les données des roulements avec feedback utilisateur
with st.spinner("🔄 Chargement des données des roulements..."):
    bearing_data = load_bearing_data()
#------------------------------------------------------------------------------------------------------------------------------------------------------FIN



# Charger les données des roulements
bearing_data = load_bearing_data()

# Fonctions de traitement du signal améliorées
def advanced_signal_stats(signal):
    """Calcule des statistiques avancées du signal"""
    return {
        'RMS': np.sqrt(np.mean(signal**2)),
        'Peak': np.max(np.abs(signal)),
        'Crest Factor': np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
        'Kurtosis': kurtosis(signal),
        'Skewness': skew(signal),
        'Energy': float(np.sum(signal**2))
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

def apply_hanning_window(signal):
    """Applique une fenêtre de Hanning au signal"""
    window = get_window('hann', len(signal))
    return signal * window

def calculate_fft(signal, fs, apply_window=True):
    """Calcule la FFT du signal avec option de fenêtrage"""
    n = len(signal)
    if apply_window:
        signal = apply_hanning_window(signal)
    yf = np.fft.fft(signal)
    xf = np.fft.fftfreq(n, 1/fs)[:n//2]
    return xf, 2.0/n * np.abs(yf[0:n//2])

# Interface utilisateur améliorée
def create_sidebar():
    """Crée une sidebar améliorée avec validation"""
    st.sidebar.header("🔧 Paramètres de Configuration")
    
    # Chargement des données
    with st.sidebar.expander("📊 Base de Données", expanded=True):
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
    
    # Paramètres des filtres avec validation
    st.sidebar.subheader("🔧 Paramètres de Filtrage")
    #----------------------------------------------------------------------------En entier TOUS
    filter_params = {
        'highpass_freq': st.sidebar.slider("Passe-haut (Hz)", 10.0, 2000.0, 500.0),
        'lowpass_freq': st.sidebar.slider("Passe-bas (Hz)", 10.0, 1000.0, 200.0),
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
    
    return selected_bearing, filter_params, wavelet_params

# Interface principale
def main():
    # Création de la sidebar
    bearing_info, filter_params, wavelet_params = create_sidebar()
    
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
                
                # Analyse spectrale du signal original avec fenêtre de Hanning
                if st.checkbox("Afficher l'analyse spectrale (avec fenêtre de Hanning)"):
                    freqs_fft, psd = welch(amplitude, fs, window='hann', nperseg=min(2048, len(amplitude)//4))
                    
                    fig_fft = go.Figure()
                    fig_fft.add_trace(go.Scatter(
                        x=freqs_fft, y=10*np.log10(psd),
                        mode='lines',
                        name='PSD'
                    ))
                    
                    fig_fft.update_layout(
                        title="Densité Spectrale de Puissance (Fenêtre de Hanning)",
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
                        'Amélioration': [f"{((v2-v1)/v1*100):+.1f}%" if v1 != 0 else 'N/A'
                                        for v1, v2 in zip(stats_orig.values(), stats_proc.values())]
                    }, index=stats_orig.keys())
                    
                    st.dataframe(comparison_df)
                    
                    ########################################################################
                    # SECTION: OPTIONS D'AFFICHAGE DU SPECTRE
                    ########################################################################
                    st.subheader("🎯 Options d'Affichage du Spectre")
                    
                    # Entrée personnalisée pour la vitesse de rotation
                    # Choix de l'unité d'entrée
                    unit_choice = st.radio(
                        "Unité de fréquence de rotation:",
                        ["Hz", "RPM"],
                        horizontal=True
                        )
        
                    # Entrée personnalisée pour la vitesse de rotation
                    if unit_choice == "RPM":
                        rotation_input = st.number_input(
                        "Fréquence de rotation (RPM)",
                        min_value=1.0,
                        max_value=30000.0,
                        value=1000.0,
                        step=1.0
                        )
                     # Conversion RPM vers Hz
                        custom_hz = rotation_input / 60.0
                        st.info(f"**Fréquence de rotation:** {rotation_input:.1f} RPM = {custom_hz:.2f} Hz")
                    else:
                      custom_hz = st.number_input(
                         "Fréquence de rotation (Hz)",
                         min_value=0.017,  # ~1 RPM
                         max_value=1000.0,
                         value=16.67,
                         step=0.01
                      )
                      rotation_rpm = custom_hz * 60.0
                      st.info(f"**Fréquence de rotation:** {custom_hz:.2f} Hz = {rotation_rpm:.1f} RPM")
                    #--------------------------------------------
                    #st.info(f"**Fréquence de rotation calculée:** {custom_hz:.2f} Hz")
                    
                    # Calcul des fréquences caractéristiques personnalisées
                    frequencies = {
                        'FTF': bearing_info['FTF'] * custom_hz,
                        'BSF': bearing_info['BSF'] * custom_hz,
                        'BPFO': bearing_info['BPFO'] * custom_hz,
                        'BPFI': bearing_info['BPFI'] * custom_hz
                    }
                    
                    # Sélection des fréquences à afficher - SUR UNE SEULE LIGNE
                    st.write("**Sélectionnez les fréquences caractéristiques à afficher:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        show_ftf = st.checkbox("FTF", False, key='ftf_spectrum')
                    with col2:
                        show_bsf = st.checkbox("BSF", False, key='bsf_spectrum')
                    with col3:
                        show_bpfo = st.checkbox("BPFO", False, key='bpfo_spectrum')
                    with col4:
                        show_bpfi = st.checkbox("BPFI", False, key='bpfi_spectrum')
                    
                    # Options pour les harmoniques - APRÈS LA SÉLECTION DES FRÉQUENCES
                    st.subheader("📐 Options des Harmoniques")
                    
                    show_harmonics = st.checkbox("Afficher les harmoniques des fréquences caractéristiques", False)
                    if show_harmonics:
                        harmonics_count = st.slider("Nombre d'harmoniques à afficher", 1, 5, 3)
                    
                    show_speed_harmonics = st.checkbox("Afficher les harmoniques de vitesse", False)
                    if show_speed_harmonics:
                        speed_harmonics_count = st.slider("Nombre d'harmoniques de vitesse", 1, 5, 3)
                        speed_harmonics_color = st.color_picker("Couleur des harmoniques de vitesse", "#FFA500")
                    
                    ########################################################################
                    # SPECTRE DU SIGNAL TRAITÉ AVEC LES OPTIONS PERSONNALISÉES
                    ########################################################################
                    st.subheader("📈 Spectre du Signal Traité")
                    
                    # Calcul de la FFT avec fenêtre de Hanning
                    fft_freq, fft_amp = calculate_fft(signal_processed, fs, apply_window=True)
                    
                    # Création du graphique
                    fig_fft_proc = go.Figure()
                    fig_fft_proc.add_trace(go.Scatter(
                        x=fft_freq, 
                        y=fft_amp,
                        mode='lines',
                        name='Spectre FFT'
                    ))
                    
                    # Couleurs pour les fréquences caractéristiques
                    freq_colors = {
                        'FTF': 'violet',
                        'BSF': 'green',
                        'BPFO': 'blue',
                        'BPFI': 'red'
                    }
                    
                    # Ajout des fréquences caractéristiques sélectionnées
                    if show_ftf:
                        fig_fft_proc.add_vline(
                            x=frequencies['FTF'],
                            line_dash="dash",
                            line_color=freq_colors['FTF'],
                            annotation_text="FTF",
                            annotation_position="top right"
                        )
                        if show_harmonics:
                            for h in range(2, harmonics_count + 1):
                                fig_fft_proc.add_vline(
                                    x=frequencies['FTF'] * h,
                                    line_dash="dot",
                                    line_color=freq_colors['FTF'],
                                    annotation_text=f"{h}×FTF",
                                    annotation_position="top right"
                                )
                    
                    if show_bsf:
                        fig_fft_proc.add_vline(
                            x=frequencies['BSF'],
                            line_dash="dash",
                            line_color=freq_colors['BSF'],
                            annotation_text="BSF",
                            annotation_position="top right"
                        )
                        if show_harmonics:
                            for h in range(2, harmonics_count + 1):
                                fig_fft_proc.add_vline(
                                    x=frequencies['BSF'] * h,
                                    line_dash="dot",
                                    line_color=freq_colors['BSF'],
                                    annotation_text=f"{h}×BSF",
                                    annotation_position="top right"
                                )
                    
                    if show_bpfo:
                        fig_fft_proc.add_vline(
                            x=frequencies['BPFO'],
                            line_dash="dash",
                            line_color=freq_colors['BPFO'],
                            annotation_text="BPFO",
                            annotation_position="top right"
                        )
                        if show_harmonics:
                            for h in range(2, harmonics_count + 1):
                                fig_fft_proc.add_vline(
                                    x=frequencies['BPFO'] * h,
                                    line_dash="dot",
                                    line_color=freq_colors['BPFO'],
                                    annotation_text=f"{h}×BPFO",
                                    annotation_position="top right"
                                )
                    
                    if show_bpfi:
                        fig_fft_proc.add_vline(
                            x=frequencies['BPFI'],
                            line_dash="dash",
                            line_color=freq_colors['BPFI'],
                            annotation_text="BPFI",
                            annotation_position="top right"
                        )
                        if show_harmonics:
                            for h in range(2, harmonics_count + 1):
                                fig_fft_proc.add_vline(
                                    x=frequencies['BPFI'] * h,
                                    line_dash="dot",
                                    line_color=freq_colors['BPFI'],
                                    annotation_text=f"{h}×BPFI",
                                    annotation_position="top right"
                                )
                    
                    # Ajout des harmoniques de vitesse si activé
                    if show_speed_harmonics:
                        for h in range(1, speed_harmonics_count + 1):
                            harmonic_freq = h * custom_hz
                            fig_fft_proc.add_vline(
                                x=harmonic_freq,
                                line_dash="dash",
                                line_color=speed_harmonics_color,
                                annotation_text=f"{h}×Vit. Rot.",
                                annotation_position="bottom right"
                            )
                    
                    fig_fft_proc.update_layout(
                        title="Spectre FFT du Signal Traité (Fenêtre de Hanning)",
                        xaxis_title="Fréquence (Hz)",
                        yaxis_title="Amplitude",
                        height=500
                    )
                    
                    st.plotly_chart(fig_fft_proc, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Oh Oh Erreur lors du traitement: {str(e)}")
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
                            display_opts = {
                              'FTF': show_ftf,
                              'BSF': show_bsf,
                              'BPFO': show_bpfo,
                              'BPFI': show_bpfi,
                              'harmonics': show_harmonics,
                              'harmonics_count': harmonics_count,
                              'show_speed_harmonics': show_speed_harmonics,
                              'speed_harmonics_count': speed_harmonics_count,
                              'speed_harmonics_color': speed_harmonics_color
                            }
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
                            #---------------------------------------------------------------------
                            freq_options = {
                                'FTF': show_ftf,
                                'BSF': show_bsf,
                                'BPFO': show_bpfo,
                                'BPFI': show_bpfi
                            }
                            
                            for freq_type, show in freq_options.items():
                                if show and freq_type in frequencies:
                                    freq_val = frequencies[freq_type]
                                    
                                    # Ligne principale
                                    fig_cwt.add_hline(
                                        y=freq_val,
                                        line=dict(color=freq_colors[freq_type], width=2, dash='dot'),
                                        annotation_text=freq_type,
                                        annotation_position="right"
                                    )

                            #---------------------------------------------------------------------
                            
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
                                            # Affichage du scalogramme amélioré
                st.plotly_chart(fig_cwt, use_container_width=True)
                
                ########################################################################
                # NOUVELLE SECTION: ANALYSE À FRÉQUENCE SPÉCIFIQUE
                ########################################################################
                st.subheader("🎯 Analyse à Fréquence Spécifique")
                
                # Sélection de la fréquence d'analyse
                min_freq = float(np.min(freqs_cwt))
                max_freq = float(np.max(freqs_cwt))
                
                # Valeur par défaut intelligente
                default_freq = frequencies.get('BPFO', min_freq + (max_freq - min_freq)/3)
                
                selected_freq = st.slider(
                    "Sélectionnez une fréquence pour l'analyse (Hz)",
                    min_value=min_freq,
                    max_value=max_freq,
                    value=default_freq,
                    step=0.1,
                    format="%.1f"
                )
                
                # Trouver l'index le plus proche de la fréquence sélectionnée
                idx_freq = np.abs(freqs_cwt - selected_freq).argmin()
                coeffs_at_freq = coeffs[idx_freq, :]
                
                # Graphique des coefficients à cette fréquence
                fig_freq = go.Figure()
                fig_freq.add_trace(go.Scatter(
                    x=time,
                    y=np.abs(coeffs_at_freq),
                    mode='lines',
                    name=f'Fréquence {freqs_cwt[idx_freq]:.1f} Hz',
                    line=dict(width=2),
                    hovertemplate='Temps: %{x:.3f} s<br>Amplitude: %{y:.2f}<extra></extra>'
                ))
                
                # Ajouter les lignes verticales pour les fréquences caractéristiques
                for freq_type, freq_val in frequencies.items():
                    if freq_val < max_freq and freq_val > min_freq:
                        fig_freq.add_vline(
                            x=freq_val,
                            line_dash="dash",
                            line_color=freq_colors.get(freq_type, 'gray'),
                            annotation_text=freq_type,
                            annotation_position="top"
                        )
                
                fig_freq.update_layout(
                    title=f'Coefficients d\'ondelettes à {freqs_cwt[idx_freq]:.1f} Hz',
                    xaxis_title='Temps (s)',
                    yaxis_title='Amplitude',
                    height=400
                )
                
                st.plotly_chart(fig_freq, use_container_width=True)
                
                ########################################################################
                # ANALYSE DE L'ENVELOPPE
                ########################################################################
                st.subheader("📊 Analyse de l'Enveloppe du Signal")
                
                # Calcul de l'enveloppe du signal à cette fréquence
                envelope = np.abs(coeffs_at_freq)
                
                # FFT de l'enveloppe pour détecter les fréquences de modulation
                n = len(envelope)
                fft_envelope = np.fft.fft(envelope)
                freqs_envelope = np.fft.fftfreq(n, d=1/fs)[:n//2]
                fft_magnitude = 2.0/n * np.abs(fft_envelope[0:n//2])
                
                # Graphique de la FFT de l'enveloppe
                fig_envelope_fft = go.Figure()
                fig_envelope_fft.add_trace(go.Scatter(
                    x=freqs_envelope,
                    y=fft_magnitude,
                    mode='lines',
                    name='FFT de l\'enveloppe',
                    line=dict(width=2),
                    hovertemplate='Fréquence: %{x:.1f} Hz<br>Amplitude: %{y:.2f}<extra></extra>'
                ))
                
                # Ajouter les fréquences caractéristiques
                for freq_type, freq_val in frequencies.items():
                    if freq_val < max(freqs_envelope) and freq_val > min(freqs_envelope):
                        fig_envelope_fft.add_vline(
                            x=freq_val,
                            line_dash="dash",
                            line_color=freq_colors.get(freq_type, 'gray'),
                            annotation_text=freq_type,
                            annotation_position="top"
                        )
                
                # Ajouter les harmoniques de vitesse
                if show_speed_harmonics:
                    for h in range(1, speed_harmonics_count + 1):
                        harmonic_freq = h * custom_hz
                        fig_envelope_fft.add_vline(
                            x=harmonic_freq,
                            line_dash="dash",
                            line_color=speed_harmonics_color,
                            annotation_text=f"{h}×Vit. Rot.",
                            annotation_position="bottom"
                        )
                
                fig_envelope_fft.update_layout(
                    title='FFT de l\'enveloppe du signal à la fréquence sélectionnée',
                    xaxis_title='Fréquence (Hz)',
                    yaxis_title='Amplitude',
                    height=400
                )
                
                st.plotly_chart(fig_envelope_fft, use_container_width=True)
                
                ########################################################################
                # INTERPRÉTATION DES RÉSULTATS
                ########################################################################
                st.subheader("🔍 Interprétation des Résultats")
                
                # Trouver le pic principal dans la FFT de l'enveloppe
                main_peak_idx = np.argmax(fft_magnitude)
                main_peak_freq = freqs_envelope[main_peak_idx]
                main_peak_amp = fft_magnitude[main_peak_idx]
                
                # Comparer avec les fréquences caractéristiques
                closest_freq = ""
                min_diff = float('inf')
                for freq_type, freq_val in frequencies.items():
                    diff = abs(main_peak_freq - freq_val)
                    if diff < min_diff:
                        min_diff = diff
                        closest_freq = freq_type
                
                # Calculer le pourcentage d'écart
                deviation = min_diff / frequencies[closest_freq] * 100 if frequencies[closest_freq] != 0 else 100
                
                # Afficher les conclusions
                if main_peak_amp > np.mean(fft_magnitude) * 5:  # Seuil d'amplitude significative
                    if deviation < 5:  # Écart de moins de 5%
                        st.success(f"✅ Forte corrélation détectée avec la fréquence {closest_freq} ({frequencies[closest_freq]:.2f} Hz)")
                        st.markdown(f"""
                        **Diagnostic préliminaire:**
                        - Modulation détectée à **{main_peak_freq:.2f} Hz** (écart: {deviation:.1f}%)
                        - Correspond à la fréquence caractéristique **{closest_freq}**
                        - Suggère un défaut de type **{get_fault_type(closest_freq)}**
                        """)
                    else:
                        st.warning(f"⚠️ Modulation détectée à {main_peak_freq:.2f} Hz (écart: {deviation:.1f}% avec {closest_freq})")
                else:
                    st.info("ℹ️ Aucune modulation significative détectée dans la bande de fréquence analysée")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse CWT: {str(e)}")

                            
                        except Exception as e:
                            st.error(f"❌ TAB4 Erreur lors de l'analyse CWT: {str(e)}")
            with tab5:
                st.subheader("📈 Diagnostic Automatisé")
                
                # Calcul des indicateurs de santé
                health_indicators = {
                    'RMS Global': stats['RMS'],
                    'Facteur de Crête': stats['Crest Factor'],
                    'Kurtosis': stats['Kurtosis'],
                    'Énergie Total': stats['Energy']
                }
                
                # Seuils d'alerte
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
                    - Planifier le remplacement immédiat
                    """)
                elif overall_status == "🟡 ATTENTION":
                    st.warning("""
                    ⚠️ **Surveillance requise**
                    - Augmenter la fréquence de surveillance
                    - Vérifier les conditions de lubrification
                    - Planifier une intervention prochaine
                    """)
                else:
                    st.success("""
                    ✅ **État normal**
                    - Continuer la surveillance selon le planning
                    - Vérifier les paramètres de fonctionnement
                    """)
                
        except Exception as e:
            st.error(f"❌ OH!TAB5 Erreur lors du traitement du fichierr: {str(e)}")
    else:
        st.info("ℹ️ Veuillez télécharger un fichier CSV pour commencer l'analyse")

if __name__ == "__main__":
    main()
