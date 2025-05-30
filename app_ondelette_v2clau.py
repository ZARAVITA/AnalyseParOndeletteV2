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
- 🎯 **Diagnostic automatisé**
- 📱 **Interface responsive améliorée**
- 🪟 **Fenêtrage de Hanning** pour les analyses spectrales

*Développé par **M. A Angelico** et **ZARAVITA** - Version Améliorée*
""")


#-------------------------------------------------------------------------claude------------------------------------------------------------------------------------
def load_bearing_data():
    """Charge les données des roulements depuis GitHub avec gestion d'erreurs robuste"""
    
    # URLs alternatives à tester
    urls = [
        "https://raw.githubusercontent.com/ZARAVITA/AnalyseParOndeletteV2/main/Bearing%20data%20Base.csv"
    ]
    
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Décodage du contenu
            content = response.content.decode('utf-8')
            
            # Test de différents séparateurs
            separators = [',', ';', '\t']
            
            for sep in separators:
                try:
                    # Lecture du CSV avec différents paramètres
                    bearing_data = pd.read_csv(
                        BytesIO(response.content), 
                        sep=sep,
                        encoding='utf-8',
                        on_bad_lines='skip'
                    )
                    
                    # Vérification de la structure minimale
                    if len(bearing_data.columns) >= 5 and len(bearing_data) > 0:
                        st.success(f"✅ Données chargées avec succès! ({len(bearing_data)} lignes)")
                        
                        # Nettoyage des données
                        bearing_data = bearing_data.dropna(subset=[bearing_data.columns[0]])  # Première colonne non nulle
                        
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
                
                except Exception as e:
                    continue
            
        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ Échec URL {i+1}: {str(e)}")
            continue
        except Exception as e:
            st.warning(f"⚠️ Erreur URL {i+1}: {str(e)}")
            continue
    
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
                    custom_rpm = st.number_input(
                        "Vitesse de rotation personnalisée (RPM)",
                        min_value=1,
                        max_value=10000,
                        value=1000,
                        step=10
                    )
                    custom_hz = custom_rpm / 60
                    st.info(f"**Fréquence de rotation calculée:** {custom_hz:.2f} Hz")
                    
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
                        show_ftf = st.checkbox("FTF", True, key='ftf_spectrum')
                    with col2:
                        show_bsf = st.checkbox("BSF", True, key='bsf_spectrum')
                    with col3:
                        show_bpfo = st.checkbox("BPFO", True, key='bpfo_spectrum')
                    with col4:
                        show_bpfi = st.checkbox("BPFI", True, key='bpfi_spectrum')
                    
                    # Options pour les harmoniques - APRÈS LA SÉLECTION DES FRÉQUENCES
                    st.subheader("📐 Options des Harmoniques")
                    
                    show_harmonics = st.checkbox("Afficher les harmoniques des fréquences caractéristiques", False)
                    if show_harmonics:
                        harmonics_count = st.slider("Nombre d'harmoniques à afficher", 1, 10, 3)
                    
                    show_speed_harmonics = st.checkbox("Afficher les harmoniques de vitesse", False)
                    if show_speed_harmonics:
                        speed_harmonics_count = st.slider("Nombre d'harmoniques de vitesse", 1, 10, 3)
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
                                annotation_text=f"{h}×Vitesse",
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
                    st.error(f"❌ Erreur lors du traitement: {str(e)}")
                    return
            
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
            st.error(f"❌ Erreur lors du traitement du fichier: {str(e)}")
    else:
        st.info("ℹ️ Veuillez télécharger un fichier CSV pour commencer l'analyse")

if __name__ == "__main__":
    main()
