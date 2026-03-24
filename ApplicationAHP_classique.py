import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="AHP Classique - Système d'Aide à la Décision",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette de couleurs professionnelle
COLORS = {
    'primary': '#D81B60',
    'secondary': '#F8BBD9',
    'accent': '#8E24AA',
    'success': '#4CAF50',
    'warning': '#FF9800',
    'info': '#2196F3',
    'dark': '#2C3E50',
    'light': '#FCE4EC',
    'gradient1': '#EC407A',
    'gradient2': '#F06292',
    'gradient3': '#F48FB1'
}

# CSS Personnalisé
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto:wght@300;400;500;700&display=swap');
    
    .stApp {{
        background: linear-gradient(135deg, {COLORS['light']} 0%, #FFFFFF 100%);
    }}
    
    .main-header {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['accent']});
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        animation: fadeInDown 0.8s ease-out;
    }}
    
    @keyframes fadeInDown {{
        from {{ opacity: 0; transform: translateY(-30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .title-main {{
        font-family: 'Roboto', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    
    .subtitle {{
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
    }}
    
    .welcome-card {{
        background: linear-gradient(135deg, {COLORS['primary']}10, {COLORS['secondary']}30);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border-left: 5px solid {COLORS['primary']};
        animation: slideInRight 0.6s ease-out;
    }}
    
    @keyframes slideInRight {{
        from {{ opacity: 0; transform: translateX(30px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    
    .card-elegant {{
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid {COLORS['secondary']};
        transition: all 0.3s ease;
    }}
    
    .card-elegant:hover {{
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(216,27,96,0.15);
        border-color: {COLORS['primary']};
    }}
    
    .metric-container {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['accent']});
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        color: white;
        transition: all 0.3s ease;
    }}
    
    .metric-container:hover {{
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(216,27,96,0.3);
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['accent']});
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(216,27,96,0.4);
    }}
    
    .stSlider > div > div > div {{
        background: {COLORS['primary']};
    }}
    
    .stRadio > div {{
        gap: 1rem;
    }}
    
    .stRadio label {{
        background: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        border: 2px solid {COLORS['secondary']};
        transition: all 0.3s ease;
    }}
    
    .stRadio label:hover {{
        border-color: {COLORS['primary']};
        background: {COLORS['primary']}10;
    }}
    
    .streamlit-expanderHeader {{
        background: linear-gradient(135deg, {COLORS['primary']}10, {COLORS['secondary']}20);
        border-radius: 15px;
        font-weight: 600;
        color: {COLORS['primary']};
    }}
    
    .dataframe-container {{
        overflow-x: auto;
        margin: 1rem 0;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }}
    
    .footer {{
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: linear-gradient(135deg, {COLORS['primary']}10, {COLORS['secondary']}20);
        border-radius: 20px;
        font-family: 'Inter', sans-serif;
    }}
    
    .thankyou-card {{
        background: linear-gradient(135deg, #4CAF50, #2196F3);
        border-radius: 15px;
        padding: 1rem 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.02); }}
    }}
</style>
""", unsafe_allow_html=True)

# Initialisation des variables de session
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'sensitivity_active' not in st.session_state:
    st.session_state.sensitivity_active = False
if 'export_data' not in st.session_state:
    st.session_state.export_data = False
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = 'standard'
if 'animation_speed' not in st.session_state:
    st.session_state.animation_speed = 1.0

def calculate_consistency_with_metrics(A):
    """Calcul avancé des métriques de cohérence"""
    n = A.shape[0]
    e_vals, e_vecs = np.linalg.eig(A)
    lambda_max = np.max(e_vals.real)
    idx_max = np.argmax(e_vals.real)
    weights = np.abs(e_vecs[:, idx_max].real)
    weights = weights / np.sum(weights)
    
    CI = (lambda_max - n) / (n - 1)
    RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 
               7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.54}
    RI = RI_dict.get(n, 1.54)
    CR = CI / RI if RI != 0 else 0
    
    confidence_level = "Élevé" if CR < 0.05 else "Moyen" if CR < 0.1 else "Faible"
    
    if CR >= 0.1:
        recommendations = [
            "📝 Revoir les comparaisons les plus contradictoires",
            "🔄 Vérifier la transitivité des jugements",
            "💡 Réduire l'échelle de comparaison si nécessaire",
            "🎯 Se concentrer sur les critères les plus importants"
        ]
    else:
        recommendations = [
            "✅ Matrice cohérente - Résultats fiables",
            "📊 Poids des critères bien équilibrés",
            "🎯 Décision optimale atteinte"
        ]
    
    return {
        'lambda_max': lambda_max,
        'weights': weights,
        'CI': CI,
        'RI': RI,
        'CR': CR,
        'is_consistent': CR < 0.1,
        'confidence_level': confidence_level,
        'recommendations': recommendations
    }

def display_matrix_table(matrix, rows, columns, title):
    """Affiche une matrice de comparaison sous forme de tableau"""
    df = pd.DataFrame(matrix, index=rows, columns=columns)
    st.markdown(f"**📋 {title}**")
    st.dataframe(df.style.format("{:.3f}").background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)

def create_visualization_gallery(weights_criteria, weights_alternatives, criterias, alternatives, final_scores):
    """Crée une galerie de visualisations interactives avec explications"""
    
    # 1. Graphique Radar
    fig_radar = go.Figure()
    for i, alt in enumerate(alternatives):
        fig_radar.add_trace(go.Scatterpolar(
            r=weights_alternatives[:, i] * 100,
            theta=criterias,
            fill='toself',
            name=alt,
            line=dict(width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{theta}</b><br>Score: %{r:.1f}%<extra></extra>'
        ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=COLORS['secondary'], linecolor=COLORS['primary']),
            angularaxis=dict(gridcolor=COLORS['secondary'], linecolor=COLORS['primary'])
        ),
        title='📊 Profil Comparatif des Alternatives',
        height=550,
        template='plotly_white',
        showlegend=True,
        legend=dict(x=1.1, y=1, bgcolor='rgba(255,255,255,0.8)')
    )
    
    # 2. Graphique 3D
    x = np.arange(len(alternatives))
    y = np.arange(len(criterias))
    X, Y = np.meshgrid(x, y)
    Z = weights_alternatives.T * 100
    
    fig_3d = go.Figure(data=[go.Surface(
        z=Z, x=X, y=Y, colorscale='Viridis',
        contours=dict(z=dict(show=True, usecolormap=True)),
        hovertemplate='Alternative: %{x}<br>Critère: %{y}<br>Score: %{z:.1f}%<extra></extra>'
    )])
    fig_3d.update_layout(
        title='✨ Visualisation 3D des Scores',
        scene=dict(
            xaxis=dict(title='Alternatives', ticktext=alternatives, tickvals=x),
            yaxis=dict(title='Critères', ticktext=criterias, tickvals=y),
            zaxis=dict(title='Score (%)', range=[0, 100]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        template='plotly_white'
    )
    
    # 3. Heatmap
    decision_matrix = weights_alternatives.T * weights_criteria * 100
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=decision_matrix, x=criterias, y=alternatives,
        text=[[f'{val:.1f}%' for val in row] for row in decision_matrix],
        texttemplate='%{text}', textfont={"size": 12},
        colorscale='RdYlGn', colorbar=dict(title="Contribution (%)"),
        hovertemplate='Critère: %{x}<br>Alternative: %{y}<br>Contribution: %{z:.1f}%<extra></extra>'
    ))
    fig_heatmap.update_layout(title='🗺️ Matrice de Décision', height=500, template='plotly_white', xaxis=dict(tickangle=45))
    
    # 4. Graphique à barres
    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(
        x=alternatives, y=final_scores * 100,
        marker=dict(color=final_scores * 100, colorscale='Viridis', showscale=True, colorbar=dict(title="Score (%)"), line=dict(color=COLORS['primary'], width=2)),
        text=[f'{score:.1f}%' for score in final_scores * 100], textposition='auto',
        hovertemplate='Alternative: %{x}<br>Score: %{y:.1f}%<extra></extra>'
    ))
    fig_bars.update_layout(title='🏆 Classement Final des Alternatives', xaxis_title='Alternatives', yaxis_title='Score (%)', height=500, template='plotly_white', hovermode='x unified')
    
    # 5. Graphique circulaire
    fig_pie = go.Figure(data=[go.Pie(
        labels=alternatives, values=final_scores * 100, hole=0.4,
        marker=dict(colors=px.colors.qualitative.Pastel),
        textinfo='label+percent', textposition='auto',
        hovertemplate='<b>%{label}</b><br>Score: %{value:.1f}%<extra></extra>'
    )])
    fig_pie.update_layout(title='📈 Répartition des Scores', height=500, template='plotly_white')
    
    # 6. Graphique d'évolution
    fig_spider = go.Figure()
    for i, crit in enumerate(criterias):
        fig_spider.add_trace(go.Scatter(
            x=alternatives, y=weights_alternatives[i] * 100,
            mode='lines+markers', name=crit, line=dict(width=3), marker=dict(size=10),
            hovertemplate='Critère: ' + crit + '<br>Alternative: %{x}<br>Score: %{y:.1f}%<extra></extra>'
        ))
    fig_spider.update_layout(title='📊 Performance par Critère', xaxis_title='Alternatives', yaxis_title='Score (%)', height=500, template='plotly_white', hovermode='x unified')
    
    # 7. Graphique des poids des critères
    fig_criteria_weights = go.Figure()
    fig_criteria_weights.add_trace(go.Bar(
        x=weights_criteria * 100, y=criterias, orientation='h',
        marker=dict(color=weights_criteria * 100, colorscale='Viridis'),
        text=[f'{w:.1f}%' for w in weights_criteria * 100], textposition='outside',
        hovertemplate='Critère: %{y}<br>Poids: %{x:.1f}%<extra></extra>'
    ))
    fig_criteria_weights.update_layout(title='📊 Poids des Critères', xaxis_title='Poids (%)', yaxis_title='Critères', height=400, template='plotly_white')
    
    # 8. Graphique Sankey (flux de décision)
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=criterias + alternatives,
            color=[COLORS['primary']] * len(criterias) + [COLORS['accent']] * len(alternatives)
        ),
        link=dict(
            source=[i for i in range(len(criterias)) for _ in range(len(alternatives))],
            target=[len(criterias) + j for _ in range(len(criterias)) for j in range(len(alternatives))],
            value=[weights_criteria[i] * weights_alternatives[i][j] * 100 for i in range(len(criterias)) for j in range(len(alternatives))]
        )
    )])
    fig_sankey.update_layout(title='🌊 Flux de Décision (Sankey)', height=500, template='plotly_white')
    
    return fig_radar, fig_3d, fig_heatmap, fig_bars, fig_pie, fig_spider, fig_criteria_weights, fig_sankey

def display_visualization_explanations():
    """Affiche les explications des visualisations"""
    with st.expander("📖 **Comprendre les visualisations - Guide d'interprétation**", expanded=False):
        st.markdown("""
        ### 📊 **Guide d'interprétation des graphiques**
        
        | Graphique | Signification | Comment l'interpréter |
        |-----------|---------------|----------------------|
        | **📊 Profil Radar** | Performance multidimensionnelle | Plus la zone couverte est grande, meilleure est l'alternative |
        | **✨ Vue 3D** | Distribution spatiale des scores | Visualisation immersive des performances |
        | **🗺️ Heatmap** | Contribution des critères | Vert = forte contribution, Rouge = faible contribution |
        | **🏆 Classement** | Score final pondéré | La barre la plus haute représente la meilleure décision |
        | **📈 Distribution** | Répartition des scores | Visualisation proportionnelle des résultats |
        | **📊 Performance** | Évolution par critère | Compare les alternatives critère par critère |
        | **📊 Poids Critères** | Importance relative | Plus la barre est longue, plus le critère est déterminant |
        | **🌊 Sankey** | Flux de décision | Visualise comment les critères influencent chaque alternative |
        
        ### 🔍 **Critères de cohérence**
        - **CR < 0.1** : Les jugements sont cohérents ✅
        - **CR ≥ 0.1** : Les jugements sont incohérents, veuillez réviser ⚠️
        - **λ max** : Valeur propre maximale, proche de n pour une matrice cohérente
        """)

def sensitivity_analysis(weights_criteria, weights_alternatives, criterias, alternatives, final_scores):
    """Analyse de sensibilité interactive"""
    st.markdown("### 🔬 Analyse de Sensibilité")
    st.markdown("*Cette analyse montre comment la variation du poids d'un critère affecte le classement final*")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_criterion = st.selectbox("Choisir un critère à analyser", criterias)
    
    idx = criterias.index(selected_criterion)
    variations = np.linspace(0, 2, 50)
    scores_variations = []
    
    for var in variations:
        temp_weights = weights_criteria.copy()
        temp_weights[idx] = weights_criteria[idx] * var
        temp_weights = temp_weights / np.sum(temp_weights)
        scores = np.dot(temp_weights, weights_alternatives)
        scores_variations.append(scores)
    
    fig_sensitivity = go.Figure()
    for i, alt in enumerate(alternatives):
        fig_sensitivity.add_trace(go.Scatter(
            x=variations * 100, y=[s[i] * 100 for s in scores_variations],
            mode='lines', name=alt, line=dict(width=3),
            hovertemplate='Variation: %{x:.0f}%<br>Score: %{y:.1f}%<extra></extra>'
        ))
    
    fig_sensitivity.add_vline(x=100, line_dash="dash", line_color=COLORS['primary'], 
                              annotation_text="Poids actuel", annotation_position="top right")
    
    fig_sensitivity.update_layout(
        title=f'Sensibilité - Variation du critère "{selected_criterion}"',
        xaxis_title='Variation du poids du critère (%)', yaxis_title='Score des alternatives (%)',
        height=500, template='plotly_white', hovermode='x unified'
    )
    
    st.plotly_chart(fig_sensitivity, use_container_width=True)
    
    # Analyse des points d'intersection
    st.info(f"💡 **Analyse**: La variation du critère '{selected_criterion}' peut modifier le classement final. Observez les croisements des courbes pour identifier les seuils critiques.")

def main():
    # Header Principal
    st.markdown("""
    <div class='main-header'>
        <h1 class='title-main'>📊 AHP Classique</h1>
        <p class='subtitle'>Analytic Hierarchy Process - Système d'Aide à la Décision Multicritère</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Message de bienvenue avec remerciements
    st.markdown(f"""
    <div class='welcome-card'>
        <div style='display: flex; align-items: center; gap: 1rem;'>
            <div style='font-size: 2.5rem;'>👋</div>
            <div>
                <h2 style='color: {COLORS['primary']}; margin: 0;'>Bienvenue dans votre interface AHP Classique</h2>
                <p style='margin: 0.5rem 0 0 0; font-size: 1rem;'>
                    Cette application vous aide à prendre des décisions multicritères de manière structurée et scientifique.
                    Utilisez la méthodologie AHP de Saaty pour comparer vos critères et alternatives.
                </p>
            </div>
        </div>
        <hr style='margin: 1rem 0; border-color: {COLORS['secondary']};'>
        <div style='text-align: center;'>
            <p style='margin: 0; font-weight: 500;'>
                🎓 <strong>Réalisé par : Fatiha Mahnoun • Karima Abdous • Oumaima Khair</strong>
            </p>
            <p style='margin: 0; font-size: 0.9rem; color: #666;'>
                Étudiantes Ingénieures à l'ENSAM Meknès | Encadré par Madame Dadda
            </p>
            <p style='margin-top: 0.5rem; font-size: 0.85rem; color: {COLORS['primary']};'>
                📊 Méthodologie AHP (Analytic Hierarchy Process) - Thomas L. Saaty
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Carte de remerciement
    st.markdown("""
    <div class='thankyou-card'>
        <h3>🙏 Merci à notre encadrante Madame Dadda</h3>
        <p>Pour son accompagnement précieux, ses conseils avisés et l'opportunité de réaliser ce projet enrichissant.</p>
        <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Votre soutien nous a permis de développer cette application professionnelle.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h2 style='color: #D81B60;'>⚙️ Configuration</h2>
            <div style='width: 50px; height: 3px; background: linear-gradient(90deg, #D81B60, #8E24AA); margin: 1rem auto;'></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎨 Paramètres")
        st.session_state.comparison_mode = st.select_slider(
            "Mode de comparaison", options=['Standard', 'Détaillé', 'Expert'], value='Standard'
        )
        
        st.markdown("---")
        
        st.markdown("### 📝 Données d'entrée")
        cri = st.text_input("📋 Critères", placeholder="Ex: Prix, Qualité, Design, Service", help="Séparez par des virgules")
        alt = st.text_input("🎯 Alternatives", placeholder="Ex: Option A, Option B, Option C", help="Séparez par des virgules")
        
        st.markdown("---")
        
        with st.expander("📚 Échelle de Saaty"):
            st.markdown("""
            | Intensité | Définition |
            |-----------|------------|
            | 1 | Égale importance |
            | 3 | Légère importance |
            | 5 | Forte importance |
            | 7 | Très forte importance |
            | 9 | Extrême importance |
            | 2,4,6,8 | Valeurs intermédiaires |
            
            **CR < 0.1** : Jugements cohérents ✅
            """)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 0.85rem; padding: 1rem;'>
            <p>📊 <strong>AHP Classique</strong></p>
            <p>ENSAM Meknès | Méthodologie de Saaty</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Contenu principal
    if cri and alt:
        criterias = [c.strip() for c in cri.split(",")]
        alternatives = [a.strip() for a in alt.split(",")]
        
        n = len(criterias)
        m = len(alternatives)
        
        # Statistiques rapides
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='card-elegant' style='text-align: center;'>
                <h3 style='color: #D81B60;'>📊 {n}</h3>
                <p>Critères analysés</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='card-elegant' style='text-align: center;'>
                <h3 style='color: #D81B60;'>🎯 {m}</h3>
                <p>Alternatives comparées</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='card-elegant' style='text-align: center;'>
                <h3 style='color: #D81B60;'>🔍 {n*m}</h3>
                <p>Comparaisons possibles</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Comparaison des critères
        with st.expander("📊 **Comparaison des Critères**", expanded=True):
            st.markdown("<p style='color: #D81B60;'>✨ Comparez l'importance relative des critères selon l'échelle de Saaty ✨</p>", unsafe_allow_html=True)
            
            A = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i+1, n):
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.markdown(f"**{criterias[i]}** vs **{criterias[j]}**")
                    with col2:
                        direction = st.radio(
                            "Qui est plus important ?",
                            (criterias[i], criterias[j]),
                            key=f"dir_c_{i}_{j}",
                            horizontal=True
                        )
                        if direction == criterias[i]:
                            value = st.slider(f"Importance de {criterias[i]}", 1, 9, 3, key=f"slider_c_{i}_{j}", label_visibility="collapsed")
                            A[i][j] = value
                            A[j][i] = 1/value
                        else:
                            value = st.slider(f"Importance de {criterias[j]}", 1, 9, 3, key=f"slider_c_{i}_{j}", label_visibility="collapsed")
                            A[j][i] = value
                            A[i][j] = 1/value
            
            for i in range(n):
                A[i][i] = 1
            
            # Affichage de la matrice des critères
            display_matrix_table(A, criterias, criterias, "Matrice de comparaison des critères")
        
        # Comparaison des alternatives
        B = np.zeros((n, m, m))
        
        with st.expander("🎯 **Comparaison des Alternatives**", expanded=True):
            for k in range(n):
                st.markdown(f"#### ✨ Critère : {criterias[k]} ✨")
                
                for i in range(m):
                    for j in range(i+1, m):
                        col1, col2 = st.columns([2, 3])
                        with col1:
                            st.markdown(f"**{alternatives[i]}** vs **{alternatives[j]}**")
                        with col2:
                            direction = st.radio(
                                "Qui est meilleur ?",
                                (alternatives[i], alternatives[j]),
                                key=f"dir_a_{k}_{i}_{j}",
                                horizontal=True
                            )
                            if direction == alternatives[i]:
                                value = st.slider(f"Supériorité de {alternatives[i]}", 1, 9, 3, key=f"slider_a_{k}_{i}_{j}", label_visibility="collapsed")
                                B[k][i][j] = value
                                B[k][j][i] = 1/value
                            else:
                                value = st.slider(f"Supériorité de {alternatives[j]}", 1, 9, 3, key=f"slider_a_{k}_{i}_{j}", label_visibility="collapsed")
                                B[k][j][i] = value
                                B[k][i][j] = 1/value
                
                for i in range(m):
                    B[k][i][i] = 1
                
                # Affichage de la matrice des alternatives pour ce critère
                display_matrix_table(B[k], alternatives, alternatives, f"Matrice de comparaison des alternatives - Critère : {criterias[k]}")
                st.markdown("---")
        
        # Bouton de calcul
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 **Lancer l'analyse AHP**", use_container_width=True):
                with st.spinner("📊 Calcul en cours... Analyse de cohérence..."):
                    time.sleep(0.5)
                
                metrics_criteria = calculate_consistency_with_metrics(A)
                weights_criteria = metrics_criteria['weights']
                
                weights_alternatives = np.zeros((n, m))
                consistency_results = []
                
                for i in range(n):
                    metrics = calculate_consistency_with_metrics(B[i])
                    weights_alternatives[i] = metrics['weights']
                    consistency_results.append(metrics)
                
                final_scores = np.dot(weights_criteria, weights_alternatives)
                
                st.session_state.show_results = True
                st.session_state.weights_criteria = weights_criteria
                st.session_state.weights_alternatives = weights_alternatives
                st.session_state.final_scores = final_scores
                st.session_state.metrics_criteria = metrics_criteria
                st.session_state.consistency_results = consistency_results
                st.session_state.A = A
                st.session_state.B = B
                st.session_state.criterias = criterias
                st.session_state.alternatives = alternatives
    
    # Affichage des résultats
    if st.session_state.get('show_results', False):
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; margin: 2rem 0; color: #D81B60;'>📈 Résultats de l'Analyse</h2>", unsafe_allow_html=True)
        
        # Métriques de cohérence
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-container'>
                <h4>λ max</h4>
                <h2>{st.session_state.metrics_criteria['lambda_max']:.4f}</h2>
                <p style='font-size:0.7rem;'>Valeur propre maximale</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-container'>
                <h4>CR</h4>
                <h2>{st.session_state.metrics_criteria['CR']:.4f}</h2>
                <p style='font-size:0.8rem;'>{'✅ Cohérent' if st.session_state.metrics_criteria['is_consistent'] else '⚠️ À revoir'}</p>
                <p style='font-size:0.7rem;'>Ratio de Cohérence</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-container'>
                <h4>CI</h4>
                <h2>{st.session_state.metrics_criteria['CI']:.4f}</h2>
                <p style='font-size:0.7rem;'>Indice de Cohérence</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-container'>
                <h4>Niveau de confiance</h4>
                <h2>{st.session_state.metrics_criteria['confidence_level']}</h2>
                <p style='font-size:0.7rem;'>Fiabilité des jugements</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Interprétation du CR
        if st.session_state.metrics_criteria['CR'] < 0.1:
            st.success(f"✅ **Interprétation**: CR = {st.session_state.metrics_criteria['CR']:.4f} < 0.1. Les jugements sont cohérents, les résultats sont fiables.")
        else:
            st.warning(f"⚠️ **Interprétation**: CR = {st.session_state.metrics_criteria['CR']:.4f} ≥ 0.1. Les jugements manquent de cohérence. Veuillez réviser les comparaisons.")
        
        # Recommandations
        st.markdown("""
        <div class='card-elegant'>
            <h3 style='color: #D81B60;'>💡 Recommandations</h3>
        """, unsafe_allow_html=True)
        for rec in st.session_state.metrics_criteria['recommendations']:
            st.markdown(f"- {rec}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Matrices de décision finales
        st.markdown("### 📋 Matrices de Décision Finales")
        
        col1, col2 = st.columns(2)
        with col1:
            weights_criteria_df = pd.DataFrame({
                'Critère': st.session_state.criterias,
                'Poids': st.session_state.weights_criteria,
                'Poids (%)': st.session_state.weights_criteria * 100
            }).sort_values('Poids', ascending=False)
            st.dataframe(weights_criteria_df.style.format({'Poids': '{:.4f}', 'Poids (%)': '{:.2f}%'}).background_gradient(cmap='RdYlGn', subset=['Poids (%)']), use_container_width=True)
        
        with col2:
            final_scores_df = pd.DataFrame({
                'Alternative': st.session_state.alternatives,
                'Score': st.session_state.final_scores,
                'Score (%)': st.session_state.final_scores * 100
            }).sort_values('Score', ascending=False)
            st.dataframe(final_scores_df.style.format({'Score': '{:.4f}', 'Score (%)': '{:.2f}%'}).background_gradient(cmap='RdYlGn', subset=['Score (%)']), use_container_width=True)
        
        # Visualisations
        st.markdown("### 🎨 Galerie de Visualisations")
        
        fig_radar, fig_3d, fig_heatmap, fig_bars, fig_pie, fig_spider, fig_criteria_weights, fig_sankey = create_visualization_gallery(
            st.session_state.weights_criteria,
            st.session_state.weights_alternatives,
            st.session_state.criterias,
            st.session_state.alternatives,
            st.session_state.final_scores
        )
        
        # Onglets pour les visualisations
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Profil Radar", "✨ Vue 3D", "🗺️ Heatmap", "🏆 Classement", 
            "📈 Distribution", "📊 Performance", "📊 Poids Critères", "🌊 Flux Décision"
        ])
        
        with tab1:
            st.plotly_chart(fig_radar, use_container_width=True)
        with tab2:
            st.plotly_chart(fig_3d, use_container_width=True)
        with tab3:
            st.plotly_chart(fig_heatmap, use_container_width=True)
        with tab4:
            st.plotly_chart(fig_bars, use_container_width=True)
        with tab5:
            st.plotly_chart(fig_pie, use_container_width=True)
        with tab6:
            st.plotly_chart(fig_spider, use_container_width=True)
        with tab7:
            st.plotly_chart(fig_criteria_weights, use_container_width=True)
        with tab8:
            st.plotly_chart(fig_sankey, use_container_width=True)
        
        # Explications des visualisations
        display_visualization_explanations()
        
        # Analyse de sensibilité
        st.markdown("---")
        if st.checkbox("🔬 Activer l'Analyse de Sensibilité"):
            sensitivity_analysis(
                st.session_state.weights_criteria,
                st.session_state.weights_alternatives,
                st.session_state.criterias,
                st.session_state.alternatives,
                st.session_state.final_scores
            )
        
        # Export des résultats
        if st.checkbox("📥 Exporter les résultats"):
            results_df = pd.DataFrame({
                'Alternative': st.session_state.alternatives,
                'Score': st.session_state.final_scores,
                'Score (%)': st.session_state.final_scores * 100
            }).sort_values('Score', ascending=False)
            
            st.dataframe(results_df.style.format({'Score': '{:.4f}', 'Score (%)': '{:.2f}%'}).background_gradient(cmap='RdYlGn', subset=['Score (%)']), use_container_width=True)
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv,
                file_name=f"ahp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Meilleure alternative
        best_idx = np.argmax(st.session_state.final_scores)
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #D81B6020, #8E24AA20); border-radius: 25px; margin: 2rem 0;'>
            <h2 style='color: #D81B60;'>🏆 Alternative Recommandée</h2>
            <h1 style='font-size: 2.5rem; margin: 1rem 0;'>{st.session_state.alternatives[best_idx]}</h1>
            <p style='font-size: 1.2rem;'>Score: {st.session_state.final_scores[best_idx]*100:.2f}%</p>
            <p>✨ Choix optimal basé sur l'analyse multicritère AHP ✨</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.balloons()
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <p>📊 <strong>AHP Classique</strong> - Analytic Hierarchy Process</p>
        <p style='font-size: 0.8rem;'>ENSAM Meknès | Fatiha Mahnoun • Karima ABadous • Oumaima Khair | Encadré par Madame Dadda</p>
        <p style='font-size: 0.8rem; margin-top: 0.5rem;'>Méthodologie AHP de Saaty | Analyse Cohérente | Visualisations Interactives</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
