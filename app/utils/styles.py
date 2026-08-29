import streamlit as st

F1_CSS = """
<style>
/* Import F1 Titillium / Orbitron Fonts */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800;900&family=Titillium+Web:wght@300;400;600;700&display=swap');

:root {
    --f1-red: #e10600;
    --f1-dark-red: #8f0400;
    --f1-bg-dark: #0b0e14;
    --f1-card-bg: rgba(20, 24, 33, 0.85);
    --f1-card-border: rgba(225, 6, 0, 0.25);
    --f1-text-primary: #f0f3f8;
    --f1-text-muted: #8c9ba5;
    --pirelli-soft: #ff1801;
    --pirelli-medium: #ffd700;
    --pirelli-hard: #ffffff;
    --pirelli-inter: #39b54a;
    --pirelli-wet: #00a0de;
}

/* Base App Styling */
.stApp {
    background-color: #0b0e14;
    color: #f0f3f8;
    font-family: 'Titillium Web', sans-serif;
}

/* Sidebar Dark Theme */
section[data-testid="stSidebar"] {
    background-color: #080a0f !important;
    border-right: 1px solid rgba(225, 6, 0, 0.2);
}

/* Headers */
h1, h2, h3, h4, .f1-title {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Top Banner Header */
.f1-header-container {
    background: linear-gradient(135deg, rgba(225, 6, 0, 0.2) 0%, rgba(11, 14, 20, 0.95) 70%);
    border-left: 5px solid var(--f1-red);
    border-bottom: 1px solid var(--f1-card-border);
    padding: 18px 24px;
    border-radius: 8px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}

/* Glassmorphism Metric Cards */
.f1-metric-card {
    background: var(--f1-card-bg);
    border: 1px solid var(--f1-card-border);
    backdrop-filter: blur(12px);
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.f1-metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--f1-red);
}

.f1-metric-label {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--f1-text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

.f1-metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.2;
}

.f1-metric-delta {
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 6px;
}

/* Pirelli Compound Badges */
.compound-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 14px;
    font-weight: 800;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-family: 'Orbitron', sans-serif;
}
.compound-soft { background-color: rgba(255, 24, 1, 0.2); color: #ff1801; border: 1.5px solid #ff1801; }
.compound-medium { background-color: rgba(255, 215, 0, 0.2); color: #ffd700; border: 1.5px solid #ffd700; }
.compound-hard { background-color: rgba(255, 255, 255, 0.2); color: #ffffff; border: 1.5px solid #ffffff; }

/* Pit Wall Strategy Card */
.strategy-box {
    background: linear-gradient(145deg, rgba(25, 30, 42, 0.95), rgba(14, 18, 26, 0.95));
    border-left: 4px solid #00d2be;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
}
</style>
"""


def apply_f1_theme():
    st.markdown(F1_CSS, unsafe_allow_html=True)
