"""
🧭 Compass POC - E2E LLM Deployment Recommendation System

A beautiful, presentation-ready Streamlit application demonstrating:
1. Business Context Extraction (Qwen 2.5 7B @ 95.1% accuracy)
2. MCDM Scoring (206 Open-Source Models)
3. Full Explainability with visual score breakdowns
4. SLO & Workload Impact Analysis
5. Hardware-aware recommendations

Usage:
    streamlit run ui/poc_app.py
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DATA_DIR = Path(__file__).parent.parent / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="🧭 Compass - LLM Deployment Advisor",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS - Beautiful Enterprise Styling
# =============================================================================

st.markdown("""
<style>
    /* Import Google Fonts - Qualifire & HuggingFace Inspired Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=DM+Sans:wght@400;500;600;700&display=swap');
    
    /* Global Styles - Qualifire.ai Inspired Clean Theme */
    .stApp {
        font-family: 'Inter', 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        /* Metallic dark background like Artificial Analysis */
        background: 
            radial-gradient(ellipse at 20% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 100%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(30, 41, 59, 0.4) 0%, transparent 70%),
            linear-gradient(180deg, #0a0a0f 0%, #0f172a 25%, #1e293b 50%, #0f172a 75%, #0a0a0f 100%);
        background-attachment: fixed;
        color: #f0f6fc;
        font-size: 16px;
        line-height: 1.7;
        letter-spacing: -0.01em;
    }
    
    /* Keyframe Animations - Subtle, Professional */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 25px rgba(99, 102, 241, 0.15); }
        50% { box-shadow: 0 0 40px rgba(99, 102, 241, 0.25), 0 0 60px rgba(16, 185, 129, 0.1); }
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    @keyframes slide-up {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fade-in {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes scale-in {
        from { opacity: 0; transform: scale(0.97); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes border-glow {
        0%, 100% { border-color: rgba(99, 102, 241, 0.3); }
        50% { border-color: rgba(99, 102, 241, 0.6); }
    }
    
    /* Corporate Color Palette - Qualifire/HuggingFace Inspired */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #111827;
        --bg-tertiary: #1f2937;
        --bg-card: rgba(17, 24, 39, 0.8);
        --bg-card-hover: rgba(31, 41, 55, 0.9);
        --border-default: rgba(75, 85, 99, 0.4);
        --border-accent: rgba(99, 102, 241, 0.5);
        --border-success: rgba(16, 185, 129, 0.5);
        --text-primary: #f9fafb;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
        --accent-indigo: #6366f1;
        --accent-purple: #8b5cf6;
        --accent-blue: #3b82f6;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --accent-green: #22c55e;
        --accent-yellow: #f59e0b;
        --accent-orange: #f97316;
        --accent-rose: #f43f5e;
        --accent-pink: #ec4899;
        --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        --gradient-success: linear-gradient(135deg, #10b981 0%, #22c55e 100%);
        --gradient-hero: linear-gradient(135deg, #1e1b4b 0%, #312e81 25%, #4c1d95 50%, #5b21b6 75%, #6d28d9 100%);
        --gradient-card: linear-gradient(145deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.03) 100%);
        --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        --shadow-glow: 0 0 40px rgba(99, 102, 241, 0.15);
    }
    
    /* Hero Section - Enterprise Grade Design */
    .hero-container {
        background: var(--gradient-hero);
        background-size: 200% 200%;
        animation: gradient-shift 15s ease infinite;
        padding: 4.5rem 4rem;
        border-radius: 1.5rem;
        margin-bottom: 3rem;
        box-shadow: var(--shadow-lg), var(--shadow-glow);
        border: 1px solid rgba(139, 92, 246, 0.2);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 80%, rgba(139, 92, 246, 0.15) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(99, 102, 241, 0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.5;
        pointer-events: none;
    }
    .hero-emoji {
        font-size: 5rem;
        margin-bottom: 1.25rem;
        animation: float 5s ease-in-out infinite;
        filter: drop-shadow(0 10px 25px rgba(0,0,0,0.4));
        position: relative;
        z-index: 1;
    }
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 0 4px 30px rgba(0,0,0,0.4);
        letter-spacing: -2px;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        font-size: 1.4rem;
        color: rgba(255,255,255,0.85);
        font-weight: 400;
        max-width: 700px;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }
    .hero-badges {
        display: flex;
        gap: 1rem;
        margin-top: 2.5rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    .hero-badge {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(16px);
        padding: 0.75rem 1.5rem;
        border-radius: 100px;
        font-size: 0.95rem;
        font-weight: 600;
        color: white;
        border: 1px solid rgba(255,255,255,0.15);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: all 0.25s ease;
        font-family: 'Inter', sans-serif;
    }
    .hero-badge:hover {
        background: rgba(255,255,255,0.18);
        transform: translateY(-3px);
        border-color: rgba(255,255,255,0.3);
    }
    
    /* Stats Cards - HuggingFace Leaderboard Inspired */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1.25rem;
        margin: 2.5rem 0;
    }
    .stat-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        padding: 1.75rem 1.25rem;
        border-radius: 1rem;
        text-align: center;
        border: 1px solid var(--border-default);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-primary);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .stat-card::after {
        content: '';
        position: absolute;
        inset: 0;
        background: var(--gradient-card);
        opacity: 0;
        transition: opacity 0.3s ease;
        pointer-events: none;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        border-color: var(--border-accent);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.1), 0 8px 20px rgba(0,0,0,0.2);
    }
    .stat-card:hover::before {
        opacity: 1;
    }
    .stat-card:hover::after {
        opacity: 1;
    }
    .stat-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        display: block;
        filter: drop-shadow(0 4px 8px rgba(0,0,0,0.15));
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-indigo), var(--accent-purple), var(--accent-emerald));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        letter-spacing: -1px;
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-weight: 600;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Editable Fields with Pencil Icons */
    .editable-field {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .editable-field:hover {
        background: rgba(102, 126, 234, 0.15);
    }
    .pencil-icon {
        font-size: 0.9rem;
        color: #38ef7d;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .editable-field:hover .pencil-icon {
        opacity: 1;
    }
    .edit-input {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(102, 126, 234, 0.4);
        border-radius: 0.5rem;
        padding: 0.5rem 0.75rem;
        color: white;
        font-weight: 600;
        width: 100px;
        text-align: center;
    }
    .edit-input:focus {
        border-color: #38ef7d;
        outline: none;
        box-shadow: 0 0 0 3px rgba(56, 239, 125, 0.25);
    }
    
    /* TOP LEADERBOARD TABLE - HuggingFace/Qualifire Inspired */
    .leaderboard-container {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border-radius: 1rem;
        border: 1px solid var(--border-default);
        overflow-x: auto;
        overflow-y: hidden;
        margin: 2rem 0;
        animation: fade-in 0.5s ease-out;
        box-shadow: 0 4px 24px rgba(0,0,0,0.15);
        width: 100%;
    }
    .leaderboard-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
        padding: 1.25rem 1.75rem;
        border-bottom: 1px solid var(--border-default);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }
    .leaderboard-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .leaderboard-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        table-layout: fixed;
    }
    .leaderboard-table th {
        background: rgba(88, 166, 255, 0.08);
        color: var(--text-secondary);
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 1rem 0.5rem;
        text-align: center;
        border-bottom: 1px solid var(--border-default);
        white-space: nowrap;
    }
    /* Column widths - FIXED layout with explicit widths (must add to 100%) */
    .leaderboard-table th:nth-child(1),
    .leaderboard-table td:nth-child(1) { width: 5%; }     /* Rank */
    .leaderboard-table th:nth-child(2),
    .leaderboard-table td:nth-child(2) { width: 18%; text-align: left; }  /* Model */
    .leaderboard-table th:nth-child(3),
    .leaderboard-table td:nth-child(3) { width: 10%; }    /* Quality */
    .leaderboard-table th:nth-child(4),
    .leaderboard-table td:nth-child(4) { width: 10%; }    /* Latency */
    .leaderboard-table th:nth-child(5),
    .leaderboard-table td:nth-child(5) { width: 10%; }    /* Cost */
    .leaderboard-table th:nth-child(6),
    .leaderboard-table td:nth-child(6) { width: 10%; }    /* Capacity */
    .leaderboard-table th:nth-child(7),
    .leaderboard-table td:nth-child(7) { width: 10%; }    /* Final Score */
    .leaderboard-table th:nth-child(8),
    .leaderboard-table td:nth-child(8) { width: 17%; }    /* Pros & Cons */
    .leaderboard-table th:nth-child(9),
    .leaderboard-table td:nth-child(9) { width: 10%; }    /* Action */
    .leaderboard-table td {
        padding: 1rem 0.5rem;
        border-bottom: 1px solid var(--border-default);
        color: var(--text-primary);
        vertical-align: middle;
        text-align: center;
    }
    .leaderboard-table tr:hover td {
        background: rgba(88, 166, 255, 0.05);
    }
    .leaderboard-table tr:last-child td {
        border-bottom: none;
    }
    /* Consistent row height */
    .leaderboard-table tr {
        height: 85px;
    }
    
    /* Rank Badges - HuggingFace Style Medals */
    .rank-badge {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1rem;
        color: white;
        margin: 0 auto;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .rank-badge:hover {
        transform: scale(1.08);
    }
    .rank-1 {
        background: linear-gradient(145deg, #fcd34d, #f59e0b);
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
        color: #78350f;
    }
    .rank-2 {
        background: linear-gradient(145deg, #e5e7eb, #9ca3af);
        box-shadow: 0 4px 12px rgba(156, 163, 175, 0.4);
        color: #374151;
    }
    .rank-3 {
        background: linear-gradient(145deg, #fdba74, #ea580c);
        box-shadow: 0 4px 12px rgba(234, 88, 12, 0.35);
        color: #7c2d12;
    }
    /* Ranks 4-10 - Indigo gradient */
    .rank-4, .rank-5, .rank-6, .rank-7, .rank-8, .rank-9, .rank-10 {
        background: linear-gradient(145deg, #6366f1, #8b5cf6);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    /* Score Bars - HuggingFace Inspired Progress Bars */
    .score-mini-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 5px;
        width: 100%;
        max-width: 100%;
        margin: 0 auto;
    }
    .score-mini-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.06);
        overflow: hidden;
        width: 100%;
        position: relative;
    }
    .score-mini-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .score-mini-label {
        font-size: 0.9rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', 'Inter', monospace;
    }
    .score-num {
        display: none;
    }
    .fill-quality { background: linear-gradient(90deg, #059669, #10b981); }
    .fill-latency { background: linear-gradient(90deg, #2563eb, #3b82f6); }
    .fill-cost { background: linear-gradient(90deg, #ea580c, #f97316); }
    .fill-capacity { background: linear-gradient(90deg, #7c3aed, #8b5cf6); }
    
    /* Score label colors */
    .label-quality { color: #10b981; }
    .label-latency { color: #3b82f6; }
    .label-cost { color: #f97316; }
    .label-capacity { color: #8b5cf6; }
    
    /* Model Card in Table - Clean Typography */
    .model-cell {
        display: flex;
        align-items: center;
        gap: 0.875rem;
    }
    .model-info {
        display: flex;
        flex-direction: column;
        gap: 3px;
    }
    .model-name {
        font-weight: 600;
        font-size: 1rem;
        color: #f9fafb;
        font-family: 'Inter', sans-serif;
        line-height: 1.3;
    }
    .model-provider {
        font-size: 0.8rem;
        color: #6b7280;
        font-weight: 500;
    }
    
    /* Final Score Display - BIG and prominent */
    .final-score {
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--accent-green) !important;
        font-family: 'Inter', sans-serif;
        text-shadow: 0 0 20px rgba(63, 185, 80, 0.4);
        display: block;
        text-align: center;
    }
    
    /* Enhanced Slider Styling */
    .stSlider {
        padding: 0.5rem 0;
    }
    .stSlider > div > div > div {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.3), rgba(56, 239, 125, 0.3)) !important;
        height: 8px !important;
        border-radius: 4px !important;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #38ef7d) !important;
        height: 8px !important;
        border-radius: 4px !important;
    }
    .stSlider [data-testid="stThumbValue"] {
        background: linear-gradient(135deg, #667eea, #38ef7d) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 0.5rem !important;
        padding: 0.25rem 0.5rem !important;
    }
    .stSlider label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Pros/Cons Tags - Clean pill design like HuggingFace */
    .tag-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 6px;
        flex-wrap: wrap;
    }
    .tag {
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        white-space: nowrap;
        text-align: center;
        transition: all 0.2s ease;
    }
    .tag:hover {
        transform: scale(1.05);
    }
    .tag-pro {
        background: rgba(63, 185, 80, 0.15);
        color: var(--accent-green);
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    .tag-con {
        background: rgba(248, 81, 73, 0.15);
        color: var(--accent-orange);
        border: 1px solid rgba(248, 81, 73, 0.3);
    }
    .tag-neutral {
        background: rgba(255,255,255,0.08);
        color: var(--text-secondary);
        border: 1px solid var(--border-default);
    }
    
    /* Action Button - HuggingFace style rounded */
    .action-cell {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .select-btn {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        color: white;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        border: none;
        cursor: pointer;
        transition: all 0.2s ease;
        white-space: nowrap;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        font-family: 'Inter', sans-serif;
        margin: 0 auto;
    }
    .select-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(88, 166, 255, 0.35);
        filter: brightness(1.1);
    }
    
    /* Extraction Card - Clean, spacious design */
    .extraction-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 1.25rem;
        border: 1px solid var(--border-default);
        margin: 2rem 0;
        animation: slide-up 0.4s ease-out;
    }
    .extraction-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
    }
    .extraction-item {
        background: var(--bg-secondary);
        padding: 1.5rem;
        border-radius: 1rem;
        display: flex;
        align-items: center;
        gap: 1.25rem;
        border: 1px solid var(--border-default);
        transition: all 0.25s ease;
    }
    .extraction-item:hover {
        border-color: var(--accent-blue);
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(88, 166, 255, 0.1);
    }
    .extraction-icon {
        font-size: 2.25rem;
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
    }
    .extraction-icon-usecase { background: linear-gradient(135deg, var(--accent-purple), #7c3aed); }
    .extraction-icon-users { background: linear-gradient(135deg, #059669, var(--accent-green)); }
    .extraction-icon-priority { background: linear-gradient(135deg, var(--accent-orange), var(--accent-pink)); }
    .extraction-icon-hardware { background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)); }
    .extraction-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }
    .extraction-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 4px;
        font-family: 'Inter', sans-serif;
    }
    
    /* SLO & Workload Cards - Clean, readable design */
    .slo-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }
    .slo-card {
        background: var(--bg-card);
        backdrop-filter: blur(10px);
        padding: 1.75rem;
        border-radius: 1rem;
        border: 1px solid var(--border-default);
        transition: all 0.25s ease;
        position: relative;
    }
    .slo-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-4px);
        box-shadow: 0 12px 35px rgba(88, 166, 255, 0.12);
    }
    .slo-card-editable {
        cursor: pointer;
    }
    .slo-card .edit-indicator {
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 0.9rem;
        color: var(--accent-green);
        opacity: 0;
        transition: opacity 0.2s ease;
    }
    .slo-card:hover .edit-indicator {
        opacity: 1;
    }
    .slo-header {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 1.25rem;
    }
    .slo-icon {
        font-size: 1.75rem;
    }
    .slo-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--accent-purple);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .slo-metrics {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .slo-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        white-space: nowrap;
        padding: 0.5rem;
        border-radius: 8px;
        transition: background 0.2s ease;
    }
    .slo-metric:hover {
        background: rgba(88, 166, 255, 0.08);
    }
    .slo-metric-name {
        font-size: 0.95rem;
        color: var(--text-secondary);
        flex-shrink: 0;
    }
    .slo-metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        text-align: right;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    .slo-metric-value .edit-icon {
        font-size: 0.85rem;
        color: var(--accent-green);
        opacity: 0;
        cursor: pointer;
        transition: opacity 0.2s ease;
    }
    .slo-metric:hover .edit-icon {
        opacity: 1;
    }
    .slo-metric-good {
        color: var(--accent-green);
    }
    .slo-metric-warn {
        color: var(--accent-orange);
    }
    
    /* Impact Visualization */
    .impact-container {
        background: rgba(255,255,255,0.02);
        border-radius: 1.25rem;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        margin: 1rem 0;
    }
    .impact-header {
        font-size: 1rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .impact-item {
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: center;
        gap: 1rem;
        padding: 0.85rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .impact-item:last-child {
        border-bottom: none;
    }
    .impact-factor {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .impact-icon {
        font-size: 1.25rem;
    }
    .impact-name {
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
    }
    .impact-effect {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.4rem 0.8rem;
        background: rgba(56, 239, 125, 0.1);
        border-radius: 0.5rem;
        min-width: 180px;
        text-align: right;
    }
    .impact-positive { 
        color: #38ef7d; 
        background: rgba(56, 239, 125, 0.1);
    }
    .impact-negative { 
        color: #f5576c; 
        background: rgba(245, 87, 108, 0.1);
    }
    .impact-neutral { 
        color: rgba(255,255,255,0.7); 
        background: rgba(255,255,255,0.05);
    }
    
    /* Priority Badges - Clean pill design */
    .priority-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 18px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        color: white;
        transition: transform 0.2s ease;
    }
    .priority-badge:hover {
        transform: scale(1.03);
    }
    .priority-low_latency { background: linear-gradient(135deg, #059669, var(--accent-green)); }
    .priority-cost_saving { background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)); }
    .priority-high_quality { background: linear-gradient(135deg, var(--accent-purple), #7c3aed); }
    .priority-high_throughput { background: linear-gradient(135deg, var(--accent-orange), var(--accent-pink)); }
    .priority-balanced { background: linear-gradient(135deg, #6b7280, #4b5563); }
    
    /* Score Bars - Clean, readable with AA-style */
    .score-container {
        margin: 1.25rem 0;
        animation: slide-up 0.4s ease-out;
    }
    .score-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        color: var(--text-primary);
        font-weight: 500;
        font-size: 1rem;
    }
    .score-bar-bg {
        background: var(--bg-tertiary);
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .score-bar-quality { background: linear-gradient(90deg, #059669, var(--accent-green)); }
    .score-bar-latency { background: linear-gradient(90deg, #3b82f6, var(--accent-blue)); }
    .score-bar-cost { background: linear-gradient(90deg, #f97316, var(--accent-orange)); }
    .score-bar-capacity { background: linear-gradient(90deg, #8b5cf6, var(--accent-purple)); }
    
    /* Section Headers - Clean, modern */
    .section-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 2.5rem 0 2rem 0;
        color: var(--text-primary);
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, rgba(88, 166, 255, 0.08), rgba(163, 113, 247, 0.05));
        padding: 1.25rem 1.75rem;
        border-radius: 1rem;
        border: 1px solid var(--border-default);
        font-family: 'Inter', sans-serif;
    }
    .section-header span:first-child {
        font-size: 2rem;
    }
    
    /* Pipeline Steps - Clean card design */
    .pipeline-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 2rem;
        margin: 2.5rem 0;
    }
    .pipeline-step {
        background: var(--bg-card);
        padding: 2.5rem 2rem;
        border-radius: 1rem;
        border: 1px solid var(--border-default);
        text-align: center;
        transition: all 0.25s ease;
        position: relative;
    }
    .pipeline-step:hover {
        transform: translateY(-6px);
        border-color: var(--accent-blue);
        box-shadow: 0 15px 45px rgba(88, 166, 255, 0.12);
    }
    .pipeline-number {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        color: white;
        font-size: 1.75rem;
        margin: 0 auto 1.25rem;
        font-family: 'Inter', sans-serif;
    }
    .pipeline-number-1 { background: linear-gradient(135deg, var(--accent-purple), #7c3aed); }
    .pipeline-number-2 { background: linear-gradient(135deg, #059669, var(--accent-green)); }
    .pipeline-number-3 { background: linear-gradient(135deg, var(--accent-orange), var(--accent-pink)); }
    .pipeline-title {
        font-weight: 700;
        font-size: 1.25rem;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        font-family: 'Inter', sans-serif;
    }
    .pipeline-desc {
        font-size: 1rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* Input Container - Qualifire-inspired clean design */
    .input-container {
        background: var(--bg-card);
        padding: 2.5rem;
        border-radius: 1.25rem;
        border: 1px solid var(--border-default);
        margin: 2rem 0;
    }
    
    /* Sidebar Styling - Clean dark theme matching main */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-default) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        color: var(--text-secondary) !important;
    }
    .sidebar-section {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.25rem 0;
        border: 1px solid var(--border-default);
    }
    .sidebar-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--text-muted) !important;
        font-weight: 700;
        margin-bottom: 1.25rem;
    }
    
    /* Weight Bars - Clean design */
    .weight-item {
        margin: 1rem 0;
    }
    .weight-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        font-size: 0.9rem;
    }
    .weight-name {
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    .weight-value {
        font-weight: 700;
        color: var(--accent-blue) !important;
        font-family: 'Inter', sans-serif;
    }
    .weight-bar {
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
        background: var(--bg-tertiary);
    }
    .weight-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.4s ease;
    }
    .weight-fill-quality { background: linear-gradient(90deg, #059669, var(--accent-green)); }
    .weight-fill-latency { background: linear-gradient(90deg, #3b82f6, var(--accent-blue)); }
    .weight-fill-cost { background: linear-gradient(90deg, #f97316, var(--accent-orange)); }
    .weight-fill-capacity { background: linear-gradient(90deg, #8b5cf6, var(--accent-purple)); }
    
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 5px; }
    ::-webkit-scrollbar-thumb { 
        background: linear-gradient(135deg, #667eea, #764ba2); 
        border-radius: 5px; 
    }
    
    /* Tabs Styling - HuggingFace inspired */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-card);
        border-radius: 12px;
        padding: 8px;
        border: 1px solid var(--border-default);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        color: var(--text-secondary);
        transition: all 0.2s ease;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: rgba(88, 166, 255, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(88, 166, 255, 0.3);
    }
    
    /* Buttons - HuggingFace style rounded */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
        border: 1px solid var(--border-default);
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        padding: 12px 24px;
        font-size: 1rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(88, 166, 255, 0.2);
        border-color: var(--accent-blue);
        color: white !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
        color: white !important;
        border: none;
        box-shadow: 0 4px 15px rgba(88, 166, 255, 0.35);
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 30px rgba(88, 166, 255, 0.45);
        transform: translateY(-3px);
        filter: brightness(1.1);
        color: white !important;
    }
    
    /* Number Input Styling - Clean design */
    .stNumberInput input {
        background: var(--bg-tertiary) !important;
        border: 2px solid var(--border-default) !important;
        border-radius: 8px !important;
        color: var(--accent-green) !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stNumberInput input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.2) !important;
    }
    .stNumberInput [data-testid="stNumberInputContainer"] {
        background: transparent !important;
    }
    .stNumberInput button {
        background: var(--bg-tertiary) !important;
        color: var(--accent-blue) !important;
        border-radius: 6px !important;
    }
    .stNumberInput button:hover {
        background: rgba(88, 166, 255, 0.2) !important;
    }
    
    /* Enhanced Selectbox Styling */
    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 10px !important;
    }
    .stSelectbox > div > div:hover {
        border-color: var(--accent-blue) !important;
    }
    
    /* Enhanced Slider Styling */
    .stSlider > div > div > div {
        background: var(--bg-tertiary) !important;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), var(--accent-green)) !important;
    }
    .stSlider > div > div > div > div > div {
        background: white !important;
        border: 3px solid var(--accent-blue) !important;
        box-shadow: 0 2px 8px rgba(88, 166, 255, 0.3) !important;
    }
    
    /* Filter Panel Styling */
    .filter-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-default);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }
    
    /* Metric Badges - Clean pill design */
    .metric-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }
    .metric-badge:hover {
        transform: scale(1.03);
    }
    .metric-badge-quality {
        background: rgba(63, 185, 80, 0.12);
        color: var(--accent-green);
        border: 1px solid rgba(63, 185, 80, 0.25);
    }
    .metric-badge-latency {
        background: rgba(88, 166, 255, 0.12);
        color: var(--accent-blue);
        border: 1px solid rgba(88, 166, 255, 0.25);
    }
    .metric-badge-cost {
        background: rgba(249, 115, 22, 0.12);
        color: var(--accent-orange);
        border: 1px solid rgba(249, 115, 22, 0.25);
    }
    .metric-badge-capacity {
        background: rgba(163, 113, 247, 0.12);
        color: var(--accent-purple);
        border: 1px solid rgba(163, 113, 247, 0.25);
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), var(--accent-green)) !important;
        border-radius: 8px;
    }
    .stProgress > div {
        background: var(--bg-tertiary) !important;
        border-radius: 8px;
    }
    
    /* Info/Warning/Error Message Styling */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid var(--border-default) !important;
    }
    .stAlert > div {
        padding: 1.25rem 1.5rem !important;
    }
    
    /* Caption Styling */
    .stCaption {
        color: var(--text-muted) !important;
    }
    
    /* Text Area Focus Styling */
    .stTextArea textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15) !important;
    }
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* Disabled Button Styling */
    .stButton > button:disabled {
        opacity: 0.4;
        cursor: not-allowed;
    }
    
    /* Smooth Page Transitions */
    .main .block-container {
        animation: fade-in 0.25s ease-out;
        max-width: 1400px;
        padding: 2rem 3rem;
    }
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Better Tooltip Styling */
    [data-testid="stTooltipIcon"] {
        color: var(--text-muted) !important;
    }
    [data-testid="stTooltipIcon"]:hover {
        color: var(--accent-blue) !important;
    }
    
    /* Custom Scrollbar - Subtle */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); border-radius: 4px; }
    ::-webkit-scrollbar-thumb { 
        background: var(--bg-tertiary); 
        border-radius: 4px; 
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-blue);
    }
    
    /* Legend for charts - AA inspired */
    .chart-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        margin: 1.5rem 0;
        padding: 1rem 1.5rem;
        background: var(--bg-card);
        border-radius: 10px;
        border: 1px solid var(--border-default);
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }
    .legend-color {
        width: 14px;
        height: 14px;
        border-radius: 4px;
    }
    .legend-color-quality { background: var(--accent-green); }
    .legend-color-latency { background: var(--accent-blue); }
    .legend-color-cost { background: var(--accent-orange); }
    .legend-color-capacity { background: var(--accent-purple); }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================

if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None
if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "models_df" not in st.session_state:
    st.session_state.models_df = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
# Workflow approval state
if "extraction_approved" not in st.session_state:
    st.session_state.extraction_approved = None  # None = pending, True = approved, False = editing
if "slo_approved" not in st.session_state:
    st.session_state.slo_approved = None
if "edited_extraction" not in st.session_state:
    st.session_state.edited_extraction = None
# Editable SLO values
if "edit_slo" not in st.session_state:
    st.session_state.edit_slo = False
if "custom_ttft" not in st.session_state:
    st.session_state.custom_ttft = None
if "custom_itl" not in st.session_state:
    st.session_state.custom_itl = None
if "custom_e2e" not in st.session_state:
    st.session_state.custom_e2e = None
if "custom_qps" not in st.session_state:
    st.session_state.custom_qps = None

# Ranking weights (for balanced score calculation)
if "weight_accuracy" not in st.session_state:
    st.session_state.weight_accuracy = 4
if "weight_cost" not in st.session_state:
    st.session_state.weight_cost = 4
if "weight_latency" not in st.session_state:
    st.session_state.weight_latency = 1
if "weight_simplicity" not in st.session_state:
    st.session_state.weight_simplicity = 1
if "include_near_miss" not in st.session_state:
    st.session_state.include_near_miss = False

# Category expansion state (for inline expand/collapse of additional options)
if "expanded_categories" not in st.session_state:
    st.session_state.expanded_categories = set()

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_206_models() -> pd.DataFrame:
    """Load all 206 models from backend API."""
    try:
        # Fetch benchmark data from backend API
        response = requests.get(
            f"{API_BASE_URL}/api/v1/benchmarks",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # Convert JSON to DataFrame
        if data.get("success") and data.get("benchmarks"):
            df = pd.DataFrame(data["benchmarks"])
            # Ensure Model Name column exists and is clean
            if 'Model Name' in df.columns:
                df = df.dropna(subset=['Model Name'])
                df = df[df['Model Name'].str.strip() != '']
            return df
        else:
            logger.warning("No benchmark data returned from API")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load benchmarks from API: {e}")
        return pd.DataFrame()

@st.cache_data
def load_slo_templates():
    """Load SLO templates for all 9 use cases."""
    return {
        "chatbot_conversational": {"ttft": 150, "itl": 30, "e2e": 500, "qps": 100},
        "code_completion": {"ttft": 100, "itl": 20, "e2e": 300, "qps": 200},
        "code_generation_detailed": {"ttft": 200, "itl": 30, "e2e": 800, "qps": 50},
        "document_analysis_rag": {"ttft": 200, "itl": 40, "e2e": 1000, "qps": 50},
        "summarization_short": {"ttft": 300, "itl": 50, "e2e": 1500, "qps": 30},
        "long_document_summarization": {"ttft": 500, "itl": 60, "e2e": 5000, "qps": 10},
        "translation": {"ttft": 200, "itl": 40, "e2e": 1000, "qps": 80},
        "content_generation": {"ttft": 300, "itl": 50, "e2e": 2000, "qps": 40},
        "research_legal_analysis": {"ttft": 500, "itl": 60, "e2e": 5000, "qps": 10},
    }

@st.cache_data
def load_research_slo_ranges():
    """Load research-backed SLO ranges from JSON file (includes BLIS data)."""
    try:
        json_path = DATA_DIR / "research" / "slo_ranges.json"
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data  
def load_research_workload_patterns():
    """Load research-backed workload patterns from JSON file (includes BLIS data)."""
    try:
        json_path = DATA_DIR / "research" / "workload_patterns.json"
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data
def load_blis_benchmarks():
    """Load BLIS benchmark data for real hardware/model performance validation."""
    try:
        # BLIS file is in data/ root, not data/benchmarks/
        json_path = DATA_DIR / "benchmarks_BLIS.json"
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load BLIS data: {e}")
        return None

def get_blis_benchmark_for_config(prompt_tokens: int, output_tokens: int, hardware: str = None):
    """Get relevant BLIS benchmarks for a specific token configuration."""
    blis_data = load_blis_benchmarks()
    if not blis_data or 'benchmarks' not in blis_data:
        return None
    
    benchmarks = blis_data['benchmarks']
    
    # Filter by token config
    matching = [b for b in benchmarks 
                if b['prompt_tokens'] == prompt_tokens and b['output_tokens'] == output_tokens]
    
    # Further filter by hardware if specified
    if hardware and matching:
        hw_filtered = [b for b in matching if hardware.upper() in b['hardware'].upper()]
        if hw_filtered:
            matching = hw_filtered
    
    return matching if matching else None

def get_slo_targets_for_use_case(use_case: str, priority: str = "balanced") -> dict:
    """Get research-backed SLO targets (min/max) for a use case with priority adjustment.
    
    Returns dict with:
    - ttft_target: {"min": X, "max": Y} - the acceptable range
    - itl_target: {"min": X, "max": Y}
    - e2e_target: {"min": X, "max": Y}
    - token_config: {"prompt": X, "output": Y}
    """
    research_data = load_research_slo_ranges()
    if not research_data:
        return None
    
    slo_ranges = research_data.get('slo_ranges', {})
    priority_adjustments = research_data.get('priority_adjustments', {})
    
    use_case_data = slo_ranges.get(use_case)
    if not use_case_data:
        return None
    
    # Get priority adjustment factors
    priority_factor = priority_adjustments.get(priority, {})
    ttft_factor = priority_factor.get('ttft_factor', 1.0)
    itl_factor = priority_factor.get('itl_factor', 1.0)
    e2e_factor = priority_factor.get('e2e_factor', 1.0)
    
    return {
        "use_case": use_case,
        "priority": priority,
        "description": use_case_data.get('description', ''),
        "ttft_target": {
            "min": int(use_case_data['ttft_ms']['min'] * ttft_factor),
            "max": int(use_case_data['ttft_ms']['max'] * ttft_factor),
        },
        "itl_target": {
            "min": int(use_case_data['itl_ms']['min'] * itl_factor),
            "max": int(use_case_data['itl_ms']['max'] * itl_factor),
        },
        "e2e_target": {
            "min": int(use_case_data['e2e_ms']['min'] * e2e_factor),
            "max": int(use_case_data['e2e_ms']['max'] * e2e_factor),
        },
        "token_config": use_case_data.get('token_config', {"prompt": 512, "output": 256}),
        "tokens_per_sec_target": use_case_data.get('tokens_per_sec', {}).get('target', 100),
        "research_note": use_case_data.get('research_note', ''),
        "recommended_hardware": priority_factor.get('recommended_hardware', 'H100_x2'),
    }


def recommend_optimal_hardware(use_case: str, priority: str, user_hardware: str = None) -> dict:
    """Recommend optimal hardware from BLIS benchmarks based on SLO requirements.

    DEPRECATED: This function is kept for potential future use. The UI now uses
    the backend API via fetch_ranked_recommendations() instead.

    Logic:
    - cost_saving: Find CHEAPEST hardware that meets MAX SLO (slowest acceptable)
    - low_latency: Find hardware that meets MIN SLO (fastest required)
    - balanced: Find hardware that meets MEAN of SLO range
    - high_quality: Relax latency, focus on larger models
    - high_throughput: Focus on tokens/sec capacity

    Returns hardware recommendation with BLIS benchmark data.
    """
    # Get SLO targets
    slo_targets = get_slo_targets_for_use_case(use_case, priority)
    if not slo_targets:
        return None

    # Get token config
    prompt_tokens = slo_targets['token_config']['prompt']
    output_tokens = slo_targets['token_config']['output']

    # Load BLIS benchmarks
    blis_data = load_blis_benchmarks()
    if not blis_data or 'benchmarks' not in blis_data:
        return None

    benchmarks = blis_data['benchmarks']

    # Filter by token config
    matching = [b for b in benchmarks
                if b['prompt_tokens'] == prompt_tokens and b['output_tokens'] == output_tokens]

    if not matching:
        return None

    # Define hardware costs (approximate monthly cost)
    # Both H100 and A100-80 are REAL BLIS benchmarks from Andre's data
    hardware_costs = {
        ("H100", 1): {"cost": 2500, "tier": 2},
        ("H100", 2): {"cost": 5000, "tier": 3},
        ("H100", 4): {"cost": 10000, "tier": 4},
        ("H100", 8): {"cost": 20000, "tier": 5},
        ("A100-80", 1): {"cost": 1600, "tier": 1},
        ("A100-80", 2): {"cost": 3200, "tier": 2},
        ("A100-80", 4): {"cost": 6400, "tier": 3},
    }

    # Determine target SLO based on priority
    if priority == "cost_saving":
        # Target MAX SLO (slowest acceptable) to use cheapest hardware
        target_ttft = slo_targets['ttft_target']['max']
        target_e2e = slo_targets['e2e_target']['max']
        sort_by = "cost"  # Sort by cost ascending
    elif priority == "low_latency":
        # Target MIN SLO (fastest required)
        target_ttft = slo_targets['ttft_target']['min']
        target_e2e = slo_targets['e2e_target']['min']
        sort_by = "latency"  # Sort by latency ascending
    elif priority == "high_throughput":
        # Target tokens/sec
        target_ttft = slo_targets['ttft_target']['max']  # Relax latency
        target_e2e = slo_targets['e2e_target']['max']
        sort_by = "throughput"  # Sort by tokens/sec descending
    else:  # balanced, high_quality
        # Target MEAN of range
        target_ttft = (slo_targets['ttft_target']['min'] + slo_targets['ttft_target']['max']) // 2
        target_e2e = (slo_targets['e2e_target']['min'] + slo_targets['e2e_target']['max']) // 2
        sort_by = "balanced"

    # Group benchmarks by hardware config
    hw_benchmarks = {}
    for b in matching:
        hw_key = (b['hardware'], b['hardware_count'])
        if hw_key not in hw_benchmarks:
            hw_benchmarks[hw_key] = []
        hw_benchmarks[hw_key].append(b)

    # Evaluate each hardware option
    viable_options = []
    for hw_key, benches in hw_benchmarks.items():
        # Get best benchmark (lowest TTFT at reasonable RPS)
        best = min(benches, key=lambda x: x['ttft_mean'])

        hw_cost = hardware_costs.get(hw_key, {"cost": 99999, "tier": 99})

        # Check if meets SLO requirements
        meets_ttft = best['ttft_p95'] <= target_ttft * 1.2  # 20% buffer
        meets_e2e = best['e2e_p95'] <= target_e2e * 1.2

        # Don't recommend hardware that's WAY faster than needed (over-provisioning)
        too_fast = False
        if priority == "cost_saving":
            # If TTFT is less than 50% of max, it's over-provisioned
            if best['ttft_mean'] < slo_targets['ttft_target']['max'] * 0.3:
                too_fast = True

        viable_options.append({
            "hardware": hw_key[0],
            "hardware_count": hw_key[1],
            "cost_monthly": hw_cost['cost'],
            "tier": hw_cost['tier'],
            "ttft_mean": best['ttft_mean'],
            "ttft_p95": best['ttft_p95'],
            "e2e_mean": best['e2e_mean'],
            "e2e_p95": best['e2e_p95'],
            "itl_mean": best['itl_mean'],
            "tokens_per_sec": best['tokens_per_second'],
            "meets_slo": meets_ttft and meets_e2e,
            "over_provisioned": too_fast,
            "benchmark_count": len(benches),
            "model_repo": best['model_hf_repo'],
        })

    # Filter to only viable options (meets SLO)
    viable = [v for v in viable_options if v['meets_slo']]

    # If no viable options, return best available
    if not viable:
        viable = viable_options

    # Sort based on priority
    if sort_by == "cost":
        # For cost_saving: prefer cheapest that meets SLO, not over-provisioned
        viable = [v for v in viable if not v['over_provisioned']] or viable
        viable.sort(key=lambda x: (x['cost_monthly'], x['ttft_mean']))
    elif sort_by == "latency":
        viable.sort(key=lambda x: x['ttft_mean'])
    elif sort_by == "throughput":
        viable.sort(key=lambda x: -x['tokens_per_sec'])
    else:  # balanced
        # Balance cost and latency
        viable.sort(key=lambda x: (x['tier'], x['ttft_mean']))

    if not viable:
        return None

    best_option = viable[0]
    alternatives = viable[1:4] if len(viable) > 1 else []

    return {
        "recommended": best_option,
        "alternatives": alternatives,
        "slo_targets": slo_targets,
        "selection_reason": _get_hardware_selection_reason(priority, best_option, slo_targets),
    }


def _get_hardware_selection_reason(priority: str, hw_option: dict, slo_targets: dict) -> str:
    """Generate explanation for why this hardware was selected.

    DEPRECATED: Helper for recommend_optimal_hardware().
    """
    hw_name = f"{hw_option['hardware']} x{hw_option['hardware_count']}"
    ttft = hw_option['ttft_mean']
    cost = hw_option['cost_monthly']
    target_max = slo_targets['ttft_target']['max']
    target_min = slo_targets['ttft_target']['min']

    if priority == "cost_saving":
        return f"💰 {hw_name} is the cheapest option (${cost:,}/mo) that meets your SLO max ({target_max}ms TTFT). Actual TTFT: {ttft:.0f}ms - good value!"
    elif priority == "low_latency":
        return f"⚡ {hw_name} achieves {ttft:.0f}ms TTFT, meeting your aggressive target ({target_min}ms). Fastest option for your use case."
    elif priority == "high_throughput":
        return f"📈 {hw_name} offers {hw_option['tokens_per_sec']:.0f} tokens/sec - best throughput for high-volume workloads."
    elif priority == "high_quality":
        return f"⭐ {hw_name} provides headroom for larger, higher-quality models with {ttft:.0f}ms TTFT."
    else:  # balanced
        return f"⚖️ {hw_name} balances cost (${cost:,}/mo) and latency ({ttft:.0f}ms TTFT) - optimal for balanced priority."


# =============================================================================
# RANKED RECOMMENDATIONS (Backend API Integration)
# =============================================================================

def fetch_ranked_recommendations(
    use_case: str,
    user_count: int,
    priority: str,
    prompt_tokens: int,
    output_tokens: int,
    expected_qps: float,
    ttft_p95_target_ms: int,
    itl_p95_target_ms: int,
    e2e_p95_target_ms: int,
    weights: dict = None,
    include_near_miss: bool = False,
) -> dict | None:
    """Fetch ranked recommendations from the backend API.

    Args:
        use_case: Use case identifier (e.g., "chatbot_conversational")
        user_count: Number of concurrent users
        priority: UI priority (maps to latency_requirement/budget_constraint)
        prompt_tokens: Input prompt token count
        output_tokens: Output generation token count
        expected_qps: Queries per second
        ttft_p95_target_ms: TTFT SLO target (p95)
        itl_p95_target_ms: ITL SLO target (p95)
        e2e_p95_target_ms: E2E SLO target (p95)
        weights: Optional dict with accuracy, price, latency, complexity weights (0-10)
        include_near_miss: Whether to include near-SLO configurations

    Returns:
        RankedRecommendationsResponse as dict, or None on error
    """
    import requests

    # Map UI priority to backend latency_requirement and budget_constraint
    priority_mapping = {
        "low_latency": {"latency_requirement": "very_high", "budget_constraint": "flexible"},
        "balanced": {"latency_requirement": "high", "budget_constraint": "moderate"},
        "cost_saving": {"latency_requirement": "medium", "budget_constraint": "strict"},
        "high_throughput": {"latency_requirement": "high", "budget_constraint": "moderate"},
        "high_quality": {"latency_requirement": "medium", "budget_constraint": "flexible"},
    }

    mapping = priority_mapping.get(priority, priority_mapping["balanced"])

    # Build request payload
    payload = {
        "use_case": use_case,
        "user_count": user_count,
        "latency_requirement": mapping["latency_requirement"],
        "budget_constraint": mapping["budget_constraint"],
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "expected_qps": expected_qps,
        "ttft_p95_target_ms": ttft_p95_target_ms,
        "itl_p95_target_ms": itl_p95_target_ms,
        "e2e_p95_target_ms": e2e_p95_target_ms,
        "include_near_miss": include_near_miss,
    }

    if weights:
        payload["weights"] = weights

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/ranked-recommend-from-spec",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch ranked recommendations: {e}")
        return None


def render_weight_controls() -> None:
    """Render weight controls and configuration options in a collapsible section.

    Updates session state directly:
    - weight_accuracy, weight_cost, weight_latency, weight_simplicity (0-10 each)
    - include_near_miss (bool)
    """
    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("""
            <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">
                <strong>Ranking Weights</strong> (0-10) — adjust to customize balanced score
            </div>
            """, unsafe_allow_html=True)

            wcol1, wcol2, wcol3, wcol4 = st.columns(4)

            with wcol1:
                accuracy = st.number_input(
                    "Accuracy",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.weight_accuracy,
                    key="input_accuracy",
                    help="Higher = prefer more capable models"
                )
                if accuracy != st.session_state.weight_accuracy:
                    st.session_state.weight_accuracy = accuracy

            with wcol2:
                cost = st.number_input(
                    "Cost",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.weight_cost,
                    key="input_cost",
                    help="Higher = prefer cheaper configurations"
                )
                if cost != st.session_state.weight_cost:
                    st.session_state.weight_cost = cost

            with wcol3:
                latency = st.number_input(
                    "Latency",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.weight_latency,
                    key="input_latency",
                    help="Higher = prefer lower latency"
                )
                if latency != st.session_state.weight_latency:
                    st.session_state.weight_latency = latency

            with wcol4:
                simplicity = st.number_input(
                    "Simplicity",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.weight_simplicity,
                    key="input_simplicity",
                    help="Higher = prefer simpler deployments"
                )
                if simplicity != st.session_state.weight_simplicity:
                    st.session_state.weight_simplicity = simplicity

        with col2:
            st.markdown("""
            <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">
                <strong>Options</strong>
            </div>
            """, unsafe_allow_html=True)

            include_near_miss = st.checkbox(
                "Include near-miss configs",
                value=st.session_state.include_near_miss,
                key="checkbox_near_miss",
                help="Include configurations that nearly meet SLO targets"
            )
            if include_near_miss != st.session_state.include_near_miss:
                st.session_state.include_near_miss = include_near_miss

            # Re-evaluate button to apply weight changes (styled as primary/blue)
            if st.button("🔄 Re-Evaluate", key="re_evaluate_btn", type="primary", help="Apply weight changes and re-fetch recommendations"):
                st.rerun()


def render_recommendation_category_card(
    category_name: str,
    category_emoji: str,
    category_color: str,
    recommendations: list,
):
    """Render a single recommendation category card with top pick and expander.

    Args:
        category_name: Display name (e.g., "Balanced", "Best Accuracy")
        category_emoji: Emoji for the category
        category_color: Hex color for styling
        recommendations: List of recommendation dicts from backend
    """
    if not recommendations:
        st.markdown(f"""
        <div style="background: var(--bg-card); padding: 1rem; border-radius: 0.75rem;
                    border: 1px solid {category_color}40; min-height: 150px;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                <span style="font-size: 1.25rem;">{category_emoji}</span>
                <span style="color: {category_color}; font-weight: 700;">{category_name}</span>
            </div>
            <p style="color: rgba(255,255,255,0.5); font-style: italic;">No configurations found</p>
        </div>
        """, unsafe_allow_html=True)
        return

    top = recommendations[0]
    model_name = top.get("model_name", "Unknown")
    gpu_type = top.get("gpu_config", {}).get("gpu_type", "Unknown") if isinstance(top.get("gpu_config"), dict) else "Unknown"
    gpu_count = top.get("gpu_config", {}).get("gpu_count", 1) if isinstance(top.get("gpu_config"), dict) else 1
    ttft = top.get("predicted_ttft_p95_ms", 0)
    cost = top.get("cost_per_month_usd", 0)
    meets_slo = top.get("meets_slo", False)
    scores = top.get("scores", {})
    balanced_score = scores.get("balanced_score", 0) if isinstance(scores, dict) else 0
    accuracy_score = scores.get("accuracy_score", 0) if isinstance(scores, dict) else 0

    slo_badge = "✅" if meets_slo else "⚠️"

    st.markdown(f"""
    <div style="background: var(--bg-card); padding: 1rem; border-radius: 0.75rem;
                border: 1px solid {category_color}40;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
            <span style="font-size: 1.25rem;">{category_emoji}</span>
            <span style="color: {category_color}; font-weight: 700;">{category_name}</span>
            <span style="margin-left: auto; font-size: 0.9rem;">{slo_badge}</span>
        </div>
        <div style="margin-bottom: 0.5rem;">
            <span style="color: white; font-weight: 600; font-size: 1.1rem;">{model_name}</span>
        </div>
        <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">
            {gpu_count}x {gpu_type}
        </div>
        <div style="display: flex; gap: 1rem; font-size: 0.85rem; color: rgba(255,255,255,0.6);">
            <span>⏱️ {ttft:.0f}ms</span>
            <span>💰 ${cost:,.0f}/mo</span>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.85rem;">
            <span style="color: {category_color};">Score: {balanced_score:.1f}</span>
            <span style="color: rgba(255,255,255,0.5);"> | Accuracy: {accuracy_score:.0f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Expander for additional options
    if len(recommendations) > 1:
        with st.expander(f"Show {len(recommendations) - 1} more options"):
            for rec in recommendations[1:]:
                rec_model = rec.get("model_name", "Unknown")
                rec_gpu_type = rec.get("gpu_config", {}).get("gpu_type", "Unknown") if isinstance(rec.get("gpu_config"), dict) else "Unknown"
                rec_gpu_count = rec.get("gpu_config", {}).get("gpu_count", 1) if isinstance(rec.get("gpu_config"), dict) else 1
                rec_ttft = rec.get("predicted_ttft_p95_ms", 0)
                rec_cost = rec.get("cost_per_month_usd", 0)
                rec_meets_slo = "✅" if rec.get("meets_slo", False) else "⚠️"
                rec_scores = rec.get("scores", {})
                rec_balanced = rec_scores.get("balanced_score", 0) if isinstance(rec_scores, dict) else 0

                st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.25rem 0; background: rgba(255,255,255,0.03);
                            border-radius: 0.5rem; font-size: 0.9rem;">
                    <span style="color: white; font-weight: 500;">{rec_meets_slo} {rec_model}</span>
                    <span style="color: rgba(255,255,255,0.5);"> — {rec_gpu_count}x {rec_gpu_type},
                          {rec_ttft:.0f}ms, ${rec_cost:,.0f}/mo (Score: {rec_balanced:.1f})</span>
                </div>
                """, unsafe_allow_html=True)


def render_ranked_recommendations(response: dict, show_config: bool = True):
    """Render all 5 ranked recommendation categories in a table format.

    Args:
        response: RankedRecommendationsResponse from backend
        show_config: Whether to show the configuration controls (default True)
    """
    total_configs = response.get("total_configs_evaluated", 0)
    configs_after_filters = response.get("configs_after_filters", 0)

    st.markdown(
        '<div class="section-header" style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.15), rgba(16, 185, 129, 0.1)); border: 1px solid rgba(6, 182, 212, 0.2);"><span>🖥️</span> Recommended Solutions</div>',
        unsafe_allow_html=True
    )

    # Render configuration controls right under the header
    if show_config:
        render_weight_controls()

    st.markdown(
        f'<div style="color: rgba(255,255,255,0.6); margin-bottom: 1rem; font-size: 0.9rem;">Evaluated <span style="color: #06b6d4; font-weight: 600;">{total_configs}</span> viable configurations, showing <span style="color: #10b981; font-weight: 600;">{configs_after_filters}</span> unique options</div>',
        unsafe_allow_html=True
    )

    # Define categories in the requested order
    categories = [
        ("balanced", "Balanced", "⚖️", "#8b5cf6"),
        ("best_accuracy", "Best Accuracy", "🎯", "#10b981"),
        ("lowest_cost", "Lowest Cost", "💰", "#f59e0b"),
        ("lowest_latency", "Lowest Latency", "⚡", "#06b6d4"),
        ("simplest", "Simplest", "🎛️", "#ec4899"),
    ]

    # Helper function to format GPU config with TP and replicas
    def format_gpu_config(gpu_config: dict) -> str:
        if not isinstance(gpu_config, dict):
            return "Unknown"
        gpu_type = gpu_config.get("gpu_type", "Unknown")
        gpu_count = gpu_config.get("gpu_count", 1)
        tp = gpu_config.get("tensor_parallel", 1)
        replicas = gpu_config.get("replicas", 1)
        return f"{gpu_count}x {gpu_type} (TP={tp}, R={replicas})"

    # Helper function to build a table row from a recommendation
    def build_row(rec: dict, cat_color: str, cat_name: str = "", cat_emoji: str = "", is_top: bool = False, more_count: int = 0) -> str:
        model_name = rec.get("model_name", "Unknown")
        gpu_config = rec.get("gpu_config", {})
        gpu_str = format_gpu_config(gpu_config)
        ttft = rec.get("predicted_ttft_p95_ms", 0)
        cost = rec.get("cost_per_month_usd", 0)
        meets_slo = rec.get("meets_slo", False)
        scores = rec.get("scores", {})
        accuracy_score = scores.get("accuracy_score", 0) if isinstance(scores, dict) else 0
        price_score = scores.get("price_score", 0) if isinstance(scores, dict) else 0
        latency_score = scores.get("latency_score", 0) if isinstance(scores, dict) else 0
        complexity_score = scores.get("complexity_score", 0) if isinstance(scores, dict) else 0
        balanced_score = scores.get("balanced_score", 0) if isinstance(scores, dict) else 0
        slo_badge = "✅" if meets_slo else "⚠️"

        if is_top:
            more_badge = f'<span style="color: rgba(255,255,255,0.4); font-size: 0.8rem; margin-left: 0.5rem;">(+{more_count})</span>' if more_count > 0 else ''
            cat_cell = f'<td style="padding: 0.75rem 0.5rem;"><span style="color: {cat_color}; font-weight: 600;">{cat_emoji} {cat_name}</span>{more_badge}</td>'
        else:
            cat_cell = f'<td style="padding: 0.75rem 0.5rem; padding-left: 2rem;"><span style="color: {cat_color}40;">↳</span></td>'

        return (
            f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);{"background: rgba(255,255,255,0.02);" if not is_top else ""}">'
            f'{cat_cell}'
            f'<td style="padding: 0.75rem 0.5rem; color: white; font-weight: {"500" if is_top else "400"};">{model_name}</td>'
            f'<td style="padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.8); font-size: 0.85rem;">{gpu_str}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: right; color: #06b6d4;">{ttft:.0f}ms</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: right; color: #f59e0b;">${cost:,.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; color: #10b981;">{accuracy_score:.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; color: #f59e0b;">{price_score:.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; color: #06b6d4;">{latency_score:.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; color: #ec4899;">{complexity_score:.0f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center; color: #8b5cf6; font-weight: 600;">{balanced_score:.1f}</td>'
            f'<td style="padding: 0.75rem 0.5rem; text-align: center;">{slo_badge}</td>'
            f'</tr>'
        )

    # Table header styles
    th_style = 'style="text-align: left; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;"'
    th_style_right = 'style="text-align: right; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;"'
    th_style_center = 'style="text-align: center; padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600;"'

    table_header = (
        '<thead>'
        f'<tr style="border-bottom: 2px solid rgba(255,255,255,0.2);">'
        f'<th {th_style}>Category</th>'
        f'<th {th_style}>Model</th>'
        f'<th {th_style}>GPU Config</th>'
        f'<th {th_style_right}>TTFT (p95)</th>'
        f'<th {th_style_right}>Cost/mo</th>'
        f'<th {th_style_center}>🎯</th>'
        f'<th {th_style_center}>💰</th>'
        f'<th {th_style_center}>⚡</th>'
        f'<th {th_style_center}>🎛️</th>'
        f'<th {th_style_center}>⚖️</th>'
        f'<th {th_style_center}>SLO</th>'
        '</tr>'
        '</thead>'
    )

    # Build ONE unified table with all categories
    # Add a first column for expand/collapse buttons rendered as part of the Category cell

    # Collect expansion state and category data
    category_data = []
    for cat_key, cat_name, cat_emoji, cat_color in categories:
        recs = response.get(cat_key, [])
        is_expanded = cat_key in st.session_state.expanded_categories
        more_count = len(recs) - 1 if len(recs) > 1 else 0
        category_data.append({
            "key": cat_key,
            "name": cat_name,
            "emoji": cat_emoji,
            "color": cat_color,
            "recs": recs,
            "is_expanded": is_expanded,
            "more_count": more_count,
        })

    # Render expand/collapse toggle buttons in a compact row
    # These appear above the table, one per category with expandable options
    expandable_cats = [c for c in category_data if c["more_count"] > 0]
    if expandable_cats:
        st.markdown(
            '<div style="margin-bottom: 0.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center;">',
            unsafe_allow_html=True
        )
        toggle_cols = st.columns(len(expandable_cats) + 1)
        toggle_cols[0].markdown(
            '<span style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">Expand:</span>',
            unsafe_allow_html=True
        )
        for idx, cat in enumerate(expandable_cats):
            with toggle_cols[idx + 1]:
                btn_label = f"−{cat['emoji']}" if cat["is_expanded"] else f"+{cat['emoji']}"
                if st.button(
                    btn_label,
                    key=f"toggle_{cat['key']}",
                    help=f"{'Collapse' if cat['is_expanded'] else 'Expand'} {cat['name']} (+{cat['more_count']})"
                ):
                    if cat["is_expanded"]:
                        st.session_state.expanded_categories.discard(cat["key"])
                    else:
                        st.session_state.expanded_categories.add(cat["key"])
                    st.rerun()

    # Build all rows for the unified table
    all_rows = []

    for cat in category_data:
        cat_key = cat["key"]
        cat_name = cat["name"]
        cat_emoji = cat["emoji"]
        cat_color = cat["color"]
        recs = cat["recs"]
        is_expanded = cat["is_expanded"]
        more_count = cat["more_count"]

        if not recs:
            # Empty category row
            all_rows.append(
                f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">'
                f'<td style="padding: 0.75rem 0.5rem;"><span style="color: {cat_color}; font-weight: 600;">{cat_emoji} {cat_name}</span></td>'
                f'<td colspan="10" style="padding: 0.75rem 0.5rem; color: rgba(255,255,255,0.5); font-style: italic;">No configurations found</td>'
                f'</tr>'
            )
        else:
            # Top recommendation for this category
            all_rows.append(build_row(recs[0], cat_color, cat_name, cat_emoji, is_top=True, more_count=more_count))

            # Add additional rows if expanded
            if is_expanded and len(recs) > 1:
                for rec in recs[1:]:
                    all_rows.append(build_row(rec, cat_color, "", "", is_top=False, more_count=0))

    # Render the single unified table
    unified_table_html = (
        f'<table style="width: 100%; border-collapse: collapse;">'
        + table_header +
        '<tbody>' + ''.join(all_rows) + '</tbody>'
        '</table>'
    )

    st.markdown(unified_table_html, unsafe_allow_html=True)


def get_blis_slo_for_model(model_name: str, use_case: str, hardware: str = "H100") -> dict:
    """Get REAL BLIS benchmark SLO data for a specific model and use case.
    
    IMPORTANT: Only returns data if we have ACTUAL benchmarks for this model.
    Returns None if no matching BLIS data exists (we don't fake data).
    
    BLIS dataset contains only these models:
    - qwen2.5-7b, llama-3.1-8b, llama-3.3-70b, phi-4
    - mistral-small-24b, mixtral-8x7b, granite-3.1-8b
    - gpt-oss-120b, gpt-oss-20b
    """
    blis_data = load_blis_benchmarks()
    if not blis_data or 'benchmarks' not in blis_data:
        return None
    
    benchmarks = blis_data['benchmarks']
    
    # Map use case to token config
    token_configs = {
        "code_completion": (512, 256),
        "chatbot_conversational": (512, 256),
        "code_generation_detailed": (1024, 1024),
        "translation": (512, 256),
        "content_generation": (512, 256),
        "summarization_short": (4096, 512),
        "document_analysis_rag": (4096, 512),
        "long_document_summarization": (10240, 1536),
        "research_legal_analysis": (10240, 1536),
    }
    
    prompt_tokens, output_tokens = token_configs.get(use_case, (512, 256))
    
    # STRICT mapping: Only map if we're confident it's the same model family
    # Key = pattern to find in recommended model name, Value = BLIS repo
    model_mapping = {
        "qwen2.5": "qwen/qwen2.5-7b-instruct",
        "qwen 2.5": "qwen/qwen2.5-7b-instruct",
        "llama 3.1 8b": "meta-llama/llama-3.1-8b-instruct",
        "llama 3.1 instruct 8b": "meta-llama/llama-3.1-8b-instruct",
        "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
        "llama 3.3": "meta-llama/llama-3.3-70b-instruct",
        "llama-3.3": "meta-llama/llama-3.3-70b-instruct",
        "granite": "ibm-granite/granite-3.1-8b-instruct",
        "phi-4": "microsoft/phi-4",
        "phi 4": "microsoft/phi-4",
        "mistral small": "mistralai/mistral-small-24b-instruct-2501",
        "mistral-small": "mistralai/mistral-small-24b-instruct-2501",
        "mixtral": "mistralai/mixtral-8x7b-instruct-v0.1",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
    }
    
    # Find matching model in BLIS data - STRICT matching
    model_lower = model_name.lower()
    blis_model_repo = None
    is_exact_match = False
    
    for key, repo in model_mapping.items():
        if key in model_lower:
            blis_model_repo = repo
            is_exact_match = True
            break
    
    # If no exact match, return None - we don't want to show misleading data
    if not blis_model_repo:
        return None
    
    # Filter benchmarks by token config
    matching = [b for b in benchmarks 
                if b['prompt_tokens'] == prompt_tokens 
                and b['output_tokens'] == output_tokens]
    
    # MUST match the specific model - no fallbacks
    model_matches = [b for b in matching if b['model_hf_repo'] == blis_model_repo]
    if not model_matches:
        return None  # No BLIS data for this model
    
    matching = model_matches
    
    # Filter by hardware if specified
    if hardware:
        hw_matches = [b for b in matching if hardware.upper() in b['hardware'].upper()]
        if hw_matches:
            matching = hw_matches
    
    if not matching:
        return None
    
    # Get best benchmark (lowest E2E at reasonable RPS)
    matching.sort(key=lambda x: x['e2e_mean'])
    best = matching[0]
    
    return {
        "model_repo": best['model_hf_repo'],
        "is_exact_match": is_exact_match,
        "recommended_model": model_name,
        "hardware": best['hardware'],
        "hardware_count": best['hardware_count'],
        "token_config": {"prompt": prompt_tokens, "output": output_tokens},
        "slo_actual": {
            "ttft_mean_ms": round(best['ttft_mean'], 1),
            "ttft_p95_ms": round(best['ttft_p95'], 1),
            "ttft_p99_ms": round(best['ttft_p99'], 1),
            "itl_mean_ms": round(best['itl_mean'], 1),
            "itl_p95_ms": round(best['itl_p95'], 1),
            "itl_p99_ms": round(best['itl_p99'], 1),
            "e2e_mean_ms": round(best['e2e_mean'], 0),
            "e2e_p95_ms": round(best['e2e_p95'], 0),
            "e2e_p99_ms": round(best['e2e_p99'], 0),
        },
        "throughput": {
            "tokens_per_sec": round(best['tokens_per_second'], 0),
            "requests_per_sec": round(best['requests_per_second'], 2),
            "responses_per_sec": round(best['responses_per_second'], 2),
        },
        "benchmark_samples": len(matching),
        "ranges": {
            "ttft_range_ms": [round(min(b['ttft_mean'] for b in matching), 1), 
                             round(max(b['ttft_mean'] for b in matching), 1)],
            "e2e_range_ms": [round(min(b['e2e_mean'] for b in matching), 0),
                           round(max(b['e2e_mean'] for b in matching), 0)],
        }
    }

def validate_slo_against_research(use_case: str, ttft: int, itl: int, e2e: int, priority: str = "balanced") -> list:
    """Validate SLO values against research-backed ranges and return warnings/info messages.
    
    Returns list of tuples: (icon, color, message, severity)
    Severity: 'error' (red), 'warning' (orange), 'info' (blue), 'success' (green)
    """
    messages = []
    research_data = load_research_slo_ranges()
    
    if not research_data or 'slo_ranges' not in research_data:
        return messages
    
    slo_ranges = research_data.get('slo_ranges', {})
    priority_adjustments = research_data.get('priority_adjustments', {})
    
    # Get use case specific ranges
    use_case_ranges = slo_ranges.get(use_case)
    if not use_case_ranges:
        return messages
    
    # Get priority factor
    priority_factor = priority_adjustments.get(priority, {})
    ttft_factor = priority_factor.get('ttft_factor', 1.0)
    itl_factor = priority_factor.get('itl_factor', 1.0)
    e2e_factor = priority_factor.get('e2e_factor', 1.0)
    
    # Adjust ranges based on priority
    ttft_min = int(use_case_ranges['ttft_ms']['min'] * ttft_factor)
    ttft_max = int(use_case_ranges['ttft_ms']['max'] * ttft_factor)
    itl_min = int(use_case_ranges['itl_ms']['min'] * itl_factor)
    itl_max = int(use_case_ranges['itl_ms']['max'] * itl_factor)
    e2e_min = int(use_case_ranges['e2e_ms']['min'] * e2e_factor)
    e2e_max = int(use_case_ranges['e2e_ms']['max'] * e2e_factor)
    
    # Get BLIS observed values for context
    blis_ttft = use_case_ranges.get('ttft_ms', {}).get('blis_observed', {})
    blis_itl = use_case_ranges.get('itl_ms', {}).get('blis_observed', {})
    blis_e2e = use_case_ranges.get('e2e_ms', {}).get('blis_observed', {})
    
    # TTFT validation with BLIS context
    if ttft < ttft_min:
        blis_min = blis_ttft.get('min', 'N/A')
        messages.append((
            "🔬", "#f5576c", 
            f"TTFT ({ttft}ms) is BELOW min ({ttft_min}ms). BLIS observed min: {blis_min}ms on H100x8!",
            "error"
        ))
    elif ttft > ttft_max:
        blis_mean = blis_ttft.get('mean', 'N/A')
        messages.append((
            "💸", "#fbbf24",
            f"TTFT ({ttft}ms) is ABOVE max ({ttft_max}ms). BLIS avg: {blis_mean}ms - you're over-provisioning!",
            "warning"
        ))
    else:
        messages.append((
            "✅", "#10b981",
            f"TTFT ({ttft}ms) ✓ within range ({ttft_min}-{ttft_max}ms)",
            "success"
        ))
    
    # ITL validation with BLIS context
    if itl < itl_min:
        blis_min = blis_itl.get('min', 'N/A')
        messages.append((
            "🔬", "#f5576c",
            f"ITL ({itl}ms) is BELOW min ({itl_min}ms). BLIS observed min: {blis_min}ms - needs batch size 1!",
            "error"
        ))
    elif itl > itl_max:
        blis_mean = blis_itl.get('mean', 'N/A')
        messages.append((
            "💸", "#fbbf24",
            f"ITL ({itl}ms) is ABOVE max ({itl_max}ms). BLIS avg: {blis_mean}ms - streaming may feel slow.",
            "warning"
        ))
    else:
        messages.append((
            "✅", "#10b981",
            f"ITL ({itl}ms) ✓ within range ({itl_min}-{itl_max}ms)",
            "success"
        ))
    
    # E2E validation with BLIS context
    if e2e < e2e_min:
        blis_min = blis_e2e.get('min', 'N/A')
        messages.append((
            "🔬", "#f5576c",
            f"E2E ({e2e}ms) is BELOW min ({e2e_min}ms). BLIS best: {blis_min}ms - very aggressive!",
            "error"
        ))
    elif e2e > e2e_max:
        blis_mean = blis_e2e.get('mean', 'N/A')
        messages.append((
            "💸", "#fbbf24",
            f"E2E ({e2e}ms) is ABOVE max ({e2e_max}ms). BLIS avg: {blis_mean}ms - over-provisioned!",
            "warning"
        ))
    else:
        messages.append((
            "✅", "#10b981",
            f"E2E ({e2e}ms) ✓ within range ({e2e_min}-{e2e_max}ms)",
            "success"
        ))
    
    # Add research note
    if use_case_ranges.get('research_note'):
        messages.append((
            "📚", "#a371f7",
            f"{use_case_ranges['research_note']}",
            "info"
        ))
    
    return messages

def validate_hardware_efficiency(use_case: str, hardware: str, ttft: int, qps: int) -> list:
    """Validate if hardware choice is efficient for the use case and SLOs.
    Uses BLIS benchmark data for accurate recommendations.
    
    Returns list of tuples: (icon, color, message, severity)
    """
    messages = []
    
    if not hardware:
        return messages
    
    # Load BLIS hardware benchmarks from research data
    research_data = load_research_slo_ranges()
    blis_hw = research_data.get('hardware_benchmarks', {}) if research_data else {}
    
    # Hardware capabilities from BLIS benchmarks
    hardware_specs = {
        "H100": {
            "cost_per_hour": 3.50, 
            "tokens_per_sec": blis_hw.get('H100_x1', {}).get('tokens_per_sec_mean', 808),
            "ttft_mean": blis_hw.get('H100_x1', {}).get('ttft_mean_ms', 87.6),
            "tier": "premium",
            "best_for": blis_hw.get('H100_x1', {}).get('best_for', [])
        },
        "A100": {
            "cost_per_hour": 2.20, 
            "tokens_per_sec": blis_hw.get('A100_x1', {}).get('tokens_per_sec_mean', 412),
            "ttft_mean": blis_hw.get('A100_x1', {}).get('ttft_mean_ms', 88.8),
            "tier": "high",
            "best_for": blis_hw.get('A100_x1', {}).get('best_for', [])
        },
        "L40S": {"cost_per_hour": 1.50, "tokens_per_sec": 100, "ttft_mean": 150, "tier": "mid", "best_for": []},
    }
    
    # Use cases that DON'T need premium hardware
    simple_use_cases = ["code_completion", "chatbot_conversational", "summarization_short", "translation"]
    complex_use_cases = ["research_legal_analysis", "long_document_summarization", "document_analysis_rag"]
    
    hw_spec = hardware_specs.get(hardware, {})
    blis_tokens_sec = hw_spec.get('tokens_per_sec', 100)
    blis_ttft = hw_spec.get('ttft_mean', 100)
    
    # Check for over-provisioning using BLIS data
    if hardware == "H100":
        if use_case in simple_use_cases and ttft > 150 and qps < 100:
            a100_cost = hardware_specs['A100']['cost_per_hour']
            h100_cost = hardware_specs['H100']['cost_per_hour']
            savings = int((1 - a100_cost/h100_cost) * 100)
            messages.append((
                "💸", "#f5576c",
                f"H100 OVERKILL! BLIS shows A100 achieves {hardware_specs['A100']['ttft_mean']:.0f}ms TTFT. Save {savings}% with A100!",
                "error"
            ))
        elif use_case in complex_use_cases:
            messages.append((
                "✅", "#10b981",
                f"H100 ✓ Good choice! BLIS: {blis_tokens_sec:.0f} tokens/sec, {blis_ttft:.0f}ms TTFT",
                "success"
            ))
        else:
            messages.append((
                "📊", "#6366f1",
                f"H100 BLIS benchmarks: {blis_tokens_sec:.0f} tokens/sec, {blis_ttft:.0f}ms avg TTFT",
                "info"
            ))
    
    if hardware == "A100":
        if use_case in simple_use_cases and ttft > 200 and qps < 50:
            messages.append((
                "💡", "#fbbf24",
                f"A100 may be overkill. BLIS shows {blis_ttft:.0f}ms TTFT - consider smaller GPU for {use_case.replace('_', ' ')}.",
                "warning"
            ))
        elif use_case in complex_use_cases and qps > 100:
            h100_tokens = hardware_specs['H100']['tokens_per_sec']
            messages.append((
                "⚡", "#3b82f6",
                f"High QPS ({qps})! H100 offers {h100_tokens:.0f} tokens/sec vs A100's {blis_tokens_sec:.0f}.",
                "info"
            ))
        else:
            messages.append((
                "📊", "#6366f1",
                f"A100 BLIS benchmarks: {blis_tokens_sec:.0f} tokens/sec, {blis_ttft:.0f}ms avg TTFT",
                "info"
            ))
    
    if hardware == "L40S":
        if use_case in complex_use_cases:
            messages.append((
                "⚠️", "#f5576c",
                f"L40S may struggle with {use_case.replace('_', ' ')}. BLIS shows A100/H100 needed for 10K+ context.",
                "warning"
            ))
        elif use_case in simple_use_cases:
            messages.append((
                "✅", "#10b981",
                f"L40S ✓ Cost-efficient for {use_case.replace('_', ' ')}!",
                "success"
            ))
    
    return messages

def get_workload_insights(use_case: str, qps: int, user_count: int) -> list:
    """Get workload pattern insights based on research data and BLIS benchmarks.
    
    Returns list of tuples: (icon, color, message, severity)
    """
    messages = []
    workload_data = load_research_workload_patterns()
    
    if not workload_data or 'workload_distributions' not in workload_data:
        return messages
    
    workload_patterns = workload_data.get('workload_distributions', {})
    traffic_profiles = workload_data.get('traffic_profiles', {})
    hardware_throughput = workload_data.get('hardware_throughput', {})
    capacity_guidance = workload_data.get('capacity_planning', {}).get('blis_guidance', {})
    
    # Get use case specific pattern
    pattern = workload_patterns.get(use_case)
    traffic = traffic_profiles.get(use_case)
    
    if pattern:
        distribution = pattern.get('distribution', 'poisson')
        active_fraction = pattern.get('active_fraction', {}).get('mean', 0.2)
        peak_multiplier = pattern.get('peak_multiplier', 2.0)
        req_per_min = pattern.get('requests_per_active_user_per_min', {}).get('mean', 0.5)
        
        # Get BLIS benchmark for this use case
        blis_bench = pattern.get('blis_benchmark', {})
        blis_optimal_rps = blis_bench.get('optimal_rps', 1.0)
        blis_max_rps = blis_bench.get('max_rps_tested', 10)
        blis_e2e_p95 = blis_bench.get('e2e_p95_at_optimal', 5000)
        
        # Calculate expected metrics
        expected_concurrent = int(user_count * active_fraction)
        expected_rps = (expected_concurrent * req_per_min) / 60
        expected_peak_rps = expected_rps * peak_multiplier
        
        messages.append((
            "📊", "#8b5cf6",
            f"Pattern: {distribution.replace('_', ' ').title()} | {int(active_fraction*100)}% concurrent users",
            "info"
        ))
        
        # Add BLIS E2E latency at optimal load
        if blis_e2e_p95:
            messages.append((
                "⏱️", "#06b6d4",
                f"BLIS E2E p95 at {blis_optimal_rps} RPS: {blis_e2e_p95}ms",
                "info"
            ))
    
    if traffic:
        prompt_tokens = traffic.get('prompt_tokens', 512)
        output_tokens = traffic.get('output_tokens', 256)
        blis_samples = traffic.get('blis_samples', 0)
        sample_info = f" ({blis_samples} BLIS samples)" if blis_samples else ""
        messages.append((
            "📝", "#3b82f6",
            f"Traffic: {prompt_tokens} → {output_tokens} tokens{sample_info}",
            "info"
        ))
    
    # Add hardware recommendation from BLIS
    if hardware_throughput and capacity_guidance:
        h100_max = capacity_guidance.get('H100_x1_max_rps', 10)
        if qps > h100_max:
            messages.append((
                "🔧", "#f97316",
                f"QPS {qps} > H100x1 max ({h100_max}). Recommend H100x2 or horizontal scaling.",
                "info"
            ))
    
    return messages

@st.cache_data
def load_weighted_scores(use_case: str) -> pd.DataFrame:
    """Load use-case-specific weighted scores from backend API."""
    try:
        # Fetch weighted scores from backend API
        response = requests.get(
            f"{API_BASE_URL}/api/v1/weighted-scores/{use_case}",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # Convert JSON to DataFrame
        if data.get("success") and data.get("scores"):
            df = pd.DataFrame(data["scores"])
            return df
        else:
            logger.warning(f"No weighted scores returned from API for use case: {use_case}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load weighted scores from API for {use_case}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_model_pricing() -> pd.DataFrame:
    """Load model pricing and latency data from model_pricing.csv.
    
    This provides REAL data for:
    - Cost scoring: price_blended ($/1M tokens)
    - Latency scoring: median_output_tokens_per_sec, median_ttft_seconds
    """
    csv_path = DATA_DIR / "benchmarks" / "models" / "model_pricing.csv"
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['model_name'])
        df = df[df['model_name'].str.strip() != '']
        return df
    except Exception:
        return pd.DataFrame()

# =============================================================================
# API FUNCTIONS (with Mock fallback for demo)
# =============================================================================

from typing import Optional, Dict, List

def extract_business_context(user_input: str) -> Optional[dict]:
    """Extract business context using Qwen 2.5 7B."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/extract",
            json={"text": user_input},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    
    # Mock response for demo
    return mock_extraction(user_input)


def mock_extraction(user_input: str) -> dict:
    """Mock extraction for demo purposes - supports all 9 use cases with robust error handling."""
    import re
    
    # Input validation - handle edge cases
    if not user_input or not isinstance(user_input, str):
        return {
            "use_case": "chatbot_conversational",
            "user_count": 100,
            "hardware": None,
            "priority": "balanced",
        }
    
    # Clean and normalize input
    text_lower = user_input.lower().strip()
    
    # USE CASE DETECTION - Order matters! More specific patterns first
    use_case = "chatbot_conversational"  # default
    
    # 1. Translation - check first (very specific)
    if any(kw in text_lower for kw in ["translat", "language pair", "multilingual", "locali"]):
        use_case = "translation"
    
    # 2. Long Document Summarization - check before short summarization
    elif any(kw in text_lower for kw in ["long summar", "long document", "book summar", "report summar", 
                                          "chapter", "extensive summar", "lengthy", "50+ page", "research paper"]):
        use_case = "long_document_summarization"
    
    # 3. Short Summarization - check BEFORE content generation (article summarization != content generation)
    elif any(kw in text_lower for kw in ["summar", "tldr", "brief", "condense", "digest", "news summar"]):
        use_case = "summarization_short"
    
    # 4. Content Generation - check before code
    elif any(kw in text_lower for kw in ["content generat", "content creation", "creative writ", "marketing content", 
                                          "blog post", "copywriting", "content tool"]):
        use_case = "content_generation"
    
    # 5. Code Generation (detailed) - check before code completion
    elif any(kw in text_lower for kw in ["code generat", "full code", "implement", "build software", 
                                          "create application", "write program"]):
        use_case = "code_generation_detailed"
    
    # 6. Code Completion - IDE autocomplete
    elif any(kw in text_lower for kw in ["code complet", "autocomplete", "code assist", "ide", "copilot",
                                          "code suggestion", "developer tool"]):
        use_case = "code_completion"
    
    # 7. Research/Legal Analysis
    elif any(kw in text_lower for kw in ["legal", "research", "analys", "contract", "compliance", 
                                          "academic", "scientific", "review paper"]):
        use_case = "research_legal_analysis"
    
    # 8. Document RAG
    elif any(kw in text_lower for kw in ["rag", "retriev", "knowledge base", "document q&a", 
                                          "document search", "enterprise search"]):
        use_case = "document_analysis_rag"
    
    # 9. Chatbot Conversational (default for chat-related)
    elif any(kw in text_lower for kw in ["chatbot", "chat", "customer support", "virtual assist", 
                                          "conversation", "support bot", "help desk"]):
        use_case = "chatbot_conversational"
    
    # Detect user count - look for numbers followed by user-related words
    # Avoid matching "50+ pages", "7B params", "80GB", etc.
    user_patterns = [
        r'(\d+[,.]?\d*)\s*k?\s*(users|developers|researchers|engineers|team members|people|employees|customers|daily users)',
        r'(\d+[,.]?\d*)\s*(k|thousand)\s*(users|developers|researchers|people)?',
        r'for\s+(\d+[,.]?\d*)\s*(k)?\s*(users|developers|researchers|people)?',
        r'(\d+[,.]?\d*)\s+team',
    ]
    
    user_count = 1000  # default
    for pattern in user_patterns:
        user_match = re.search(pattern, text_lower)
        if user_match:
            num_str = user_match.group(1).replace(',', '')
            num = float(num_str)
            # Check if 'k' or 'thousand' is in the match
            full_match = user_match.group(0).lower()
            if 'k ' in full_match or 'k\t' in full_match or full_match.endswith('k') or 'thousand' in full_match:
                num *= 1000
            user_count = int(num)
            break
    
    # Detect hardware
    hardware = None
    if "h100" in text_lower:
        hardware = "H100"
    elif "a100" in text_lower:
        hardware = "A100"
    elif "l40" in text_lower:
        hardware = "L40S"
    
    # Detect priority from user input
    priority = "balanced"  # default
    latency_keywords = ["latency", "fast", "speed", "quick", "responsive", "real-time", "instant", "low latency", "critical"]
    cost_keywords = ["cost", "cheap", "budget", "efficient", "affordable", "save money", "cost-effective"]
    quality_keywords = ["quality", "accurate", "best", "precision", "top quality", "high quality", "most important"]
    throughput_keywords = ["throughput", "scale", "high volume", "capacity", "concurrent", "many users"]
    
    # Check for latency priority
    if any(kw in text_lower for kw in latency_keywords):
        priority = "low_latency"
    # Check for cost priority
    elif any(kw in text_lower for kw in cost_keywords):
        priority = "cost_saving"
    # Check for quality priority
    elif any(kw in text_lower for kw in quality_keywords):
        priority = "high_quality"
    # Check for throughput priority
    elif any(kw in text_lower for kw in throughput_keywords):
        priority = "high_throughput"
    
    return {
        "use_case": use_case,
        "user_count": user_count,
        "hardware": hardware,
        "priority": priority,
    }


def get_enhanced_recommendation(business_context: dict) -> Optional[dict]:
    """Get enhanced recommendation with explainability."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v2/recommend",
            json={
                "business_context": business_context,
                "include_explanation": True,
                "top_k": 5,
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    
    # Use BLIS-based recommendation with ACTUAL data
    return blis_recommendation(business_context)


# =============================================================================
# BLIS MODEL NAME MAPPING
# Maps BLIS repo names to our quality CSV model names
# =============================================================================
BLIS_TO_QUALITY_MODEL_MAP = {
    'ibm-granite/granite-3.1-8b-instruct': 'Granite 3.3 8B (Non-reasoning)',
    'meta-llama/llama-3.1-8b-instruct': 'Llama 3.1 8B Instruct',
    'meta-llama/llama-3.3-70b-instruct': 'Llama 3.3 70B Instruct',
    'microsoft/phi-4': 'Phi-4',
    'mistralai/mistral-small-24b-instruct-2501': 'Mistral Small 3.1',
    'mistralai/mistral-small-3.1-24b-instruct-2503': 'Mistral Small 3.2',
    'mistralai/mixtral-8x7b-instruct-v0.1': 'Mixtral 8x7B Instruct',
    'openai/gpt-oss-120b': 'gpt-oss-120B (high)',
    'openai/gpt-oss-20b': 'gpt-oss-20B (high)',
    'qwen/qwen2.5-7b-instruct': 'Qwen 2.5 7B Instruct',
}

# Hardware costs (monthly) - BOTH H100 and A100-80 are real BLIS data
HARDWARE_COSTS = {
    ('H100', 1): 2500,
    ('H100', 2): 5000,
    ('H100', 4): 10000,
    ('H100', 8): 20000,
    ('A100-80', 1): 1600,
    ('A100-80', 2): 3200,
    ('A100-80', 4): 6400,
}


def blis_recommendation(context: dict) -> dict:
    """BLIS-based recommendation using ACTUAL benchmark data.
    
    NEW ARCHITECTURE:
    - Model quality: from weighted_scores CSVs (use-case specific)
    - Latency/throughput: from ACTUAL BLIS benchmarks (model+hardware specific)
    - Cost: from hardware tier (cheaper hardware = higher cost score)
    
    Creates MODEL+HARDWARE combinations ranked by priority:
    - cost_saving: cheapest hardware that meets SLO for best models
    - low_latency: fastest hardware (lowest TTFT) for best models
    - high_quality: best model quality with hardware that meets SLO
    - balanced: weighted combination of all factors
    """
    use_case = context.get("use_case", "chatbot_conversational")
    priority = context.get("priority", "balanced")
    user_count = context.get("user_count", 1000)
    
    # Load BLIS benchmark data
    blis_data = load_blis_benchmarks()
    if not blis_data or 'benchmarks' not in blis_data:
        return mock_recommendation_fallback(context)
    
    benchmarks = blis_data['benchmarks']
    
    # Load quality scores for this use case
    weighted_df = load_weighted_scores(use_case)
    quality_lookup = {}
    if not weighted_df.empty:
        for _, row in weighted_df.iterrows():
            model_name = row.get('Model Name', '')
            score_str = row.get('Use Case Score', '0')
            if isinstance(score_str, str):
                score = float(score_str.replace('%', '')) if '%' in score_str else float(score_str) * 100
            else:
                score = float(score_str) * 100 if score_str <= 1 else float(score_str)
            quality_lookup[model_name] = score
    
    # Get SLO targets for this use case
    slo_data = get_slo_targets_for_use_case(use_case, priority)
    ttft_max = slo_data.get('ttft_target', {}).get('max', 200)
    e2e_max = slo_data.get('e2e_target', {}).get('max', 5000)
    
    # Priority weights for MCDM
    weights = {
        "balanced": {"quality": 0.30, "latency": 0.30, "cost": 0.25, "throughput": 0.15},
        "low_latency": {"quality": 0.15, "latency": 0.50, "cost": 0.15, "throughput": 0.20},
        "cost_saving": {"quality": 0.20, "latency": 0.15, "cost": 0.50, "throughput": 0.15},
        "high_quality": {"quality": 0.50, "latency": 0.20, "cost": 0.15, "throughput": 0.15},
        "high_throughput": {"quality": 0.15, "latency": 0.15, "cost": 0.15, "throughput": 0.55},
    }[priority]
    
    # Aggregate BLIS data by model+hardware (use best config per combo)
    model_hw_combos = {}
    for b in benchmarks:
        model_repo = b['model_hf_repo']
        hw = b['hardware']
        hw_count = b['hardware_count']
        key = (model_repo, hw, hw_count)
        
        # Keep the benchmark with lowest TTFT for each combo
        if key not in model_hw_combos or b['ttft_p95'] < model_hw_combos[key]['ttft_p95']:
            model_hw_combos[key] = {
                'model_repo': model_repo,
                'model_name': model_repo.split('/')[-1],
                'hardware': hw,
                'hardware_count': hw_count,
                'ttft_mean': b['ttft_mean'],
                'ttft_p95': b['ttft_p95'],
                'itl_mean': b['itl_mean'],
                'itl_p95': b['itl_p95'],
                'e2e_mean': b['e2e_mean'],
                'e2e_p95': b['e2e_p95'],
                'tokens_per_second': b['tokens_per_second'],
                'prompt_tokens': b['prompt_tokens'],
                'output_tokens': b['output_tokens'],
            }
    
    # Calculate scores for each model+hardware combo
    scored_combos = []
    
    # Find max values for normalization
    max_ttft = max(c['ttft_p95'] for c in model_hw_combos.values()) or 1
    max_tps = max(c['tokens_per_second'] for c in model_hw_combos.values()) or 1
    max_cost = max(HARDWARE_COSTS.values()) or 1
    
    for key, combo in model_hw_combos.items():
        # Get quality score from CSV (mapped model name)
        quality_model = BLIS_TO_QUALITY_MODEL_MAP.get(combo['model_repo'], combo['model_name'])
        quality_score = quality_lookup.get(quality_model, 30.0)  # Default 30 if not found
        
        # Latency score: lower TTFT = higher score (inverted, normalized 0-100)
        latency_score = 100 - (combo['ttft_p95'] / max_ttft * 100)
        latency_score = max(10, min(100, latency_score))
        
        # Throughput score: higher TPS = higher score
        throughput_score = (combo['tokens_per_second'] / max_tps) * 100
        throughput_score = max(10, min(100, throughput_score))
        
        # Cost score: lower hardware cost = higher score (inverted)
        hw_cost = HARDWARE_COSTS.get((combo['hardware'], combo['hardware_count']), 10000)
        cost_score = 100 - (hw_cost / max_cost * 80)  # Leave headroom
        cost_score = max(10, min(100, cost_score))
        
        # Check if meets SLO
        meets_slo = combo['ttft_p95'] <= ttft_max and combo['e2e_p95'] <= e2e_max
        
        # Calculate weighted MCDM score
        final_score = (
            weights['quality'] * quality_score +
            weights['latency'] * latency_score +
            weights['cost'] * cost_score +
            weights['throughput'] * throughput_score
        )
        
        # Bonus for meeting SLO
        if meets_slo:
            final_score += 5
        
        scored_combos.append({
            **combo,
            'quality_score': round(quality_score, 1),
            'latency_score': round(latency_score, 1),
            'cost_score': round(cost_score, 1),
            'throughput_score': round(throughput_score, 1),
            'final_score': round(final_score, 2),
            'meets_slo': meets_slo,
            'hw_cost_monthly': hw_cost,
            'quality_model_name': quality_model,
        })
    
    # Sort by final score (highest first)
    scored_combos.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Get top recommendation
    if not scored_combos:
        return mock_recommendation_fallback(context)
    
    top = scored_combos[0]
    
    # Build recommendation response
    return {
        "model_name": top['model_name'].replace('-', ' ').title(),
        "model_hf_repo": top['model_repo'],
        "score": top['final_score'],
        "intent": {
            "use_case": use_case,
            "priority": priority,
            "user_count": user_count,
        },
        "gpu_config": {
            "gpu_type": top['hardware'],
            "gpu_count": top['hardware_count'],
        },
        "meets_slo": top['meets_slo'],
        "slo_targets": slo_data,
        "hardware_recommendation": {
            "recommended": {
                "hardware": top['hardware'],
                "hardware_count": top['hardware_count'],
                "ttft_p95": top['ttft_p95'],
                "itl_mean": top['itl_mean'],
                "e2e_p95": top['e2e_p95'],
                "tokens_per_second": top['tokens_per_second'],
                "cost_monthly": top['hw_cost_monthly'],
            },
            "selection_reason": get_selection_reason(top, priority),
            "alternatives": [
                {
                    "model": c['model_name'],
                    "hardware": c['hardware'],
                    "hardware_count": c['hardware_count'],
                    "ttft_p95": c['ttft_p95'],
                    "e2e_p95": c['e2e_p95'],
                    "cost_monthly": c['hw_cost_monthly'],
                    "score": c['final_score'],
                    "meets_slo": c['meets_slo'],
                }
                for c in scored_combos[1:6]  # Top 5 alternatives
            ],
        },
        "score_breakdown": {
            "quality": {"score": top['quality_score'], "weight": weights['quality']},
            "latency": {"score": top['latency_score'], "weight": weights['latency']},
            "cost": {"score": top['cost_score'], "weight": weights['cost']},
            "throughput": {"score": top['throughput_score'], "weight": weights['throughput']},
        },
        "blis_actual": {
            "ttft_mean": top['ttft_mean'],
            "ttft_p95": top['ttft_p95'],
            "itl_mean": top['itl_mean'],
            "itl_p95": top['itl_p95'],
            "e2e_mean": top['e2e_mean'],
            "e2e_p95": top['e2e_p95'],
            "tokens_per_second": top['tokens_per_second'],
            "prompt_tokens": top['prompt_tokens'],
            "output_tokens": top['output_tokens'],
        },
        "recommendations": [
            {
                "rank": i + 1,
                "model_name": f"{c['model_name'].replace('-', ' ').title()} on {c['hardware']} x{c['hardware_count']}",
                "model_id": c['model_repo'],
                "hardware": c['hardware'],
                "hardware_count": c['hardware_count'],
                "final_score": c['final_score'],
                "score_breakdown": {
                    "quality_score": c['quality_score'],
                    "latency_score": c['latency_score'],
                    "cost_score": c['cost_score'],
                    "capacity_score": c['throughput_score'],
                    "quality_contribution": round(c['quality_score'] * weights['quality'] / 100 * c['final_score'], 1),
                    "latency_contribution": round(c['latency_score'] * weights['latency'] / 100 * c['final_score'], 1),
                    "cost_contribution": round(c['cost_score'] * weights['cost'] / 100 * c['final_score'], 1),
                    "capacity_contribution": round(c['throughput_score'] * weights['throughput'] / 100 * c['final_score'], 1),
                },
                "blis_metrics": {
                    "ttft_p95_ms": c['ttft_p95'],
                    "e2e_p95_ms": c['e2e_p95'],
                    "tokens_per_second": c['tokens_per_second'],
                },
                "cost_monthly": c['hw_cost_monthly'],
                "meets_slo": c['meets_slo'],
                "pros": get_model_pros(c, priority),
                "cons": get_model_cons(c, priority),
            }
            for i, c in enumerate(scored_combos[:10])
        ],
    }


def get_selection_reason(top: dict, priority: str) -> str:
    """Generate human-readable selection reason."""
    model = top['model_name'].replace('-', ' ').title()
    hw = f"{top['hardware']} x{top['hardware_count']}"
    ttft = top['ttft_p95']
    cost = top['hw_cost_monthly']
    tps = top['tokens_per_second']
    
    if priority == "cost_saving":
        return f"💰 {model} on {hw} is the most cost-effective option (${cost:,}/mo) that meets your SLO requirements with {ttft:.0f}ms TTFT."
    elif priority == "low_latency":
        return f"⚡ {model} on {hw} delivers the lowest latency ({ttft:.0f}ms TTFT P95) from actual BLIS benchmarks."
    elif priority == "high_quality":
        return f"⭐ {model} has the highest quality score for your use case, running on {hw} with {ttft:.0f}ms TTFT."
    elif priority == "high_throughput":
        return f"📈 {model} on {hw} achieves {tps:.0f} tokens/sec throughput from actual BLIS benchmarks."
    else:  # balanced
        return f"⚖️ {model} on {hw} provides optimal balance: {ttft:.0f}ms TTFT, {tps:.0f} tokens/sec, ${cost:,}/mo."


def get_model_pros(combo: dict, priority: str) -> list:
    """Generate pros based on ACTUAL BLIS metrics."""
    pros = []
    ttft = combo['ttft_p95']
    tps = combo['tokens_per_second']
    cost = combo['hw_cost_monthly']
    quality = combo['quality_score']
    
    if ttft < 50:
        pros.append(f"⚡ Ultra-fast TTFT ({ttft:.0f}ms)")
    elif ttft < 100:
        pros.append(f"⚡ Fast TTFT ({ttft:.0f}ms)")
    
    if tps > 400:
        pros.append(f"🚀 High throughput ({tps:.0f} tok/s)")
    elif tps > 200:
        pros.append(f"📈 Good throughput ({tps:.0f} tok/s)")
    
    if cost < 3000:
        pros.append(f"💰 Cost-efficient (${cost:,}/mo)")
    
    if quality > 50:
        pros.append(f"⭐ High quality ({quality:.0f}%)")
    
    if combo['meets_slo']:
        pros.append("✅ Meets SLO targets")
    
    return pros[:4] if pros else ["📊 BLIS benchmarked"]


def get_model_cons(combo: dict, priority: str) -> list:
    """Generate cons based on ACTUAL BLIS metrics."""
    cons = []
    ttft = combo['ttft_p95']
    tps = combo['tokens_per_second']
    cost = combo['hw_cost_monthly']
    quality = combo['quality_score']
    
    if ttft > 200:
        cons.append(f"⏱️ Higher latency ({ttft:.0f}ms)")
    
    if tps < 100:
        cons.append(f"📉 Lower throughput ({tps:.0f} tok/s)")
    
    if cost > 10000:
        cons.append(f"💸 Premium cost (${cost:,}/mo)")
    
    if quality < 40:
        cons.append(f"📊 Lower quality score ({quality:.0f}%)")
    
    if not combo['meets_slo']:
        cons.append("⚠️ May not meet SLO")
    
    return cons[:2]


def mock_recommendation_fallback(context: dict) -> dict:
    """Fallback recommendation when BLIS data unavailable."""
    return mock_recommendation(context)


def mock_recommendation(context: dict) -> dict:
    """FALLBACK: Recommendation using CSV data when BLIS unavailable.
    
    Data sources:
    - Quality: weighted_scores/{use_case}.csv (task-specific benchmark scores)
    - Cost: model_pricing.csv (price_blended - $/1M tokens)
    - Latency: model_pricing.csv (median_output_tokens_per_sec, median_ttft_seconds)
    
    Enterprise-grade error handling with graceful fallbacks.
    """
    # Validate context input
    if not context or not isinstance(context, dict):
        context = {}
    
    use_case = context.get("use_case", "chatbot_conversational")
    priority = context.get("priority", "balanced")
    
    # Validate use_case is in allowed list
    valid_use_cases = [
        "chatbot_conversational", "code_completion", "code_generation_detailed",
        "document_analysis_rag", "summarization_short", "long_document_summarization",
        "translation", "content_generation", "research_legal_analysis"
    ]
    if use_case not in valid_use_cases:
        use_case = "chatbot_conversational"
    
    # Validate priority is in allowed list
    valid_priorities = ["balanced", "low_latency", "cost_saving", "high_quality", "high_throughput"]
    if priority not in valid_priorities:
        priority = "balanced"
    
    # Load use-case-specific weighted scores (the QUALITY component)
    weighted_df = load_weighted_scores(use_case)
    # Also load 206-model benchmark for validation
    all_models_df = load_206_models()
    # Load REAL pricing and latency data
    pricing_df = load_model_pricing()
    
    # Create pricing lookup dict (model_name -> pricing data)
    pricing_lookup = {}
    if not pricing_df.empty:
        for _, row in pricing_df.iterrows():
            model_name = row.get('model_name', '')
            if model_name:
                pricing_lookup[model_name] = {
                    'price_blended': row.get('price_blended', 0),
                    'price_input': row.get('price_per_1m_input_tokens', 0),
                    'price_output': row.get('price_per_1m_output_tokens', 0),
                    'tokens_per_sec': row.get('median_output_tokens_per_sec', 0),
                    'ttft_seconds': row.get('median_ttft_seconds', 0),
                }
    
    # Calculate normalization ranges for scoring
    # Cost: Lower is better (0 = most expensive, 100 = cheapest/free)
    max_price = max((p['price_blended'] for p in pricing_lookup.values() if p['price_blended'] > 0), default=10)
    # Latency: Higher tokens/sec is better (faster = higher score)
    max_tokens_sec = max((p['tokens_per_sec'] for p in pricing_lookup.values() if p['tokens_per_sec'] > 0), default=500)
    
    # Priority-based weights for MCDM scoring
    weights = {
        "balanced": {"quality": 0.30, "latency": 0.25, "cost": 0.25, "capacity": 0.20},
        "low_latency": {"quality": 0.20, "latency": 0.45, "cost": 0.15, "capacity": 0.20},
        "cost_saving": {"quality": 0.20, "latency": 0.15, "cost": 0.50, "capacity": 0.15},
        "high_quality": {"quality": 0.50, "latency": 0.20, "cost": 0.15, "capacity": 0.15},
        "high_throughput": {"quality": 0.20, "latency": 0.15, "cost": 0.15, "capacity": 0.50},
    }[priority]
    
    # Parse use case score from weighted_scores CSV
    def parse_score(x):
        if pd.isna(x) or x == 'N/A':
            return 0
        if isinstance(x, str):
            return float(x.replace('%', '')) if '%' in x else float(x) * 100
        return float(x) * 100 if x <= 1 else float(x)
    
    def calculate_cost_score(model_name: str) -> float:
        """Calculate cost score from REAL pricing data.
        Lower price = higher score (100 = free, 0 = most expensive)
        """
        pricing = pricing_lookup.get(model_name, {})
        price = pricing.get('price_blended', 0)
        
        if price <= 0:
            # Free model = perfect cost score
            return 95
        
        # Normalize: lower price = higher score
        # Score = 100 - (price / max_price * 100)
        cost_score = 100 - (price / max_price * 80)  # Cap at 80% reduction
        return max(min(cost_score, 100), 20)  # Range: 20-100
    
    def calculate_latency_score(model_name: str) -> float:
        """Calculate latency score from REAL speed data.
        Higher tokens/sec = higher score
        """
        import math
        pricing = pricing_lookup.get(model_name, {})
        tokens_sec = pricing.get('tokens_per_sec', 0) or 0
        ttft = pricing.get('ttft_seconds', 0) or 0
        
        # Handle NaN values
        if isinstance(tokens_sec, float) and math.isnan(tokens_sec):
            tokens_sec = 0
        if isinstance(ttft, float) and math.isnan(ttft):
            ttft = 0
        
        if tokens_sec <= 0:
            # No data - estimate based on model size
            model_lower = model_name.lower()
            is_large = any(s in model_lower for s in ["405b", "70b", "72b", "235b", "120b", "80b"])
            is_medium = any(s in model_lower for s in ["32b", "27b", "22b", "14b", "49b", "20b"])
            is_small = any(s in model_lower for s in ["7b", "8b", "3b", "4b", "1.7b", "1b"])
            if is_large:
                return 45.0
            elif is_medium:
                return 60.0
            elif is_small:
                return 80.0
            return 55.0  # Default for unknown size
        
        # Normalize: higher tokens/sec = higher score
        safe_max = max_tokens_sec if max_tokens_sec > 0 else 500
        latency_score = (tokens_sec / safe_max) * 100
        
        # Bonus for low TTFT (< 0.5s = +10, < 1s = +5)
        if 0 < ttft < 0.5:
            latency_score += 10
        elif 0 < ttft < 1.0:
            latency_score += 5
        
        return float(max(min(latency_score, 100), 20))
    
    models = []
    
    # Use weighted_scores CSV for quality (already ranked by use case)
    if not weighted_df.empty:
        # Get valid model names from the 206-model benchmark
        valid_models = set(all_models_df['Model Name'].dropna().tolist()) if not all_models_df.empty else set()
        
        # Filter weighted_scores to only include models in the 206 benchmark
        weighted_df = weighted_df[weighted_df['Model Name'].isin(valid_models)] if valid_models else weighted_df
        
        # Get top 10 models from weighted scores (already sorted by use case quality)
        top_models = weighted_df.head(10)
        
        for _, row in top_models.iterrows():
            model_name = row.get("Model Name", "")
            provider = row.get("Provider", "Unknown")
            
            # Get QUALITY from USE-CASE-SPECIFIC weighted score!
            quality_score = parse_score(row.get('Use Case Score', 0))
            
            # Skip models with no quality score
            if quality_score == 0:
                continue
            
            # Get REAL cost and latency scores from model_pricing.csv
            cost_score = calculate_cost_score(model_name)
            latency_score = calculate_latency_score(model_name)
            
            # Capacity score based on throughput and model architecture
            model_lower = model_name.lower()
            is_moe = ("a" in model_lower and "b" in model_lower) or "moe" in model_lower or "mixture" in model_lower
            is_small = any(s in model_lower for s in ["7b", "8b", "3b", "4b", "1.7b", "1b"])
            
            # Ensure latency_score is a valid number
            safe_latency = latency_score if latency_score and latency_score > 0 else 55.0
            
            if is_moe:
                capacity_score = 80 + (safe_latency / 10)  # MoE = high capacity
            elif is_small:
                capacity_score = 75 + (safe_latency / 8)
            else:
                capacity_score = 50 + (safe_latency / 5)
            
            # Ensure all scores are valid floats
            import math
            quality_score = float(quality_score) if quality_score and not (isinstance(quality_score, float) and math.isnan(quality_score)) else 50.0
            latency_score = float(safe_latency)
            cost_score = float(cost_score) if cost_score and not (isinstance(cost_score, float) and math.isnan(cost_score)) else 50.0
            capacity_score = float(capacity_score) if capacity_score and not (isinstance(capacity_score, float) and math.isnan(capacity_score)) else 60.0
            
            models.append({
                "name": model_name,
                "provider": provider,
                "quality": min(max(quality_score, 0), 100),
                "latency": min(max(latency_score, 0), 100),
                "cost": min(max(cost_score, 0), 100),
                "capacity": min(max(capacity_score, 0), 100),
            })
    
    # Fallback to hardcoded models if CSV fails (using exact names from CSV)
    if not models:
        models = [
            {"name": "Kimi K2 Thinking", "provider": "Moonshot AI", "quality": 94, "latency": 62, "cost": 52, "capacity": 65},
            {"name": "MiniMax-M2", "provider": "MiniMax", "quality": 92, "latency": 68, "cost": 58, "capacity": 72},
            {"name": "DeepSeek V3 (Dec '24)", "provider": "DeepSeek", "quality": 95, "latency": 55, "cost": 45, "capacity": 58},
            {"name": "Qwen2.5 Instruct 72B", "provider": "Alibaba", "quality": 90, "latency": 58, "cost": 48, "capacity": 62},
            {"name": "Llama 3.1 Instruct 70B", "provider": "Meta", "quality": 88, "latency": 60, "cost": 50, "capacity": 65},
        ]
    
    # Calculate final MCDM score (ensure no NaN values)
    import math
    for m in models:
        # Ensure all component scores are valid numbers
        quality = m["quality"] if m["quality"] and not math.isnan(m["quality"]) else 50.0
        latency = m["latency"] if m["latency"] and not math.isnan(m["latency"]) else 50.0
        cost = m["cost"] if m["cost"] and not math.isnan(m["cost"]) else 50.0
        capacity = m["capacity"] if m["capacity"] and not math.isnan(m["capacity"]) else 50.0
        
        m["final_score"] = (
            quality * weights["quality"] +
            latency * weights["latency"] +
            cost * weights["cost"] +
            capacity * weights["capacity"]
        )
        
        # Ensure final_score is valid
        if math.isnan(m["final_score"]):
            m["final_score"] = 50.0
    
    # Sort by score
    models.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Get hardware from context
    user_hardware = context.get("hardware", None)
    
    # Get optimal hardware recommendation based on priority and SLO requirements
    hw_recommendation = recommend_optimal_hardware(use_case, priority, user_hardware)
    
    # Use recommended hardware or user-specified
    if user_hardware:
        hardware = user_hardware
    elif hw_recommendation and hw_recommendation.get('recommended'):
        hardware = hw_recommendation['recommended']['hardware']
    else:
        hardware = "H100"
    
    # Build recommendations (top 10 to support filtering)
    recommendations = []
    for m in models[:10]:
        pros = []
        cons = []
        
        if m["quality"] >= 90:
            pros.append("⭐ Top Quality")
        elif m["quality"] >= 80:
            pros.append("✅ Good Quality")
        if m["latency"] >= 85:
            pros.append("⚡ Ultra Fast")
        elif m["latency"] >= 75:
            pros.append("🚀 Fast")
        if m["cost"] >= 80:
            pros.append("💰 Cost-Efficient")
        if m["capacity"] >= 85:
            pros.append("📈 High Capacity")
        
        if m["quality"] < 75:
            cons.append("📉 Lower Quality")
        if m["latency"] < 55:
            cons.append("🐢 Slower")
        if m["cost"] < 45:
            cons.append("💸 Expensive")
        if m["capacity"] < 50:
            cons.append("📊 Limited Capacity")
        
        # Get REAL BLIS benchmark SLO data for this model
        blis_slo = get_blis_slo_for_model(m["name"], use_case, hardware)
        
        recommendation = {
            "model_name": m["name"],
            "provider": m["provider"],
            "final_score": m["final_score"],
            "score_breakdown": {
                "quality_score": m["quality"],
                "latency_score": m["latency"],
                "cost_score": m["cost"],
                "capacity_score": m["capacity"],
                "quality_contribution": m["quality"] * weights["quality"],
                "latency_contribution": m["latency"] * weights["latency"],
                "cost_contribution": m["cost"] * weights["cost"],
                "capacity_contribution": m["capacity"] * weights["capacity"],
            },
            "pros": pros if pros else ["✅ Balanced Performance"],
            "cons": cons if cons else ["⚖️ No significant weaknesses"],
        }
        
        # Add BLIS SLO data if available
        if blis_slo:
            recommendation["blis_slo"] = blis_slo
        
        recommendations.append(recommendation)
    
    # Build response with hardware recommendation
    response = {"recommendations": recommendations}
    
    # Add hardware recommendation details
    if hw_recommendation:
        response["hardware_recommendation"] = hw_recommendation
        response["slo_targets"] = hw_recommendation.get("slo_targets")
    
    return response

# =============================================================================
# VISUAL COMPONENTS
# =============================================================================

def render_hero():
    """Render the animated hero section with project description."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-emoji">🧭</div>
        <div class="hero-title">Compass</div>
        <div class="hero-subtitle">AI-Powered LLM Deployment Recommendations — From Natural Language to Production in Seconds</div>
        <div class="hero-badges">
            <span class="hero-badge">📦 206 Models</span>
            <span class="hero-badge">🎯 95.1% Accuracy</span>
            <span class="hero-badge">⚖️ MCDM Scoring</span>
            <span class="hero-badge">📊 15 Benchmarks</span>
            <span class="hero-badge">🎪 9 Use Cases</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Short project description - clean, readable like Qualifire
    st.markdown("""
    <div style="background: var(--bg-card); 
                padding: 1.5rem 2rem; border-radius: 12px; margin: 1.5rem 0; 
                border: 1px solid var(--border-default);">
        <p style="color: var(--text-primary); margin: 0; font-size: 1.05rem; line-height: 1.8; text-align: center;">
            <strong style="color: var(--accent-blue);">Compass</strong> uses <strong style="color: var(--accent-green);">Qwen 2.5 7B</strong> to extract your business requirements from natural language, 
            then scores <strong style="color: var(--accent-purple);">206 open-source models</strong> using <strong>Multi-Criteria Decision Making (MCDM)</strong> 
            across Quality, Latency, Cost, and Capacity to recommend the best model for your deployment. 
            All data powered by <strong style="color: var(--accent-pink);">Artificial Analysis</strong> benchmarks.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_stats(models_count: int):
    """Render statistics cards with clean design."""
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-card" title="From Artificial Analysis benchmark database">
            <span class="stat-icon">📦</span>
            <div class="stat-value">{models_count}</div>
            <div class="stat-label">Open-Source Models</div>
        </div>
        <div class="stat-card" title="Qwen 2.5 7B on 600 test cases">
            <span class="stat-icon">🎯</span>
            <div class="stat-value">95.1%</div>
            <div class="stat-label">Extraction Accuracy</div>
        </div>
        <div class="stat-card" title="Quality + Latency + Cost + Capacity">
            <span class="stat-icon">⚖️</span>
            <div class="stat-value">4</div>
            <div class="stat-label">Scoring Criteria</div>
        </div>
        <div class="stat-card" title="MMLU-Pro, GPQA, IFBench, LiveCodeBench, AIME & more">
            <span class="stat-icon">📊</span>
            <div class="stat-value">15</div>
            <div class="stat-label">Benchmark Datasets</div>
        </div>
        <div class="stat-card" title="All 9 use cases supported">
            <span class="stat-icon">🎪</span>
            <div class="stat-value">9</div>
            <div class="stat-label">Use Cases</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # MCDM Formula Explanation Expander
    # MCDM Expander with clean styling
    st.markdown("""
    <style>
        /* Expander styling - clean design */
        [data-testid="stExpander"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-default) !important;
            border-radius: 12px !important;
        }
        [data-testid="stExpander"] summary {
            background: rgba(88, 166, 255, 0.08) !important;
            border-radius: 11px 11px 0 0 !important;
            padding: 1rem 1.25rem !important;
        }
        [data-testid="stExpander"] summary span {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }
        [data-testid="stExpander"] svg {
            color: var(--accent-blue) !important;
        }
        /* Force ALL text inside expander */
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h4,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] th,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] td,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] span,
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] strong {
            color: var(--text-primary) !important;
        }
        [data-testid="stExpander"] [data-testid="stMarkdownContainer"] code {
            background: rgba(88, 166, 255, 0.1) !important;
            color: var(--accent-blue) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    with st.expander("📊 **MCDM Scoring Formula** - How each component is calculated", expanded=False):
        st.markdown('<h4 style="color: var(--accent-purple) !important; margin-bottom: 1.25rem; font-family: Inter, sans-serif;">⚖️ Multi-Criteria Decision Making (MCDM)</h4>', unsafe_allow_html=True)
        st.code("FINAL_SCORE = w_quality × Quality + w_latency × Latency + w_cost × Cost + w_capacity × Capacity", language=None)
        
        st.markdown("""
<table style="width: 100%; border-collapse: collapse; margin-top: 1.5rem; background: transparent;">
<tr style="border-bottom: 2px solid rgba(88, 166, 255, 0.25); background: transparent;">
    <th style="text-align: left; padding: 1rem; color: var(--accent-purple) !important; font-weight: 700; width: 130px; background: transparent; font-size: 0.95rem;">Component</th>
    <th style="text-align: left; padding: 1rem; color: var(--accent-purple) !important; font-weight: 700; background: transparent; font-size: 0.95rem;">Formula & Explanation</th>
</tr>
<tr style="border-bottom: 1px solid var(--border-default); background: transparent;">
    <td style="padding: 1rem; color: var(--accent-green) !important; font-weight: 700; background: transparent; font-size: 1rem;">🎯 Quality</td>
    <td style="padding: 1rem; color: var(--text-primary) !important; background: transparent; line-height: 1.7;">
        <code style="background: rgba(63, 185, 80, 0.12); padding: 6px 10px; border-radius: 6px; color: var(--accent-green); font-size: 0.9rem;">Quality = UseCase_Score(model) × 100</code><br><br>
        <span style="color: var(--text-primary);"><strong style="color: var(--accent-green);">Use-case specific score</strong> from <code style="background: rgba(163, 113, 247, 0.12); color: var(--accent-purple); padding: 2px 6px; border-radius: 4px;">weighted_scores</code> CSVs. Each use case has pre-ranked models based on relevant benchmarks (e.g., LiveCodeBench for code, MMLU for chatbot). Score range: 0-100.</span>
    </td>
</tr>
<tr style="border-bottom: 1px solid var(--border-default); background: transparent;">
    <td style="padding: 1rem; color: var(--accent-blue) !important; font-weight: 700; background: transparent; font-size: 1rem;">⚡ Latency</td>
    <td style="padding: 1rem; color: var(--text-primary) !important; background: transparent; line-height: 1.7;">
        <code style="background: rgba(88, 166, 255, 0.12); padding: 6px 10px; border-radius: 6px; color: var(--accent-blue); font-size: 0.9rem;">Latency = (tokens_per_sec / max_tokens_sec) × 100 + TTFT_bonus</code><br><br>
        <strong style="color: var(--accent-blue);">📊 Data Source:</strong> <code style="background: rgba(163, 113, 247, 0.12); color: var(--accent-purple); padding: 2px 6px; border-radius: 4px;">model_pricing.csv</code><br><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-blue);">tokens_per_sec</strong> = median_output_tokens_per_sec (real benchmark data)</span><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-blue);">TTFT_bonus</strong> = +10 if TTFT < 0.5s, +5 if TTFT < 1.0s</span><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-green);">Fast models (200+ tokens/sec)</strong>: Score 80-100</span><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-purple);">Medium models (50-200 tokens/sec)</strong>: Score 40-80</span><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-orange);">Slow models (< 50 tokens/sec)</strong>: Score 20-40</span>
    </td>
</tr>
<tr style="border-bottom: 1px solid var(--border-default); background: transparent;">
    <td style="padding: 1rem; color: var(--accent-orange) !important; font-weight: 700; background: transparent; font-size: 1rem;">💰 Cost</td>
    <td style="padding: 1rem; color: var(--text-primary) !important; background: transparent; line-height: 1.7;">
        <code style="background: rgba(249, 115, 22, 0.12); padding: 6px 10px; border-radius: 6px; color: var(--accent-orange); font-size: 0.9rem;">Cost = 100 - (price_blended / max_price) × 80</code><br><br>
        <strong style="color: var(--accent-orange);">📊 Data Source:</strong> <code style="background: rgba(163, 113, 247, 0.12); color: var(--accent-purple); padding: 2px 6px; border-radius: 4px;">model_pricing.csv</code><br><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-orange);">price_blended</strong> = Real API cost per 1M tokens (USD)</span><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-green);">Free/Open-source models</strong>: Score 95 (self-hosted)</span><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-green);">Cheap models (< $0.5/1M)</strong>: Score 75-90</span><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-purple);">Medium cost ($0.5-2/1M)</strong>: Score 50-75</span><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-orange);">Expensive models (> $2/1M)</strong>: Score 20-50</span>
    </td>
</tr>
<tr style="background: transparent;">
    <td style="padding: 1rem; color: var(--accent-purple) !important; font-weight: 700; background: transparent; font-size: 1rem;">📈 Capacity</td>
    <td style="padding: 1rem; color: var(--text-primary) !important; background: transparent; line-height: 1.7;">
        <code style="background: rgba(163, 113, 247, 0.12); padding: 6px 10px; border-radius: 6px; color: var(--accent-purple); font-size: 0.9rem;">Capacity = base_throughput × efficiency_multiplier</code><br><br>
        <span style="color: var(--text-primary);"><strong style="color: var(--accent-purple);">Throughput potential</strong> = requests per second the model can handle</span><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-purple);">base_throughput</strong> = 100 - (params_billions × 0.8)</span><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-purple);">efficiency_multiplier</strong> = 1.3 for MoE models, 1.0 for dense models</span><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-green);">Small dense models (7-8B params)</strong>: Score 75-90 (high throughput)</span><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-purple);">MoE architectures</strong>: Score 65-80 (efficient despite size)</span><br>
        <span style="color: var(--text-primary);">• <strong style="color: var(--accent-orange);">Large dense models (70B+)</strong>: Score 40-55 (lower throughput)</span>
    </td>
</tr>
</table>
        """, unsafe_allow_html=True)


def render_pipeline():
    """Render the pipeline visualization."""
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step">
            <div class="pipeline-number pipeline-number-1">1</div>
            <div class="pipeline-title">🔍 Context Extraction</div>
            <div class="pipeline-desc">Qwen 2.5 7B extracts use case, users, priority & hardware from natural language</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-number pipeline-number-2">2</div>
            <div class="pipeline-title">⚖️ MCDM Scoring</div>
            <div class="pipeline-desc">Score 206 models on Quality, Latency, Cost & Capacity with weighted criteria</div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-number pipeline-number-3">3</div>
            <div class="pipeline-title">🏆 Recommendation</div>
            <div class="pipeline-desc">Top 5 models with explainability, tradeoffs, SLO compliance & deployment config</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_top5_table(recommendations: list, priority: str):
    """Render beautiful Top 5 recommendation leaderboard table with filtering."""
    
    # Filter controls
    st.markdown("""
    <style>
        /* Filter section - ALL text white and visible */
        div[data-testid="stHorizontalBlock"] label {
            color: white !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }
        /* Selectbox text - white */
        .stSelectbox > div > div {
            background: rgba(102, 126, 234, 0.2) !important;
        }
        .stSelectbox [data-baseweb="select"] > div {
            color: white !important;
            background: rgba(102, 126, 234, 0.15) !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
        }
        .stSelectbox [data-baseweb="select"] span {
            color: white !important;
        }
        /* Slider labels white */
        .stSlider label {
            color: white !important;
            font-weight: 600 !important;
        }
        .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {
            color: rgba(255,255,255,0.6) !important;
        }
    </style>
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; padding: 1rem; 
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(56, 239, 125, 0.05)); 
                border-radius: 1rem; border: 1px solid rgba(102, 126, 234, 0.2);">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.25rem;">🔧</span>
            <span style="color: white; font-weight: 600;">Filter & Sort Options</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        sort_by = st.selectbox(
            "Sort By",
            ["Final Score", "Quality", "Latency", "Cost", "Capacity"],
            key="sort_recommendations"
        )
    
    with col2:
        priority_filter = st.selectbox(
            "Priority Focus",
            ["All Priorities", "⚖️ Balanced", "⚡ Low Latency", "💰 Cost Saving", "⭐ High Quality", "📈 High Throughput"],
            key="priority_filter"
        )
    
    with col3:
        min_score = st.slider("Min Total Score", 0, 100, 0, key="min_score_filter")
    
    with col4:
        min_quality = st.slider("Min Quality Score", 0, 100, 0, key="min_quality_filter")
    
    with col5:
        show_count = st.selectbox("Show Top", [3, 5, 10], key="show_count")
    
    # Show "Best Model for Priority" when specific priority is selected (not All Priorities)
    if priority_filter != "All Priorities" and recommendations:
        # Calculate best model for selected priority
        priority_weights_map = {
            "⚖️ Balanced": {"quality": 0.30, "latency": 0.25, "cost": 0.25, "capacity": 0.20},
            "⚡ Low Latency": {"quality": 0.20, "latency": 0.45, "cost": 0.15, "capacity": 0.20},
            "💰 Cost Saving": {"quality": 0.20, "latency": 0.15, "cost": 0.50, "capacity": 0.15},
            "⭐ High Quality": {"quality": 0.50, "latency": 0.20, "cost": 0.15, "capacity": 0.15},
            "📈 High Throughput": {"quality": 0.20, "latency": 0.15, "cost": 0.15, "capacity": 0.50},
        }
        pweights = priority_weights_map.get(priority_filter, priority_weights_map["⚖️ Balanced"])
        
        best_model = None
        best_score = 0
        for rec in recommendations:
            breakdown = rec.get("score_breakdown", {})
            score = (
                (breakdown.get("quality_score") or 0) * pweights["quality"] +
                (breakdown.get("latency_score") or 0) * pweights["latency"] +
                (breakdown.get("cost_score") or 0) * pweights["cost"] +
                (breakdown.get("capacity_score") or 0) * pweights["capacity"]
            )
            if score > best_score:
                best_score = score
                best_model = rec
        
        if best_model:
            model_name = best_model.get("model_name", "Unknown")
            provider = best_model.get("provider", "Unknown")
            breakdown = best_model.get("score_breakdown", {})
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(56, 239, 125, 0.15), rgba(102, 126, 234, 0.1)); 
                        padding: 1.25rem; border-radius: 1rem; margin-bottom: 1.5rem;
                        border: 2px solid rgba(56, 239, 125, 0.4);">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
                    <div>
                        <div style="font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-bottom: 0.25rem;">
                            🏆 Best Model for <span style="color: #f093fb; font-weight: 700;">{priority_filter}</span>
                        </div>
                        <div style="font-size: 1.5rem; font-weight: 800; color: #38ef7d;">{model_name}</div>
                        <div style="font-size: 0.85rem; color: rgba(255,255,255,0.5);">{provider}</div>
                    </div>
                    <div style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
                        <div style="text-align: center; padding: 0.5rem 1rem; background: rgba(56, 239, 125, 0.1); border-radius: 0.5rem;">
                            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.6);">Quality</div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: #38ef7d;">{breakdown.get('quality_score', 0):.0f}</div>
                        </div>
                        <div style="text-align: center; padding: 0.5rem 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 0.5rem;">
                            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.6);">Latency</div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: #667eea;">{breakdown.get('latency_score', 0):.0f}</div>
                        </div>
                        <div style="text-align: center; padding: 0.5rem 1rem; background: rgba(245, 87, 108, 0.1); border-radius: 0.5rem;">
                            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.6);">Cost</div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: #f5576c;">{breakdown.get('cost_score', 0):.0f}</div>
                        </div>
                        <div style="text-align: center; padding: 0.5rem 1rem; background: rgba(79, 172, 254, 0.1); border-radius: 0.5rem;">
                            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.6);">Capacity</div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: #4facfe;">{breakdown.get('capacity_score', 0):.0f}</div>
                        </div>
                        <div style="text-align: center; padding: 0.5rem 1rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(56, 239, 125, 0.2)); border-radius: 0.5rem; border: 1px solid rgba(56, 239, 125, 0.3);">
                            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.6);">Final Score</div>
                            <div style="font-size: 1.3rem; font-weight: 800; background: linear-gradient(135deg, #667eea, #38ef7d); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{best_score:.1f}</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Apply filters with robust error handling
    try:
        filtered_recs = recommendations.copy() if recommendations else []
        
        # Apply priority-based re-scoring if specific priority selected
        if priority_filter != "All Priorities":
            priority_weights_map = {
                "⚖️ Balanced": {"quality": 0.30, "latency": 0.25, "cost": 0.25, "capacity": 0.20},
                "⚡ Low Latency": {"quality": 0.20, "latency": 0.45, "cost": 0.15, "capacity": 0.20},
                "💰 Cost Saving": {"quality": 0.20, "latency": 0.15, "cost": 0.50, "capacity": 0.15},
                "⭐ High Quality": {"quality": 0.50, "latency": 0.20, "cost": 0.15, "capacity": 0.15},
                "📈 High Throughput": {"quality": 0.20, "latency": 0.15, "cost": 0.15, "capacity": 0.50},
            }
            weights = priority_weights_map.get(priority_filter, priority_weights_map["⚖️ Balanced"])
            
            # Re-calculate final scores based on selected priority
            for rec in filtered_recs:
                breakdown = rec.get("score_breakdown", {})
                rec["final_score"] = (
                    (breakdown.get("quality_score") or 0) * weights["quality"] +
                    (breakdown.get("latency_score") or 0) * weights["latency"] +
                    (breakdown.get("cost_score") or 0) * weights["cost"] +
                    (breakdown.get("capacity_score") or 0) * weights["capacity"]
                )
        
        # Filter by minimum scores (handle missing/None values)
        filtered_recs = [
            r for r in filtered_recs 
            if (r.get('final_score') or 0) >= min_score
        ]
        filtered_recs = [
            r for r in filtered_recs 
            if (r.get('score_breakdown', {}).get('quality_score') or 0) >= min_quality
        ]
        
        # Sort with safe key extraction
        def safe_sort_key(field):
            def get_value(x):
                if field == "final_score":
                    return float(x.get('final_score') or 0)
                return float(x.get('score_breakdown', {}).get(f'{field.lower()}_score') or 0)
            return get_value
        
        sort_map = {
            "Final Score": safe_sort_key("final_score"),
            "Quality": safe_sort_key("quality"),
            "Latency": safe_sort_key("latency"),
            "Cost": safe_sort_key("cost"),
            "Capacity": safe_sort_key("capacity"),
        }
        filtered_recs = sorted(filtered_recs, key=sort_map[sort_by], reverse=True)[:show_count]
    except Exception as e:
        st.error(f"⚠️ Error applying filters. Showing unfiltered results.")
        filtered_recs = recommendations[:5] if recommendations else []
    
    if not filtered_recs:
        st.info("🔍 No models match the selected filters. Try adjusting the criteria or lowering the minimum scores.")
        return
    
    # Add legend for score bars - AA inspired
    st.markdown("""
    <div class="chart-legend">
        <div class="legend-item">
            <div class="legend-color legend-color-quality"></div>
            <span>Quality (Benchmark Score)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-color-latency"></div>
            <span>Latency (Inference Speed)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-color-cost"></div>
            <span>Cost (GPU Efficiency)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-color-capacity"></div>
            <span>Capacity (Throughput)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Priority display info
    priority_info = f" | Priority: <strong style='color: var(--accent-purple);'>{priority_filter}</strong>" if priority_filter != "All Priorities" else ""
    
    st.markdown(f"""
    <div class="leaderboard-container">
        <div class="leaderboard-header">
            <span style="font-size: 1.75rem;">🏆</span>
            <span class="leaderboard-title">Top {len(filtered_recs)} Model Recommendations</span>
            <span style="margin-left: auto; font-size: 0.9rem; color: var(--text-secondary);">
                Sorted by: <strong style="color: var(--accent-green);">{sort_by}</strong>{priority_info}
            </span>
        </div>
        <table class="leaderboard-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>🎯 Quality</th>
                    <th>⚡ Latency</th>
                    <th>💰 Cost</th>
                    <th>📈 Capacity</th>
                    <th>Final Score</th>
                    <th>Pros & Cons</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
    """, unsafe_allow_html=True)
    
    recommendations = filtered_recs  # Use filtered list
    
    for i, rec in enumerate(recommendations, 1):
        breakdown = rec.get("score_breakdown", {})
        pros = rec.get("pros", [])
        cons = rec.get("cons", [])
        
        # Build pros/cons tags
        tags_html = ""
        for pro in pros[:2]:
            tags_html += f'<span class="tag tag-pro">{pro}</span>'
        for con in cons[:1]:
            tags_html += f'<span class="tag tag-con">{con}</span>'
        
        st.markdown(f"""
            <tr>
                <td><div class="rank-badge rank-{i}">{i}</div></td>
                <td>
                    <div class="model-cell">
                        <div class="model-info">
                            <span class="model-name">{rec.get('model_name', 'Unknown')}</span>
                            <span class="model-provider">{rec.get('provider', 'Open Source')}</span>
                        </div>
                    </div>
                </td>
                <td>
                    <div class="score-mini-container">
                        <span class="score-mini-label label-quality">{breakdown.get('quality_score', 0):.0f}%</span>
                        <div class="score-mini-bar">
                            <div class="score-mini-fill fill-quality" style="width: {breakdown.get('quality_score', 0)}%;"></div>
                        </div>
                    </div>
                </td>
                <td>
                    <div class="score-mini-container">
                        <span class="score-mini-label label-latency">{breakdown.get('latency_score', 0):.0f}%</span>
                        <div class="score-mini-bar">
                            <div class="score-mini-fill fill-latency" style="width: {breakdown.get('latency_score', 0)}%;"></div>
                        </div>
                    </div>
                </td>
                <td>
                    <div class="score-mini-container">
                        <span class="score-mini-label label-cost">{breakdown.get('cost_score', 0):.0f}%</span>
                        <div class="score-mini-bar">
                            <div class="score-mini-fill fill-cost" style="width: {breakdown.get('cost_score', 0)}%;"></div>
                        </div>
                    </div>
                </td>
                <td>
                    <div class="score-mini-container">
                        <span class="score-mini-label label-capacity">{breakdown.get('capacity_score', 0):.0f}%</span>
                        <div class="score-mini-bar">
                            <div class="score-mini-fill fill-capacity" style="width: {breakdown.get('capacity_score', 0)}%;"></div>
                        </div>
                    </div>
                </td>
                <td style="text-align: center;"><span class="final-score">{rec.get('final_score', 0):.1f}</span></td>
                <td style="text-align: center;">
                    <div class="tag-container">
                        {tags_html}
                    </div>
                </td>
                <td style="text-align: center;">
                    <div style="display: flex; justify-content: center;">
                        <button class="select-btn">Select →</button>
                    </div>
                </td>
            </tr>
        """, unsafe_allow_html=True)
    
    st.markdown("""
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


def render_score_bar(label: str, icon: str, score: float, bar_class: str, contribution: float):
    """Render score bar like Artificial Analysis - number ABOVE the bar."""
    import math
    # Handle NaN values
    if math.isnan(score) if isinstance(score, float) else False:
        score = 50  # Default to 50 if NaN
    if math.isnan(contribution) if isinstance(contribution, float) else False:
        contribution = 0
    
    score = max(0, min(100, score))  # Clamp to 0-100
    
    # Color based on bar class
    bar_colors = {
        "score-bar-quality": "#38ef7d",
        "score-bar-latency": "#667eea", 
        "score-bar-cost": "#f5576c",
        "score-bar-capacity": "#8b5cf6",
    }
    bar_color = bar_colors.get(bar_class, "#38ef7d")
    
    st.markdown(f"""
    <div style="margin-bottom: 1.25rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="color: rgba(255,255,255,0.9); font-weight: 600; font-size: 1rem;">{icon} {label}</span>
            <span style="font-size: 1.1rem;">
                <strong style="color: white; font-size: 1.3rem;">{score:.0f}</strong>
                <span style="color: rgba(255,255,255,0.5);">→</span>
                <span style="color: #38ef7d; font-weight: 700;">+{contribution:.1f}</span>
            </span>
        </div>
        <div style="position: relative; height: 28px;">
            <!-- Score number above bar (Artificial Analysis style) -->
            <div style="position: absolute; left: {min(score, 95)}%; transform: translateX(-50%); top: -2px; z-index: 10;">
                <span style="background: {bar_color}; color: white; font-weight: 700; font-size: 0.9rem; padding: 0.2rem 0.5rem; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">{score:.0f}%</span>
            </div>
            <!-- Bar background -->
            <div style="position: absolute; top: 8px; left: 0; right: 0; height: 12px; background: rgba(255,255,255,0.1); border-radius: 6px; overflow: hidden;">
                <div style="width: {score}%; height: 100%; background: linear-gradient(90deg, {bar_color}cc, {bar_color}); border-radius: 6px; transition: width 0.5s ease;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_slo_cards(use_case: str, user_count: int, priority: str = "balanced", hardware: str = None):
    """Render SLO and workload impact cards with editable fields."""
    slo_templates = load_slo_templates()
    slo = slo_templates.get(use_case, slo_templates["chatbot_conversational"])

    # Calculate QPS based on user count
    estimated_qps = max(1, user_count // 50)

    # Use custom values if set, otherwise use defaults
    ttft = st.session_state.custom_ttft if st.session_state.custom_ttft else slo['ttft']
    itl = st.session_state.custom_itl if st.session_state.custom_itl else slo['itl']
    e2e = st.session_state.custom_e2e if st.session_state.custom_e2e else slo['e2e']
    qps = st.session_state.custom_qps if st.session_state.custom_qps else estimated_qps

    # Golden styled section header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
        <span style="font-size: 1.2rem; color: #D4AF37;">✏️</span>
        <span style="color: #D4AF37; font-size: 0.85rem; font-weight: 600;">CLICK VALUES TO EDIT</span>
    </div>
    """, unsafe_allow_html=True)

    # Create 4 columns for all cards in one row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="slo-card">
            <span class="edit-indicator">✏️</span>
            <div class="slo-header">
                <span class="slo-icon">⏱️</span>
                <span class="slo-title">SLO Targets</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Editable TTFT
        new_ttft = st.number_input("TTFT (ms)", value=ttft, min_value=10, max_value=2000, step=10, key="edit_ttft", label_visibility="collapsed")
        st.markdown(f'<div style="font-size: 0.9rem; color: rgba(255,255,255,0.7); margin-top: -0.75rem; margin-bottom: 0.5rem;">⏱️ TTFT < <span style="color: #38ef7d; font-weight: 700; font-size: 1rem;">{new_ttft}ms</span></div>', unsafe_allow_html=True)
        
        # Editable ITL
        new_itl = st.number_input("ITL (ms)", value=itl, min_value=5, max_value=500, step=5, key="edit_itl", label_visibility="collapsed")
        st.markdown(f'<div style="font-size: 0.9rem; color: rgba(255,255,255,0.7); margin-top: -0.75rem; margin-bottom: 0.5rem;">⚡ ITL < <span style="color: #38ef7d; font-weight: 700; font-size: 1rem;">{new_itl}ms</span></div>', unsafe_allow_html=True)
        
        # Editable E2E
        new_e2e = st.number_input("E2E (ms)", value=e2e, min_value=100, max_value=10000, step=100, key="edit_e2e", label_visibility="collapsed")
        st.markdown(f'<div style="font-size: 0.9rem; color: rgba(255,255,255,0.7); margin-top: -0.75rem; margin-bottom: 0.5rem;">🏁 E2E < <span style="color: #38ef7d; font-weight: 700; font-size: 1rem;">{new_e2e}ms</span></div>', unsafe_allow_html=True)
        
        # Store custom values
        if new_ttft != ttft:
            st.session_state.custom_ttft = new_ttft
        if new_itl != itl:
            st.session_state.custom_itl = new_itl
        if new_e2e != e2e:
            st.session_state.custom_e2e = new_e2e
        
        # Get priority from session state
        current_priority = st.session_state.get('extracted_priority', 'balanced')
        
        # Research-backed SLO validation
        research_messages = validate_slo_against_research(use_case, new_ttft, new_itl, new_e2e, current_priority)
        
        # Separate by severity for better UX
        errors = [m for m in research_messages if m[3] == 'error']
        warnings = [m for m in research_messages if m[3] == 'warning']
        successes = [m for m in research_messages if m[3] == 'success']
        infos = [m for m in research_messages if m[3] == 'info']
        
        # Show errors first (red)
        for icon, color, text, _ in errors:
            st.markdown(f'<div style="font-size: 0.85rem; color: {color}; padding: 0.4rem 0.5rem; line-height: 1.4; background: rgba(245, 87, 108, 0.1); border-radius: 6px; margin: 4px 0; border-left: 3px solid {color};">{icon} {text}</div>', unsafe_allow_html=True)
        
        # Show warnings (orange/yellow)
        for icon, color, text, _ in warnings:
            st.markdown(f'<div style="font-size: 0.85rem; color: {color}; padding: 0.4rem 0.5rem; line-height: 1.4; background: rgba(251, 191, 36, 0.1); border-radius: 6px; margin: 4px 0; border-left: 3px solid {color};">{icon} {text}</div>', unsafe_allow_html=True)
        
        # Show successes (green) - more visible
        if successes and not errors and not warnings:
            st.markdown(f'<div style="font-size: 0.9rem; color: #38ef7d; padding: 0.4rem 0.5rem; line-height: 1.4; background: rgba(56, 239, 125, 0.1); border-radius: 6px; margin: 4px 0;">✅ All SLO values within research-backed ranges</div>', unsafe_allow_html=True)
        
        # Show research note (purple) - bigger font
        for icon, color, text, _ in infos:
            st.markdown(f'<div style="font-size: 0.8rem; color: {color}; padding: 0.35rem 0.5rem; line-height: 1.4; font-style: italic; background: rgba(139, 92, 246, 0.08); border-radius: 5px; margin: 3px 0;">{icon} {text}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="slo-card">
            <span class="edit-indicator">✏️</span>
            <div class="slo-header">
                <span class="slo-icon">📊</span>
                <span class="slo-title">Workload Profile</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Load token config and workload data from research
        research_data = load_research_slo_ranges()
        use_case_ranges = research_data.get('slo_ranges', {}).get(use_case, {}) if research_data else {}
        token_config = use_case_ranges.get('token_config', {'prompt': 512, 'output': 256})
        prompt_tokens = token_config.get('prompt', 512)
        output_tokens = token_config.get('output', 256)

        workload_data = load_research_workload_patterns()
        pattern = workload_data.get('workload_distributions', {}).get(use_case, {}) if workload_data else {}
        peak_mult = pattern.get('peak_multiplier', 2.0)

        # 1. Editable QPS - support up to 10M QPS for enterprise scale
        new_qps = st.number_input("Expected QPS", value=min(qps, 10000000), min_value=1, max_value=10000000, step=1, key="edit_qps", label_visibility="collapsed")
        st.markdown(f'<div style="font-size: 0.9rem; color: rgba(255,255,255,0.7); margin-top: -0.75rem; margin-bottom: 0.5rem;">📊 Expected QPS: <span style="color: #4facfe; font-weight: 700; font-size: 1rem;">{new_qps}</span></div>', unsafe_allow_html=True)

        if new_qps != qps:
            st.session_state.custom_qps = new_qps

        # 2-4. Fixed workload values in a styled box (Mean Prompt Tokens, Mean Output Tokens, Peak Multiplier)
        st.markdown(f"""
        <div style="margin-top: 0.5rem; background: rgba(255,255,255,0.03); padding: 0.75rem; border-radius: 8px;">
            <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; font-size: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1); cursor: help;" title="Value determined by use case">
                <span style="color: rgba(255,255,255,0.8);">📏 Mean Prompt Tokens</span>
                <span style="color: #38ef7d; font-weight: 700; font-size: 1.1rem;">{prompt_tokens}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; font-size: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1); cursor: help;" title="Value determined by use case">
                <span style="color: rgba(255,255,255,0.8);">📏 Mean Output Tokens</span>
                <span style="color: #38ef7d; font-weight: 700; font-size: 1.1rem;">{output_tokens}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; font-size: 1rem; cursor: help;" title="Value determined by use case">
                <span style="color: rgba(255,255,255,0.8);">📈 Peak Multiplier</span>
                <span style="color: #38ef7d; font-weight: 700; font-size: 1.1rem;">{peak_mult}x</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 5. Informational messages from research data
        workload_messages = get_workload_insights(use_case, new_qps, user_count)

        for icon, color, text, severity in workload_messages[:3]:  # Limit to 3 for space
            bg_color = "rgba(245, 87, 108, 0.1)" if severity == "error" else \
                       "rgba(251, 191, 36, 0.1)" if severity == "warning" else \
                       "rgba(56, 239, 125, 0.08)" if severity == "success" else "rgba(88, 166, 255, 0.08)"
            st.markdown(f'<div style="font-size: 0.85rem; color: {color}; padding: 0.4rem 0.5rem; line-height: 1.4; background: {bg_color}; border-radius: 6px; margin: 4px 0;">{icon} {text}</div>', unsafe_allow_html=True)
    
    with col3:
        # Task Datasets - show which benchmarks are used for this use case
        TASK_DATASETS = {
            "chatbot_conversational": [
                ("MMLU-Pro", 30, "#38ef7d"),
                ("IFBench", 30, "#4facfe"),
                ("HLE", 20, "#a855f7"),
                ("Intelligence Index", 15, "#f59e0b"),
                ("GPQA", 5, "#667eea"),
            ],
            "code_completion": [
                ("LiveCodeBench", 35, "#38ef7d"),
                ("SciCode", 30, "#4facfe"),
                ("Coding Index", 20, "#a855f7"),
                ("Terminal-Bench", 10, "#f59e0b"),
                ("IFBench", 5, "#667eea"),
            ],
            "code_generation_detailed": [
                ("LiveCodeBench", 30, "#38ef7d"),
                ("SciCode", 25, "#4facfe"),
                ("IFBench", 20, "#a855f7"),
                ("Coding Index", 15, "#f59e0b"),
                ("HLE", 10, "#667eea"),
            ],
            "translation": [
                ("IFBench", 35, "#38ef7d"),
                ("MMLU-Pro", 30, "#4facfe"),
                ("HLE", 20, "#a855f7"),
                ("Intelligence Index", 15, "#f59e0b"),
            ],
            "content_generation": [
                ("MMLU-Pro", 30, "#38ef7d"),
                ("HLE", 25, "#4facfe"),
                ("IFBench", 25, "#a855f7"),
                ("Intelligence Index", 20, "#f59e0b"),
            ],
            "summarization_short": [
                ("HLE", 30, "#38ef7d"),
                ("MMLU-Pro", 25, "#4facfe"),
                ("IFBench", 25, "#a855f7"),
                ("Intelligence Index", 20, "#f59e0b"),
            ],
            "document_analysis_rag": [
                ("AA-LCR", 40, "#38ef7d"),
                ("MMLU-Pro", 20, "#4facfe"),
                ("HLE", 20, "#a855f7"),
                ("IFBench", 10, "#f59e0b"),
                ("τ²-Bench", 10, "#667eea"),
            ],
            "long_document_summarization": [
                ("AA-LCR", 45, "#38ef7d"),
                ("MMLU-Pro", 20, "#4facfe"),
                ("HLE", 20, "#a855f7"),
                ("IFBench", 15, "#f59e0b"),
            ],
            "research_legal_analysis": [
                ("AA-LCR", 40, "#38ef7d"),
                ("MMLU-Pro", 25, "#4facfe"),
                ("HLE", 15, "#a855f7"),
                ("GPQA", 10, "#f59e0b"),
                ("IFBench", 5, "#667eea"),
                ("τ²-Bench", 5, "#f5576c"),
            ],
        }
        
        datasets = TASK_DATASETS.get(use_case, TASK_DATASETS["chatbot_conversational"])
        
        st.markdown("""
        <div class="slo-card">
            <div class="slo-header">
                <span class="slo-icon">📊</span>
                <span class="slo-title">Task Datasets</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display datasets with weights - build HTML as single string
        datasets_items = []
        for name, weight, color in datasets:
            datasets_items.append(f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);"><span style="color: rgba(255,255,255,0.9); font-size: 0.9rem;">{name}</span><span style="color: {color}; font-weight: 700; font-size: 0.9rem; background: {color}22; padding: 2px 8px; border-radius: 4px;">{weight}%</span></div>')
        
        datasets_html = "".join(datasets_items)
        full_html = f'<div style="background: rgba(255,255,255,0.03); padding: 0.75rem; border-radius: 8px; margin-top: 0.5rem;">{datasets_html}</div><div style="font-size: 0.75rem; color: rgba(255,255,255,0.5); margin-top: 0.5rem; font-style: italic;">📖 Weights from Artificial Analysis Intelligence Index methodology</div>'
        st.markdown(full_html, unsafe_allow_html=True)

    with col4:
        # Technical Spec (Optional Fields) - same style as other cards
        st.markdown("""
        <div class="slo-card">
            <div class="slo-header">
                <span class="slo-icon">📋</span>
                <span class="slo-title">Technical Spec (Optional Fields)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Build items based on what user mentioned
        items = []

        # Priority - only show if not balanced
        if priority and priority != "balanced":
            priority_display = priority.replace('_', ' ').title()
            priority_color = {
                "low_latency": "#667eea",
                "cost_saving": "#f5576c",
                "high_quality": "#38ef7d",
                "high_throughput": "#4facfe"
            }.get(priority, "#9ca3af")
            priority_icon = {
                "low_latency": "⚡",
                "cost_saving": "💰",
                "high_quality": "⭐",
                "high_throughput": "📈"
            }.get(priority, "🎯")
            items.append((priority_icon, "Priority", priority_display, priority_color))

        # Hardware - only show if user explicitly mentioned it
        if hardware and hardware not in ["Any GPU", "Any", None, ""]:
            items.append(("🖥️", "Hardware", hardware, "#38ef7d"))

        # Build content HTML
        if items:
            items_html = "".join([
                f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);"><span style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">{icon} {label}</span><span style="color: {color}; font-weight: 700; font-size: 0.9rem; background: {color}22; padding: 4px 10px; border-radius: 6px;">{value}</span></div>'
                for icon, label, value, color in items
            ])
        else:
            items_html = '<div style="display: flex; justify-content: center; padding: 1rem 0;"><span style="color: rgba(255,255,255,0.5); font-size: 0.85rem; font-style: italic;">Default settings applied</span></div>'

        full_html = f'<div style="background: rgba(255,255,255,0.03); padding: 0.75rem; border-radius: 8px; margin-top: 0.5rem;">{items_html}</div>'
        st.markdown(full_html, unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Load models
    if st.session_state.models_df is None:
        st.session_state.models_df = load_206_models()
    models_df = st.session_state.models_df
    models_count = 206  # Always show 206 from our Artificial Analysis catalog
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        priority = st.selectbox(
            "🎯 Optimization Priority",
            ["balanced", "low_latency", "cost_saving", "high_quality", "high_throughput"],
            format_func=lambda x: {
                "balanced": "⚖️ Balanced",
                "low_latency": "⚡ Low Latency",
                "cost_saving": "💰 Cost Saving",
                "high_quality": "⭐ High Quality",
                "high_throughput": "📈 High Throughput"
            }.get(x, x)
        )
        
        # Weight Profile Section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">⚖️ Weight Profile</div>', unsafe_allow_html=True)
        
        weights = {
            "low_latency": {"Quality": 20, "Latency": 45, "Cost": 15, "Capacity": 20},
            "cost_saving": {"Quality": 20, "Latency": 15, "Cost": 50, "Capacity": 15},
            "high_quality": {"Quality": 50, "Latency": 20, "Cost": 15, "Capacity": 15},
            "high_throughput": {"Quality": 20, "Latency": 15, "Cost": 15, "Capacity": 50},
            "balanced": {"Quality": 30, "Latency": 25, "Cost": 25, "Capacity": 20},
        }[priority]
        
        icons = {"Quality": "🎯", "Latency": "⚡", "Cost": "💰", "Capacity": "📈"}
        classes = {"Quality": "quality", "Latency": "latency", "Cost": "cost", "Capacity": "capacity"}
        
        for metric, weight in weights.items():
            st.markdown(f"""
            <div class="weight-item">
                <div class="weight-label">
                    <span class="weight-name">{icons[metric]} {metric}</span>
                    <span class="weight-value">{weight}%</span>
                </div>
                <div class="weight-bar">
                    <div class="weight-fill weight-fill-{classes[metric]}" style="width: {weight}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Database Section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📦 Model Database</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align: center; padding: 0.75rem 0;">
            <div style="font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Inter', sans-serif;">206</div>
            <div style="color: var(--text-secondary); font-size: 0.9rem; font-weight: 500;">Open-Source Models</div>
            <div style="color: var(--text-muted); font-size: 0.8rem; margin-top: 0.75rem;">Meta • Alibaba • DeepSeek • Google • Mistral</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Extractor Section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🤖 LLM Extractor</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; padding: 0.75rem 0;">
            <div style="font-weight: 700; color: var(--text-primary); font-size: 1.2rem; font-family: 'Inter', sans-serif;">Qwen 2.5 7B</div>
            <div style="color: var(--accent-green); font-weight: 800; font-size: 1.75rem; margin: 0.5rem 0; font-family: 'Inter', sans-serif;">95.1%</div>
            <div style="color: var(--text-muted); font-size: 0.8rem;">accuracy on 600 test cases</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main Content
    render_hero()
    render_stats(models_count)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendation", "📦 Model Catalog", "📖 How It Works"])
    
    with tab1:
        render_recommendation_tab(priority, models_df)
    
    with tab2:
        render_catalog_tab(models_df)
    
    with tab3:
        render_how_it_works_tab()


def render_recommendation_tab(priority: str, models_df: pd.DataFrame):
    """Main recommendation interface with clean task buttons."""
    
    st.markdown('<div class="section-header"><span>🎯</span> Select Your Use Case</div>', unsafe_allow_html=True)
    
    # Row 1: 5 task buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("💬 Chat Completion", use_container_width=True, key="task_chat"):
            st.session_state.user_input = "Customer service chatbot for 5000 users. Latency is critical - responses under 200ms. Using H100 GPUs."
    
    with col2:
        if st.button("💻 Code Completion", use_container_width=True, key="task_code"):
            st.session_state.user_input = "IDE code completion tool for 200 developers. Need fast autocomplete suggestions, low latency is key."
    
    with col3:
        if st.button("📄 Document Q&A", use_container_width=True, key="task_rag"):
            st.session_state.user_input = "RAG system for enterprise document Q&A, 1000 users, cost-efficient preferred, A100 GPUs available."
    
    with col4:
        if st.button("📝 Summarization", use_container_width=True, key="task_summ"):
            st.session_state.user_input = "News article summarization for 2000 daily users. Quick summaries, cost-effective solution needed."
    
    with col5:
        if st.button("⚖️ Legal Analysis", use_container_width=True, key="task_legal"):
            st.session_state.user_input = "Legal document analysis for 50 lawyers. Accuracy is critical, budget is flexible."
    
    # Row 2: 4 more task buttons
    col6, col7, col8, col9 = st.columns(4)
    
    with col6:
        if st.button("🌐 Translation", use_container_width=True, key="task_trans"):
            st.session_state.user_input = "Multi-language translation service for 3000 users. Need to translate between 10 language pairs accurately."
    
    with col7:
        if st.button("✍️ Content Generation", use_container_width=True, key="task_content"):
            st.session_state.user_input = "Content generation tool for marketing team, 100 users. Need creative blog posts and social media content."
    
    with col8:
        if st.button("📚 Long Doc Summary", use_container_width=True, key="task_longdoc"):
            st.session_state.user_input = "Long document summarization for research papers (50+ pages). 200 researchers, quality is most important."
    
    with col9:
        if st.button("🔧 Code Generation", use_container_width=True, key="task_codegen"):
            st.session_state.user_input = "Full code generation tool for implementing features from specs. 50 developers, high quality code needed."
    
    # Input area with validation
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    user_input = st.text_area(
        "Your requirements:",
        value=st.session_state.user_input,
        height=120,
        max_chars=2000,  # Corporate standard: limit input length
        placeholder="✨ Describe your LLM use case in natural language...\n\nExample: I need a chatbot for customer support with 10,000 users. Low latency is important, and we have H100 GPUs available.",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show character count
    char_count = len(user_input) if user_input else 0
    char_color = "#38ef7d" if char_count < 1500 else "#f5576c" if char_count > 1800 else "#f093fb"
    st.markdown(f'<div style="text-align: right; font-size: 0.75rem; color: {char_color}; margin-top: -0.5rem;">{char_count}/2000 characters</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1.5, 1, 2])
    with col1:
        # Disable button if input is too short
        analyze_disabled = len(user_input.strip()) < 10 if user_input else True
        analyze_clicked = st.button("🚀 Analyze & Recommend", type="primary", use_container_width=True, disabled=analyze_disabled)
        if analyze_disabled and user_input and len(user_input.strip()) < 10:
            st.caption("⚠️ Please enter at least 10 characters")
    with col2:
        if st.button("🔄 Clear", use_container_width=True):
            # Complete session state reset
            for key in ['user_input', 'extraction_result', 'recommendation_result', 
                       'extraction_approved', 'slo_approved', 'edited_extraction',
                       'custom_ttft', 'custom_itl', 'custom_e2e', 'custom_qps', 'used_priority']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.user_input = ""
            st.rerun()
    
    # Input validation before analysis
    if analyze_clicked and user_input and len(user_input.strip()) >= 10:
        st.session_state.user_input = user_input
        # Reset workflow state
        st.session_state.extraction_approved = None
        st.session_state.slo_approved = None
        st.session_state.recommendation_result = None
        st.session_state.edited_extraction = None
        
        # Show progress bar for better UX
        progress_container = st.empty()
        with progress_container:
            progress_bar = st.progress(0, text="🔍 Initializing extraction...")
            
        try:
            progress_bar.progress(20, text="🔍 Analyzing input text...")
            extraction = extract_business_context(user_input)
            progress_bar.progress(80, text="✅ Extraction complete!")
            
            if extraction:
                st.session_state.extraction_result = extraction
                st.session_state.used_priority = extraction.get("priority", priority)
                progress_bar.progress(100, text="🎉 Ready!")
            else:
                st.error("❌ Could not extract business context. Please try rephrasing your input.")
                progress_bar.empty()
                
        except Exception as e:
            st.error(f"❌ An error occurred during analysis. Please try again.")
            progress_bar.empty()
        finally:
            # Clean up progress bar after brief delay
            import time
            time.sleep(0.5)
            progress_container.empty()
    
    # Get the priority that was actually used
    used_priority = st.session_state.get("used_priority", priority)
    
    # === STEP 1: Show Extraction Result with Approval ===
    if st.session_state.extraction_result and st.session_state.extraction_approved is None:
        render_extraction_with_approval(st.session_state.extraction_result, used_priority, models_df)
        return  # Don't show anything else until approved
    
    # === STEP 2: If editing, show edit form ===
    if st.session_state.extraction_approved == False:
        render_extraction_edit_form(st.session_state.extraction_result, models_df)
        return
    
    # === STEP 3: Show SLO/Workload (after extraction approved) ===
    if st.session_state.extraction_approved == True and st.session_state.slo_approved is None:
        # Get final extraction (edited or original)
        final_extraction = st.session_state.edited_extraction or st.session_state.extraction_result
        render_extraction_result(final_extraction, used_priority)
        render_slo_with_approval(final_extraction, used_priority, models_df)
        return
    
    # === STEP 4: Show Full Results (after SLO approved) ===
    if st.session_state.slo_approved == True:
        final_extraction = st.session_state.edited_extraction or st.session_state.extraction_result
        render_extraction_result(final_extraction, used_priority)
        
        if not st.session_state.recommendation_result:
            # Generate recommendations now
            business_context = {
                "use_case": final_extraction.get("use_case", "chatbot_conversational"),
                "user_count": final_extraction.get("user_count", 1000),
                "priority": used_priority,
                "hardware_preference": final_extraction.get("hardware"),
            }
            with st.spinner(f"🧠 Scoring {len(models_df)} models with MCDM..."):
                recommendation = get_enhanced_recommendation(business_context)
            if recommendation:
                st.session_state.recommendation_result = recommendation
        
        if st.session_state.recommendation_result:
            render_recommendation_result(st.session_state.recommendation_result, used_priority, final_extraction)


def render_extraction_result(extraction: dict, priority: str):
    """Render beautiful extraction results."""
    st.markdown('<div class="section-header"><span>📋</span> Step 1: Extracted Business Context</div>', unsafe_allow_html=True)
    
    use_case = extraction.get("use_case", "unknown")
    user_count = extraction.get("user_count", 0)
    hardware = extraction.get("hardware")
    
    st.markdown(f"""
    <div class="extraction-card">
        <div class="extraction-grid">
            <div class="extraction-item">
                <div class="extraction-icon extraction-icon-usecase">🎯</div>
                <div>
                    <div class="extraction-label">Use Case</div>
                    <div class="extraction-value">{use_case.replace("_", " ").title() if use_case else "Unknown"}</div>
                </div>
            </div>
            <div class="extraction-item">
                <div class="extraction-icon extraction-icon-users">👥</div>
                <div>
                    <div class="extraction-label">Expected Users</div>
                    <div class="extraction-value">{user_count:,}</div>
                </div>
            </div>
            <div class="extraction-item">
                <div class="extraction-icon extraction-icon-priority">⚡</div>
                <div>
                    <div class="extraction-label">Priority</div>
                    <div class="extraction-value"><span class="priority-badge priority-{priority}">{priority.replace("_", " ").title()}</span></div>
                </div>
            </div>
            <div class="extraction-item">
                <div class="extraction-icon extraction-icon-hardware">🖥️</div>
                <div>
                    <div class="extraction-label">Hardware</div>
                    <div class="extraction-value">{hardware if hardware else "Any GPU"}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_extraction_with_approval(extraction: dict, priority: str, models_df: pd.DataFrame):
    """Render extraction results with YES/NO approval buttons."""
    st.markdown('<div class="section-header"><span>📋</span> Step 1: Extracted Business Context</div>', unsafe_allow_html=True)
    
    use_case = extraction.get("use_case", "unknown")
    user_count = extraction.get("user_count", 0)
    hardware = extraction.get("hardware")
    
    st.markdown(f"""
    <div class="extraction-card" style="border: 2px solid #D4AF37;">
        <div class="extraction-grid">
            <div class="extraction-item">
                <div class="extraction-icon extraction-icon-usecase">🎯</div>
                <div>
                    <div class="extraction-label">Use Case</div>
                    <div class="extraction-value">{use_case.replace("_", " ").title() if use_case else "Unknown"}</div>
                </div>
            </div>
            <div class="extraction-item">
                <div class="extraction-icon extraction-icon-users">👥</div>
                <div>
                    <div class="extraction-label">Expected Users</div>
                    <div class="extraction-value">{user_count:,}</div>
                </div>
            </div>
            <div class="extraction-item">
                <div class="extraction-icon extraction-icon-priority">⚡</div>
                <div>
                    <div class="extraction-label">Priority</div>
                    <div class="extraction-value"><span class="priority-badge priority-{priority}">{priority.replace("_", " ").title()}</span></div>
                </div>
            </div>
            <div class="extraction-item">
                <div class="extraction-icon extraction-icon-hardware">🖥️</div>
                <div>
                    <div class="extraction-label">Hardware</div>
                    <div class="extraction-value">{hardware if hardware else "Any GPU"}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Approval question
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(212, 175, 55, 0.15), rgba(56, 239, 125, 0.1)); 
                padding: 1.25rem; border-radius: 1rem; margin: 1.5rem 0; text-align: center;
                border: 1px solid rgba(212, 175, 55, 0.3);">
        <p style="color: white; font-size: 1.2rem; font-weight: 600; margin: 0;">
            ✅ Is this extraction correct?
        </p>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-top: 0.5rem;">
            Verify the extracted business context before proceeding to recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("✅ Yes, Continue", type="primary", use_container_width=True, key="approve_extraction"):
            st.session_state.extraction_approved = True
            st.rerun()
    with col2:
        if st.button("✏️ No, Edit", use_container_width=True, key="edit_extraction"):
            st.session_state.extraction_approved = False
            st.rerun()
    with col3:
        if st.button("🔄 Start Over", use_container_width=True, key="restart"):
            st.session_state.extraction_result = None
            st.session_state.extraction_approved = None
            st.session_state.recommendation_result = None
            st.session_state.user_input = ""
            st.rerun()


def render_extraction_edit_form(extraction: dict, models_df: pd.DataFrame):
    """Render editable form for extraction correction."""
    st.markdown('<div class="section-header"><span>✏️</span> Edit Business Context</div>', unsafe_allow_html=True)
    
    # CSS to make form inputs visible
    st.markdown("""
    <style>
        /* Edit form - make all text white and visible */
        .stSelectbox > div > div {
            color: white !important;
        }
        .stSelectbox [data-baseweb="select"] > div {
            background: rgba(102, 126, 234, 0.2) !important;
            color: white !important;
        }
        .stSelectbox [data-baseweb="select"] span {
            color: white !important;
        }
        .stNumberInput input {
            color: white !important;
            background: rgba(102, 126, 234, 0.2) !important;
        }
        /* Dropdown menu items */
        [data-baseweb="menu"] {
            background: #1a1a2e !important;
        }
        [data-baseweb="menu"] li {
            color: white !important;
        }
        [data-baseweb="menu"] li:hover {
            background: rgba(102, 126, 234, 0.3) !important;
        }
    </style>
    <div style="background: rgba(245, 87, 108, 0.1); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #f5576c;">
        <p style="color: white; margin: 0;">Correct the extracted values below and click "Apply Changes" to continue.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use case dropdown
    use_cases = [
        "chatbot_conversational", "code_completion", "code_generation_detailed",
        "document_analysis_rag", "summarization_short", "long_document_summarization",
        "translation", "content_generation", "research_legal_analysis"
    ]
    use_case_labels = {
        "chatbot_conversational": "💬 Chatbot / Conversational AI",
        "code_completion": "💻 Code Completion (IDE autocomplete)",
        "code_generation_detailed": "🔧 Code Generation (full implementations)",
        "document_analysis_rag": "📄 Document RAG / Q&A",
        "summarization_short": "📝 Short Summarization (<10 pages)",
        "long_document_summarization": "📚 Long Document Summarization (10+ pages)",
        "translation": "🌐 Translation",
        "content_generation": "✍️ Content Generation",
        "research_legal_analysis": "⚖️ Research / Legal Analysis"
    }
    
    current_use_case = extraction.get("use_case", "chatbot_conversational")
    current_idx = use_cases.index(current_use_case) if current_use_case in use_cases else 0
    
    col1, col2 = st.columns(2)
    with col1:
        new_use_case = st.selectbox(
            "🎯 Use Case",
            use_cases,
            index=current_idx,
            format_func=lambda x: use_case_labels.get(x, x),
            key="edit_use_case"
        )
        
        new_user_count = st.number_input(
            "👥 User Count",
            min_value=1,
            max_value=1000000,
            value=extraction.get("user_count", 1000),
            step=100,
            key="edit_user_count"
        )
    
    with col2:
        priorities = ["balanced", "low_latency", "cost_saving", "high_quality", "high_throughput"]
        priority_labels = {
            "balanced": "⚖️ Balanced",
            "low_latency": "⚡ Low Latency",
            "cost_saving": "💰 Cost Saving",
            "high_quality": "⭐ High Quality",
            "high_throughput": "📈 High Throughput"
        }
        current_priority = extraction.get("priority", "balanced")
        priority_idx = priorities.index(current_priority) if current_priority in priorities else 0
        
        new_priority = st.selectbox(
            "⚡ Priority",
            priorities,
            index=priority_idx,
            format_func=lambda x: priority_labels.get(x, x),
            key="edit_priority"
        )
        
        hardware_options = ["Any GPU", "H100", "A100", "A10G", "L4", "T4"]
        current_hardware = extraction.get("hardware") or "Any GPU"
        hw_idx = hardware_options.index(current_hardware) if current_hardware in hardware_options else 0
        
        new_hardware = st.selectbox(
            "🖥️ Hardware",
            hardware_options,
            index=hw_idx,
            key="edit_hardware"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Apply Changes", type="primary", use_container_width=True, key="apply_edit"):
            st.session_state.edited_extraction = {
                "use_case": new_use_case,
                "user_count": new_user_count,
                "priority": new_priority,
                "hardware": new_hardware if new_hardware != "Any GPU" else None
            }
            st.session_state.used_priority = new_priority
            st.session_state.extraction_approved = True
            st.rerun()
    with col2:
        if st.button("🔙 Cancel", use_container_width=True, key="cancel_edit"):
            st.session_state.extraction_approved = None
            st.rerun()


def render_slo_with_approval(extraction: dict, priority: str, models_df: pd.DataFrame):
    """Render SLO section with approval to proceed to recommendations."""
    use_case = extraction.get("use_case", "chatbot_conversational")
    user_count = extraction.get("user_count", 1000)
    hardware = extraction.get("hardware")

    # SLO and Impact Cards - all 4 cards in one row
    render_slo_cards(use_case, user_count, priority, hardware)
    
    # Proceed button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True, key="generate_recs"):
            st.session_state.slo_approved = True
            st.rerun()


def render_recommendation_result(result: dict, priority: str, extraction: dict):
    """Render beautiful recommendation results with Top 5 table."""

    # Get SLO targets from result
    slo_targets = result.get("slo_targets", {})

    # === Ranked Hardware Recommendations (Backend API) ===
    # Gather data from extraction and slo_targets
    use_case = extraction.get("use_case", "chatbot_conversational")
    user_count = extraction.get("user_count", 1000)

    # Get token config and SLO targets
    token_config = slo_targets.get("token_config", {"prompt": 512, "output": 256})
    prompt_tokens = token_config.get("prompt", 512)
    output_tokens = token_config.get("output", 256)

    # Get SLO target values (use max as the target)
    ttft_target = slo_targets.get("ttft_target", {}).get("max", 200)
    itl_target = slo_targets.get("itl_target", {}).get("max", 50)
    e2e_target = slo_targets.get("e2e_target", {}).get("max", 5000)

    # Calculate expected QPS from user count (rough estimate: ~1 query per 100 users per second)
    expected_qps = max(1.0, user_count / 100.0)

    # Get current weights and include_near_miss settings from session state
    weights = {
        "accuracy": st.session_state.weight_accuracy,
        "price": st.session_state.weight_cost,
        "latency": st.session_state.weight_latency,
        "complexity": st.session_state.weight_simplicity,
    }
    include_near_miss = st.session_state.include_near_miss

    # Fetch ranked recommendations from backend
    with st.spinner("Fetching ranked recommendations from backend..."):
        ranked_response = fetch_ranked_recommendations(
            use_case=use_case,
            user_count=user_count,
            priority=priority,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            expected_qps=expected_qps,
            ttft_p95_target_ms=ttft_target,
            itl_p95_target_ms=itl_target,
            e2e_p95_target_ms=e2e_target,
            weights=weights,
            include_near_miss=include_near_miss,
        )

    if ranked_response:
        render_ranked_recommendations(ranked_response)
    else:
        st.warning("Could not fetch ranked recommendations from backend. Ensure the backend is running.")

    st.markdown("---")
    
    st.markdown('<div class="section-header"><span>🏆</span> Step 2: Top 5 Model Recommendations</div>', unsafe_allow_html=True)
    
    recommendations = result.get("recommendations", [])
    if not recommendations:
        st.warning("No recommendations found. Try adjusting your requirements.")
        return
    
    # Render Top 5 Leaderboard Table
    render_top5_table(recommendations, priority)
    
    # Winner details
    winner = recommendations[0]
    breakdown = winner.get("score_breakdown", {})
    
    st.markdown("---")
    st.markdown('<div class="section-header" style="background: linear-gradient(135deg, rgba(212, 175, 55, 0.15), rgba(166, 124, 0, 0.1)); border: 1px solid rgba(212, 175, 55, 0.2);"><span>🏆</span> Winner Details: Score Breakdown</div>', unsafe_allow_html=True)
    
    # Add explanation for the score notation - golden styled
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(212, 175, 55, 0.1), rgba(166, 124, 0, 0.05)); padding: 0.75rem 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #D4AF37;">
        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;">
            <strong style="color: #D4AF37;">📊 Score Format:</strong> <code style="background: rgba(212, 175, 55, 0.2); color: #F4E4BA; padding: 0.1rem 0.4rem; border-radius: 0.25rem;">87 → +17.4</code> means the model scored <strong>87/100</strong> in this category, contributing <strong style="color: #38ef7d;">+17.4 points</strong> to the final weighted score.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f'<h3 style="color: white; font-size: 1.8rem; font-weight: 700; margin-bottom: 1rem; background: linear-gradient(135deg, #D4AF37, #F4E4BA, #D4AF37); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; text-shadow: 0 2px 10px rgba(212, 175, 55, 0.3);">🏆 {winner.get("model_name", "Unknown")}</h3>', unsafe_allow_html=True)
        render_score_bar("Quality", "🎯", breakdown.get("quality_score", 0), "score-bar-quality", breakdown.get("quality_contribution", 0))
        render_score_bar("Latency", "⚡", breakdown.get("latency_score", 0), "score-bar-latency", breakdown.get("latency_contribution", 0))
        render_score_bar("Cost", "💰", breakdown.get("cost_score", 0), "score-bar-cost", breakdown.get("cost_contribution", 0))
        render_score_bar("Capacity", "📈", breakdown.get("capacity_score", 0), "score-bar-capacity", breakdown.get("capacity_contribution", 0))
    
    with col2:
        st.markdown('<h3 style="color: white;">🎯 Why This Model?</h3>', unsafe_allow_html=True)
        
        # Get use case from extraction context for use-case-specific summary
        use_case = extraction.get('use_case', 'chatbot_conversational')
        model_name = winner.get('model_name', 'Unknown')
        
        # Use-case specific model summaries
        use_case_context = {
            "code_completion": f"excels at real-time code suggestions with fast TTFT, ideal for IDE integrations where developer productivity depends on instant completions",
            "code_generation_detailed": f"provides detailed, well-documented code generation with strong reasoning capabilities, suitable for complex software engineering tasks",
            "chatbot_conversational": f"delivers natural, engaging conversations with consistent response quality, perfect for customer service and interactive applications",
            "translation": f"handles multilingual translation tasks with high accuracy, supporting document localization and cross-language communication",
            "content_generation": f"creates compelling marketing copy and creative content with style consistency and brand voice alignment",
            "summarization_short": f"efficiently compresses documents while preserving key insights, ideal for quick document digestion",
            "document_analysis_rag": f"excels at RAG-based document Q&A with accurate information retrieval and coherent answer synthesis",
            "long_document_summarization": f"handles long-context documents (10K+ tokens) with comprehensive summarization maintaining narrative flow",
            "research_legal_analysis": f"provides thorough analysis for legal and research documents requiring high accuracy and nuanced understanding",
        }
        
        # Model-specific traits
        model_traits = {
            "deepseek": "Known for exceptional coding and math performance at lower cost.",
            "qwen": "Top performer on reasoning benchmarks with strong multilingual support.",
            "llama": "Industry standard with excellent instruction-following.",
            "gemma": "Lightweight yet powerful, optimized for efficiency.",
            "mistral": "Efficient MoE architecture balancing speed and quality.",
            "doubao": "ByteDance model excelling in code tasks.",
            "kimi": "Moonshot AI model with strong reasoning.",
            "phi": "Microsoft's compact model for edge deployment.",
            "gpt-oss": "Open-source model benchmarked on diverse tasks.",
        }
        
        # Build use-case specific summary
        use_case_desc = use_case_context.get(use_case, "optimized for your specific task requirements")
        model_trait = ""
        for key, trait in model_traits.items():
            if key.lower() in model_name.lower():
                model_trait = trait
                break
        
        summary = f"🎯 <strong>{model_name}</strong> {use_case_desc}."
        if model_trait:
            summary += f"<br><br>🔬 <em>{model_trait}</em>"
        
        # Display model summary
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(56, 239, 125, 0.1), rgba(102, 126, 234, 0.1)); padding: 1rem; border-radius: 0.75rem; margin-bottom: 1rem; border-left: 4px solid #38ef7d;">
            <p style="color: rgba(255,255,255,0.95); margin: 0; font-size: 0.95rem; line-height: 1.6;">{summary}</p>
        </div>
        """, unsafe_allow_html=True)
        
        pros = winner.get("pros", ["⭐ Top Quality", "⚡ Fast Responses"])
        cons = winner.get("cons", [])
        
        st.markdown('<p style="color: white; font-weight: 600; margin-bottom: 0.5rem;">✅ Strengths:</p>', unsafe_allow_html=True)
        pros_html = '<div style="display: flex; flex-direction: column; gap: 0.4rem; margin-bottom: 1rem;">'
        for pro in pros:
            pros_html += f'<span class="tag tag-pro">{pro}</span>'
        pros_html += '</div>'
        st.markdown(pros_html, unsafe_allow_html=True)
        
        if cons:
            st.markdown('<p style="color: white; font-weight: 600; margin-bottom: 0.5rem;">⚠️ Trade-offs:</p>', unsafe_allow_html=True)
            cons_html = '<div style="display: flex; flex-direction: column; gap: 0.4rem;">'
            for con in cons:
                cons_html += f'<span class="tag tag-con">{con}</span>'
            cons_html += '</div>'
            st.markdown(cons_html, unsafe_allow_html=True)
        
        st.markdown('<hr style="border-color: rgba(212, 175, 55, 0.3); margin: 1rem 0;">', unsafe_allow_html=True)
        st.markdown(f"""
        <p style="color: white;"><strong style="color: #D4AF37;">🏆 Final Score:</strong> <code style="background: linear-gradient(135deg, #D4AF37, #F4E4BA); color: #1a1a2e; padding: 0.35rem 0.75rem; border-radius: 0.25rem; font-weight: 700; font-size: 1.1rem;">{winner.get('final_score', 0):.1f}/100</code></p>
        <p style="color: rgba(212, 175, 55, 0.8); font-style: italic;">Based on {priority.replace('_', ' ').title()} priority weighting</p>
        """, unsafe_allow_html=True)
    
    # Display BLIS SLO data if available (REAL benchmark data)
    blis_slo = winner.get("blis_slo")
    if blis_slo:
        st.markdown("---")
        st.markdown("""
        <div class="section-header" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(16, 185, 129, 0.1)); border: 1px solid rgba(99, 102, 241, 0.2);">
            <span>📊</span> Real BLIS Benchmark SLOs (Actual Achievable Performance)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(16, 185, 129, 0.05)); padding: 0.75rem 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #6366f1;">
            <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;">
                <strong style="color: #6366f1;">🔬 BLIS Benchmarks:</strong> These are <strong>real measured values</strong> from the BLIS simulator across 591 benchmark samples. 
                Unlike research-backed <em>targets</em>, these represent <strong style="color: #10b981;">actual achievable SLOs</strong> for this model/hardware configuration.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        slo_actual = blis_slo.get("slo_actual", {})
        throughput = blis_slo.get("throughput", {})
        token_config = blis_slo.get("token_config", {})
        hardware = blis_slo.get("hardware", "H100")
        hw_count = blis_slo.get("hardware_count", 1)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: var(--bg-card); padding: 1.25rem; border-radius: 0.75rem; border: 1px solid rgba(99, 102, 241, 0.3);">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1.5rem;">⏱️</span>
                    <span style="color: #6366f1; font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">TTFT (Time to First Token)</span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 0.4rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">Mean:</span>
                        <span style="color: #10b981; font-weight: 700; font-size: 1rem;">{slo_actual.get('ttft_mean_ms', 'N/A')}ms</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">P95:</span>
                        <span style="color: #f59e0b; font-weight: 600;">{slo_actual.get('ttft_p95_ms', 'N/A')}ms</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">P99:</span>
                        <span style="color: #f97316; font-weight: 600;">{slo_actual.get('ttft_p99_ms', 'N/A')}ms</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: var(--bg-card); padding: 1.25rem; border-radius: 0.75rem; border: 1px solid rgba(16, 185, 129, 0.3);">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1.5rem;">⚡</span>
                    <span style="color: #10b981; font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">ITL (Inter-Token Latency)</span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 0.4rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">Mean:</span>
                        <span style="color: #10b981; font-weight: 700; font-size: 1rem;">{slo_actual.get('itl_mean_ms', 'N/A')}ms</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">P95:</span>
                        <span style="color: #f59e0b; font-weight: 600;">{slo_actual.get('itl_p95_ms', 'N/A')}ms</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">P99:</span>
                        <span style="color: #f97316; font-weight: 600;">{slo_actual.get('itl_p99_ms', 'N/A')}ms</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: var(--bg-card); padding: 1.25rem; border-radius: 0.75rem; border: 1px solid rgba(245, 158, 11, 0.3);">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 1.5rem;">🏁</span>
                    <span style="color: #f59e0b; font-weight: 700; font-size: 0.9rem; text-transform: uppercase;">E2E (End-to-End)</span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 0.4rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">Mean:</span>
                        <span style="color: #10b981; font-weight: 700; font-size: 1rem;">{slo_actual.get('e2e_mean_ms', 'N/A')}ms</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">P95:</span>
                        <span style="color: #f59e0b; font-weight: 600;">{slo_actual.get('e2e_p95_ms', 'N/A')}ms</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">P99:</span>
                        <span style="color: #f97316; font-weight: 600;">{slo_actual.get('e2e_p99_ms', 'N/A')}ms</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Throughput and Config row
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
            <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(99, 102, 241, 0.05)); padding: 1rem; border-radius: 0.75rem; text-align: center; border: 1px solid rgba(139, 92, 246, 0.2);">
                <span style="font-size: 1.25rem;">🚀</span>
                <p style="color: rgba(255,255,255,0.7); margin: 0.25rem 0 0 0; font-size: 0.75rem; text-transform: uppercase;">Tokens/sec</p>
                <p style="color: #8b5cf6; font-weight: 800; font-size: 1.5rem; margin: 0;">{throughput.get('tokens_per_sec', 'N/A')}</p>
            </div>
            <div style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(14, 165, 233, 0.05)); padding: 1rem; border-radius: 0.75rem; text-align: center; border: 1px solid rgba(6, 182, 212, 0.2);">
                <span style="font-size: 1.25rem;">🖥️</span>
                <p style="color: rgba(255,255,255,0.7); margin: 0.25rem 0 0 0; font-size: 0.75rem; text-transform: uppercase;">Hardware</p>
                <p style="color: #06b6d4; font-weight: 800; font-size: 1.25rem; margin: 0;">{hardware} x{hw_count}</p>
            </div>
            <div style="background: linear-gradient(135deg, rgba(244, 114, 182, 0.1), rgba(236, 72, 153, 0.05)); padding: 1rem; border-radius: 0.75rem; text-align: center; border: 1px solid rgba(244, 114, 182, 0.2);">
                <span style="font-size: 1.25rem;">📝</span>
                <p style="color: rgba(255,255,255,0.7); margin: 0.25rem 0 0 0; font-size: 0.75rem; text-transform: uppercase;">Token Config</p>
                <p style="color: #f472b6; font-weight: 700; font-size: 1rem; margin: 0;">{token_config.get('prompt', '?')} → {token_config.get('output', '?')}</p>
            </div>
        </div>
        
        <div style="margin-top: 1rem; padding: 0.75rem; background: rgba(16, 185, 129, 0.08); border-radius: 0.5rem; border: 1px solid rgba(16, 185, 129, 0.2);">
            <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.8rem; text-align: center;">
                <strong style="color: #10b981;">📊 BLIS Samples:</strong> {blis_slo.get('benchmark_samples', 0)} benchmarks | 
                <strong style="color: #6366f1;">Model:</strong> {blis_slo.get('model_repo', 'N/A').split('/')[-1]}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # No BLIS data available for this model
        st.markdown("---")
        model_name = winner.get('model_name', 'Unknown')
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(245, 158, 11, 0.05)); padding: 1.25rem; border-radius: 0.75rem; border: 1px solid rgba(251, 191, 36, 0.3);">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">⚠️</span>
                <span style="color: #fbbf24; font-weight: 700; font-size: 1.1rem;">No BLIS Benchmark Data Available</span>
            </div>
            <p style="color: rgba(255,255,255,0.85); margin: 0 0 0.75rem 0; font-size: 0.95rem; line-height: 1.5;">
                <strong>{model_name}</strong> is not in the BLIS benchmark dataset. 
                The quality, latency, and cost scores above are derived from <strong style="color: #667eea;">Artificial Analysis</strong> benchmarks and model characteristics.
            </p>
            <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.85rem;">
                📊 <strong>BLIS models available:</strong> Qwen2.5-7B, Llama-3.1-8B, Llama-3.3-70B, Phi-4, Mistral-Small-24B, Mixtral-8x7B, Granite-3.1-8B
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_catalog_tab(models_df: pd.DataFrame):
    """Model catalog browser - shows ALL columns from the CSV."""
    st.markdown('<div class="section-header" style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(212, 175, 55, 0.1));"><span>📦</span> 206 Open-Source Model Catalog</div>', unsafe_allow_html=True)
    
    # Short project description
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(56, 239, 125, 0.05)); padding: 1rem 1.5rem; border-radius: 1rem; margin-bottom: 1.5rem; border: 1px solid rgba(102, 126, 234, 0.2);">
        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.95rem; line-height: 1.6;">
            <strong style="color: #D4AF37;">🧭 Compass Model Database:</strong> Complete benchmark data from <strong>Artificial Analysis</strong> covering 
            <span style="color: #38ef7d; font-weight: 700;">206 open-source LLMs</span> across 
            <span style="color: #667eea; font-weight: 700;">15 benchmark datasets</span> including MMLU-Pro, GPQA, IFBench, LiveCodeBench, AIME, Math-500, and more.
            Filter and search to find the perfect model for your use case.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if models_df.empty:
        st.warning("Could not load model catalog")
        return
    
    # Custom CSS for catalog tab
    st.markdown("""
    <style>
        .stMultiSelect label, .stTextInput label {
            color: white !important;
            font-weight: 600 !important;
        }
        .stMultiSelect [data-baseweb="select"] {
            background: rgba(255,255,255,0.95) !important;
        }
        .stTextInput input {
            background: rgba(255,255,255,0.95) !important;
            color: #1a1a2e !important;
        }
        /* Table header styling */
        .stDataFrame th {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            color: white !important;
            font-weight: 700 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        providers = sorted(models_df['Provider'].dropna().unique())
        selected = st.multiselect("🏢 Filter by Provider", providers)
    with col2:
        search = st.text_input("🔍 Search Models", placeholder="e.g., Llama, Qwen, DeepSeek...")
    
    filtered = models_df.copy()
    if selected:
        filtered = filtered[filtered['Provider'].isin(selected)]
    if search:
        filtered = filtered[filtered['Model Name'].str.contains(search, case=False, na=False)]
    
    is_filtered = bool(selected or search)
    shown_count = len(filtered) if is_filtered else 206
    
    st.markdown(f'<p style="color: white; font-size: 1.1rem;">📊 Showing <strong style="color: #38ef7d;">{shown_count}</strong> of <strong style="color: #38ef7d;">206</strong> models</p>', unsafe_allow_html=True)
    
    # Show ALL columns from the CSV
    all_benchmark_cols = ['mmlu_pro', 'gpqa', 'ifbench', 'livecodebench', 'aime', 'aime_25', 
                          'math_500', 'artificial_analysis_intelligence_index', 
                          'artificial_analysis_coding_index', 'artificial_analysis_math_index',
                          'lcr', 'tau2', 'scicode', 'hle', 'terminalbench_hard']
    
    # Core columns + all available benchmarks
    display_cols = ['Model Name', 'Provider'] + [c for c in all_benchmark_cols if c in filtered.columns]
    available = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[available].copy()
    
    # Sort by mmlu_pro descending
    if 'mmlu_pro' in display_df.columns:
        def parse_pct(x):
            if pd.isna(x) or x == 'N/A':
                return 0
            if isinstance(x, str):
                return float(x.replace('%', '')) / 100 if '%' in x else float(x)
            return float(x)
        display_df['_sort'] = display_df['mmlu_pro'].apply(parse_pct)
        display_df = display_df.sort_values('_sort', ascending=False)
        display_df = display_df.drop(columns=['_sort'])
    
    # Column icons for better readability
    column_icons = {
        'Model Name': '🤖 Model',
        'Provider': '🏢 Provider',
        'mmlu_pro': '🎯 MMLU-Pro',
        'gpqa': '📚 GPQA', 
        'ifbench': '💬 IFBench',
        'livecodebench': '💻 LiveCode',
        'aime': '🧮 AIME',
        'aime_25': '🧮 AIME-25',
        'math_500': '📐 Math-500',
        'artificial_analysis_intelligence_index': '🧠 Intel-Idx',
        'artificial_analysis_coding_index': '💻 Code-Idx',
        'artificial_analysis_math_index': '📐 Math-Idx',
        'lcr': '📖 LCR',
        'tau2': '🌊 TAU2',
        'scicode': '🔬 SciCode',
        'hle': '🎓 HLE',
        'terminalbench_hard': '🖥️ Terminal'
    }
    display_df = display_df.rename(columns={k: v for k, v in column_icons.items() if k in display_df.columns})
    
    st.dataframe(display_df.head(100), use_container_width=True, hide_index=True, height=500)
    
    # Artificial Analysis-style Benchmark Leaderboards
    st.markdown("""
    <div style="margin-top: 2rem;">
        <h3 style="color: white; margin-bottom: 0.5rem; font-weight: 700;">📊 Benchmark Leaderboards</h3>
        <p style="color: #9ca3af; font-size: 0.9rem; margin-bottom: 1.5rem;">Research-grade benchmarks across various domains — Data from <strong style="color: #a855f7;">Artificial Analysis</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for benchmark charts (like Artificial Analysis layout)
    bench_col1, bench_col2 = st.columns(2)
    
    # Helper function to create AA-style bar chart
    def create_benchmark_chart(df, col_name, title, color_scale):
        if col_name not in df.columns:
            return None
        chart_df = df[['Model Name', col_name]].dropna().head(15)
        if chart_df.empty:
            return None
        
        # Parse percentage values
        def parse_val(x):
            if pd.isna(x) or x == 'N/A':
                return 0
            if isinstance(x, str):
                return float(x.replace('%', '')) if '%' in x else float(x) * 100
            return float(x) * 100 if float(x) <= 1 else float(x)
        
        chart_df['score'] = chart_df[col_name].apply(parse_val)
        chart_df = chart_df.sort_values('score', ascending=True).tail(12)
        
        fig = px.bar(
            chart_df,
            x='score',
            y='Model Name',
            orientation='h',
            color='score',
            color_continuous_scale=color_scale,
            text=chart_df['score'].apply(lambda x: f'{x:.0f}%')
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=False,
            coloraxis_showscale=False,
            title=dict(text=title, font=dict(size=14, color='white')),
            xaxis=dict(title='', showgrid=False, range=[0, 100]),
            yaxis=dict(title='', showgrid=False),
            margin=dict(l=10, r=10, t=40, b=10),
            height=350
        )
        fig.update_traces(textposition='outside', textfont_size=10, textfont_color='white')
        return fig
    
    with bench_col1:
        # MMLU-Pro Chart
        fig = create_benchmark_chart(models_df, 'mmlu_pro', '🎯 MMLU-Pro (Knowledge)', ['#3b82f6', '#1d4ed8', '#1e40af'])
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="mmlu_chart")
        
        # LiveCodeBench Chart
        fig = create_benchmark_chart(models_df, 'livecodebench', '💻 LiveCodeBench (Coding)', ['#10b981', '#059669', '#047857'])
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="code_chart")
    
    with bench_col2:
        # GPQA Chart
        fig = create_benchmark_chart(models_df, 'gpqa', '📚 GPQA Diamond (Science)', ['#8b5cf6', '#7c3aed', '#6d28d9'])
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="gpqa_chart")
        
        # IFBench Chart
        fig = create_benchmark_chart(models_df, 'ifbench', '💬 IFBench (Instruction Following)', ['#f59e0b', '#d97706', '#b45309'])
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="if_chart")
    
    # Provider Chart with custom colors
    st.markdown('<h4 style="color: white; margin-top: 2rem;">🏢 Models by Provider</h4>', unsafe_allow_html=True)
    counts = models_df['Provider'].value_counts().head(10)
    
    # Create a colored bar chart using plotly for better visibility
    fig = px.bar(
        x=counts.index, 
        y=counts.values,
        color=counts.values,
        color_continuous_scale=['#667eea', '#764ba2', '#f093fb', '#38ef7d'],
        labels={'x': 'Provider', 'y': 'Number of Models', 'color': 'Count'}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(tickfont=dict(color='white', size=12)),
        yaxis=dict(tickfont=dict(color='white', size=12), gridcolor='rgba(255,255,255,0.1)'),
        margin=dict(l=40, r=40, t=20, b=60)
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


def render_how_it_works_tab():
    """Documentation tab with collapsible sections."""
    st.markdown('<div class="section-header" style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(212, 175, 55, 0.1));"><span>📖</span> How Compass Works</div>', unsafe_allow_html=True)
    
    # Collapsible pipeline section
    with st.expander("🔄 **E2E Pipeline Visualization** - Click to expand", expanded=False):
        render_pipeline()
    
    # Styled tables for visibility on dark background
    st.markdown("""
    <style>
        .doc-table { width: 100%; border-collapse: collapse; margin: 1.5rem 0; }
        .doc-table th { background: rgba(102, 126, 234, 0.3); color: white; padding: 1rem; text-align: left; font-weight: 600; }
        .doc-table td { background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.9); padding: 0.75rem 1rem; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .doc-table tr:hover td { background: rgba(102, 126, 234, 0.15); }
        .doc-section { color: white; font-size: 1.5rem; font-weight: 700; margin: 2rem 0 1rem 0; }
        .doc-formula { background: rgba(102, 126, 234, 0.2); color: #38ef7d; padding: 1rem; border-radius: 0.5rem; font-family: monospace; margin: 1rem 0; }
    </style>
    
    <div class="doc-section">🎯 Supported Use Cases (9 Total)</div>
    <table class="doc-table">
        <tr><th>Use Case</th><th>Description</th><th>Key Benchmarks</th><th>Typical SLO</th></tr>
        <tr><td>💬 Chatbot Conversational</td><td>Customer service, virtual assistants, Q&A bots</td><td>MMLU, IFBench</td><td>TTFT &lt; 150ms</td></tr>
        <tr><td>💻 Code Completion</td><td>IDE autocomplete, real-time code suggestions</td><td>LiveCodeBench</td><td>TTFT &lt; 100ms</td></tr>
        <tr><td>🔧 Code Generation</td><td>Full code generation, detailed implementations</td><td>LiveCodeBench, GPQA</td><td>TTFT &lt; 200ms</td></tr>
        <tr><td>📄 Document RAG</td><td>Retrieval-augmented Q&A, knowledge base search</td><td>GPQA, LCR</td><td>E2E &lt; 1000ms</td></tr>
        <tr><td>📝 Short Summarization</td><td>News, articles, brief documents (&lt;10 pages)</td><td>Tau2, LCR</td><td>E2E &lt; 1500ms</td></tr>
        <tr><td>📚 Long Doc Summarization</td><td>Reports, books, chapters (10+ pages)</td><td>LCR, Tau2</td><td>E2E &lt; 5000ms</td></tr>
        <tr><td>🌐 Translation</td><td>Multi-language text translation</td><td>MMLU, IFBench</td><td>TTFT &lt; 200ms</td></tr>
        <tr><td>✍️ Content Generation</td><td>Creative writing, marketing content</td><td>IFBench, MMLU</td><td>TTFT &lt; 300ms</td></tr>
        <tr><td>⚖️ Legal/Research Analysis</td><td>Complex legal & research document analysis</td><td>GPQA, MMLU</td><td>Quality &gt; Speed</td></tr>
    </table>
    
    <div class="doc-section">⚖️ MCDM Scoring Formula</div>
    <div class="doc-formula">FINAL_SCORE = w_quality × Quality + w_latency × Latency + w_cost × Cost + w_capacity × Capacity</div>
    
    <p style="color: white; font-weight: 600; margin: 1rem 0;">Priority-based weight adjustment:</p>
    <table class="doc-table">
        <tr><th>Priority</th><th>Quality</th><th>Latency</th><th>Cost</th><th>Capacity</th></tr>
        <tr><td>⚖️ Balanced</td><td>30%</td><td>25%</td><td>25%</td><td>20%</td></tr>
        <tr><td>⚡ Low Latency</td><td>20%</td><td style="color: #38ef7d; font-weight: 700;">45%</td><td>15%</td><td>20%</td></tr>
        <tr><td>💰 Cost Saving</td><td>20%</td><td>15%</td><td style="color: #38ef7d; font-weight: 700;">50%</td><td>15%</td></tr>
        <tr><td>⭐ High Quality</td><td style="color: #38ef7d; font-weight: 700;">50%</td><td>20%</td><td>15%</td><td>15%</td></tr>
        <tr><td>📈 High Throughput</td><td>20%</td><td>15%</td><td>15%</td><td style="color: #38ef7d; font-weight: 700;">50%</td></tr>
    </table>
    
    <div class="doc-section">📊 How Factors Affect Scoring</div>
    <table class="doc-table">
        <tr><th>Factor</th><th>Impact on Recommendation</th><th>Example</th></tr>
        <tr><td><strong>🎯 Use Case</strong></td><td>Models are ranked by use-case-specific benchmarks from our 206-model evaluation. <span style="color: #38ef7d;">Higher-ranked models for your use case get better Quality scores.</span></td><td>Code Completion → LiveCodeBench weighted heavily</td></tr>
        <tr><td><strong>👥 User Count</strong></td><td>High user counts increase importance of Capacity & Latency. <span style="color: #38ef7d;">More users = need for faster, scalable models.</span></td><td>10K users → Capacity weight +15%</td></tr>
        <tr><td><strong>🖥️ Hardware</strong></td><td>GPU type affects Cost & Throughput calculations. <span style="color: #38ef7d;">Premium GPUs enable larger models.</span></td><td>H100 → Can run 70B+ models efficiently</td></tr>
        <tr><td><strong>⚡ Priority</strong></td><td>Dynamically shifts MCDM weight distribution. <span style="color: #38ef7d;">Your priority becomes the dominant factor (45-50%).</span></td><td>"Cost Saving" → Cost weight = 50%</td></tr>
    </table>
    
    <div class="doc-section">🔬 Use-Case Quality Scoring</div>
    <p style="color: rgba(255,255,255,0.9); line-height: 1.8; margin-bottom: 1rem;">
        Each use case has a dedicated <strong style="color: #38ef7d;">Weighted Scores CSV</strong> (e.g., <code style="background: rgba(255,255,255,0.1); padding: 0.2rem 0.4rem; border-radius: 0.25rem;">opensource_chatbot_conversational.csv</code>) 
        that ranks all 206 models based on relevant benchmarks for that task:
    </p>
    <table class="doc-table">
        <tr><th>Use Case</th><th>Primary Benchmarks</th><th>Top Model (Example)</th></tr>
        <tr><td>💬 Chatbot</td><td>MMLU Pro, IFBench, GPQA</td><td>Kimi K2 Thinking (64.6%)</td></tr>
        <tr><td>💻 Code Completion</td><td>LiveCodeBench, GPQA</td><td>Doubao Seed Code (72.1%)</td></tr>
        <tr><td>🌐 Translation</td><td>MMLU, IFBench</td><td>Kimi K2 Thinking (63.8%)</td></tr>
        <tr><td>✍️ Content Gen</td><td>IFBench, MMLU Pro</td><td>Kimi K2 Thinking (61.4%)</td></tr>
    </table>
    <p style="color: rgba(255,255,255,0.7); font-style: italic; margin-top: 1rem;">
        📈 The use-case quality score becomes the "Quality" component in the MCDM formula, ensuring models best suited for your task rank highest.
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
