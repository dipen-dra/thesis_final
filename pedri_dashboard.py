






# import os
# import base64
# from typing import Optional, Tuple

# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import percentileofscore

# from mplsoccer import Pitch
# from sklearn.cluster import KMeans

# # =======================================
# # STREAMLIT CONFIG
# # =======================================
# st.set_page_config(
#     page_title="Pedri Spatial Intelligence Dashboard",
#     page_icon="⚽",
#     layout="wide",
# )

# PRIMARY_BG = "#f0f2ff"
# FOOTBALL_GREEN = "#28a745"
# st.markdown(
#     f"""
#     <style>
#         /* =============================
#            GLOBAL APP SETTINGS
#         ============================== */
#         .stApp {{
#             background-color: {PRIMARY_BG} !important;
#             color: #111827 !important;
#         }}
#         .block-container {{
#             max-width: 1200px !important; margin: auto !important;
#             padding-top: 2rem !important; padding-bottom: 3rem !important;
#         }}
#         /* =============================
#            VISIBILITY FIXES FOR ALL TEXT
#         ============================== */
#         h1, h2, h3, h4, h5, h6 {{
#             color: #1f2937 !important;
#         }}
#         div[data-testid="stMarkdown"] label {{
#             color: #374151 !important;
#         }}
#         div[data-testid="stExpander"] {{
#              border-color: #d1d5db !important;
#         }}
#         div[data-testid="stAlert"] * {{ color: #31333F !important; }}
#         /* =============================
#            PLAYER IMAGE & HERO CARD
#         ============================== */
#         .player-image {{
#             width: 160px; height: 160px; border-radius: 50%; object-fit: cover;
#             border: 4px solid white; box-shadow: 0 12px 35px rgba(15, 23, 42, 0.25);
#         }}
#         .hero-card {{
#             background: white; border-radius: 22px; padding: 1.5rem 1.75rem;
#             box-shadow: 0 15px 40px rgba(15, 23, 42, 0.12); border: 1px solid #e5e7eb;
#         }}
#         .hero-name {{ font-size: 1.8rem; font-weight: 800; margin-bottom: 0.25rem; color: #111827; }}
#         .hero-subtitle {{ font-size: 0.98rem; color: #4b5563; margin-bottom: 0.5rem; }}
#         .hero-tag {{
#             display: inline-block; padding: 0.25rem 0.65rem; border-radius: 999px;
#             font-size: 0.75rem; font-weight: 600; background: rgba(40,167,69,0.08);
#             color: {FOOTBALL_GREEN}; margin-right: 0.4rem;
#         }}
#         /* =============================
#            KPI METRIC CARDS
#         ============================== */
#         .metric-card {{
#             background: white; border-radius: 18px; padding: 1rem 1.25rem;
#             box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08); border: 1px solid #e5e7eb;
#             text-align: center; transition: 0.2s ease-in-out;
#         }}
#         .metric-card:hover {{ transform: translateY(-4px); box-shadow: 0 14px 36px rgba(15, 23, 42, 0.18); }}
#         .metric-label {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7280; margin-bottom: 0.15rem; }}
#         .metric-value {{ font-size: 1.3rem; font-weight: 700; color: #111827; }}
#         .metric-sub {{ font-size: 0.78rem; color: #9ca3af; margin-top: 0.1rem; }}
#         /* =============================
#            SECTION TITLES
#         ============================== */
#         .section-title {{ font-size: 1.1rem; font-weight: 700; margin-top: 1.5rem; margin-bottom: 0.3rem; color: #111827; }}
#         .section-caption {{ font-size: 0.85rem; color: #6b7280; margin-bottom: 0.75rem; }}
#         /* =============================
#            AI INSIGHT CARD
#         ============================== */
#         .insight-card {{
#             background: white; border-radius: 18px; padding: 1.25rem 1.5rem;
#             box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08); border: 1px solid #e5e7eb;
#         }}
#         .insight-title {{
#             font-size: 0.9rem; font-weight: 600; text-transform: uppercase;
#             color: {FOOTBALL_GREEN}; letter-spacing: 0.15em; margin-bottom: 0.5rem;
#         }}
#         .insight-body {{ font-size: 0.95rem; color: #111827; line-height: 1.6; }}
#         .insight-body h4 {{ font-size: 1rem; color: #1f2937; margin-top: 1rem; margin-bottom: 0.5rem;}}
#         /* =============================
#            STREAMLIT TABS
#         ============================== */
#         .stTabs [role="tab"] {{
#             background-color: white !important; color: #374151 !important; border-radius: 999px !important;
#             padding: 6px 18px !important; border: 1px solid #d1d5db !important;
#             font-weight: 600 !important; transition: all 0.2s ease-in-out !important;
#         }}
#         .stTabs [role="tab"][aria-selected="true"] {{
#             background-color: rgba(40,167,69,0.15) !important; color: #1f2937 !important;
#             border-color: #28a745 !important; font-weight: 700 !important;
#         }}
#         .stTabs [role="tab"]:hover {{ background-color: rgba(40,167,69,0.10) !important; color: #1f2937 !important; }}
#         /* =============================
#            CUSTOM PERCENTILE BARS
#         ============================== */
#         .percentile-container {{ display: flex; align-items: center; margin-bottom: 4px; height: 24px; }}
#         .percentile-label {{ width: 220px; font-size: 0.8rem; padding-right: 10px; text-align: right; color: #4b5563;}}
#         .percentile-bar-bg {{ flex-grow: 1; background-color: #e9ecef; border-radius: 5px; height: 18px; }}
#         .percentile-bar-fg {{
#             height: 100%; border-radius: 5px; text-align: right; color: white;
#             font-size: 0.75rem; padding-right: 5px; line-height: 18px; font-weight: 600;
#         }}
#         .percentile-value {{ width: 50px; text-align: right; font-size: 0.8rem; font-weight: 600; padding-left: 10px; color: #374151; }}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# # =======================================
# # UTILS (UNCHANGED)
# # =======================================
# def load_image_base64(path: str) -> str:
#     with open(path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode()

# def get_xt_column(df: pd.DataFrame) -> Optional[str]:
#     candidates = ["xT", "xt", "xT_value", "xT_impact", "expected_threat"]
#     for c in candidates:
#         if c in df.columns: return c
#     return None

# def get_xy_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
#     candidates_x = ["x", "X", "pos_x", "start_x"]
#     candidates_y = ["y", "Y", "pos_y", "start_y"]
#     x_col = next((c for c in candidates_x if c in df.columns), None)
#     y_col = next((c for c in candidates_y if c in df.columns), None)
#     return x_col, y_col

# def compute_event_kpis(df: pd.DataFrame) -> dict:
#     kpis = {}
#     kpis["Total Actions"] = len(df)
#     xt_col = get_xt_column(df)
#     kpis["Avg. xT Impact"] = float(df[xt_col].mean()) if xt_col and df[xt_col].notna().any() else None
#     pass_mask = df["event_type"].astype(str).str.contains("pass", case=False, na=False)
#     total_passes = pass_mask.sum()
#     if total_passes > 0:
#         if "outcome" in df.columns:
#             completed = df.loc[pass_mask, "outcome"].astype(str).str.contains("success|complete|accurate|on_target", case=False, na=False).sum()
#         elif "is_complete" in df.columns:
#             completed = df.loc[pass_mask, "is_complete"].sum()
#         else:
#             completed = None
#         kpis["Pass Accuracy (%)"] = 100 * completed / total_passes if completed is not None else None
#     else:
#         kpis["Pass Accuracy (%)"] = None
#     if "is_progressive" in df.columns:
#         kpis["Progressive Actions"] = int(df["is_progressive"].sum())
#     elif "end_x" in df.columns and "x" in df.columns:
#         kpis["Progressive Actions"] = int((df["end_x"] - df["x"] >= 15).sum())
#     else:
#         kpis["Progressive Actions"] = 0
#     if "is_recovery" in df.columns:
#         kpis["Ball Recoveries"] = int(df["is_recovery"].sum())
#     elif "event_type" in df.columns:
#         kpis["Ball Recoveries"] = int(df["event_type"].astype(str).str.contains("recovery", case=False, na=False).sum())
#     else:
#         kpis["Ball Recoveries"] = 0
#     kpis["Dribbles"] = int(df["event_type"].astype(str).str.contains("dribble", case=False, na=False).sum()) if "event_type" in df.columns else 0
#     return kpis

# # =======================================
# # NEW: ADVANCED AI INSIGHT FUNCTION
# # =======================================
# def ai_insight_event(df: pd.DataFrame, kpis: dict, seasonal_df: pd.DataFrame) -> str:
#     # --- 1. Get Pedri's seasonal averages as a benchmark ---
#     try:
#         pedri_avg = seasonal_df[seasonal_df['player'] == 'Pedri'].iloc[0]
#     except (IndexError, KeyError):
#         return "Could not find Pedri's seasonal data in player_comparison_data.csv to generate a comparative insight."

#     lines = []

#     # --- 2. Overall Performance Headline ---
#     match_xt = kpis.get("Avg. xT Impact", 0)
#     avg_xt = pedri_avg.get('avg_xt_impact', 0)
    
#     headline = ""
#     if match_xt > avg_xt * 1.3: headline = "🔥 **Exceptional Creative Performance:** A standout match where he significantly exceeded his usual creative output."
#     elif match_xt > avg_xt * 1.1: headline = "🟢 **Strong Attacking Influence:** An above-average performance in terms of creating threat and influencing the final third."
#     elif match_xt < avg_xt * 0.7: headline = "🧊 **Subdued Creative Role:** A quieter game where his creative impact was notably below his seasonal average, suggesting a different tactical role."
#     else: headline = "🟡 **Solid & Controlled Performance:** A typical display, balancing creative duties with maintaining team structure, in line with his seasonal averages."
#     lines.append(headline)

#     # --- 3. Detailed Breakdown vs. Seasonal Average ---
#     lines.append("<h4>Detailed Performance Analysis</h4>")

#     # Creative Impact (xT)
#     xt_diff = ((match_xt / avg_xt) - 1) * 100 if avg_xt > 0 else 0
#     xt_eval = "well above" if xt_diff > 25 else "above" if xt_diff > 10 else "well below" if xt_diff < -25 else "below" if xt_diff < -10 else "in line with"
#     lines.append(f" • **Creative Impact:** His average xT of **{match_xt:.2f}** in this match was **{xt_eval}** his seasonal average of {avg_xt:.2f} ({xt_diff:+.0f}%).")

#     # Ball Progression
#     match_prog = kpis.get("Progressive Actions", 0)
#     avg_prog = pedri_avg.get('progressive_passes_p90', 0) + pedri_avg.get('progressive_carries_p90', 0)
#     prog_diff = ((match_prog / avg_prog) - 1) * 100 if avg_prog > 0 else 0
#     prog_eval = "a significantly higher" if prog_diff > 20 else "a higher" if prog_diff > 10 else "a lower" if prog_diff < -10 else "a similar"
#     lines.append(f" • **Ball Progression:** He completed **{match_prog}** progressive actions, indicating **{prog_eval}** volume of forward carries and passes compared to his average of {avg_prog:.1f} per 90.")

#     # Defensive Contribution
#     match_recov = kpis.get("Ball Recoveries", 0)
#     # Note: Recoveries aren't in seasonal data, so we create a realistic benchmark based on his tackles + interceptions.
#     avg_def = pedri_avg.get('successful_tackles_p90', 0) + pedri_avg.get('interceptions_p90', 0)
#     avg_recov_benchmark = avg_def * 2.5 # Analytical assumption: recoveries are multiples of direct defensive actions
#     def_eval = "high level" if match_recov > avg_recov_benchmark * 1.2 else "reduced" if match_recov < avg_recov_benchmark * 0.8 else "solid"
#     lines.append(f" • **Defensive Workrate:** With **{match_recov}** ball recoveries, he showed a **{def_eval}** level of defensive engagement in this match.")
    
#     # --- 4. Tactical Role Interpretation (from K-Means) ---
#     lines.append("<h4>Tactical Role Interpretation</h4>")
#     x_col, y_col = get_xy_columns(df)
#     if x_col and y_col:
#         coords = df[[x_col, y_col]].dropna()
#         if len(coords) >= 25:
#             kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(coords)
#             avg_x = kmeans.cluster_centers_[:, 0].mean()
#             if avg_x < 45: role = "a deeper **#6 Deep-Lying Playmaker** role, focusing on build-up."
#             elif avg_x < 75: role = "a classic **#8 Hybrid Midfielder** role, connecting defense and attack."
#             else: role = "an advanced **#10 Attacking Midfielder** role, operating primarily in the final third."
#             lines.append(f" • **Positional Analysis:** K-Means clustering of his on-ball actions reveals an average x-position of **{avg_x:.1f}**. This suggests he adopted {role}")
    
#     # --- 5. Final Synthesis ---
#     synthesis = ""
#     if xt_diff > 20 and prog_diff > 15:
#         synthesis = "This was a dominant, high-volume performance where he was both the primary progressor and creator."
#     elif xt_diff > 20 and prog_diff < 0:
#         synthesis = "This performance was defined by efficiency over volume. He may have had fewer touches but was highly effective in creating danger when on the ball."
#     elif xt_diff < -20 and prog_diff > 15:
#         synthesis = "The data points to a 'Workhorse' performance. While his final-third creativity was limited, his main duty was evidently to progress the ball relentlessly through the midfield."
#     else:
#         synthesis = "This was a balanced performance, fulfilling his tactical instructions without deviating significantly from his established player profile."
#     lines.append(f" • **Overall Verdict:** {synthesis}")

#     return "\n".join(lines)


# # =======================================
# # PITCH DRAWING, HEADER, etc. (UNCHANGED)
# # =======================================
# # All your other functions from the previous final code remain exactly the same.
# # I am including them all here for a complete, copy-paste ready script.

# def draw_event_heatmap(df: pd.DataFrame, title: str):
#     x_col, y_col = get_xy_columns(df)
#     if not x_col or not y_col or df[[x_col, y_col]].dropna().empty: return
#     pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
#     fig, ax = pitch.draw(figsize=(6, 8)); sns.kdeplot(x=df[x_col], y=df[y_col], fill=True, cmap="inferno", alpha=0.9, ax=ax, thresh=0.05, bw_adjust=0.45)
#     ax.set_title(title, color="white", fontsize=12, fontweight="bold"); st.pyplot(fig)

# def draw_passing_network(df: pd.DataFrame, title: str):
#     x_col, y_col = get_xy_columns(df)
#     if not x_col or not y_col or "end_x" not in df.columns or "end_y" not in df.columns: return
#     mask = df["event_type"].astype(str).str.contains("pass", case=False, na=False)
#     if "is_complete" in df.columns: mask &= df["is_complete"] == 1
#     passes = df.loc[mask].dropna(subset=[x_col, y_col, "end_x", "end_y"])
#     if passes.empty: return
#     pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
#     fig, ax = pitch.draw(figsize=(6, 8))
#     pitch.lines(passes[x_col], passes[y_col], passes["end_x"], passes["end_y"], ax=ax, comet=True, color="gold", lw=1.2, alpha=0.7)
#     ax.set_title(title, color="white", fontsize=12, fontweight="bold"); st.pyplot(fig)

# def draw_ball_progression(df: pd.DataFrame, title: str):
#     x_col, y_col = get_xy_columns(df)
#     if not x_col or not y_col or "end_x" not in df.columns or "end_y" not in df.columns: return
#     data = df[df["event_type"].astype(str).str.contains("pass|carry|dribble", case=False, na=False)].copy()
#     xt_col = get_xt_column(data)
#     if xt_col and data[xt_col].notna().any(): data = data[data[xt_col] >= data[xt_col].quantile(0.75)]
#     data.dropna(subset=[x_col, y_col, "end_x", "end_y"], inplace=True)
#     if data.empty: return
#     pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
#     fig, ax = pitch.draw(figsize=(6, 8))
#     pitch.arrows(data[x_col], data[y_col], data["end_x"], data["end_y"], ax=ax, width=1.2, headwidth=2.5, headlength=2.5, color="#ffdd57", alpha=0.9)
#     ax.set_title(title, color="white", fontsize=12, fontweight="bold"); st.pyplot(fig)

# def render_header():
#     st.markdown('<p style="font-size:0.85rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.20em;">FC Barcelona • 2024-25 La Liga</p>', unsafe_allow_html=True)
#     col_img, col_info = st.columns([1, 2], gap="large")
#     with col_img:
#         if os.path.exists("pedri.png"):
#             img64 = load_image_base64("pedri.png")
#             st.markdown(f'<div style="display:flex; justify-content:center; align-items:center;"><img src="data:image/png;base64,{img64}" class="player-image" /></div>', unsafe_allow_html=True)
#     with col_info:
#         st.markdown("""
#             <div class="hero-card">
#                 <div class="hero-name">Pedro González López</div>
#                 <div class="hero-subtitle">FC Barcelona • Midfielder • #8</div>
#                 <div style="margin-bottom:0.6rem;"><span class="hero-tag">Positional Play Engine</span><span class="hero-tag">Left Half-Space Creator</span></div>
#                 <div style="font-size:0.9rem; color:#4b5563;">Premium spatial-intelligence dashboard for match-by-match heatmap analysis, role detection, and seasonal benchmarking of Pedri’s influence.</div>
#             </div>""", unsafe_allow_html=True)
#     st.markdown("<br/>", unsafe_allow_html=True)
#     st.markdown('<div class="section-title">2024-25 Season Snapshot (Static)</div>', unsafe_allow_html=True)
#     cols = st.columns(6)
#     metrics_data = [("Matches Played", "34", "La Liga"), ("Goals", "7", "Box arrivals"), ("Assists", "9", "Final ball"), ("Avg. xT / 90", "0.18", "Threat impact"), ("Prog Actions / 90", "11.1", "Carries & passes"), ("Tkl+Int / 90", "2.5", "Defensive work")]
#     for col, (label, value, sub) in zip(cols, metrics_data):
#         with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

# def render_match_analyzer():
#     st.markdown('<div class="section-title">Match Analyzer</div><div class="section-caption">Upload a single-match event CSV to generate a dynamic, AI-powered performance evaluation.</div>', unsafe_allow_html=True)
    
#     # --- MODIFICATION: Load seasonal data for context ---
#     seasonal_df = None
#     if os.path.exists("player_comparison_data.csv"):
#         seasonal_df = pd.read_csv("player_comparison_data.csv")

#     uploaded = st.file_uploader("Upload Match Data", type=["csv"], label_visibility="collapsed")
    
#     if uploaded is None:
#         st.info("Upload a match file to start the analysis."); return
    
#     try: df = pd.read_csv(uploaded)
#     except Exception as e:
#         st.error(f"Could not read CSV file: {e}"); return
    
#     mode = "event" if {"event_type", "end_x"}.issubset(df.columns) else None # Simplified to only support event data for this advanced insight
    
#     if mode != "event":
#         st.error("The AI Insight feature currently only supports Event Data CSVs. Please upload a valid file."); return
        
#     kpis = compute_event_kpis(df)
#     labels = ["Total Actions", "Avg. xT Impact", "Pass Accuracy (%)", "Progressive Actions", "Ball Recoveries", "Dribbles"]
#     cols = st.columns(len(labels))
#     for col, label in zip(cols, labels):
#         val = kpis.get(label); disp = f"{val:.2f}" if isinstance(val, (float, np.floating)) and not np.isnan(val) else str(val) if val is not None else "–"
#         with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{disp}</div></div>', unsafe_allow_html=True)
    
#     tab_plots, tab_insight = st.tabs(["📊 Match Plots", "🧠 AI Performance Report"])

#     with tab_plots:
#         c1, c2, c3 = st.columns(3)
#         with c1: draw_event_heatmap(df, "On-ball Action Density")
#         with c2: draw_passing_network(df, "Completed Passing Network")
#         with c3: draw_ball_progression(df, "High-Value Ball Progression")

#     with tab_insight:
#         if seasonal_df is not None:
#             insight_html = ai_insight_event(df, kpis, seasonal_df).replace("\n", "<br/>")
#             st.markdown(f'<div class="insight-card"><div class="insight-title">AI MATCH ANALYSIS vs. 24-25 SEASON</div><div class="insight-body">{insight_html}</div></div>', unsafe_allow_html=True)
#         else:
#             st.warning("`player_comparison_data.csv` not found. AI insights will be limited without seasonal benchmarks.")

# # All the functions for the Advanced Analytics Board remain the same
# def get_percentile_color(p):
#     if p > 90: return "#059669";
#     if p > 75: return "#10B981";
#     if p > 50: return "#F59E0B";
#     if p > 25: return "#F97316";
#     return "#EF4444"

# def draw_percentile_profile(player_data: pd.Series, all_data: pd.DataFrame, metrics: list):
#     st.markdown(f"<h4 style='font-size: 1.1rem; text-align: center; margin-bottom: 1rem; color: #1f2937;'>{player_data['player']}</h4>", unsafe_allow_html=True)
#     for metric in metrics:
#         player_value = player_data[metric]
#         percentile = percentileofscore(all_data[metric], player_value)
#         bar_color = get_percentile_color(percentile)
#         label = metric.replace('_', ' ').replace(' p90', '').title()
#         st.markdown(f'''
#             <div class="percentile-container">
#                 <div class="percentile-label">{label}</div>
#                 <div class="percentile-bar-bg">
#                     <div class="percentile-bar-fg" style="width: {percentile}%; background-color: {bar_color};">{int(percentile)}</div>
#                 </div>
#                 <div class="percentile-value">{player_value:.2f}</div>
#             </div>''', unsafe_allow_html=True)

# def draw_radar_chart(df: pd.DataFrame, metrics: list, title: str):
#     labels = [m.replace('_', ' ').replace(' p90', '').title() for m in metrics]
#     num_vars = len(labels)
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
#     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
#     df_norm = df.copy()
#     for m in metrics:
#         min_val, max_val = all_data_for_radar[m].min(), all_data_for_radar[m].max()
#         df_norm[m] = (df_norm[m] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
#     for _, row in df_norm.iterrows():
#         values = row[metrics].tolist() + [row[metrics].tolist()[0]]
#         ax.plot(angles, values, linewidth=1.5, label=row['player'])
#         ax.fill(angles, values, alpha=0.15)
#     ax.set_yticklabels([]); ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
#     ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
#     ax.set_title(title, size=12, y=1.12); st.pyplot(fig)

# def draw_scatter_plot(df: pd.DataFrame, x_metric: str, y_metric: str, title: str, xlabel: str, ylabel: str):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.scatterplot(data=df, x=x_metric, y=y_metric, hue="player", s=120, palette="viridis", legend=False, ax=ax)
#     for _, row in df.iterrows():
#         ax.text(row[x_metric], row[y_metric], row['player'], fontdict={'size': 9}, ha='left', va='bottom')
#     ax.set_title(title, fontsize=12, fontweight='bold'); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
#     ax.grid(True, which='both', linestyle='--', linewidth=0.3); st.pyplot(fig)

# all_data_for_radar = pd.DataFrame()

# def render_season_comparison():
#     global all_data_for_radar
#     st.markdown('<div class="section-title">Advanced Midfielder Analytics Board</div><div class="section-caption">Benchmark elite midfielders from the 2024-25 season. Analyze player profiles, compare KPIs, and identify tactical archetypes.</div>', unsafe_allow_html=True)
#     csv_path = "player_comparison_data.csv"
#     if not os.path.exists(csv_path):
#         st.error(f"`{csv_path}` not found."); return
#     try:
#         df = pd.read_csv(csv_path)
#         all_data_for_radar = df.copy()
#     except Exception as e:
#         st.error(f"Could not read `{csv_path}`: {e}"); return
#     players = sorted(df["player"].unique().tolist())
#     st.markdown("---")
#     st.markdown("<h3 style='font-size: 1.1rem;'>Player Profile Analysis</h3>", unsafe_allow_html=True)
#     profile_players_selected = st.multiselect("Select players to view their detailed percentile profile (2-3 recommended):", players, default=["Pedri", "Vitinha"] if "Pedri" in players and "Vitinha" in players else players[:2])
#     if profile_players_selected:
#         cols = st.columns(len(profile_players_selected))
#         for i, player_name in enumerate(profile_players_selected):
#             with cols[i]:
#                 player_data = df[df['player'] == player_name].iloc[0]
#                 profile_metrics = ['avg_xt_impact', 'key_passes_p90', 'non_penalty_xg_p90', 'shots_p90', 'progressive_passes_p90', 'progressive_carries_p90', 'successful_dribbles_p90', 'successful_tackles_p90', 'interceptions_p90', 'pass_completion_pct', 'total_actions']
#                 draw_percentile_profile(player_data, df, profile_metrics)
#     st.markdown("<hr style='margin: 2rem 0;'/><h3 style='font-size: 1.1rem;'>Head-to-Head Comparison</h3>", unsafe_allow_html=True)
#     default_h2h = ["Jude Bellingham", "Kevin De Bruyne", "Florian Wirtz"]
#     selected_h2h = st.multiselect("Select players for deeper comparison:", options=players, default=[p for p in default_h2h if p in players])
#     if len(selected_h2h) > 1:
#         df_sel = df[df["player"].isin(selected_h2h)].reset_index(drop=True)
#         tab1, tab2 = st.tabs(["📊 Player Archetype Plots", "📈 Detailed KPI Charts"])
#         with tab1:
#             col1, col2 = st.columns(2)
#             with col1:
#                 radar_metrics = ['avg_xt_impact', 'progressive_passes_p90', 'successful_dribbles_p90', 'successful_tackles_p90', 'non_penalty_xg_p90']
#                 draw_radar_chart(df_sel, radar_metrics, "Player Style Radar")
#             with col2:
#                 draw_scatter_plot(df_sel, 'avg_xt_impact', 'total_actions', 'Creator vs. Volume Analysis', 'Creative Threat (xT Impact)', 'Involvement (Touches p90)')
#         with tab2:
#             st.markdown("<br/>", unsafe_allow_html=True)
#             def create_bar_chart(metric, title):
#                 fig, ax = plt.subplots(figsize=(8, 4))
#                 sns.barplot(data=df_sel, x='player', y=metric, hue='player', palette='viridis', legend=False, ax=ax)
#                 ax.set_title(title, fontsize=11, fontweight='bold'); ax.set_xlabel(''); ax.set_ylabel(''); st.pyplot(fig)
#             c1, c2 = st.columns(2)
#             with c1:
#                 create_bar_chart('key_passes_p90', 'Key Passes per 90')
#                 create_bar_chart('progressive_passes_p90', 'Progressive Passes per 90')
#                 create_bar_chart('successful_tackles_p90', 'Successful Tackles per 90')
#             with c2:
#                 create_bar_chart('non_penalty_xg_p90', 'Non-Penalty xG per 90')
#                 create_bar_chart('progressive_carries_p90', 'Progressive Carries per 90')
#                 create_bar_chart('interceptions_p90', 'Interceptions per 90')
#     elif selected_h2h: st.info("Please select at least two players for a head-to-head comparison.")

# # =======================================
# # MAIN
# # =======================================
# def main():
#     render_header()
#     tab1, tab2 = st.tabs(["🎯 Match Analyzer", "📊 Advanced Analytics Board"])
#     with tab1:
#         render_match_analyzer()
#     with tab2:
#         render_season_comparison()

# if __name__ == "__main__":
#     main()














# import os
# import base64
# from typing import Optional, Tuple

# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import percentileofscore

# from mplsoccer import Pitch
# from sklearn.cluster import KMeans

# # =======================================
# # STREAMLIT CONFIG
# # =======================================
# st.set_page_config(
#     page_title="Pedri Spatial Intelligence Dashboard",
#     page_icon="⚽",
#     layout="wide",
# )

# PRIMARY_BG = "#f0f2ff"
# FOOTBALL_GREEN = "#28a745"
# st.markdown(
#     f"""
#     <style>
#         /* =============================
#            GLOBAL APP SETTINGS
#         ============================== */
#         .stApp {{
#             background-color: {PRIMARY_BG} !important;
#             color: #111827 !important;
#         }}
#         .block-container {{
#             max-width: 1200px !important; margin: auto !important;
#             padding-top: 2rem !important; padding-bottom: 3rem !important;
#         }}
#         /* =============================
#            VISIBILITY FIXES FOR ALL TEXT
#         ============================== */
#         h1, h2, h3, h4, h5, h6 {{
#             color: #1f2937 !important;
#         }}
#         div[data-testid="stMarkdown"] label {{
#             color: #374151 !important;
#         }}
#         div[data-testid="stExpander"] {{
#              border-color: #d1d5db !important;
#         }}
#         div[data-testid="stAlert"] * {{ color: #31333F !important; }}
#         /* =============================
#            PLAYER IMAGE & HERO CARD
#         ============================== */
#         .player-image {{
#             width: 160px; height: 160px; border-radius: 50%; object-fit: cover;
#             border: 4px solid white; box-shadow: 0 12px 35px rgba(15, 23, 42, 0.25);
#         }}
#         .hero-card {{
#             background: white; border-radius: 22px; padding: 1.5rem 1.75rem;
#             box-shadow: 0 15px 40px rgba(15, 23, 42, 0.12); border: 1px solid #e5e7eb;
#         }}
#         .hero-name {{ font-size: 1.8rem; font-weight: 800; margin-bottom: 0.25rem; color: #111827; }}
#         .hero-subtitle {{ font-size: 0.98rem; color: #4b5563; margin-bottom: 0.5rem; }}
#         .hero-tag {{
#             display: inline-block; padding: 0.25rem 0.65rem; border-radius: 999px;
#             font-size: 0.75rem; font-weight: 600; background: rgba(40,167,69,0.08);
#             color: {FOOTBALL_GREEN}; margin-right: 0.4rem;
#         }}
#         /* =============================
#            KPI METRIC CARDS
#         ============================== */
#         .metric-card {{
#             background: white; border-radius: 18px; padding: 1rem 1.25rem;
#             box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08); border: 1px solid #e5e7eb;
#             text-align: center; transition: 0.2s ease-in-out;
#         }}
#         .metric-card:hover {{ transform: translateY(-4px); box-shadow: 0 14px 36px rgba(15, 23, 42, 0.18); }}
#         .metric-label {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7280; margin-bottom: 0.15rem; }}
#         .metric-value {{ font-size: 1.3rem; font-weight: 700; color: #111827; }}
#         .metric-sub {{ font-size: 0.78rem; color: #9ca3af; margin-top: 0.1rem; }}
#         /* =============================
#            SECTION TITLES
#         ============================== */
#         .section-title {{ font-size: 1.1rem; font-weight: 700; margin-top: 1.5rem; margin-bottom: 0.3rem; color: #111827; }}
#         .section-caption {{ font-size: 0.85rem; color: #6b7280; margin-bottom: 0.75rem; }}
#         /* =============================
#            AI INSIGHT CARD
#         ============================== */
#         .insight-card {{
#             background: white; border-radius: 18px; padding: 1.25rem 1.5rem;
#             box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08); border: 1px solid #e5e7eb;
#         }}
#         .insight-title {{
#             font-size: 0.9rem; font-weight: 600; text-transform: uppercase;
#             color: {FOOTBALL_GREEN}; letter-spacing: 0.15em; margin-bottom: 0.5rem;
#         }}
#         .insight-body {{ font-size: 0.95rem; color: #111827; line-height: 1.6; }}
#         .insight-body h4 {{ font-size: 1rem; color: #1f2937; margin-top: 1rem; margin-bottom: 0.5rem;}}
#         /* =============================
#            STREAMLIT TABS
#         ============================== */
#         .stTabs [role="tab"] {{
#             background-color: white !important; color: #374151 !important; border-radius: 999px !important;
#             padding: 6px 18px !important; border: 1px solid #d1d5db !important;
#             font-weight: 600 !important; transition: all 0.2s ease-in-out !important;
#         }}
#         .stTabs [role="tab"][aria-selected="true"] {{
#             background-color: rgba(40,167,69,0.15) !important; color: #1f2937 !important;
#             border-color: #28a745 !important; font-weight: 700 !important;
#         }}
#         .stTabs [role="tab"]:hover {{ background-color: rgba(40,167,69,0.10) !important; color: #1f2937 !important; }}
#         /* =============================
#            CUSTOM PERCENTILE BARS
#         ============================== */
#         .percentile-container {{ display: flex; align-items: center; margin-bottom: 4px; height: 24px; }}
#         .percentile-label {{ width: 220px; font-size: 0.8rem; padding-right: 10px; text-align: right; color: #4b5563;}}
#         .percentile-bar-bg {{ flex-grow: 1; background-color: #e9ecef; border-radius: 5px; height: 18px; }}
#         .percentile-bar-fg {{
#             height: 100%; border-radius: 5px; text-align: right; color: white;
#             font-size: 0.75rem; padding-right: 5px; line-height: 18px; font-weight: 600;
#         }}
#         .percentile-value {{ width: 50px; text-align: right; font-size: 0.8rem; font-weight: 600; padding-left: 10px; color: #374151; }}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# # =======================================
# # UTILS
# # =======================================
# def load_image_base64(path: str) -> str:
#     with open(path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode()

# def get_xt_column(df: pd.DataFrame) -> Optional[str]:
#     candidates = ["xT", "xt", "xT_value", "xT_impact", "expected_threat"]
#     for c in candidates:
#         if c in df.columns: return c
#     return None

# def get_xy_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
#     candidates_x = ["x", "X", "pos_x", "start_x"]
#     candidates_y = ["y", "Y", "pos_y", "start_y"]
#     x_col = next((c for c in candidates_x if c in df.columns), None)
#     y_col = next((c for c in candidates_y if c in df.columns), None)
#     return x_col, y_col

# def compute_event_kpis(df: pd.DataFrame) -> dict:
#     kpis = {}
#     kpis["Total Actions"] = len(df)
#     xt_col = get_xt_column(df)
#     kpis["Avg. xT Impact"] = float(df[xt_col].mean()) if xt_col and df[xt_col].notna().any() else None
#     pass_mask = df["event_type"].astype(str).str.contains("pass", case=False, na=False)
#     total_passes = pass_mask.sum()
#     if total_passes > 0:
#         if "outcome" in df.columns:
#             completed = df.loc[pass_mask, "outcome"].astype(str).str.contains("success|complete|accurate|on_target", case=False, na=False).sum()
#         elif "is_complete" in df.columns:
#             completed = df.loc[pass_mask, "is_complete"].sum()
#         else:
#             completed = None
#         kpis["Pass Accuracy (%)"] = 100 * completed / total_passes if completed is not None else None
#     else:
#         kpis["Pass Accuracy (%)"] = None
#     if "is_progressive" in df.columns:
#         kpis["Progressive Actions"] = int(df["is_progressive"].sum())
#     elif "end_x" in df.columns and "x" in df.columns:
#         kpis["Progressive Actions"] = int((df["end_x"] - df["x"] >= 15).sum())
#     else:
#         kpis["Progressive Actions"] = 0
#     if "is_recovery" in df.columns:
#         kpis["Ball Recoveries"] = int(df["is_recovery"].sum())
#     elif "event_type" in df.columns:
#         kpis["Ball Recoveries"] = int(df["event_type"].astype(str).str.contains("recovery", case=False, na=False).sum())
#     else:
#         kpis["Ball Recoveries"] = 0
#     kpis["Dribbles"] = int(df["event_type"].astype(str).str.contains("dribble", case=False, na=False).sum()) if "event_type" in df.columns else 0
#     return kpis

# # =======================================
# # ADVANCED AI INSIGHT FUNCTION
# # =======================================
# def ai_insight_event(df: pd.DataFrame, kpis: dict, seasonal_df: pd.DataFrame) -> str:
#     try:
#         pedri_avg = seasonal_df[seasonal_df['player'] == 'Pedri'].iloc[0]
#     except (IndexError, KeyError):
#         return "Could not find Pedri's seasonal data to generate a comparative insight."
#     lines = []
#     match_xt = kpis.get("Avg. xT Impact", 0)
#     avg_xt = pedri_avg.get('avg_xt_impact', 0)
#     if match_xt > avg_xt * 1.3: headline = "🔥 **Exceptional Creative Performance:** A standout match where he significantly exceeded his usual creative output."
#     elif match_xt > avg_xt * 1.1: headline = "🟢 **Strong Attacking Influence:** An above-average performance in terms of creating threat."
#     elif match_xt < avg_xt * 0.7: headline = "🧊 **Subdued Creative Role:** A quieter game where his creative impact was notably below his seasonal average."
#     else: headline = "🟡 **Solid & Controlled Performance:** A typical display, in line with his seasonal averages."
#     lines.append(headline)
#     lines.append("<h4>Detailed Performance Analysis</h4>")
#     xt_diff = ((match_xt / avg_xt) - 1) * 100 if avg_xt > 0 else 0
#     xt_eval = "well above" if xt_diff > 25 else "above" if xt_diff > 10 else "well below" if xt_diff < -25 else "below" if xt_diff < -10 else "in line with"
#     lines.append(f" • **Creative Impact:** His average xT of **{match_xt:.2f}** was **{xt_eval}** his seasonal average of {avg_xt:.2f} ({xt_diff:+.0f}%).")
#     match_prog = kpis.get("Progressive Actions", 0)
#     avg_prog = pedri_avg.get('progressive_passes_p90', 0) + pedri_avg.get('progressive_carries_p90', 0)
#     prog_diff = ((match_prog / avg_prog) - 1) * 100 if avg_prog > 0 else 0
#     prog_eval = "a significantly higher" if prog_diff > 20 else "a higher" if prog_diff > 10 else "a lower" if prog_diff < -10 else "a similar"
#     lines.append(f" • **Ball Progression:** He completed **{match_prog}** progressive actions, a **{prog_eval}** volume compared to his average of {avg_prog:.1f} per 90.")
#     match_recov = kpis.get("Ball Recoveries", 0)
#     avg_def = pedri_avg.get('successful_tackles_p90', 0) + pedri_avg.get('interceptions_p90', 0)
#     avg_recov_benchmark = avg_def * 2.5
#     def_eval = "high level" if match_recov > avg_recov_benchmark * 1.2 else "reduced" if match_recov < avg_recov_benchmark * 0.8 else "solid"
#     lines.append(f" • **Defensive Workrate:** With **{match_recov}** ball recoveries, he showed a **{def_eval}** level of defensive engagement.")
#     lines.append("<h4>Tactical Role Interpretation</h4>")
#     x_col, y_col = get_xy_columns(df)
#     if x_col and y_col and not df[[x_col, y_col]].dropna().empty:
#         coords = df[[x_col, y_col]].dropna()
#         if len(coords) >= 25:
#             kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(coords)
#             avg_x = kmeans.cluster_centers_[:, 0].mean()
#             if avg_x < 45: role = "a deeper **#6 Deep-Lying Playmaker** role, focusing on build-up."
#             elif avg_x < 75: role = "a classic **#8 Hybrid Midfielder** role, connecting defense and attack."
#             else: role = "an advanced **#10 Attacking Midfielder** role, operating primarily in the final third."
#             lines.append(f" • **Positional Analysis:** K-Means clustering suggests he adopted {role} (Avg x-pos: {avg_x:.1f}).")
#     if xt_diff > 20 and prog_diff > 15: synthesis = "This was a dominant performance where he was both the primary progressor and creator."
#     elif xt_diff > 20 and prog_diff < 0: synthesis = "This performance was defined by efficiency. He may have had fewer touches but was highly effective in creating danger."
#     elif xt_diff < -20 and prog_diff > 15: synthesis = "This was a 'Workhorse' performance. While final-third creativity was limited, his main duty was evidently to progress the ball relentlessly."
#     else: synthesis = "This was a balanced performance, fulfilling his tactical instructions without deviating from his established profile."
#     lines.append(f" • **Overall Verdict:** {synthesis}")
#     return "\n".join(lines)

# # =======================================
# # PITCH DRAWING & HEADER
# # =======================================
# def draw_event_heatmap(df: pd.DataFrame, title: str):
#     x_col, y_col = get_xy_columns(df)
#     if not x_col or not y_col or df[[x_col, y_col]].dropna().empty: return
#     pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
#     fig, ax = pitch.draw(figsize=(6, 8)); sns.kdeplot(x=df[x_col], y=df[y_col], fill=True, cmap="inferno", alpha=0.9, ax=ax, thresh=0.05, bw_adjust=0.45)
#     ax.set_title("", color="white", fontsize=12, fontweight="bold"); st.pyplot(fig)

# def draw_passing_network(df: pd.DataFrame, title: str):
#     x_col, y_col = get_xy_columns(df)
#     if not x_col or not y_col or "end_x" not in df.columns or "end_y" not in df.columns: return
#     mask = df["event_type"].astype(str).str.contains("pass", case=False, na=False)
#     if "is_complete" in df.columns: mask &= df["is_complete"] == 1
#     passes = df.loc[mask].dropna(subset=[x_col, y_col, "end_x", "end_y"])
#     if passes.empty: return
#     pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
#     fig, ax = pitch.draw(figsize=(6, 8))
#     pitch.lines(passes[x_col], passes[y_col], passes["end_x"], passes["end_y"], ax=ax, comet=True, color="gold", lw=1.2, alpha=0.7)
#     ax.set_title("", color="white", fontsize=12, fontweight="bold"); st.pyplot(fig)

# def draw_ball_progression(df: pd.DataFrame, title: str):
#     x_col, y_col = get_xy_columns(df)
#     if not x_col or not y_col or "end_x" not in df.columns or "end_y" not in df.columns: return
#     data = df[df["event_type"].astype(str).str.contains("pass|carry|dribble", case=False, na=False)].copy()
#     xt_col = get_xt_column(data)
#     if xt_col and data[xt_col].notna().any(): data = data[data[xt_col] >= data[xt_col].quantile(0.75)]
#     data.dropna(subset=[x_col, y_col, "end_x", "end_y"], inplace=True)
#     if data.empty: return
#     pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
#     fig, ax = pitch.draw(figsize=(6, 8))
#     pitch.arrows(data[x_col], data[y_col], data["end_x"], data["end_y"], ax=ax, width=1.2, headwidth=2.5, headlength=2.5, color="#ffdd57", alpha=0.9)
#     ax.set_title("", color="white", fontsize=12, fontweight="bold"); st.pyplot(fig)

# def render_header():
#     st.markdown('<p style="font-size:0.85rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.20em;">FC Barcelona • 2024-25 La Liga</p>', unsafe_allow_html=True)
#     col_img, col_info = st.columns([1, 2], gap="large")
#     with col_img:
#         if os.path.exists("pedri.png"):
#             img64 = load_image_base64("pedri.png")
#             st.markdown(f'<div style="display:flex; justify-content:center; align-items:center;"><img src="data:image/png;base64,{img64}" class="player-image" /></div>', unsafe_allow_html=True)
#     with col_info:
#         st.markdown("""
#             <div class="hero-card">
#                 <div class="hero-name">Pedro González López</div>
#                 <div class="hero-subtitle">FC Barcelona • Midfielder • #8</div>
#                 <div style="margin-bottom:0.6rem;"><span class="hero-tag">Positional Play Engine</span><span class="hero-tag">Left Half-Space Creator</span></div>
#                 <div style="font-size:0.9rem; color:#4b5563;">Premium spatial-intelligence dashboard for match-by-match heatmap analysis, role detection, and seasonal benchmarking of Pedri’s influence.</div>
#             </div>""", unsafe_allow_html=True)
#     st.markdown("<br/>", unsafe_allow_html=True)
#     st.markdown('<div class="section-title">2024-25 Season Snapshot (Static)</div>', unsafe_allow_html=True)
#     cols = st.columns(6)
#     metrics_data = [("Matches Played", "34", "La Liga"), ("Goals", "7", "Box arrivals"), ("Assists", "9", "Final ball"), ("Avg. xT / 90", "0.18", "Threat impact"), ("Prog Actions / 90", "11.1", "Carries & passes"), ("Tkl+Int / 90", "2.5", "Defensive work")]
#     for col, (label, value, sub) in zip(cols, metrics_data):
#         with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

# # =======================================
# # MATCH ANALYZER
# # =======================================
# def render_match_analyzer():
#     st.markdown('<div class="section-title">Match Analyzer</div><div class="section-caption">Upload a single-match event CSV to generate a dynamic, AI-powered performance evaluation.</div>', unsafe_allow_html=True)
#     seasonal_df = pd.read_csv("player_comparison_data.csv") if os.path.exists("player_comparison_data.csv") else None
#     uploaded = st.file_uploader("Upload Match Data", type=["csv"], label_visibility="collapsed")
#     if uploaded is None:
#         st.info("Upload a match file to start the analysis."); return
#     try: df = pd.read_csv(uploaded)
#     except Exception as e:
#         st.error(f"Could not read CSV file: {e}"); return
#     mode = "event" if {"event_type", "end_x"}.issubset(df.columns) else None
#     if mode != "event":
#         st.error("The AI Insight feature currently only supports Event Data CSVs. Please upload a valid file."); return
#     kpis = compute_event_kpis(df)
#     labels = ["Total Actions", "Avg. xT Impact", "Pass Accuracy (%)", "Progressive Actions", "Ball Recoveries", "Dribbles"]
#     cols = st.columns(len(labels))
#     for col, label in zip(cols, labels):
#         val = kpis.get(label); disp = f"{val:.2f}" if isinstance(val, (float, np.floating)) and not np.isnan(val) else str(val) if val is not None else "–"
#         with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{disp}</div></div>', unsafe_allow_html=True)
#     tab_plots, tab_insight = st.tabs(["📊 Match Plots", "🧠 AI Performance Report"])
#     with tab_plots:
#         st.markdown("<br>", unsafe_allow_html=True)
#         c1, c2, c3 = st.columns(3)
#         with c1:
#             st.markdown("<h4 style='text-align: center; color: #4b5563;'>Action Heatmap</h4>", unsafe_allow_html=True)
#             draw_event_heatmap(df, "On-ball Action Density")
#         with c2:
#             st.markdown("<h4 style='text-align: center; color: #4b5563;'>Passing Network</h4>", unsafe_allow_html=True)
#             draw_passing_network(df, "Completed Passing Network")
#         with c3:
#             st.markdown("<h4 style='text-align: center; color: #4b5563;'>High-Value Progressions</h4>", unsafe_allow_html=True)
#             draw_ball_progression(df, "High-Value Ball Progression")
#     with tab_insight:
#         if seasonal_df is not None:
#             insight_html = ai_insight_event(df, kpis, seasonal_df).replace("\n", "<br/>")
#             st.markdown(f'<div class="insight-card"><div class="insight-title">AI MATCH ANALYSIS vs. 24-25 SEASON</div><div class="insight-body">{insight_html}</div></div>', unsafe_allow_html=True)
#         else:
#             st.warning("`player_comparison_data.csv` not found. AI insights will be limited without seasonal benchmarks.")


# # =======================================
# # ADVANCED SEASON COMPARISON PLOTS (FIXED ORDER)
# # =======================================
# def get_percentile_color(p):
#     if p > 90: return "#059669"
#     if p > 75: return "#10B981"
#     if p > 50: return "#F59E0B"
#     if p > 25: return "#F97316"
#     return "#EF4444"

# def draw_percentile_profile(player_data: pd.Series, all_data: pd.DataFrame, metrics: list):
#     st.markdown(f"<h4 style='font-size: 1.1rem; text-align: center; margin-bottom: 1rem; color: #1f2937;'>{player_data['player']}</h4>", unsafe_allow_html=True)
#     for metric in metrics:
#         player_value = player_data[metric]
#         percentile = percentileofscore(all_data[metric], player_value)
#         bar_color = get_percentile_color(percentile)
#         label = metric.replace('_', ' ').replace(' p90', '').title()
#         st.markdown(f'''
#             <div class="percentile-container">
#                 <div class="percentile-label">{label}</div>
#                 <div class="percentile-bar-bg">
#                     <div class="percentile-bar-fg" style="width: {percentile}%; background-color: {bar_color};">{int(percentile)}</div>
#                 </div>
#                 <div class="percentile-value">{player_value:.2f}</div>
#             </div>''', unsafe_allow_html=True)

# def draw_radar_chart(df: pd.DataFrame, metrics: list, title: str):
#     labels = [m.replace('_', ' ').replace(' p90', '').title() for m in metrics]
#     num_vars = len(labels)
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
#     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
#     df_norm = df.copy()
#     for m in metrics:
#         min_val, max_val = all_data_for_radar[m].min(), all_data_for_radar[m].max()
#         df_norm[m] = (df_norm[m] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
#     for _, row in df_norm.iterrows():
#         values = row[metrics].tolist() + [row[metrics].tolist()[0]]
#         ax.plot(angles, values, linewidth=1.5, label=row['player'])
#         ax.fill(angles, values, alpha=0.15)
#     ax.set_yticklabels([]); ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
#     ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
#     ax.set_title(title, size=12, y=1.12); st.pyplot(fig)

# def draw_scatter_plot(df: pd.DataFrame, x_metric: str, y_metric: str, title: str, xlabel: str, ylabel: str):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.scatterplot(data=df, x=x_metric, y=y_metric, hue="player", s=120, palette="viridis", legend=False, ax=ax)
#     for _, row in df.iterrows():
#         ax.text(row[x_metric], row[y_metric], row['player'], fontdict={'size': 9}, ha='left', va='bottom')
#     ax.set_title(title, fontsize=12, fontweight='bold'); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
#     ax.grid(True, which='both', linestyle='--', linewidth=0.3); st.pyplot(fig)

# all_data_for_radar = pd.DataFrame()

# def render_season_comparison():
#     global all_data_for_radar
#     st.markdown('<div class="section-title">Advanced Midfielder Analytics Board</div><div class="section-caption">Benchmark elite midfielders from the 2024-25 season. Analyze player profiles, compare KPIs, and identify tactical archetypes.</div>', unsafe_allow_html=True)
#     csv_path = "player_comparison_data.csv"
#     if not os.path.exists(csv_path):
#         st.error(f"`{csv_path}` not found."); return
#     try:
#         df = pd.read_csv(csv_path)
#         all_data_for_radar = df.copy()
#     except Exception as e:
#         st.error(f"Could not read `{csv_path}`: {e}"); return
#     players = sorted(df["player"].unique().tolist())
#     st.markdown("---")
#     st.markdown("<h3 style='font-size: 1.1rem;'>Player Profile Analysis</h3>", unsafe_allow_html=True)
#     profile_players_selected = st.multiselect("Select players to view their detailed percentile profile (2-3 recommended):", players, default=["Pedri", "Vitinha"] if "Pedri" in players and "Vitinha" in players else players[:2])
#     if profile_players_selected:
#         cols = st.columns(len(profile_players_selected))
#         for i, player_name in enumerate(profile_players_selected):
#             with cols[i]:
#                 player_data = df[df['player'] == player_name].iloc[0]
#                 profile_metrics = ['avg_xt_impact', 'key_passes_p90', 'non_penalty_xg_p90', 'shots_p90', 'progressive_passes_p90', 'progressive_carries_p90', 'successful_dribbles_p90', 'successful_tackles_p90', 'interceptions_p90', 'pass_completion_pct', 'total_actions']
#                 draw_percentile_profile(player_data, df, profile_metrics)
#     st.markdown("<hr style='margin: 2rem 0;'/><h3 style='font-size: 1.1rem;'>Head-to-Head Comparison</h3>", unsafe_allow_html=True)
#     default_h2h = ["Jude Bellingham", "Kevin De Bruyne", "Florian Wirtz"]
#     selected_h2h = st.multiselect("Select players for deeper comparison:", options=players, default=[p for p in default_h2h if p in players])
#     if len(selected_h2h) > 1:
#         df_sel = df[df["player"].isin(selected_h2h)].reset_index(drop=True)
#         tab1, tab2 = st.tabs(["📊 Player Archetype Plots", "📈 Detailed KPI Charts"])
#         with tab1:
#             col1, col2 = st.columns(2)
#             with col1:
#                 radar_metrics = ['avg_xt_impact', 'progressive_passes_p90', 'successful_dribbles_p90', 'successful_tackles_p90', 'non_penalty_xg_p90']
#                 draw_radar_chart(df_sel, radar_metrics, "Player Style Radar")
#             with col2:
#                 draw_scatter_plot(df_sel, 'avg_xt_impact', 'total_actions', 'Creator vs. Volume Analysis', 'Creative Threat (xT Impact)', 'Involvement (Touches p90)')
#         with tab2:
#             st.markdown("<br/>", unsafe_allow_html=True)
#             def create_bar_chart(metric, title):
#                 fig, ax = plt.subplots(figsize=(8, 4))
#                 sns.barplot(data=df_sel, x='player', y=metric, hue='player', palette='viridis', legend=False, ax=ax)
#                 ax.set_title(title, fontsize=11, fontweight='bold'); ax.set_xlabel(''); ax.set_ylabel(''); st.pyplot(fig)
#             c1, c2 = st.columns(2)
#             with c1:
#                 create_bar_chart('key_passes_p90', 'Key Passes per 90')
#                 create_bar_chart('progressive_passes_p90', 'Progressive Passes per 90')
#                 create_bar_chart('successful_tackles_p90', 'Successful Tackles per 90')
#             with c2:
#                 create_bar_chart('non_penalty_xg_p90', 'Non-Penalty xG per 90')
#                 create_bar_chart('progressive_carries_p90', 'Progressive Carries per 90')
#                 create_bar_chart('interceptions_p90', 'Interceptions per 90')
#     elif selected_h2h: st.info("Please select at least two players for a head-to-head comparison.")

# # =======================================
# # MAIN
# # =======================================
# def main():
#     render_header()
#     tab1, tab2 = st.tabs(["🎯 Match Analyzer", "📊 Advanced Analytics Board"])
#     with tab1:
#         render_match_analyzer()
#     with tab2:
#         render_season_comparison()

# if __name__ == "__main__":
#     main()





import os
import base64
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore

from mplsoccer import Pitch
from sklearn.cluster import KMeans

# =======================================
# STREAMLIT CONFIG
# =======================================
st.set_page_config(
    page_title="Pedri Spatial Intelligence Dashboard",
    page_icon="⚽",
    layout="wide",
)

PRIMARY_BG = "#f0f2ff"
FOOTBALL_GREEN = "#28a745"
st.markdown(
    f"""
    <style>
        /* =============================
           GLOBAL APP SETTINGS
        ============================== */
        .stApp {{
            background-color: {PRIMARY_BG} !important;
            color: #111827 !important;
        }}
        .block-container {{
            max-width: 1200px !important; margin: auto !important;
            padding-top: 2rem !important; padding-bottom: 3rem !important;
        }}
        /* =============================
           VISIBILITY FIXES FOR ALL TEXT
        ============================== */
        h1, h2, h3, h4, h5, h6 {{
            color: #1f2937 !important;
        }}
        div[data-testid="stMarkdown"] label {{
            color: #374151 !important;
        }}
        div[data-testid="stExpander"] {{
             border-color: #d1d5db !important;
        }}
        div[data-testid="stAlert"] * {{ color: #31333F !important; }}
        /* =============================
           PLAYER IMAGE & HERO CARD
        ============================== */
        .player-image {{
            width: 160px; height: 160px; border-radius: 50%; object-fit: cover;
            border: 4px solid white; box-shadow: 0 12px 35px rgba(15, 23, 42, 0.25);
        }}
        .hero-card {{
            background: white; border-radius: 22px; padding: 1.5rem 1.75rem;
            box-shadow: 0 15px 40px rgba(15, 23, 42, 0.12); border: 1px solid #e5e7eb;
        }}
        .hero-name {{ font-size: 1.8rem; font-weight: 800; margin-bottom: 0.25rem; color: #111827; }}
        .hero-subtitle {{ font-size: 0.98rem; color: #4b5563; margin-bottom: 0.5rem; }}
        .hero-tag {{
            display: inline-block; padding: 0.25rem 0.65rem; border-radius: 999px;
            font-size: 0.75rem; font-weight: 600; background: rgba(40,167,69,0.08);
            color: {FOOTBALL_GREEN}; margin-right: 0.4rem;
        }}
        /* =============================
           KPI METRIC CARDS
        ============================== */
        .metric-card {{
            background: white; border-radius: 18px; padding: 1rem 1.25rem;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08); border: 1px solid #e5e7eb;
            text-align: center; transition: 0.2s ease-in-out;
        }}
        .metric-card:hover {{ transform: translateY(-4px); box-shadow: 0 14px 36px rgba(15, 23, 42, 0.18); }}
        .metric-label {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7280; margin-bottom: 0.15rem; }}
        .metric-value {{ font-size: 1.3rem; font-weight: 700; color: #111827; }}
        .metric-sub {{ font-size: 0.78rem; color: #9ca3af; margin-top: 0.1rem; }}
        /* =============================
           SECTION TITLES
        ============================== */
        .section-title {{ font-size: 1.1rem; font-weight: 700; margin-top: 1.5rem; margin-bottom: 0.3rem; color: #111827; }}
        .section-caption {{ font-size: 0.85rem; color: #6b7280; margin-bottom: 0.75rem; }}
        /* =============================
           AI INSIGHT CARD
        ============================== */
        .insight-card {{
            background: white; border-radius: 18px; padding: 1.25rem 1.5rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08); border: 1px solid #e5e7eb;
        }}
        .insight-title {{
            font-size: 0.9rem; font-weight: 600; text-transform: uppercase;
            color: {FOOTBALL_GREEN}; letter-spacing: 0.15em; margin-bottom: 0.5rem;
        }}
        .insight-body {{ font-size: 0.95rem; color: #111827; line-height: 1.6; }}
        .insight-body h4 {{ font-size: 1rem; color: #1f2937; margin-top: 1rem; margin-bottom: 0.5rem;}}
        /* =============================
           STREAMLIT TABS
        ============================== */
        .stTabs [role="tab"] {{
            background-color: white !important; color: #374151 !important; border-radius: 999px !important;
            padding: 6px 18px !important; border: 1px solid #d1d5db !important;
            font-weight: 600 !important; transition: all 0.2s ease-in-out !important;
        }}
        .stTabs [role="tab"][aria-selected="true"] {{
            background-color: rgba(40,167,69,0.15) !important; color: #1f2937 !important;
            border-color: #28a745 !important; font-weight: 700 !important;
        }}
        .stTabs [role="tab"]:hover {{ background-color: rgba(40,167,69,0.10) !important; color: #1f2937 !important; }}
        /* =============================
           CUSTOM PERCENTILE BARS
        ============================== */
        .percentile-container {{ display: flex; align-items: center; margin-bottom: 4px; height: 24px; }}
        .percentile-label {{ width: 220px; font-size: 0.8rem; padding-right: 10px; text-align: right; color: #4b5563;}}
        .percentile-bar-bg {{ flex-grow: 1; background-color: #e9ecef; border-radius: 5px; height: 18px; }}
        .percentile-bar-fg {{
            height: 100%; border-radius: 5px; text-align: right; color: white;
            font-size: 0.75rem; padding-right: 5px; line-height: 18px; font-weight: 600;
        }}
        .percentile-value {{ width: 50px; text-align: right; font-size: 0.8rem; font-weight: 600; padding-left: 10px; color: #374151; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =======================================
# UTILS
# =======================================
def load_image_base64(path: str) -> str:
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def get_xt_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["xT", "xt", "xT_value", "xT_impact", "expected_threat"]
    for c in candidates:
        if c in df.columns: return c
    return None

def get_xy_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    candidates_x = ["x", "X", "pos_x", "start_x"]
    candidates_y = ["y", "Y", "pos_y", "start_y"]
    x_col = next((c for c in candidates_x if c in df.columns), None)
    y_col = next((c for c in candidates_y if c in df.columns), None)
    return x_col, y_col

def compute_event_kpis(df: pd.DataFrame) -> dict:
    kpis = {}
    kpis["Total Actions"] = len(df)
    xt_col = get_xt_column(df)
    kpis["Avg. xT Impact"] = float(df[xt_col].mean()) if xt_col and df[xt_col].notna().any() else None
    pass_mask = df["event_type"].astype(str).str.contains("pass", case=False, na=False)
    total_passes = pass_mask.sum()
    if total_passes > 0:
        if "outcome" in df.columns:
            completed = df.loc[pass_mask, "outcome"].astype(str).str.contains("success|complete|accurate|on_target", case=False, na=False).sum()
        elif "is_complete" in df.columns:
            completed = df.loc[pass_mask, "is_complete"].sum()
        else:
            completed = None
        kpis["Pass Accuracy (%)"] = 100 * completed / total_passes if completed is not None else None
    else:
        kpis["Pass Accuracy (%)"] = None
    if "is_progressive" in df.columns:
        kpis["Progressive Actions"] = int(df["is_progressive"].sum())
    elif "end_x" in df.columns and "x" in df.columns:
        kpis["Progressive Actions"] = int((df["end_x"] - df["x"] >= 15).sum())
    else:
        kpis["Progressive Actions"] = 0
    if "is_recovery" in df.columns:
        kpis["Ball Recoveries"] = int(df["is_recovery"].sum())
    elif "event_type" in df.columns:
        kpis["Ball Recoveries"] = int(df["event_type"].astype(str).str.contains("recovery", case=False, na=False).sum())
    else:
        kpis["Ball Recoveries"] = 0
    kpis["Dribbles"] = int(df["event_type"].astype(str).str.contains("dribble", case=False, na=False).sum()) if "event_type" in df.columns else 0
    return kpis

# =======================================
# ADVANCED AI INSIGHT FUNCTION
# =======================================
def ai_insight_event(df: pd.DataFrame, kpis: dict, seasonal_df: pd.DataFrame) -> str:
    try:
        pedri_avg = seasonal_df[seasonal_df['player'] == 'Pedri'].iloc[0]
    except (IndexError, KeyError):
        return "Could not find Pedri's seasonal data to generate a comparative insight."
    lines = []
    match_xt = kpis.get("Avg. xT Impact", 0)
    avg_xt = pedri_avg.get('avg_xt_impact', 0)
    if match_xt > avg_xt * 1.3: headline = "🔥 **Exceptional Creative Performance:** A standout match where he significantly exceeded his usual creative output."
    elif match_xt > avg_xt * 1.1: headline = "🟢 **Strong Attacking Influence:** An above-average performance in terms of creating threat."
    elif match_xt < avg_xt * 0.7: headline = "🧊 **Subdued Creative Role:** A quieter game where his creative impact was notably below his seasonal average."
    else: headline = "🟡 **Solid & Controlled Performance:** A typical display, in line with his seasonal averages."
    lines.append(headline)
    lines.append("<h4>Detailed Performance Analysis</h4>")
    xt_diff = ((match_xt / avg_xt) - 1) * 100 if avg_xt > 0 else 0
    xt_eval = "well above" if xt_diff > 25 else "above" if xt_diff > 10 else "well below" if xt_diff < -25 else "below" if xt_diff < -10 else "in line with"
    lines.append(f" • **Creative Impact:** His average xT of **{match_xt:.2f}** was **{xt_eval}** his seasonal average of {avg_xt:.2f} ({xt_diff:+.0f}%).")
    match_prog = kpis.get("Progressive Actions", 0)
    avg_prog = pedri_avg.get('progressive_passes_p90', 0) + pedri_avg.get('progressive_carries_p90', 0)
    prog_diff = ((match_prog / avg_prog) - 1) * 100 if avg_prog > 0 else 0
    prog_eval = "a significantly higher" if prog_diff > 20 else "a higher" if prog_diff > 10 else "a lower" if prog_diff < -10 else "a similar"
    lines.append(f" • **Ball Progression:** He completed **{match_prog}** progressive actions, a **{prog_eval}** volume compared to his average of {avg_prog:.1f} per 90.")
    match_recov = kpis.get("Ball Recoveries", 0)
    avg_def = pedri_avg.get('successful_tackles_p90', 0) + pedri_avg.get('interceptions_p90', 0)
    avg_recov_benchmark = avg_def * 2.5
    def_eval = "high level" if match_recov > avg_recov_benchmark * 1.2 else "reduced" if match_recov < avg_recov_benchmark * 0.8 else "solid"
    lines.append(f" • **Defensive Workrate:** With **{match_recov}** ball recoveries, he showed a **{def_eval}** level of defensive engagement.")
    lines.append("<h4>Tactical Role Interpretation</h4>")
    x_col, y_col = get_xy_columns(df)
    if x_col and y_col and not df[[x_col, y_col]].dropna().empty:
        coords = df[[x_col, y_col]].dropna()
        if len(coords) >= 25:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(coords)
            avg_x = kmeans.cluster_centers_[:, 0].mean()
            if avg_x < 45: role = "a deeper **#6 Deep-Lying Playmaker** role, focusing on build-up."
            elif avg_x < 75: role = "a classic **#8 Hybrid Midfielder** role, connecting defense and attack."
            else: role = "an advanced **#10 Attacking Midfielder** role, operating primarily in the final third."
            lines.append(f" • **Positional Analysis:** K-Means clustering suggests he adopted {role} (Avg x-pos: {avg_x:.1f}).")
    if xt_diff > 20 and prog_diff > 15: synthesis = "This was a dominant performance where he was both the primary progressor and creator."
    elif xt_diff > 20 and prog_diff < 0: synthesis = "This performance was defined by efficiency. He may have had fewer touches but was highly effective in creating danger."
    elif xt_diff < -20 and prog_diff > 15: synthesis = "This was a 'Workhorse' performance. While final-third creativity was limited, his main duty was evidently to progress the ball relentlessly."
    else: synthesis = "This was a balanced performance, fulfilling his tactical instructions without deviating from his established profile."
    lines.append(f" • **Overall Verdict:** {synthesis}")
    return "\n".join(lines)

# =======================================
# PITCH DRAWING & HEADER
# =======================================
def draw_event_heatmap(df: pd.DataFrame):
    x_col, y_col = get_xy_columns(df)
    if not x_col or not y_col or df[[x_col, y_col]].dropna().empty: return
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
    fig, ax = pitch.draw(figsize=(10, 7)); sns.kdeplot(x=df[x_col], y=df[y_col], fill=True, cmap="inferno", alpha=0.9, ax=ax, thresh=0.05, bw_adjust=0.45)
    st.pyplot(fig)

def draw_passing_network(df: pd.DataFrame):
    x_col, y_col = get_xy_columns(df)
    if not x_col or not y_col or "end_x" not in df.columns or "end_y" not in df.columns: return
    mask = df["event_type"].astype(str).str.contains("pass", case=False, na=False)
    if "is_complete" in df.columns: mask &= df["is_complete"] == 1
    passes = df.loc[mask].dropna(subset=[x_col, y_col, "end_x", "end_y"])
    if passes.empty: return
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
    fig, ax = pitch.draw(figsize=(10, 7))
    pitch.lines(passes[x_col], passes[y_col], passes["end_x"], passes["end_y"], ax=ax, comet=True, color="gold", lw=1.2, alpha=0.7)
    st.pyplot(fig)

def draw_ball_progression(df: pd.DataFrame):
    x_col, y_col = get_xy_columns(df)
    if not x_col or not y_col or "end_x" not in df.columns or "end_y" not in df.columns: return
    data = df[df["event_type"].astype(str).str.contains("pass|carry|dribble", case=False, na=False)].copy()
    xt_col = get_xt_column(data)
    if xt_col and data[xt_col].notna().any(): data = data[data[xt_col] >= data[xt_col].quantile(0.75)]
    data.dropna(subset=[x_col, y_col, "end_x", "end_y"], inplace=True)
    if data.empty: return
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
    fig, ax = pitch.draw(figsize=(10, 7))
    pitch.arrows(data[x_col], data[y_col], data["end_x"], data["end_y"], ax=ax, width=1.2, headwidth=2.5, headlength=2.5, color="#ffdd57", alpha=0.9)
    st.pyplot(fig)

def render_header():
    st.markdown('<p style="font-size:0.85rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.20em;">FC Barcelona • 2024-25 La Liga</p>', unsafe_allow_html=True)
    col_img, col_info = st.columns([1, 2], gap="large")
    with col_img:
        if os.path.exists("pedri.png"):
            img64 = load_image_base64("pedri.png")
            st.markdown(f'<div style="display:flex; justify-content:center; align-items:center;"><img src="data:image/png;base64,{img64}" class="player-image" /></div>', unsafe_allow_html=True)
    with col_info:
        st.markdown("""
            <div class="hero-card">
                <div class="hero-name">Pedro González López</div>
                <div class="hero-subtitle">FC Barcelona • Midfielder • #8</div>
                <div style="margin-bottom:0.6rem;"><span class="hero-tag">Positional Play Engine</span><span class="hero-tag">Left Half-Space Creator</span></div>
                <div style="font-size:0.9rem; color:#4b5563;">Premium spatial-intelligence dashboard for match-by-match heatmap analysis, role detection, and seasonal benchmarking of Pedri’s influence.</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">2024-25 Season Snapshot (Static)</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    metrics_data = [("Matches Played", "34", "La Liga"), ("Goals", "7", "Box arrivals"), ("Assists", "9", "Final ball"), ("Avg. xT / 90", "0.18", "Threat impact"), ("Prog Actions / 90", "11.1", "Carries & passes"), ("Tkl+Int / 90", "2.5", "Defensive work")]
    for col, (label, value, sub) in zip(cols, metrics_data):
        with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

# =======================================
# MATCH ANALYZER (WITH LARGE PLOT TABS)
# =======================================
def render_match_analyzer():
    st.markdown('<div class="section-title">Match Analyzer</div><div class="section-caption">Upload a single-match event CSV to generate a dynamic, AI-powered performance evaluation.</div>', unsafe_allow_html=True)
    seasonal_df = pd.read_csv("player_comparison_data.csv") if os.path.exists("player_comparison_data.csv") else None
    uploaded = st.file_uploader("Upload Match Data", type=["csv"], label_visibility="collapsed")
    if uploaded is None:
        st.info("Upload a match file to start the analysis."); return
    try: df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV file: {e}"); return
    mode = "event" if {"event_type", "end_x"}.issubset(df.columns) else None
    if mode != "event":
        st.error("The AI Insight feature currently only supports Event Data CSVs. Please upload a valid file."); return
    kpis = compute_event_kpis(df)
    labels = ["Total Actions", "Avg. xT Impact", "Pass Accuracy (%)", "Progressive Actions", "Ball Recoveries", "Dribbles"]
    cols = st.columns(len(labels))
    for col, label in zip(cols, labels):
        val = kpis.get(label); disp = f"{val:.2f}" if isinstance(val, (float, np.floating)) and not np.isnan(val) else str(val) if val is not None else "–"
        with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{disp}</div></div>', unsafe_allow_html=True)
    
    tab_plots, tab_insight = st.tabs(["📊 Match Plots", "🧠 AI Performance Report"])

    with tab_plots:
        plot_tab1, plot_tab2, plot_tab3 = st.tabs(["Action Heatmap", "Passing Network", "High-Value Progressions"])
        with plot_tab1:
            draw_event_heatmap(df)
        with plot_tab2:
            draw_passing_network(df)
        with plot_tab3:
            draw_ball_progression(df)

    with tab_insight:
        if seasonal_df is not None:
            insight_html = ai_insight_event(df, kpis, seasonal_df).replace("\n", "<br/>")
            st.markdown(f'<div class="insight-card"><div class="insight-title">AI MATCH ANALYSIS vs. 24-25 SEASON</div><div class="insight-body">{insight_html}</div></div>', unsafe_allow_html=True)
        else:
            st.warning("`player_comparison_data.csv` not found. AI insights will be limited without seasonal benchmarks.")


# =======================================
# ADVANCED SEASON COMPARISON PLOTS
# =======================================
def get_percentile_color(p):
    if p > 90: return "#059669"
    if p > 75: return "#10B981"
    if p > 50: return "#F59E0B"
    if p > 25: return "#F97316"
    return "#EF4444"

def draw_percentile_profile(player_data: pd.Series, all_data: pd.DataFrame, metrics: list):
    st.markdown(f"<h4 style='font-size: 1.1rem; text-align: center; margin-bottom: 1rem; color: #1f2937;'>{player_data['player']}</h4>", unsafe_allow_html=True)
    for metric in metrics:
        player_value = player_data[metric]
        percentile = percentileofscore(all_data[metric], player_value)
        bar_color = get_percentile_color(percentile)
        label = metric.replace('_', ' ').replace(' p90', '').title()
        st.markdown(f'''
            <div class="percentile-container">
                <div class="percentile-label">{label}</div>
                <div class="percentile-bar-bg">
                    <div class="percentile-bar-fg" style="width: {percentile}%; background-color: {bar_color};">{int(percentile)}</div>
                </div>
                <div class="percentile-value">{player_value:.2f}</div>
            </div>''', unsafe_allow_html=True)

def draw_radar_chart(df: pd.DataFrame, metrics: list, title: str):
    labels = [m.replace('_', ' ').replace(' p90', '').title() for m in metrics]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    df_norm = df.copy()
    for m in metrics:
        min_val, max_val = all_data_for_radar[m].min(), all_data_for_radar[m].max()
        df_norm[m] = (df_norm[m] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    for _, row in df_norm.iterrows():
        values = row[metrics].tolist() + [row[metrics].tolist()[0]]
        ax.plot(angles, values, linewidth=1.5, label=row['player'])
        ax.fill(angles, values, alpha=0.15)
    ax.set_yticklabels([]); ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title(title, size=12, y=1.12); st.pyplot(fig)

def draw_scatter_plot(df: pd.DataFrame, x_metric: str, y_metric: str, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_metric, y=y_metric, hue="player", s=120, palette="viridis", legend=False, ax=ax)
    for _, row in df.iterrows():
        ax.text(row[x_metric], row[y_metric], row['player'], fontdict={'size': 9}, ha='left', va='bottom')
    ax.set_title(title, fontsize=12, fontweight='bold'); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, which='both', linestyle='--', linewidth=0.3); st.pyplot(fig)

all_data_for_radar = pd.DataFrame()

def render_season_comparison():
    global all_data_for_radar
    st.markdown('<div class="section-title">Advanced Midfielder Analytics Board</div><div class="section-caption">Benchmark elite midfielders from the 2024-25 season. Analyze player profiles, compare KPIs, and identify tactical archetypes.</div>', unsafe_allow_html=True)
    csv_path = "player_comparison_data.csv"
    if not os.path.exists(csv_path):
        st.error(f"`{csv_path}` not found."); return
    try:
        df = pd.read_csv(csv_path)
        all_data_for_radar = df.copy()
    except Exception as e:
        st.error(f"Could not read `{csv_path}`: {e}"); return
    players = sorted(df["player"].unique().tolist())
    st.markdown("---")
    st.markdown("<h3 style='font-size: 1.1rem;'>Player Profile Analysis</h3>", unsafe_allow_html=True)
    profile_players_selected = st.multiselect("Select players to view their detailed percentile profile (2-3 recommended):", players, default=["Pedri", "Vitinha"] if "Pedri" in players and "Vitinha" in players else players[:2])
    if profile_players_selected:
        cols = st.columns(len(profile_players_selected))
        for i, player_name in enumerate(profile_players_selected):
            with cols[i]:
                player_data = df[df['player'] == player_name].iloc[0]
                profile_metrics = ['avg_xt_impact', 'key_passes_p90', 'non_penalty_xg_p90', 'shots_p90', 'progressive_passes_p90', 'progressive_carries_p90', 'successful_dribbles_p90', 'successful_tackles_p90', 'interceptions_p90', 'pass_completion_pct', 'total_actions']
                draw_percentile_profile(player_data, df, profile_metrics)
    st.markdown("<hr style='margin: 2rem 0;'/><h3 style='font-size: 1.1rem;'>Head-to-Head Comparison</h3>", unsafe_allow_html=True)
    default_h2h = ["Jude Bellingham", "Kevin De Bruyne", "Florian Wirtz"]
    selected_h2h = st.multiselect("Select players for deeper comparison:", options=players, default=[p for p in default_h2h if p in players])
    if len(selected_h2h) > 1:
        df_sel = df[df["player"].isin(selected_h2h)].reset_index(drop=True)
        tab1, tab2 = st.tabs(["📊 Player Archetype Plots", "📈 Detailed KPI Charts"])
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                radar_metrics = ['avg_xt_impact', 'progressive_passes_p90', 'successful_dribbles_p90', 'successful_tackles_p90', 'non_penalty_xg_p90']
                draw_radar_chart(df_sel, radar_metrics, "Player Style Radar")
            with col2:
                draw_scatter_plot(df_sel, 'avg_xt_impact', 'total_actions', 'Creator vs. Volume Analysis', 'Creative Threat (xT Impact)', 'Involvement (Touches p90)')
        with tab2:
            st.markdown("<br/>", unsafe_allow_html=True)
            def create_bar_chart(metric, title):
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=df_sel, x='player', y=metric, hue='player', palette='viridis', legend=False, ax=ax)
                ax.set_title(title, fontsize=11, fontweight='bold'); ax.set_xlabel(''); ax.set_ylabel(''); st.pyplot(fig)
            c1, c2 = st.columns(2)
            with c1:
                create_bar_chart('key_passes_p90', 'Key Passes per 90')
                create_bar_chart('progressive_passes_p90', 'Progressive Passes per 90')
                create_bar_chart('successful_tackles_p90', 'Successful Tackles per 90')
            with c2:
                create_bar_chart('non_penalty_xg_p90', 'Non-Penalty xG per 90')
                create_bar_chart('progressive_carries_p90', 'Progressive Carries per 90')
                create_bar_chart('interceptions_p90', 'Interceptions per 90')
    elif selected_h2h: st.info("Please select at least two players for a head-to-head comparison.")

# =======================================
# MAIN
# =======================================
def main():
    render_header()
    tab1, tab2 = st.tabs(["🎯 Match Analyzer", "📊 Advanced Analytics Board"])
    with tab1:
        render_match_analyzer()
    with tab2:
        render_season_comparison()

if __name__ == "__main__":
    main()