


# import os
# import base64
# from typing import Optional, Tuple

# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns

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

# PRIMARY_BG = "#f0f2f5"
# FOOTBALL_GREEN = "#28a745"
# st.markdown(
#     f"""
#     <style>

#         /* =============================
#            GLOBAL APP SETTINGS
#         ============================== */
#         .stApp {{
#             background-color: {PRIMARY_BG} !important;
#             color: #111827 !important; /* Set a default dark text color for the whole app */
#         }}
#         .block-container {{
#             max-width: 1200px !important;
#             margin: auto !important;
#             padding-top: 2rem !important;
#             padding-bottom: 3rem !important;
#         }}


#         /* =============================
#            PLAYER IMAGE
#         ============================== */
#         .player-image {{
#             width: 160px;
#             height: 160px;
#             border-radius: 50%;
#             object-fit: cover;
#             border: 4px solid white;
#             box-shadow: 0 12px 35px rgba(15, 23, 42, 0.25);
#         }}


#         /* =============================
#            HERO PLAYER CARD
#         ============================== */
#         .hero-card {{
#             background: white;
#             border-radius: 22px;
#             padding: 1.5rem 1.75rem;
#             box-shadow: 0 15px 40px rgba(15, 23, 42, 0.12);
#             border: 1px solid #e5e7eb;
#         }}

#         .hero-name {{
#             font-size: 1.8rem;
#             font-weight: 800;
#             margin-bottom: 0.25rem;
#             color: #111827;
#         }}

#         .hero-subtitle {{
#             font-size: 0.98rem;
#             color: #4b5563;
#             margin-bottom: 0.5rem;
#         }}

#         .hero-tag {{
#             display: inline-block;
#             padding: 0.25rem 0.65rem;
#             border-radius: 999px;
#             font-size: 0.75rem;
#             font-weight: 600;
#             background: rgba(40,167,69,0.08);
#             color: {FOOTBALL_GREEN};
#             margin-right: 0.4rem;
#         }}


#         /* =============================
#            KPI METRIC CARDS
#         ============================== */
#         .metric-card {{
#             background: white;
#             border-radius: 18px;
#             padding: 1rem 1.25rem;
#             box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
#             border: 1px solid #e5e7eb;
#             text-align: center;
#             transition: 0.2s ease-in-out;
#         }}

#         .metric-card:hover {{
#             transform: translateY(-4px);
#             box-shadow: 0 14px 36px rgba(15, 23, 42, 0.18);
#         }}

#         .metric-label {{
#             font-size: 0.78rem;
#             text-transform: uppercase;
#             letter-spacing: 0.08em;
#             color: #6b7280;
#             margin-bottom: 0.15rem;
#         }}

#         .metric-value {{
#             font-size: 1.3rem;
#             font-weight: 700;
#             color: #111827;
#         }}

#         .metric-sub {{
#             font-size: 0.78rem;
#             color: #9ca3af;
#             margin-top: 0.1rem;
#         }}


#         /* =============================
#            SECTION TITLES
#         ============================== */
#         .section-title {{
#             font-size: 1.1rem;
#             font-weight: 700;
#             margin-top: 1.5rem;
#             margin-bottom: 0.3rem;
#             color: #111827;
#         }}

#         .section-caption {{
#             font-size: 0.85rem;
#             color: #6b7280;
#             margin-bottom: 0.75rem;
#         }}


#         /* =============================
#            AI INSIGHT CARD
#         ============================== */
#         .insight-card {{
#             background: white;
#             border-radius: 18px;
#             padding: 1.25rem 1.5rem;
#             box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
#             border: 1px solid #e5e7eb;
#         }}

#         .insight-title {{
#             font-size: 0.9rem;
#             font-weight: 600;
#             text-transform: uppercase;
#             color: {FOOTBALL_GREEN};
#             letter-spacing: 0.15em;
#             margin-bottom: 0.5rem;
#         }}

#         .insight-body {{
#             font-size: 0.95rem;
#             color: #111827;
#             line-height: 1.5;
#         }}


#         /* =============================
#            FIX STREAMLIT TAB VISIBILITY
#         ============================== */

#         .stTabs [role="tab"] {{
#             background-color: white !important;
#             color: #374151 !important;
#             border-radius: 999px !important;
#             padding: 6px 18px !important;
#             border: 1px solid #d1d5db !important;
#             font-weight: 600 !important;
#             transition: all 0.2s ease-in-out !important;
#         }}

#         .stTabs [role="tab"][aria-selected="true"] {{
#             background-color: rgba(40,167,69,0.15) !important;
#             color: #1f2937 !important;
#             border-color: #28a745 !important;
#             font-weight: 700 !important;
#         }}

#         .stTabs [role="tab"]:hover {{
#             background-color: rgba(40,167,69,0.10) !important;
#             color: #1f2937 !important;
#         }}

#         /* =================================
#            SOLVED: UNIVERSAL VISIBILITY FIX
#         ================================= */
#         div[data-testid="stAlert"] * {{
#             color: #31333F !important; /* Force dark text for all elements inside any alert */
#         }}

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
#         if c in df.columns:
#             return c
#     return None


# def get_xy_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
#     candidates_x = ["x", "X", "pos_x", "start_x"]
#     candidates_y = ["y", "Y", "pos_y", "start_y"]
#     x_col = next((c for c in candidates_x if c in df.columns), None)
#     y_col = next((c for c in candidates_y if c in df.columns), None)
#     return x_col, y_col


# def compute_event_kpis(df: pd.DataFrame) -> dict:
#     kpis = {}
#     total_actions = len(df)
#     kpis["Total Actions"] = total_actions

#     xt_col = get_xt_column(df)
#     if xt_col:
#         kpis["Avg. xT Impact"] = float(df[xt_col].mean())
#     else:
#         kpis["Avg. xT Impact"] = None

#     # Pass accuracy
#     pass_mask = df["event_type"].astype(str).str.contains("pass", case=False, na=False)
#     total_passes = pass_mask.sum()
#     completed = None

#     if total_passes > 0:
#         if "outcome" in df.columns:
#             completed = df.loc[pass_mask, "outcome"].astype(str).str.contains(
#                 "success|complete|accurate|on_target", case=False, na=False
#             ).sum()
#         elif "is_complete" in df.columns:
#             completed = df.loc[pass_mask, "is_complete"].sum()

#     if completed is not None and total_passes > 0:
#         kpis["Pass Accuracy (%)"] = 100 * completed / total_passes
#     else:
#         kpis["Pass Accuracy (%)"] = None

#     # Progressive actions
#     if "is_progressive" in df.columns:
#         kpis["Progressive Actions"] = int(df["is_progressive"].sum())
#     elif "end_x" in df.columns and "x" in df.columns:
#         kpis["Progressive Actions"] = int((df["end_x"] - df["x"] >= 15).sum())
#     else:
#         kpis["Progressive Actions"] = 0

#     # Ball recoveries
#     if "is_recovery" in df.columns:
#         kpis["Ball Recoveries"] = int(df["is_recovery"].sum())
#     elif "event_type" in df.columns:
#         kpis["Ball Recoveries"] = int(
#             df["event_type"].astype(str).str.contains("recovery", case=False, na=False).sum()
#         )
#     else:
#         kpis["Ball Recoveries"] = 0

#     # Dribbles
#     if "event_type" in df.columns:
#         kpis["Dribbles"] = int(
#             df["event_type"].astype(str).str.contains("dribble", case=False, na=False).sum()
#         )
#     else:
#         kpis["Dribbles"] = 0

#     return kpis


# def compute_tracking_kpis(df: pd.DataFrame) -> dict:
#     kpis = {}
#     kpis["Total Touches"] = int(df["touch"].fillna(0).sum()) if "touch" in df.columns else 0
#     kpis["Total Tackles"] = int(df["tackle"].fillna(0).sum()) if "tackle" in df.columns else 0
#     kpis["Total Passes"] = 0  # extend if you add explicit pass flag

#     if "speed_m_per_s" in df.columns:
#         kpis["Top Speed (m/s)"] = float(df["speed_m_per_s"].max())
#     else:
#         kpis["Top Speed (m/s)"] = None

#     if "in_possession" in df.columns:
#         series = df["in_possession"].fillna(0)
#         if series.max() > 1:
#             sec = series.sum()
#         else:
#             sec = series.sum() * 0.5
#         kpis["Time in Possession (min)"] = sec / 60.0
#     else:
#         kpis["Time in Possession (min)"] = None

#     return kpis


# def ai_insight_event(df: pd.DataFrame, kpis: dict) -> str:
#     xt_col = get_xt_column(df)
#     avg_xt = kpis.get("Avg. xT Impact")

#     if avg_xt is None or np.isnan(avg_xt):
#         rating = "⚪ Rating not available (xT data missing)."
#     else:
#         if avg_xt < 0.03:
#             rating = "🧊 Quiet creative influence – more of a connector than a line-breaker in this game."
#         elif avg_xt < 0.08:
#             rating = "🟡 Solid connective display – stable circulation with selective risk-taking."
#         elif avg_xt < 0.15:
#             rating = "🟢 Strong creative game – consistently moved play into valuable zones."
#         else:
#             rating = "🔥 Elite creative performance – heavily involved in high-value actions toward goal."

#     x_col, y_col = get_xy_columns(df)
#     role_line = "Tactical role inference unavailable (not enough spatial data)."

#     if x_col and y_col:
#         coords = df[[x_col, y_col]].dropna()
#         if len(coords) >= 25:
#             kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
#             kmeans.fit(coords)
#             centers = kmeans.cluster_centers_
#             avg_x = centers[:, 0].mean()

#             if avg_x < 40:
#                 role = "Deep Pivot (#6 profile)"
#             elif avg_x < 70:
#                 role = "Hybrid #8 (connector & creator)"
#             else:
#                 role = "Advanced Creator in the half-spaces"

#             role_line = (
#                 f"K-Means clustering on Pedri’s on-ball locations yields an average effective x-position "
#                 f"around **{avg_x:.1f}**, which aligns with a **{role}** in this match."
#             )

#     total_actions = kpis.get("Total Actions", 0)
#     progressive = kpis.get("Progressive Actions", 0)
#     recov = kpis.get("Ball Recoveries", 0)
#     drib = kpis.get("Dribbles", 0)
#     pass_acc = kpis.get("Pass Accuracy (%)", None)

#     prog_rate = 100 * progressive / total_actions if total_actions > 0 else 0.0

#     lines = [
#         f"**Match Rating (xT-based)**  \n{rating}",
#         "",
#         f"**Volume & Progression**  \n"
#         f"- Total actions: **{total_actions}**, with ~**{prog_rate:.1f}%** classified as progressive.\n"
#         f"- Ball recoveries: **{recov}**, indicating how often he restarted attacks from regains.\n"
#         f"- Dribbles: **{drib}**, showing how frequently he broke lines with the ball.",
#     ]
#     if pass_acc is not None:
#         lines.append(
#             f"- Pass accuracy: **{pass_acc:.1f}%**, giving context on the risk level of his distribution."
#         )
#     lines.extend([
#         "",
#         f"**Tactical Role & Positioning**  \n{role_line}",
#         "",
#         "**Impact on Barcelona’s Strategy**  \n"
#         "The heatmap and progression map together illustrate where Pedri repeatedly received possession "
#         "and how he converted those touches into territorial or chance-building value. Dense clusters "
#         "in the left half-space typically mean Barcelona funnel their positional-play structure through him; "
#         "high-xT actions into the box highlight his role as a final-third creator rather than only a metronome."
#     ])

#     return "\n".join(lines)


# def ai_insight_tracking(df: pd.DataFrame, kpis: dict) -> str:
#     top_speed = kpis.get("Top Speed (m/s)")
#     poss_min = kpis.get("Time in Possession (min)")
#     tackles = kpis.get("Total Tackles")
#     touches = kpis.get("Total Touches")

#     if top_speed is None or np.isnan(top_speed):
#         work_line = "Work-rate cannot be fully quantified from speed data here, but spatial coverage still hints at his intensity."
#     elif top_speed < 5.5:
#         work_line = f"Top recorded speed of **{top_speed:.2f} m/s** suggests a more positional, tempo-controlling performance."
#     elif top_speed < 7.5:
#         work_line = f"With a top speed of **{top_speed:.2f} m/s**, Pedri reached regular high-intensity efforts to support both phases."
#     else:
#         work_line = f"Top speed of **{top_speed:.2f} m/s** indicates repeated sprint-level efforts, likely when arriving late into attacks or counter-pressing."

#     if poss_min is None or np.isnan(poss_min):
#         poss_line = "Time in active possession cannot be precisely inferred, but touch density still shows how central he was in circulation."
#     else:
#         poss_line = f"He spent approximately **{poss_min:.1f} minutes** in active ball possession, underlining his central role as a reference point for progression."

#     if tackles is None:
#         tackle_line = "Tackle data is incomplete."
#     elif tackles == 0:
#         tackle_line = "With **0 tackles**, his defensive involvement was more about screening and occupying passing lanes."
#     elif tackles < 4:
#         tackle_line = f"He contributed **{tackles} tackles**, stepping in selectively when the ball entered his zone."
#     else:
#         tackle_line = f"With **{tackles} tackles**, he was very active defensively, frequently leaving his slot to break opposition moves."

#     touch_line = "" if touches is None else f"He registered **{touches} touches**, which links his spatial coverage with frequent involvement in build-up and consolidation phases."

#     return (
#         f"**Work Rate & Intensity**  \n"
#         f"{work_line}  \n"
#         f"{poss_line}\n\n"
#         f"**Defensive Contribution**  \n"
#         f"{tackle_line}\n\n"
#         f"**Spatial Coverage & On-Ball Profile**  \n"
#         f"{touch_line}  \n"
#         "The combined touch and tackle maps typically show how he shuttles between deeper build-up spaces and higher half-spaces, "
#         "maintaining Barcelona’s positional structure while still offering depth in both pressing and chance creation."
#     )


# # =======================================
# # PITCH DRAWING HELPERS
# # =======================================

# def draw_event_heatmap(df: pd.DataFrame, title: str):
#     x_col, y_col = get_xy_columns(df)
#     if not x_col or not y_col:
#         st.warning("No usable x/y columns found for heatmap.")
#         return

#     coords = df[[x_col, y_col]].dropna()
#     if coords.empty:
#         st.warning("No coordinate data for heatmap.")
#         return

#     pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
#     fig, ax = pitch.draw(figsize=(6, 8))
#     sns.kdeplot(
#         x=coords[x_col],
#         y=coords[y_col],
#         fill=True,
#         cmap="inferno",
#         alpha=0.9,
#         ax=ax,
#         thresh=0.05,
#         bw_adjust=0.45,
#     )
#     ax.set_title(title, color="white", fontsize=12, fontweight="bold")
#     st.pyplot(fig)


# def draw_passing_network(df: pd.DataFrame, title: str):
#     x_col, y_col = get_xy_columns(df)
#     if not x_col or not y_col or "end_x" not in df.columns or "end_y" not in df.columns:
#         st.warning("Passing network requires x/y and end_x/end_y columns.")
#         return

#     mask = df["event_type"].astype(str).str.contains("pass", case=False, na=False)
#     if "outcome" in df.columns:
#         mask &= df["outcome"].astype(str).str.contains("success|complete|accurate|on_target", case=False, na=False)
#     elif "is_complete" in df.columns:
#         mask &= df["is_complete"] == 1

#     passes = df.loc[mask, [x_col, y_col, "end_x", "end_y"]].dropna()
#     if passes.empty:
#         st.warning("No completed passes found.")
#         return

#     pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
#     fig, ax = pitch.draw(figsize=(6, 8))

#     pitch.lines(
#         passes[x_col], passes[y_col], passes["end_x"], passes["end_y"],
#         ax=ax, comet=True, color="gold", lw=1.2, alpha=0.7
#     )

#     ax.set_title(title, color="white", fontsize=12, fontweight="bold")
#     st.pyplot(fig)


# def draw_ball_progression(df: pd.DataFrame, title: str):
#     if "end_x" not in df.columns or "end_y" not in df.columns:
#         st.warning("Ball progression requires end_x/end_y.")
#         return

#     x_col, y_col = get_xy_columns(df)
#     if not x_col or not y_col:
#         st.warning("No usable x/y columns for progression map.")
#         return

#     data = df.copy()
#     xt_col = get_xt_column(df)
#     if xt_col:
#         thresh = data[xt_col].quantile(0.75)
#         data = data[data[xt_col] >= thresh]

#     if "event_type" in data.columns:
#         data = data[data["event_type"].astype(str).str.contains("pass|carry|dribble", case=False, na=False)]

#     data = data[[x_col, y_col, "end_x", "end_y"]].dropna()
#     if data.empty:
#         st.warning("No high-value progression actions found.")
#         return

#     pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
#     fig, ax = pitch.draw(figsize=(6, 8))

#     pitch.arrows(
#         data[x_col], data[y_col], data["end_x"], data["end_y"],
#         ax=ax, width=1.2, headwidth=2.5, headlength=2.5, color="#ffdd57", alpha=0.9
#     )

#     ax.set_title(title, color="white", fontsize=12, fontweight="bold")
#     st.pyplot(fig)


# def draw_touch_tackle_map(df: pd.DataFrame, title: str):
#     x_col, y_col = get_xy_columns(df)
#     if not x_col or not y_col:
#         st.warning("No usable x/y columns for touch & tackle map.")
#         return

#     pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
#     fig, ax = pitch.draw(figsize=(6, 8))

#     if "touch" in df.columns:
#         touches = df[df["touch"].fillna(0) > 0]
#         ax.scatter(touches[x_col], touches[y_col], s=18, color=FOOTBALL_GREEN, alpha=0.8, label="Touches")

#     if "tackle" in df.columns:
#         tackles = df[df["tackle"].fillna(0) > 0]
#         ax.scatter(tackles[x_col], tackles[y_col], s=32, marker="x", color="#e11d48", alpha=0.9, label="Tackles")

#     leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=9)
#     for text in leg.get_texts():
#         text.set_color("white")

#     ax.set_title(title, color="white", fontsize=12, fontweight="bold")
#     st.pyplot(fig)


# # =======================================
# # HEADER
# # =======================================

# def render_header():
#     st.markdown(
#         '<p style="font-size:0.85rem; color:#6b7280; text-transform:uppercase; '
#         'letter-spacing:0.20em;">FC Barcelona • 2024-25 La Liga</p>',
#         unsafe_allow_html=True,
#     )

#     col_img, col_info = st.columns([1, 2], gap="large")

#     with col_img:
#         img_path = "pedri.png"
#         if os.path.exists(img_path):
#             try:
#                 img64 = load_image_base64(img_path)
#                 st.markdown(f'<div style="display:flex; justify-content:center; align-items:center;"><img src="data:image/png;base64,{img64}" class="player-image" /></div>', unsafe_allow_html=True)
#             except Exception as e:
#                 st.warning(f"Could not load {img_path}: {e}")
#         else:
#             st.markdown('<div style="display:flex; justify-content:center; align-items:center;"><div style="width:160px; height:160px; border-radius:50%; background:linear-gradient(135deg,#22c55e,#16a34a); display:flex; align-items:center; justify-content:center; color:white; font-size:3rem; font-weight:800; box-shadow:0 12px 35px rgba(15,23,42,0.25);">8</div></div>', unsafe_allow_html=True)

#     with col_info:
#         st.markdown(
#             """
#             <div class="hero-card">
#                 <div class="hero-name">Pedro González López</div>
#                 <div class="hero-subtitle">FC Barcelona • Midfielder • #8</div>
#                 <div style="margin-bottom:0.6rem;">
#                     <span class="hero-tag">Positional Play Engine</span>
#                     <span class="hero-tag">Left Half-Space Creator</span>
#                     <span class="hero-tag">2024-25 La Liga</span>
#                 </div>
#                 <div style="font-size:0.9rem; color:#4b5563;">
#                     Premium spatial-intelligence dashboard for match-by-match
#                     heatmap analysis, role detection, and seasonal benchmarking
#                     of Pedri’s influence on Barcelona’s positional structure.
#                 </div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#     st.markdown("<br/>", unsafe_allow_html=True)

#     st.markdown(
#         """
#         <div class="section-title">2024-25 Season Snapshot (Static)</div>
#         <div class="section-caption">
#             Fixed KPIs so the dashboard already feels rich and analytical,
#             even before loading any match data.
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     cols = st.columns(6)
#     def metric(col, label, value, sub=None):
#         with col:
#             sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ''
#             st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div>{sub_html}</div>', unsafe_allow_html=True)

#     metric(cols[0], "Matches Played", "24", "All competitions 24/25")
#     metric(cols[1], "La Liga Goals", "6", "Late box arrivals & shots")
#     metric(cols[2], "Assists", "8", "Final ball & secondary assists")
#     metric(cols[3], "Avg. xT / 90", "0.19", "Expected Threat impact")
#     metric(cols[4], "Prog Actions / 90", "17.4", "Carries & passes")
#     metric(cols[5], "Pressing / 90", "9.1", "Out-of-possession work")


# # =======================================
# # MATCH ANALYZER
# # =======================================
# def render_match_analyzer():
#     st.markdown(
#         """
#         <div class="section-title">Match Analyzer</div>
#         <div class="section-caption">
#             Upload a single-match CSV (event or tracking data). The app will auto-detect
#             the schema and generate tailored KPIs, visuals, and AI-tactical insights.
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     uploaded = st.file_uploader("Drag and drop file here", type=["csv"], help="Use pedri_events_matchXX.csv or pedri_tracking_matchXX.csv from your generator.", label_visibility="collapsed")

#     if uploaded is None:
#         st.info("Upload a match file to start the analysis.")
#         return

#     try:
#         df = pd.read_csv(uploaded)
#     except Exception as e:
#         st.error(f"Could not read CSV file: {e}")
#         return

#     cols = set(df.columns)
#     event_sig = {"event_type", "end_x", "end_y"}
#     tracking_sig = {"speed_m_per_s", "touch", "in_possession"}

#     mode = "event" if event_sig.issubset(cols) else "tracking" if tracking_sig.issubset(cols) else None

#     st.markdown(f"**Detected data type:** `{mode or 'unknown'}`")

#     if mode is None:
#         st.error("Unknown schema. Check your columns.")
#         st.dataframe(df.head())
#         return

#     if mode == "event":
#         kpis = compute_event_kpis(df)
#         labels = ["Total Actions", "Avg. xT Impact", "Pass Accuracy (%)", "Progressive Actions", "Ball Recoveries", "Dribbles"]
#         kpi_cols = st.columns(len(labels))
#         for col, label in zip(kpi_cols, labels):
#             val = kpis.get(label)
#             disp = f"{val:.2f}" if isinstance(val, float) else str(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else "–"
#             st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{disp}</div></div>', unsafe_allow_html=True)

#         tab1, tab2, tab3 = st.tabs(["Heatmap", "Passing Network", "Ball Progression"])
#         with tab1: draw_event_heatmap(df, "Pedri – On-ball Action Density")
#         with tab2: draw_passing_network(df, "Pedri – Completed Passing Network")
#         with tab3: draw_ball_progression(df, "Pedri – High-Value Ball Progression")

#         insight_html = ai_insight_event(df, kpis).replace("\n", "<br/>")
#         st.markdown(f'<br/><div class="insight-card"><div class="insight-title">AI Match Insight</div><div class="insight-body">{insight_html}</div></div>', unsafe_allow_html=True)

#     elif mode == "tracking":
#         kpis = compute_tracking_kpis(df)
#         labels = ["Total Touches", "Total Passes", "Total Tackles", "Top Speed (m/s)", "Time in Possession (min)"]
#         kpi_cols = st.columns(len(labels))
#         for col, label in zip(kpi_cols, labels):
#             val = kpis.get(label)
#             disp = f"{val:.2f}" if isinstance(val, float) else str(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else "–"
#             st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{disp}</div></div>', unsafe_allow_html=True)
        
#         tab1, tab2 = st.tabs(["Heatmap", "Touch & Tackle Map"])
#         with tab1: draw_event_heatmap(df, "Pedri – Movement Heatmap (Tracking)")
#         with tab2: draw_touch_tackle_map(df, "Pedri – Touches & Tackles")

#         insight_html = ai_insight_tracking(df, kpis).replace("\n", "<br/>")
#         st.markdown(f'<br/><div class="insight-card"><div class="insight-title">AI Tracking Insight</div><div class="insight-body">{insight_html}</div></div>', unsafe_allow_html=True)

# # =======================================
# # ADVANCED SEASON COMPARISON PLOTS
# # =======================================
# def draw_radar_chart(df: pd.DataFrame, metrics: list, title: str):
#     labels = [m.replace('_', ' ').replace(' p90', '').replace(' pct', ' (%)').title() for m in metrics]
#     num_vars = len(labels)

#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#     angles += angles[:1]

#     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

#     # Normalize the data for radar chart plotting
#     df_norm = df.copy()
#     for m in metrics:
#         min_val, max_val = df_norm[m].min(), df_norm[m].max()
#         df_norm[m] = (df[m] - min_val) / (max_val - min_val) if max_val > min_val else 0.5

#     for i, row in df_norm.iterrows():
#         values = row[metrics].tolist()
#         values += values[:1]
#         ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=row['player'])
#         ax.fill(angles, values, alpha=0.15)

#     ax.set_yticklabels([])
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels)
#     ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
#     ax.set_title(title, size=12, y=1.12)
#     st.pyplot(fig)


# def draw_scatter_plot(df: pd.DataFrame, x_metric: str, y_metric: str, title: str, xlabel: str, ylabel: str):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.scatterplot(data=df, x=x_metric, y=y_metric, hue="player", s=120, palette="viridis", legend=False, ax=ax)

#     for i, row in df.iterrows():
#         ax.text(row[x_metric] + 0.001, row[y_metric] + 0.2, row['player'], fontdict={'size': 9})

#     ax.set_title(title, fontsize=12, fontweight='bold')
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.grid(True, which='both', linestyle='--', linewidth=0.3)
#     st.pyplot(fig)


# # =======================================
# # ENHANCED SEASON COMPARISON
# # =======================================
# def render_season_comparison():
#     st.markdown(
#         """
#         <div class="section-title">Advanced Season Benchmarking</div>
#         <div class="section-caption">
#             Compare Pedri against other elite midfielders using advanced visualizations to understand player archetypes and performance profiles.
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     csv_path = "player_comparison_data.csv"
#     if not os.path.exists(csv_path):
#         st.error(f"`{csv_path}` not found. Please ensure the comparison data file is in the app directory.")
#         return

#     try:
#         df = pd.read_csv(csv_path)
#     except Exception as e:
#         st.error(f"Could not read `{csv_path}`: {e}")
#         return

#     required_cols = {'player', 'avg_xt_impact', 'total_actions', 'progressive_passes_p90', 'progressive_carries_p90', 'pass_completion_pct', 'successful_tackles_p90'}
#     if not required_cols.issubset(df.columns):
#         st.error(f"Comparison file is missing required columns. It must contain: {', '.join(required_cols)}")
#         return

#     players = sorted(df["player"].unique().tolist())
#     default_players = [p for p in players if "Pedri" in p]
#     if not default_players:
#         default_players = players[:3] if len(players) > 2 else players

#     selected = st.multiselect("Select players to compare", options=players, default=default_players)

#     if not selected:
#         st.info("Select at least one player to begin comparison.")
#         return

#     df_sel = df[df["player"].isin(selected)].reset_index(drop=True)

#     st.markdown("---")

#     # --- New Advanced Visualizations ---
#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("<h3 style='text-align: center; font-size: 1rem;'>Midfielder Profile Radar</h3>", unsafe_allow_html=True)
#         radar_metrics = [
#             'avg_xt_impact', 'progressive_passes_p90', 'progressive_carries_p90',
#             'successful_tackles_p90', 'pass_completion_pct'
#         ]
#         draw_radar_chart(df_sel, radar_metrics, "Player Style Comparison")

#     with col2:
#         st.markdown("<h3 style='text-align: center; font-size: 1rem;'>Creator vs. Volume Analysis</h3>", unsafe_allow_html=True)
#         draw_scatter_plot(
#             df=df_sel,
#             x_metric='avg_xt_impact',
#             y_metric='total_actions',
#             title='Player Archetype Analysis',
#             xlabel='Creative Threat (xT Impact per 90)',
#             ylabel='Involvement (Total Actions per 90)'
#         )

#     st.markdown("<br/>", unsafe_allow_html=True)


# # =======================================
# # MAIN
# # =======================================
# def main():
#     render_header()
#     tab1, tab2 = st.tabs(["🎯 Match Analyzer", "📊 Season Comparison"])
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
            max-width: 1200px !important;
            margin: auto !important;
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
        }}
        /* =============================
           PLAYER IMAGE
        ============================== */
        .player-image {{
            width: 160px;
            height: 160px;
            border-radius: 50%;
            object-fit: cover;
            border: 4px solid white;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.25);
        }}
        /* =============================
           HERO PLAYER CARD
        ============================== */
        .hero-card {{
            background: white;
            border-radius: 22px;
            padding: 1.5rem 1.75rem;
            box-shadow: 0 15px 40px rgba(15, 23, 42, 0.12);
            border: 1px solid #e5e7eb;
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
        .insight-body {{ font-size: 0.95rem; color: #111827; line-height: 1.5; }}
        /* =============================
           STREAMLIT TABS & ALERTS
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
        div[data-testid="stAlert"] * {{ color: #31333F !important; }}
        /* =============================
           CUSTOM PERCENTILE BARS
        ============================== */
        .percentile-container {{ display: flex; align-items: center; margin-bottom: 4px; height: 24px; }}
        .percentile-label {{ width: 220px; font-size: 0.8rem; padding-right: 10px; text-align: right; }}
        .percentile-bar-bg {{ flex-grow: 1; background-color: #e9ecef; border-radius: 5px; height: 18px; }}
        .percentile-bar-fg {{
            height: 100%; border-radius: 5px; text-align: right; color: white;
            font-size: 0.75rem; padding-right: 5px; line-height: 18px; font-weight: 600;
        }}
        .percentile-value {{ width: 50px; text-align: right; font-size: 0.8rem; font-weight: 600; padding-left: 10px; }}
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
    kpis["Avg. xT Impact"] = float(df[xt_col].mean()) if xt_col else None
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

def compute_tracking_kpis(df: pd.DataFrame) -> dict:
    kpis = {}
    kpis["Total Touches"] = int(df["touch"].fillna(0).sum()) if "touch" in df.columns else 0
    kpis["Total Tackles"] = int(df["tackle"].fillna(0).sum()) if "tackle" in df.columns else 0
    kpis["Total Passes"] = 0
    kpis["Top Speed (m/s)"] = float(df["speed_m_per_s"].max()) if "speed_m_per_s" in df.columns else None
    if "in_possession" in df.columns:
        series = df["in_possession"].fillna(0)
        sec = series.sum() if series.max() > 1 else series.sum() * 0.5
        kpis["Time in Possession (min)"] = sec / 60.0
    else:
        kpis["Time in Possession (min)"] = None
    return kpis

def ai_insight_event(df: pd.DataFrame, kpis: dict) -> str:
    avg_xt = kpis.get("Avg. xT Impact")
    if avg_xt is None or np.isnan(avg_xt): rating = "⚪ Rating not available (xT data missing)."
    elif avg_xt < 0.03: rating = "🧊 Quiet creative influence."
    elif avg_xt < 0.08: rating = "🟡 Solid connective display."
    elif avg_xt < 0.15: rating = "🟢 Strong creative game."
    else: rating = "🔥 Elite creative performance."
    return f"**Match Rating (xT-based):** {rating}"

def ai_insight_tracking(df: pd.DataFrame, kpis: dict) -> str:
    top_speed = kpis.get("Top Speed (m/s)")
    if top_speed is None or np.isnan(top_speed): work_line = "Work-rate cannot be fully quantified."
    elif top_speed < 7.5: work_line = f"Top speed of **{top_speed:.2f} m/s** suggests a controlled, positional performance."
    else: work_line = f"Top speed of **{top_speed:.2f} m/s** indicates repeated sprint-level efforts."
    return f"**Work Rate & Intensity:** {work_line}"


# =======================================
# PITCH DRAWING HELPERS
# =======================================
def draw_event_heatmap(df: pd.DataFrame, title: str):
    x_col, y_col = get_xy_columns(df)
    if not x_col or not y_col or df[[x_col, y_col]].dropna().empty: return
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
    fig, ax = pitch.draw(figsize=(6, 8)); sns.kdeplot(x=df[x_col], y=df[y_col], fill=True, cmap="inferno", alpha=0.9, ax=ax, thresh=0.05, bw_adjust=0.45)
    ax.set_title(title, color="white", fontsize=12, fontweight="bold"); st.pyplot(fig)

def draw_passing_network(df: pd.DataFrame, title: str):
    x_col, y_col = get_xy_columns(df)
    if not x_col or not y_col or "end_x" not in df.columns or "end_y" not in df.columns: return
    mask = df["event_type"].astype(str).str.contains("pass", case=False, na=False)
    if "is_complete" in df.columns: mask &= df["is_complete"] == 1
    passes = df.loc[mask].dropna(subset=[x_col, y_col, "end_x", "end_y"])
    if passes.empty: return
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
    fig, ax = pitch.draw(figsize=(6, 8))
    pitch.lines(passes[x_col], passes[y_col], passes["end_x"], passes["end_y"], ax=ax, comet=True, color="gold", lw=1.2, alpha=0.7)
    ax.set_title(title, color="white", fontsize=12, fontweight="bold"); st.pyplot(fig)

def draw_ball_progression(df: pd.DataFrame, title: str):
    x_col, y_col = get_xy_columns(df)
    if not x_col or not y_col or "end_x" not in df.columns or "end_y" not in df.columns: return
    data = df[df["event_type"].astype(str).str.contains("pass|carry|dribble", case=False, na=False)].copy()
    xt_col = get_xt_column(data)
    if xt_col: data = data[data[xt_col] >= data[xt_col].quantile(0.75)]
    data.dropna(subset=[x_col, y_col, "end_x", "end_y"], inplace=True)
    if data.empty: return
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
    fig, ax = pitch.draw(figsize=(6, 8))
    pitch.arrows(data[x_col], data[y_col], data["end_x"], data["end_y"], ax=ax, width=1.2, headwidth=2.5, headlength=2.5, color="#ffdd57", alpha=0.9)
    ax.set_title(title, color="white", fontsize=12, fontweight="bold"); st.pyplot(fig)

def draw_touch_tackle_map(df: pd.DataFrame, title: str):
    x_col, y_col = get_xy_columns(df)
    if not x_col or not y_col: return
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
    fig, ax = pitch.draw(figsize=(6, 8))
    if "touch" in df.columns: ax.scatter(df[df["touch"] > 0][x_col], df[df["touch"] > 0][y_col], s=18, color=FOOTBALL_GREEN, alpha=0.8, label="Touches")
    if "tackle" in df.columns: ax.scatter(df[df["tackle"] > 0][x_col], df[df["tackle"] > 0][y_col], s=32, marker="x", color="#e11d48", alpha=0.9, label="Tackles")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=9)
    for text in leg.get_texts(): text.set_color("white")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold"); st.pyplot(fig)


# =======================================
# HEADER
# =======================================
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
    metrics_data = [("Matches Played", "24", "All comps"), ("La Liga Goals", "6", "Box arrivals"), ("Assists", "8", "Final ball"), ("Avg. xT / 90", "0.19", "Threat impact"), ("Prog Actions / 90", "17.4", "Carries & passes"), ("Pressing / 90", "9.1", "Defensive work")]
    for col, (label, value, sub) in zip(cols, metrics_data):
        with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)


# =======================================
# MATCH ANALYZER
# =======================================
def render_match_analyzer():
    st.markdown('<div class="section-title">Match Analyzer</div><div class="section-caption">Upload a single-match CSV to generate tailored KPIs, visuals, and AI-tactical insights.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Match Data", type=["csv"], label_visibility="collapsed")
    if uploaded is None:
        st.info("Upload a match file to start the analysis."); return
    try: df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV file: {e}"); return
    mode = "event" if {"event_type", "end_x"}.issubset(df.columns) else "tracking" if {"speed_m_per_s", "touch"}.issubset(df.columns) else None
    st.markdown(f"**Detected data type:** `{mode or 'unknown'}`")
    if mode is None:
        st.error("Unknown schema. Check your columns."); st.dataframe(df.head()); return
    if mode == "event":
        kpis = compute_event_kpis(df)
        labels = ["Total Actions", "Avg. xT Impact", "Pass Accuracy (%)", "Progressive Actions", "Ball Recoveries", "Dribbles"]
        cols = st.columns(len(labels))
        for col, label in zip(cols, labels):
            val = kpis.get(label); disp = f"{val:.2f}" if isinstance(val, float) else str(val) if val is not None and not np.isnan(val) else "–"
            with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{disp}</div></div>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Heatmap", "Passing Network", "Ball Progression"])
        with tab1: draw_event_heatmap(df, "On-ball Action Density")
        with tab2: draw_passing_network(df, "Completed Passing Network")
        with tab3: draw_ball_progression(df, "High-Value Ball Progression")
        insight = ai_insight_event(df, kpis).replace("\n", "<br/>")
        st.markdown(f'<br/><div class="insight-card"><div class="insight-title">AI Match Insight</div><div class="insight-body">{insight}</div></div>', unsafe_allow_html=True)
    elif mode == "tracking":
        kpis = compute_tracking_kpis(df)
        labels = ["Total Touches", "Total Tackles", "Top Speed (m/s)", "Time in Possession (min)"]
        cols = st.columns(len(labels))
        for col, label in zip(cols, labels):
            val = kpis.get(label); disp = f"{val:.2f}" if isinstance(val, float) else str(val) if val is not None and not np.isnan(val) else "–"
            with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{disp}</div></div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Heatmap", "Touch & Tackle Map"])
        with tab1: draw_event_heatmap(df, "Movement Heatmap (Tracking)")
        with tab2: draw_touch_tackle_map(df, "Touches & Tackles")
        insight = ai_insight_tracking(df, kpis).replace("\n", "<br/>")
        st.markdown(f'<br/><div class="insight-card"><div class="insight-title">AI Tracking Insight</div><div class="insight-body">{insight}</div></div>', unsafe_allow_html=True)


# =======================================
# ADVANCED SEASON COMPARISON PLOTS
# =======================================
def get_percentile_color(p):
    if p > 90: return "#059669";
    if p > 75: return "#10B981";
    if p > 50: return "#F59E0B";
    if p > 25: return "#F97316";
    return "#EF4444"

def draw_percentile_profile(player_data: pd.Series, all_data: pd.DataFrame, metrics: list):
    st.markdown(f"<h3 style='font-size: 1.2rem; text-align: center; margin-bottom: 1rem;'>{player_data['player']} vs. Elite Midfielders</h3>", unsafe_allow_html=True)
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
        min_val, max_val = df_norm[m].min(), df_norm[m].max()
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

# =======================================
# NEW ADVANCED ANALYTICS BOARD
# =======================================
def render_season_comparison():
    st.markdown('<div class="section-title">Advanced Midfielder Analytics Board</div><div class="section-caption">Benchmark elite midfielders from the 2024-25 season. Analyze player profiles, compare KPIs, and identify tactical archetypes.</div>', unsafe_allow_html=True)
    csv_path = "player_comparison_data.csv"
    if not os.path.exists(csv_path):
        st.error(f"`{csv_path}` not found. Please ensure the comparison data file is in the app directory."); return
    try: df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Could not read `{csv_path}`: {e}"); return
    players = sorted(df["player"].unique().tolist())
    st.markdown("---")
    st.markdown("<h3 style='font-size: 1.1rem;'>Player Profile Analysis</h3>", unsafe_allow_html=True)
    profile_player = st.selectbox("Select a player to view their detailed percentile profile:", players, index=players.index("Pedri") if "Pedri" in players else 0)
    if profile_player:
        player_data = df[df['player'] == profile_player].iloc[0]
        profile_metrics = ['avg_xt_impact', 'key_passes_p90', 'non_penalty_xg_p90', 'shots_p90', 'progressive_passes_p90', 'progressive_carries_p90', 'successful_dribbles_p90', 'successful_tackles_p90', 'interceptions_p90', 'pass_completion_pct', 'total_actions']
        draw_percentile_profile(player_data, df, profile_metrics)
    st.markdown("<hr style='margin: 2rem 0;'/><h3 style='font-size: 1.1rem;'>Head-to-Head Comparison</h3>", unsafe_allow_html=True)
    default_players = ["Pedri", "Jude Bellingham", "Kevin De Bruyne", "Florian Wirtz"]
    selected = st.multiselect("Select players for head-to-head comparison:", options=players, default=[p for p in default_players if p in players])
    if len(selected) > 1:
        df_sel = df[df["player"].isin(selected)].reset_index(drop=True)
        tab1, tab2 = st.tabs(["📊 Player Archetype Plots", "📈 Detailed KPI Charts"])
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                radar_metrics = ['avg_xt_impact', 'progressive_passes_p90', 'successful_dribbles_p90', 'successful_tackles_p90', 'non_penalty_xg_p90']
                draw_radar_chart(df_sel, radar_metrics, "Player Style Radar")
            with col2:
                draw_scatter_plot(df_sel, 'avg_xt_impact', 'total_actions', 'Creator vs. Volume Analysis', 'Creative Threat (xT Impact)', 'Involvement (Total Actions)')
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
    elif selected: st.info("Please select at least two players for a head-to-head comparison.")


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