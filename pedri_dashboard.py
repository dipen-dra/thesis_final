# import os
# from typing import Optional, Tuple

# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns

# from mplsoccer import Pitch
# from sklearn.cluster import KMeans

# # --------------------------
# # STREAMLIT PAGE CONFIG
# # --------------------------
# st.set_page_config(
#     page_title="Pedri Spatial Intelligence Lab",
#     page_icon="⚽",
#     layout="wide",
# )

# # --------------------------
# # GLOBAL STYLE (CSS)
# # --------------------------
# PRIMARY_BG = "#f0f2f5"
# FOOTBALL_GREEN = "#28a745"

# st.markdown(
#     f"""
#     <style>
#         /* Global background */
#         .stApp {{
#             background-color: {PRIMARY_BG};
#         }}

#         /* Constrain main content width and center it */
#         .block-container {{
#             max-width: 1200px;
#             padding-top: 2rem;
#             padding-bottom: 3rem;
#             margin: auto;
#         }}

#         /* Tabs styling */
#         .stTabs [role="tablist"] {{
#             gap: 1rem;
#         }}
#         .stTabs [role="tab"] {{
#             padding: 0.5rem 1.25rem;
#             border-radius: 999px;
#             border: 1px solid #d0d0d0;
#             background: white;
#         }}
#         .stTabs [aria-selected="true"] {{
#             border: 1px solid {FOOTBALL_GREEN};
#             background: rgba(40,167,69,0.1);
#             color: #111 !important;
#         }}

#         /* KPI metric cards */
#         .metric-card {{
#             background: white;
#             border-radius: 18px;
#             padding: 1rem 1.25rem;
#             box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
#             border: 1px solid #e5e7eb;
#         }}
#         .metric-label {{
#             font-size: 0.80rem;
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

#         /* Insight card */
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

#         /* Player header card */
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

#         /* Circular player image */
#         .player-image {{
#             width: 160px;
#             height: 160px;
#             border-radius: 50%;
#             object-fit: cover;
#             border: 4px solid white;
#             box-shadow: 0 12px 35px rgba(15, 23, 42, 0.18);
#         }}

#         /* Section headings */
#         .section-title {{
#             font-size: 1.1rem;
#             font-weight: 700;
#             margin-top: 1.5rem;
#             margin-bottom: 0.5rem;
#             color: #111827;
#         }}
#         .section-caption {{
#             font-size: 0.85rem;
#             color: #6b7280;
#             margin-bottom: 0.75rem;
#         }}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # --------------------------
# # UTILITY FUNCTIONS
# # --------------------------


# def get_xt_column(df: pd.DataFrame) -> Optional[str]:
#     candidates = ["xt", "xT", "xT_value", "expected_threat", "xT_impact"]
#     for c in candidates:
#         if c in df.columns:
#             return c
#     return None


# def get_xy_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
#     """
#     Try common coordinate column names for football data.
#     """
#     candidates_x = ["x", "X", "pos_x", "end_x", "start_x"]
#     candidates_y = ["y", "Y", "pos_y", "end_y", "start_y"]
#     x_col = next((c for c in candidates_x if c in df.columns), None)
#     y_col = next((c for c in candidates_y if c in df.columns), None)
#     return x_col, y_col


# def safe_int(value) -> Optional[int]:
#     try:
#         return int(value)
#     except Exception:
#         return None


# def safe_float(value) -> Optional[float]:
#     try:
#         return float(value)
#     except Exception:
#         return None


# def _event_pass_mask(df: pd.DataFrame) -> pd.Series:
#     if "event_type" not in df.columns:
#         return pd.Series(False, index=df.index)
#     return df["event_type"].astype(str).str.contains("pass", case=False, na=False)


# def _event_progressive_mask(df: pd.DataFrame, x_col: Optional[str]) -> pd.Series:
#     if "is_progressive" in df.columns:
#         return df["is_progressive"] == 1
#     if x_col and "end_x" in df.columns:
#         # simple heuristic: progressive if ball moves forward by >15% of pitch
#         try:
#             return (df["end_x"] - df[x_col]) >= 15
#         except Exception:
#             return pd.Series(False, index=df.index)
#     return pd.Series(False, index=df.index)


# def _event_ball_recovery_mask(df: pd.DataFrame) -> pd.Series:
#     if "is_recovery" in df.columns:
#         return df["is_recovery"] == 1
#     if "event_type" in df.columns:
#         return df["event_type"].astype(str).str.contains("recovery", case=False, na=False)
#     return pd.Series(False, index=df.index)


# def _event_dribble_mask(df: pd.DataFrame) -> pd.Series:
#     if "event_type" in df.columns:
#         return df["event_type"].astype(str).str.contains("dribble", case=False, na=False)
#     return pd.Series(False, index=df.index)


# def compute_event_kpis(df: pd.DataFrame) -> dict:
#     kpis = {}

#     total_actions = len(df)
#     kpis["Total Actions"] = total_actions

#     xt_col = get_xt_column(df)
#     if xt_col:
#         avg_xt = safe_float(df[xt_col].mean())
#     else:
#         avg_xt = None
#     kpis["Avg. xT Impact"] = avg_xt

#     # Pass accuracy
#     pass_mask = _event_pass_mask(df)
#     total_passes = int(pass_mask.sum())
#     completed_passes = None

#     if total_passes > 0:
#         if "outcome" in df.columns:
#             completed_passes = int(
#                 df.loc[pass_mask, "outcome"]
#                 .astype(str)
#                 .str.contains("complete|accurate|success", case=False, na=False)
#                 .sum()
#             )
#         elif "is_complete" in df.columns:
#             completed_passes = int(df.loc[pass_mask, "is_complete"].sum())
#         else:
#             completed_passes = None

#     if completed_passes is not None and total_passes > 0:
#         pass_accuracy = round(100 * completed_passes / total_passes, 1)
#     else:
#         pass_accuracy = None
#     kpis["Pass Accuracy (%)"] = pass_accuracy

#     # Progressive actions
#     x_col, _ = get_xy_columns(df)
#     progressive_mask = _event_progressive_mask(df, x_col)
#     kpis["Progressive Actions"] = int(progressive_mask.sum())

#     # Ball recoveries
#     ball_recovery_mask = _event_ball_recovery_mask(df)
#     kpis["Ball Recoveries"] = int(ball_recovery_mask.sum())

#     # Dribbles
#     dribble_mask = _event_dribble_mask(df)
#     kpis["Dribbles"] = int(dribble_mask.sum())

#     return kpis


# def compute_tracking_kpis(df: pd.DataFrame) -> dict:
#     kpis = {}

#     # Total touches
#     if "touch" in df.columns:
#         touches_series = df["touch"].fillna(0)
#         if touches_series.nunique() <= 5:  # likely 0/1 or small ints
#             total_touches = int((touches_series > 0).sum())
#         else:
#             total_touches = int(touches_series.sum())
#     else:
#         total_touches = 0

#     kpis["Total Touches"] = total_touches

#     # Total passes & tackles if flags exist
#     total_passes = 0
#     total_tackles = 0
#     for col in df.columns:
#         if "pass" in col.lower() and df[col].dtype != "O":
#             total_passes = int(df[col].fillna(0).sum())
#         if "tackle" in col.lower() and df[col].dtype != "O":
#             total_tackles = int(df[col].fillna(0).sum())

#     kpis["Total Passes"] = total_passes
#     kpis["Total Tackles"] = total_tackles

#     # Top speed
#     if "speed_m_per_s" in df.columns:
#         top_speed = safe_float(df["speed_m_per_s"].max())
#     else:
#         top_speed = None
#     kpis["Top Speed (m/s)"] = top_speed

#     # Time in possession (min)
#     if "in_possession" in df.columns:
#         series = df["in_possession"].fillna(0)
#         # heuristic: if values > 1, assume "seconds"; else assume 0/1 indicator at ~25 Hz
#         if series.max() > 1:
#             possession_seconds = float(series.sum())
#         else:
#             possession_seconds = float(series.sum() * 0.04)  # ~25 frames/s
#         kpis["Time in Possession (min)"] = possession_seconds / 60.0
#     else:
#         kpis["Time in Possession (min)"] = None

#     return kpis


# def ai_insight_event(df: pd.DataFrame, kpis: dict) -> str:
#     xt_col = get_xt_column(df)
#     avg_xt = kpis.get("Avg. xT Impact") if xt_col else None

#     # Rating based on xT
#     if avg_xt is None or np.isnan(avg_xt):
#         rating = "⚪ Data not sufficient for a reliable match rating."
#     else:
#         if avg_xt < 0.03:
#             rating = "🧊 Quiet game – Pedri had limited final-third impact."
#         elif avg_xt < 0.08:
#             rating = "🟡 Solid but unspectacular – mostly connective, low-risk play."
#         elif avg_xt < 0.15:
#             rating = "🟢 Strong creative display – consistently moved the ball into useful zones."
#         else:
#             rating = "🔥 Elite creative performance – heavily involved in high-value actions."

#     # Tactical role via K-Means on spatial data
#     x_col, y_col = get_xy_columns(df)
#     role_text = "Tactical role could not be inferred (insufficient spatial data)."

#     if x_col and y_col:
#         coords = df[[x_col, y_col]].dropna()
#         if len(coords) >= 25:
#             n_clusters = 3
#             kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
#             kmeans.fit(coords)
#             centers = kmeans.cluster_centers_
#             avg_x_center = centers[:, 0].mean()

#             # simple heuristic for StatsBomb-like 0-120 pitch
#             if avg_x_center < 40:
#                 role_label = "Deep Pivot"
#             elif avg_x_center < 65:
#                 role_label = "Hybrid #8 (Connector & Creator)"
#             else:
#                 role_label = "Advanced Creator / Half-Space #10"

#             role_text = (
#                 f"K-Means clustering on Pedri’s on-ball locations reveals an average "
#                 f"effective position around **x ≈ {avg_x_center:.1f}**. "
#                 f"This pattern aligns with a **{role_label}** profile, combining his "
#                 f"receiving zones and subsequent action points."
#             )

#     total_actions = kpis.get("Total Actions", 0)
#     progressive_actions = kpis.get("Progressive Actions", 0)
#     ball_recoveries = kpis.get("Ball Recoveries", 0)
#     dribbles = kpis.get("Dribbles", 0)
#     pass_acc = kpis.get("Pass Accuracy (%)", None)

#     if total_actions > 0:
#         prog_rate = 100 * progressive_actions / total_actions
#     else:
#         prog_rate = 0.0

#     desc_lines = []

#     desc_lines.append(f"**Match Rating (xT-based)**  \n{rating}")
#     desc_lines.append("")
#     desc_lines.append(
#         f"**Spatial & Positional Behaviour**  \n"
#         f"- Total actions: **{total_actions}** with "
#         f"~**{prog_rate:.1f}%** classified as progressive, indicating how often he "
#         f"helped the team climb through the thirds.\n"
#         f"- Ball recoveries: **{ball_recoveries}**, showing how frequently he "
#         f"re-established possession and triggered new attacking phases.\n"
#         f"- Dribbles: **{dribbles}**, which reflects his willingness to break pressure "
#         f"with the ball rather than only circulating it."
#     )

#     if pass_acc is not None:
#         desc_lines.append(
#             f"- Pass accuracy sits at **{pass_acc:.1f}%**, framing the balance between "
#             f"risk-taking and ball security."
#         )

#     desc_lines.append("")
#     desc_lines.append(f"**Tactical Role Inference**  \n{role_text}")
#     desc_lines.append("")
#     desc_lines.append(
#         "**Impact on Team Strategy**  \n"
#         "Pedri’s heatmap and progression map together show **where** he repeatedly "
#         "touched the ball and **how** those touches changed the state of the attack. "
#         "Dense activity clusters in the half-spaces usually indicate that Barcelona "
#         "are funnelling their positional play through him, while repeated high-xT "
#         "actions into the box signal that he is not only a connector but also a key "
#         "finishing passer in the final third."
#     )

#     return "\n".join(desc_lines)


# def ai_insight_tracking(df: pd.DataFrame, kpis: dict) -> str:
#     top_speed = kpis.get("Top Speed (m/s)")
#     possession_min = kpis.get("Time in Possession (min)")
#     total_tackles = kpis.get("Total Tackles")
#     total_touches = kpis.get("Total Touches")

#     # Work rate assessment
#     if top_speed is None or np.isnan(top_speed):
#         work_rate_line = (
#             "Work rate can’t be fully quantified from the available speed data, "
#             "but the spatial coverage map still shows his involvement across lanes."
#         )
#     else:
#         if top_speed < 6:
#             work_rate_line = (
#                 f"Top recorded speed of **{top_speed:.2f} m/s** suggests a more "
#                 f"controlled, tempo-setting performance rather than repeated deep sprints."
#             )
#         elif top_speed < 8:
#             work_rate_line = (
#                 f"Top speed of **{top_speed:.2f} m/s** indicates regular high-intensity "
#                 f"runs to support both sides of the ball."
#             )
#         else:
#             work_rate_line = (
#                 f"With a top speed of **{top_speed:.2f} m/s**, he reached genuine sprinting "
#                 f"intensity, often required to arrive into or recover from advanced positions."
#             )

#     if possession_min is None or np.isnan(possession_min):
#         possession_line = (
#             "Time in possession can’t be precisely inferred from the raw tracking, "
#             "but the density of touches still signals how central he was as a reference point."
#         )
#     else:
#         possession_line = (
#             f"He spent approximately **{possession_min:.1f} minutes** in active ball "
#             f"possession, underlining how often he acted as the pivot for circulation."
#         )

#     # Defensive contribution
#     if total_tackles is None:
#         tackles_line = "Tackle data is incomplete, so defensive interventions are approximated."
#     else:
#         if total_tackles == 0:
#             tackles_line = (
#                 "He recorded **no tackles**, implying his defensive role was more about "
#                 "screening lanes and positioning rather than direct duels."
#             )
#         elif total_tackles < 3:
#             tackles_line = (
#                 f"He engaged in **{total_tackles} tackles**, contributing selectively "
#                 f"when the ball entered his zone."
#             )
#         else:
#             tackles_line = (
#                 f"With **{total_tackles} tackles**, he was heavily active defensively, "
#                 f"often stepping out of the line to break opposition moves."
#             )

#     if total_touches is None:
#         touches_line = ""
#     else:
#         touches_line = (
#             f"Total touches of **{total_touches}** tie together his off-ball coverage "
#             f"with frequent on-ball involvement in the progression chains."
#         )

#     text = (
#         f"**Work Rate & Intensity**  \n"
#         f"{work_rate_line}  \n"
#         f"{possession_line}\n\n"
#         f"**Defensive Contribution**  \n"
#         f"{tackles_line}\n\n"
#         f"**Spatial Coverage & On-Ball Profile**  \n"
#         f"{touches_line}  \n"
#         f"The combined touch and tackle maps typically show how he shuttles between "
#         f"the defensive and middle thirds, plugging gaps on one side while still "
#         f"arriving in advanced pockets to maintain Barcelona’s positional structure."
#     )

#     return text


# def draw_event_heatmap(df: pd.DataFrame, title: str):
#     x_col, y_col = get_xy_columns(df)
#     if not x_col or not y_col:
#         st.warning("Could not find suitable x/y columns to render the heatmap.")
#         return

#     pitch = Pitch(
#         pitch_type="statsbomb",
#         pitch_color="#333f37",
#         line_color="white",
#     )
#     fig, ax = pitch.draw(figsize=(6, 8))
#     data = df[[x_col, y_col]].dropna()

#     if len(data) == 0:
#         st.warning("No coordinate data found for heatmap.")
#         return

#     sns.kdeplot(
#         x=data[x_col],
#         y=data[y_col],
#         fill=True,
#         shade=True,
#         thresh=0.05,
#         alpha=0.95,
#         cmap="inferno",
#         bw_adjust=0.45,
#         ax=ax,
#     )

#     ax.set_title(
#         title,
#         fontsize=12,
#         color="white",
#         pad=10,
#         fontweight="bold",
#     )
#     st.pyplot(fig)


# def draw_passing_network(df: pd.DataFrame, title: str):
#     x_col, y_col = get_xy_columns(df)
#     if not x_col or not y_col or "end_x" not in df.columns or "end_y" not in df.columns:
#         st.warning("Passing network requires x/y and end_x/end_y columns.")
#         return

#     pass_mask = _event_pass_mask(df)
#     if "outcome" in df.columns:
#         pass_mask &= df["outcome"].astype(str).str.contains(
#             "complete|accurate|success", case=False, na=False
#         )
#     elif "is_complete" in df.columns:
#         pass_mask &= df["is_complete"] == 1

#     data = df.loc[pass_mask, [x_col, y_col, "end_x", "end_y"]].dropna()
#     if len(data) == 0:
#         st.warning("No completed passes found to build passing network.")
#         return

#     pitch = Pitch(
#         pitch_type="statsbomb",
#         pitch_color="#333f37",
#         line_color="white",
#     )
#     fig, ax = pitch.draw(figsize=(6, 8))

#     # Origins & destinations
#     ax.scatter(
#         data[x_col],
#         data[y_col],
#         s=15,
#         alpha=0.8,
#     )
#     ax.scatter(
#         data["end_x"],
#         data["end_y"],
#         s=8,
#         alpha=0.7,
#     )

#     # Pass lines
#     for _, row in data.iterrows():
#         pitch.lines(
#             row[x_col],
#             row[y_col],
#             row["end_x"],
#             row["end_y"],
#             comet=True,
#             transparent=True,
#             ax=ax,
#             lw=1.2,
#             alpha=0.6,
#             color="gold",
#         )

#     ax.set_title(
#         title,
#         fontsize=12,
#         color="white",
#         pad=10,
#         fontweight="bold",
#     )
#     st.pyplot(fig)


# def draw_ball_progression(df: pd.DataFrame, title: str):
#     xt_col = get_xt_column(df)
#     x_col, y_col = get_xy_columns(df)
#     if not x_col or not y_col or "end_x" not in df.columns or "end_y" not in df.columns:
#         st.warning("Ball progression view requires x/y and end_x/end_y columns.")
#         return

#     data = df.copy()
#     if xt_col:
#         threshold = data[xt_col].quantile(0.75)
#         data = data[data[xt_col] >= threshold]

#     # focus on passes & carries if event_type present
#     if "event_type" in data.columns:
#         mask = data["event_type"].astype(str).str.contains(
#             "pass|carry|dribble", case=False, na=False
#         )
#         data = data[mask]

#     data = data[[x_col, y_col, "end_x", "end_y"]].dropna()
#     if len(data) == 0:
#         st.warning("No high-value progression actions were found.")
#         return

#     pitch = Pitch(
#         pitch_type="statsbomb",
#         pitch_color="#333f37",
#         line_color="white",
#     )
#     fig, ax = pitch.draw(figsize=(6, 8))

#     for _, row in data.iterrows():
#         pitch.arrows(
#             row[x_col],
#             row[y_col],
#             row["end_x"],
#             row["end_y"],
#             width=1.2,
#             headwidth=2.5,
#             headlength=2.5,
#             ax=ax,
#             color="#ffdd57",
#             alpha=0.85,
#         )

#     ax.set_title(
#         title,
#         fontsize=12,
#         color="white",
#         pad=10,
#         fontweight="bold",
#     )
#     st.pyplot(fig)


# def draw_tracking_heatmap(df: pd.DataFrame, title: str):
#     draw_event_heatmap(df, title)


# def draw_touch_tackle_map(df: pd.DataFrame, title: str):
#     x_col, y_col = get_xy_columns(df)
#     if not x_col or not y_col:
#         st.warning("Could not find suitable x/y columns to render touch & tackle map.")
#         return

#     pitch = Pitch(
#         pitch_type="statsbomb",
#         pitch_color="#333f37",
#         line_color="white",
#     )
#     fig, ax = pitch.draw(figsize=(6, 8))

#     if "touch" in df.columns:
#         touches = df[df["touch"].fillna(0) > 0]
#         ax.scatter(
#             touches[x_col],
#             touches[y_col],
#             s=18,
#             alpha=0.8,
#             color="#28a745",
#             label="Touches",
#         )

#     tackles = pd.DataFrame()
#     for col in df.columns:
#         if "tackle" in col.lower():
#             tackles = df[df[col].fillna(0) > 0]
#             break

#     if not tackles.empty:
#         ax.scatter(
#             tackles[x_col],
#             tackles[y_col],
#             s=32,
#             marker="x",
#             linewidths=1.5,
#             alpha=0.9,
#             color="#e11d48",
#             label="Tackles",
#         )

#     ax.legend(
#         loc="upper center",
#         bbox_to_anchor=(0.5, 1.05),
#         ncol=2,
#         frameon=False,
#         fontsize=9,
#     )

#     ax.set_title(
#         title,
#         fontsize=12,
#         color="white",
#         pad=10,
#         fontweight="bold",
#     )
#     st.pyplot(fig)


# # --------------------------
# # HEADER: STATIC PLAYER CARD
# # --------------------------


# def render_header():
#     st.markdown(
#         '<p style="font-size:0.85rem; color:#6b7280; text-transform:uppercase; '
#         'letter-spacing:0.20em;">FC BARCELONA | 2024-25 LA LIGA</p>',
#         unsafe_allow_html=True,
#     )

#     with st.container():
#         col_img, col_info = st.columns([1, 2], gap="large")

#         with col_img:
#             if os.path.exists("pedri.png"):
#                 st.markdown(
#                     """
#                     <div style="display:flex; justify-content:center; align-items:center;">
#                         <img src="pedri.png" class="player-image" />
#                     </div>
#                     """,
#                     unsafe_allow_html=True,
#                 )
#             else:
#                 st.markdown(
#                     """
#                     <div style="display:flex; justify-content:center; align-items:center;">
#                         <div style="
#                             width:160px; height:160px; border-radius:50%;
#                             background:linear-gradient(135deg,#22c55e,#16a34a);
#                             display:flex; align-items:center; justify-content:center;
#                             color:white; font-size:3rem; font-weight:800;
#                             box-shadow:0 12px 35px rgba(15,23,42,0.25);
#                         ">
#                             8
#                         </div>
#                     </div>
#                     """,
#                     unsafe_allow_html=True,
#                 )

#         with col_info:
#             st.markdown(
#                 """
#                 <div class="hero-card">
#                     <div class="hero-name">Pedro González López</div>
#                     <div class="hero-subtitle">
#                         FC Barcelona • Midfielder • #8
#                     </div>
#                     <div style="margin-bottom:0.6rem;">
#                         <span class="hero-tag">Positional Play Engine</span>
#                         <span class="hero-tag">Half-Space Specialist</span>
#                         <span class="hero-tag">2024-25 La Liga</span>
#                     </div>
#                     <div style="font-size:0.9rem; color:#4b5563;">
#                         Premium spatial-intelligence dashboard for match-by-match
#                         heatmap analysis, role detection, and seasonal benchmarking
#                         of Pedri’s influence on Barcelona’s positional structure.
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )

#     st.markdown("<br/>", unsafe_allow_html=True)

#     # Static season KPIs row
#     st.markdown(
#         """
#         <div class="section-title">2024-25 Season Snapshot (Static)</div>
#         <div class="section-caption">
#             Fixed, thesis-ready KPIs that keep the dashboard feeling rich and informative
#             even before any match file is loaded.
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     col1, col2, col3, col4, col5, col6 = st.columns(6)

#     def metric_card(col, label, value, sub=None):
#         with col:
#             st.markdown(
#                 f"""
#                 <div class="metric-card">
#                     <div class="metric-label">{label}</div>
#                     <div class="metric-value">{value}</div>
#                     {'<div class="metric-sub">'+sub+'</div>' if sub else ''}
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )

#     metric_card(col1, "Matches Played", "24", "All competitions 24/25")
#     metric_card(col2, "La Liga Goals", "6", "From open play & late arrivals")
#     metric_card(col3, "Assists", "8", "Final-pass & pre-assist influence")
#     metric_card(col4, "Avg. xT / 90", "0.19", "Model-based expected threat")
#     metric_card(col5, "Progressive Actions / 90", "17.4", "Carries & passes")
#     metric_card(col6, "Pressing Actions / 90", "9.1", "Measuring out-of-possession work")


# # --------------------------
# # MATCH ANALYZER TAB
# # --------------------------


# def render_match_analyzer():
#     st.markdown(
#         """
#         <div class="section-title">Match Analyzer</div>
#         <div class="section-caption">
#             Upload a single-match CSV to generate Pedri’s heatmap, passing patterns,
#             progression maps, and AI-driven tactical interpretation.
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     uploaded = st.file_uploader(
#         "Upload a Match CSV",
#         type=["csv"],
#         help="Drop in either event-level or tracking-level data for Pedri.",
#     )

#     if uploaded is None:
#         st.info(
#             "Upload a match file to unlock the full event or tracking analysis pipeline. "
#             "The app will auto-detect the schema and adapt the visuals and insights."
#         )
#         return

#     try:
#         df = pd.read_csv(uploaded)
#     except Exception as e:
#         st.error(f"Could not read the CSV file: {e}")
#         return

#     columns = set(df.columns)
#     event_signature = {"event_type", "end_x", "end_y"}
#     tracking_signature = {"speed_m_per_s", "touch", "in_possession"}

#     mode = None
#     if event_signature.issubset(columns):
#         mode = "event"
#     elif tracking_signature.issubset(columns):
#         mode = "tracking"

#     st.markdown(
#         f"*Detected file type:* **{mode.capitalize() if mode else 'Unknown / Custom'}**"
#     )

#     if mode is None:
#         st.error(
#             "Schema not recognised. For event data, include columns like "
#             "`event_type`, `end_x`, `end_y`. For tracking data, include "
#             "`speed_m_per_s`, `touch`, `in_possession`."
#         )
#         with st.expander("Preview first 10 rows"):
#             st.dataframe(df.head(10))
#         return

#     # ----------------------------------
#     # EVENT DATA ANALYSIS
#     # ----------------------------------
#     if mode == "event":
#         kpis = compute_event_kpis(df)

#         kpi_cols = st.columns(6)
#         labels = [
#             "Total Actions",
#             "Avg. xT Impact",
#             "Pass Accuracy (%)",
#             "Progressive Actions",
#             "Ball Recoveries",
#             "Dribbles",
#         ]

#         for col_obj, label in zip(kpi_cols, labels):
#             val = kpis.get(label)
#             if val is None or (isinstance(val, float) and np.isnan(val)):
#                 display_val = "–"
#             else:
#                 if isinstance(val, float):
#                     display_val = f"{val:.2f}"
#                 else:
#                     display_val = str(val)
#             with col_obj:
#                 st.markdown(
#                     f"""
#                     <div class="metric-card">
#                         <div class="metric-label">{label}</div>
#                         <div class="metric-value">{display_val}</div>
#                     </div>
#                     """,
#                     unsafe_allow_html=True,
#                 )

#         st.markdown("<br/>", unsafe_allow_html=True)

#         viz_tab1, viz_tab2, viz_tab3 = st.tabs(
#             ["Heatmap", "Passing Network", "Ball Progression"]
#         )

#         with viz_tab1:
#             draw_event_heatmap(
#                 df,
#                 "Pedri – On-Ball Action Heatmap (Single Match)",
#             )

#         with viz_tab2:
#             draw_passing_network(
#                 df,
#                 "Pedri – Completed Passing Network (Origins → Destinations)",
#             )

#         with viz_tab3:
#             draw_ball_progression(
#                 df,
#                 "Pedri – High-Value Ball Progression (xT-Weighted)",
#             )

#         # AI Textual Insight
#         insight_text = ai_insight_event(df, kpis)
#         st.markdown("<br/>", unsafe_allow_html=True)
#         st.markdown(
#             f"""
#             <div class="insight-card">
#                 <div class="insight-title">AI Match Insight</div>
#                 <div class="insight-body">
#                     {insight_text.replace("\n", "<br/>")}
#                 </div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#     # ----------------------------------
#     # TRACKING DATA ANALYSIS
#     # ----------------------------------
#     elif mode == "tracking":
#         kpis = compute_tracking_kpis(df)

#         kpi_cols = st.columns(5)
#         labels = [
#             "Total Touches",
#             "Total Passes",
#             "Total Tackles",
#             "Top Speed (m/s)",
#             "Time in Possession (min)",
#         ]

#         for col_obj, label in zip(kpi_cols, labels):
#             val = kpis.get(label)
#             if val is None or (isinstance(val, float) and np.isnan(val)):
#                 display_val = "–"
#             else:
#                 if isinstance(val, float):
#                     display_val = f"{val:.2f}"
#                 else:
#                     display_val = str(val)
#             with col_obj:
#                 st.markdown(
#                     f"""
#                     <div class="metric-card">
#                         <div class="metric-label">{label}</div>
#                         <div class="metric-value">{display_val}</div>
#                     </div>
#                     """,
#                     unsafe_allow_html=True,
#                 )

#         st.markdown("<br/>", unsafe_allow_html=True)

#         viz_tab1, viz_tab2 = st.tabs(
#             ["Heatmap", "Touch & Tackle Map"]
#         )

#         with viz_tab1:
#             draw_tracking_heatmap(
#                 df,
#                 "Pedri – Movement & Involvement Heatmap (Tracking)",
#             )

#         with viz_tab2:
#             draw_touch_tackle_map(
#                 df,
#                 "Pedri – Touches (●) & Tackles (×)",
#             )

#         insight_text = ai_insight_tracking(df, kpis)
#         st.markdown("<br/>", unsafe_allow_html=True)
#         st.markdown(
#             f"""
#             <div class="insight-card">
#                 <div class="insight-title">AI Tracking Insight</div>
#                 <div class="insight-body">
#                     {insight_text.replace("\n", "<br/>")}
#                 </div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )


# # --------------------------
# # SEASON COMPARISON TAB
# # --------------------------


# def render_season_comparison():
#     st.markdown(
#         """
#         <div class="section-title">Season Comparison</div>
#         <div class="section-caption">
#             Benchmark Pedri’s 2024-25 La Liga profile against other elite interiors
#             and attacking midfielders.
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     csv_path = "player_comparison_data.csv"
#     if not os.path.exists(csv_path):
#         st.error(
#             f"`{csv_path}` not found. Place your comparison dataset (e.g. Pedri, "
#             f"Bellingham, KDB, Vitinha) in the app directory."
#         )
#         return

#     try:
#         comp_df = pd.read_csv(csv_path)
#     except Exception as e:
#         st.error(f"Could not read `player_comparison_data.csv`: {e}")
#         return

#     # Player name column
#     player_col = None
#     for candidate in ["player", "Player", "name", "Name"]:
#         if candidate in comp_df.columns:
#             player_col = candidate
#             break

#     if player_col is None:
#         st.error(
#             "Could not identify a player-name column. Please include a `player` or "
#             "`name` column in `player_comparison_data.csv`."
#         )
#         st.dataframe(comp_df.head())
#         return

#     players = sorted(comp_df[player_col].unique().tolist())
#     default_players = [p for p in players if "Pedri" in str(p)] or players[:3]

#     selected_players = st.multiselect(
#         "Select players to compare",
#         options=players,
#         default=default_players,
#         help="For example: Pedri, Jude Bellingham, Kevin De Bruyne, Vitinha.",
#     )

#     if not selected_players:
#         st.info("Select at least one player to render comparison charts.")
#         return

#     comp_df = comp_df[comp_df[player_col].isin(selected_players)]

#     # Metric columns
#     xt_metric = None
#     actions_metric = None
#     pivot_metric = None
#     creator_metric = None

#     for candidate in ["avg_xt", "avg_xT", "Avg_xT", "avg_xt_impact", "Avg_xT_Impact"]:
#         if candidate in comp_df.columns:
#             xt_metric = candidate
#             break

#     for candidate in ["total_actions", "TotalActions", "actions", "Total_Actions"]:
#         if candidate in comp_df.columns:
#             actions_metric = candidate
#             break

#     for candidate in ["pivot_role_pct", "pivot%", "pivot_share", "pivot_ratio"]:
#         if candidate in comp_df.columns:
#             pivot_metric = candidate
#             break

#     for candidate in ["creator_role_pct", "creator%", "creator_share", "creator_ratio"]:
#         if candidate in comp_df.columns:
#             creator_metric = candidate
#             break

#     def bar_chart(metric_col: str, title: str, ylabel: str):
#         fig, ax = plt.subplots(figsize=(6, 4))
#         sns.barplot(
#             data=comp_df,
#             x=player_col,
#             y=metric_col,
#             ax=ax,
#         )
#         ax.set_title(title, fontsize=11, fontweight="bold")
#         ax.set_xlabel("")
#         ax.set_ylabel(ylabel)
#         ax.tick_params(axis="x", rotation=20)
#         ax.grid(axis="y", linestyle="--", alpha=0.25)
#         st.pyplot(fig)

#     col1, col2 = st.columns(2)

#     with col1:
#         if xt_metric:
#             bar_chart(
#                 xt_metric,
#                 "Avg. xT Impact per 90",
#                 "xT / 90",
#             )
#         else:
#             st.warning(
#                 "No Avg. xT column found. Add e.g. `avg_xt_impact` to your CSV."
#             )

#     with col2:
#         if actions_metric:
#             bar_chart(
#                 actions_metric,
#                 "Total Actions per 90",
#                 "Actions / 90",
#             )
#         else:
#             st.warning(
#                 "No total actions column found. Add e.g. `total_actions` to your CSV."
#             )

#     st.markdown("<br/>", unsafe_allow_html=True)

#     col3, col4 = st.columns(2)

#     with col3:
#         if pivot_metric:
#             bar_chart(
#                 pivot_metric,
#                 "Tactical Role Share – Pivot",
#                 "Pivot Role (%)",
#             )
#         else:
#             st.warning(
#                 "No 'pivot role %' metric found. Add e.g. `pivot_role_pct` to your CSV."
#             )

#     with col4:
#         if creator_metric:
#             bar_chart(
#                 creator_metric,
#                 "Tactical Role Share – Creator",
#                 "Creator Role (%)",
#             )
#         else:
#             st.warning(
#                 "No 'creator role %' metric found. Add e.g. `creator_role_pct` to your CSV."
#             )


# # --------------------------
# # MAIN ENTRY POINT
# # --------------------------

# def main():
#     render_header()

#     tab_match, tab_season = st.tabs(["🎯 Match Analyzer", "📊 Season Comparison"])

#     with tab_match:
#         render_match_analyzer()

#     with tab_season:
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

PRIMARY_BG = "#f0f2f5"
FOOTBALL_GREEN = "#28a745"

# =======================================
# GLOBAL CSS
# =======================================
# st.markdown(
#     f"""
#     <style>
#         .stApp {{
#             background-color: {PRIMARY_BG};
#         }}
#         .block-container {{
#             max-width: 1200px;
#             margin: auto;
#             padding-top: 2rem;
#             padding-bottom: 3rem;
#         }}
#         .player-image {{
#             width: 160px;
#             height: 160px;
#             border-radius: 50%;
#             object-fit: cover;
#             border: 4px solid white;
#             box-shadow: 0 12px 35px rgba(15, 23, 42, 0.25);
#         }}
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
#         .metric-card {{
#             background: white;
#             border-radius: 18px;
#             padding: 1rem 1.25rem;
#             box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
#             border: 1px solid #e5e7eb;
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
#         .stTabs [role="tablist"] {{
#             gap: 0.5rem;
#         }}
#         .stTabs [role="tab"] {{
#             padding: 0.4rem 1rem;
#             border-radius: 999px;
#             border: 1px solid #d1d5db;
#             background: white;
#         }}
#         .stTabs [aria-selected="true"] {{
#             border-color: {FOOTBALL_GREEN};
#             background: rgba(40,167,69,0.08);
#         }}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# =======================================
# GLOBAL CSS
# =======================================
st.markdown(
    f"""
    <style>

        /* =============================
           GLOBAL APP SETTINGS
        ============================== */
        .stApp {{
            background-color: {PRIMARY_BG} !important;
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

        .hero-name {{
            font-size: 1.8rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
            color: #111827;
        }}

        .hero-subtitle {{
            font-size: 0.98rem;
            color: #4b5563;
            margin-bottom: 0.5rem;
        }}

        .hero-tag {{
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(40,167,69,0.08);
            color: {FOOTBALL_GREEN};
            margin-right: 0.4rem;
        }}


        /* =============================
           KPI METRIC CARDS
        ============================== */
        .metric-card {{
            background: white;
            border-radius: 18px;
            padding: 1rem 1.25rem;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
            border: 1px solid #e5e7eb;
            text-align: center;
            transition: 0.2s ease-in-out;
        }}

        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.18);
        }}

        .metric-label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6b7280;
            margin-bottom: 0.15rem;
        }}

        .metric-value {{
            font-size: 1.3rem;
            font-weight: 700;
            color: #111827;
        }}

        .metric-sub {{
            font-size: 0.78rem;
            color: #9ca3af;
            margin-top: 0.1rem;
        }}


        /* =============================
           SECTION TITLES
        ============================== */
        .section-title {{
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 1.5rem;
            margin-bottom: 0.3rem;
            color: #111827;
        }}

        .section-caption {{
            font-size: 0.85rem;
            color: #6b7280;
            margin-bottom: 0.75rem;
        }}


        /* =============================
           AI INSIGHT CARD
        ============================== */
        .insight-card {{
            background: white;
            border-radius: 18px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            border: 1px solid #e5e7eb;
        }}

        .insight-title {{
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            color: {FOOTBALL_GREEN};
            letter-spacing: 0.15em;
            margin-bottom: 0.5rem;
        }}

        .insight-body {{
            font-size: 0.95rem;
            color: #111827;
            line-height: 1.5;
        }}


        /* =============================
           FIX STREAMLIT TAB VISIBILITY
        ============================== */

        .stTabs [role="tab"] {{
            background-color: white !important;
            color: #374151 !important;   /* Visible dark text */
            border-radius: 999px !important;
            padding: 6px 18px !important;
            border: 1px solid #d1d5db !important;
            font-weight: 600 !important;
            transition: all 0.2s ease-in-out !important;
        }}

        .stTabs [role="tab"][aria-selected="true"] {{
            background-color: rgba(40,167,69,0.15) !important;  /* Soft green */
            color: #1f2937 !important;      /* Darker text */
            border-color: #28a745 !important;  /* Green border */
            font-weight: 700 !important;
        }}

        .stTabs [role="tab"]:hover {{
            background-color: rgba(40,167,69,0.10) !important;
            color: #1f2937 !important;
        }}

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
        if c in df.columns:
            return c
    return None


def get_xy_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    candidates_x = ["x", "X", "pos_x", "start_x"]
    candidates_y = ["y", "Y", "pos_y", "start_y"]
    x_col = next((c for c in candidates_x if c in df.columns), None)
    y_col = next((c for c in candidates_y if c in df.columns), None)
    return x_col, y_col


def compute_event_kpis(df: pd.DataFrame) -> dict:
    kpis = {}
    total_actions = len(df)
    kpis["Total Actions"] = total_actions

    xt_col = get_xt_column(df)
    if xt_col:
        kpis["Avg. xT Impact"] = float(df[xt_col].mean())
    else:
        kpis["Avg. xT Impact"] = None

    # Pass accuracy
    pass_mask = df["event_type"].astype(str).str.contains("pass", case=False, na=False)
    total_passes = pass_mask.sum()
    completed = None

    if total_passes > 0:
        if "outcome" in df.columns:
            completed = df.loc[pass_mask, "outcome"].astype(str).str.contains(
                "success|complete|accurate|on_target", case=False, na=False
            ).sum()
        elif "is_complete" in df.columns:
            completed = df.loc[pass_mask, "is_complete"].sum()

    if completed is not None and total_passes > 0:
        kpis["Pass Accuracy (%)"] = 100 * completed / total_passes
    else:
        kpis["Pass Accuracy (%)"] = None

    # Progressive actions
    if "is_progressive" in df.columns:
        kpis["Progressive Actions"] = int(df["is_progressive"].sum())
    elif "end_x" in df.columns and "x" in df.columns:
        kpis["Progressive Actions"] = int((df["end_x"] - df["x"] >= 15).sum())
    else:
        kpis["Progressive Actions"] = 0

    # Ball recoveries
    if "is_recovery" in df.columns:
        kpis["Ball Recoveries"] = int(df["is_recovery"].sum())
    elif "event_type" in df.columns:
        kpis["Ball Recoveries"] = int(
            df["event_type"].astype(str).str.contains("recovery", case=False, na=False).sum()
        )
    else:
        kpis["Ball Recoveries"] = 0

    # Dribbles
    if "event_type" in df.columns:
        kpis["Dribbles"] = int(
            df["event_type"].astype(str).str.contains("dribble", case=False, na=False).sum()
        )
    else:
        kpis["Dribbles"] = 0

    return kpis


def compute_tracking_kpis(df: pd.DataFrame) -> dict:
    kpis = {}
    kpis["Total Touches"] = int(df["touch"].fillna(0).sum()) if "touch" in df.columns else 0
    kpis["Total Tackles"] = int(df["tackle"].fillna(0).sum()) if "tackle" in df.columns else 0
    kpis["Total Passes"] = 0  # extend if you add explicit pass flag

    if "speed_m_per_s" in df.columns:
        kpis["Top Speed (m/s)"] = float(df["speed_m_per_s"].max())
    else:
        kpis["Top Speed (m/s)"] = None

    if "in_possession" in df.columns:
        series = df["in_possession"].fillna(0)
        # If values > 1, assume seconds; else indicator at ~2Hz
        if series.max() > 1:
            sec = series.sum()
        else:
            sec = series.sum() * 0.5  # 2Hz tracking (from generator)
        kpis["Time in Possession (min)"] = sec / 60.0
    else:
        kpis["Time in Possession (min)"] = None

    return kpis


def ai_insight_event(df: pd.DataFrame, kpis: dict) -> str:
    xt_col = get_xt_column(df)
    avg_xt = kpis.get("Avg. xT Impact")

    if avg_xt is None or np.isnan(avg_xt):
        rating = "⚪ Rating not available (xT data missing)."
    else:
        if avg_xt < 0.03:
            rating = "🧊 Quiet creative influence – more of a connector than a line-breaker in this game."
        elif avg_xt < 0.08:
            rating = "🟡 Solid connective display – stable circulation with selective risk-taking."
        elif avg_xt < 0.15:
            rating = "🟢 Strong creative game – consistently moved play into valuable zones."
        else:
            rating = "🔥 Elite creative performance – heavily involved in high-value actions toward goal."

    # K-Means on spatial locations
    x_col, y_col = get_xy_columns(df)
    role_line = "Tactical role inference unavailable (not enough spatial data)."

    if x_col and y_col:
        coords = df[[x_col, y_col]].dropna()
        if len(coords) >= 25:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(coords)
            centers = kmeans.cluster_centers_
            avg_x = centers[:, 0].mean()

            if avg_x < 40:
                role = "Deep Pivot (#6 profile)"
            elif avg_x < 70:
                role = "Hybrid #8 (connector & creator)"
            else:
                role = "Advanced Creator in the half-spaces"

            role_line = (
                f"K-Means clustering on Pedri’s on-ball locations yields an average effective x-position "
                f"around **{avg_x:.1f}**, which aligns with a **{role}** in this match."
            )

    total_actions = kpis.get("Total Actions", 0)
    progressive = kpis.get("Progressive Actions", 0)
    recov = kpis.get("Ball Recoveries", 0)
    drib = kpis.get("Dribbles", 0)
    pass_acc = kpis.get("Pass Accuracy (%)", None)

    prog_rate = 100 * progressive / total_actions if total_actions > 0 else 0.0

    lines = []
    lines.append(f"**Match Rating (xT-based)**  \n{rating}")
    lines.append("")
    lines.append(
        f"**Volume & Progression**  \n"
        f"- Total actions: **{total_actions}**, with ~**{prog_rate:.1f}%** classified as progressive.\n"
        f"- Ball recoveries: **{recov}**, indicating how often he restarted attacks from regains.\n"
        f"- Dribbles: **{drib}**, showing how frequently he broke lines with the ball."
    )
    if pass_acc is not None:
        lines.append(
            f"- Pass accuracy: **{pass_acc:.1f}%**, giving context on the risk level of his distribution."
        )
    lines.append("")
    lines.append(f"**Tactical Role & Positioning**  \n{role_line}")
    lines.append("")
    lines.append(
        "**Impact on Barcelona’s Strategy**  \n"
        "The heatmap and progression map together illustrate where Pedri repeatedly received possession "
        "and how he converted those touches into territorial or chance-building value. Dense clusters "
        "in the left half-space typically mean Barcelona funnel their positional-play structure through him; "
        "high-xT actions into the box highlight his role as a final-third creator rather than only a metronome."
    )

    return "\n".join(lines)


def ai_insight_tracking(df: pd.DataFrame, kpis: dict) -> str:
    top_speed = kpis.get("Top Speed (m/s)")
    poss_min = kpis.get("Time in Possession (min)")
    tackles = kpis.get("Total Tackles")
    touches = kpis.get("Total Touches")

    # Work rate line
    if top_speed is None or np.isnan(top_speed):
        work_line = "Work-rate cannot be fully quantified from speed data here, but spatial coverage still hints at his intensity."
    else:
        if top_speed < 5.5:
            work_line = (
                f"Top recorded speed of **{top_speed:.2f} m/s** suggests a more positional, tempo-controlling performance."
            )
        elif top_speed < 7.5:
            work_line = (
                f"With a top speed of **{top_speed:.2f} m/s**, Pedri reached regular high-intensity efforts to support both phases."
            )
        else:
            work_line = (
                f"Top speed of **{top_speed:.2f} m/s** indicates repeated sprint-level efforts, likely when arriving late into attacks or counter-pressing."
            )

    if poss_min is None or np.isnan(poss_min):
        poss_line = "Time in active possession cannot be precisely inferred, but touch density still shows how central he was in circulation."
    else:
        poss_line = (
            f"He spent approximately **{poss_min:.1f} minutes** in active ball possession, underlining his central role as a reference point for progression."
        )

    if tackles is None:
        tackle_line = "Tackle data is incomplete."
    else:
        if tackles == 0:
            tackle_line = "With **0 tackles**, his defensive involvement was more about screening and occupying passing lanes."
        elif tackles < 4:
            tackle_line = f"He contributed **{tackles} tackles**, stepping in selectively when the ball entered his zone."
        else:
            tackle_line = f"With **{tackles} tackles**, he was very active defensively, frequently leaving his slot to break opposition moves."

    if touches is None:
        touch_line = ""
    else:
        touch_line = (
            f"He registered **{touches} touches**, which links his spatial coverage with frequent involvement in build-up and consolidation phases."
        )

    text = (
        f"**Work Rate & Intensity**  \n"
        f"{work_line}  \n"
        f"{poss_line}\n\n"
        f"**Defensive Contribution**  \n"
        f"{tackle_line}\n\n"
        f"**Spatial Coverage & On-Ball Profile**  \n"
        f"{touch_line}  \n"
        "The combined touch and tackle maps typically show how he shuttles between deeper build-up spaces and higher half-spaces, "
        "maintaining Barcelona’s positional structure while still offering depth in both pressing and chance creation."
    )

    return text


# =======================================
# PITCH DRAWING HELPERS
# =======================================

def draw_event_heatmap(df: pd.DataFrame, title: str):
    x_col, y_col = get_xy_columns(df)
    if not x_col or not y_col:
        st.warning("No usable x/y columns found for heatmap.")
        return

    coords = df[[x_col, y_col]].dropna()
    if coords.empty:
        st.warning("No coordinate data for heatmap.")
        return

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
    fig, ax = pitch.draw(figsize=(6, 8))
    sns.kdeplot(
        x=coords[x_col],
        y=coords[y_col],
        fill=True,
        cmap="inferno",
        alpha=0.9,
        ax=ax,
        thresh=0.05,
        bw_adjust=0.45,
    )
    ax.set_title(title, color="white", fontsize=12, fontweight="bold")
    st.pyplot(fig)


def draw_passing_network(df: pd.DataFrame, title: str):
    x_col, y_col = get_xy_columns(df)
    if not x_col or not y_col or "end_x" not in df.columns or "end_y" not in df.columns:
        st.warning("Passing network requires x/y and end_x/end_y columns.")
        return

    mask = df["event_type"].astype(str).str.contains("pass", case=False, na=False)
    if "outcome" in df.columns:
        mask &= df["outcome"].astype(str).str.contains("success|complete|accurate|on_target", case=False, na=False)
    elif "is_complete" in df.columns:
        mask &= df["is_complete"] == 1

    passes = df.loc[mask, [x_col, y_col, "end_x", "end_y"]].dropna()
    if passes.empty:
        st.warning("No completed passes found.")
        return

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
    fig, ax = pitch.draw(figsize=(6, 8))

    for _, row in passes.iterrows():
        pitch.lines(
            row[x_col],
            row[y_col],
            row["end_x"],
            row["end_y"],
            ax=ax,
            comet=True,
            color="gold",
            lw=1.2,
            alpha=0.7,
        )

    ax.set_title(title, color="white", fontsize=12, fontweight="bold")
    st.pyplot(fig)


def draw_ball_progression(df: pd.DataFrame, title: str):
    if "end_x" not in df.columns or "end_y" not in df.columns:
        st.warning("Ball progression requires end_x/end_y.")
        return

    x_col, y_col = get_xy_columns(df)
    if not x_col or not y_col:
        st.warning("No usable x/y columns for progression map.")
        return

    data = df.copy()
    xt_col = get_xt_column(df)
    if xt_col:
        thresh = data[xt_col].quantile(0.75)
        data = data[data[xt_col] >= thresh]

    if "event_type" in data.columns:
        data = data[
            data["event_type"]
            .astype(str)
            .str.contains("pass|carry|dribble", case=False, na=False)
        ]

    data = data[[x_col, y_col, "end_x", "end_y"]].dropna()
    if data.empty:
        st.warning("No high-value progression actions found.")
        return

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
    fig, ax = pitch.draw(figsize=(6, 8))

    for _, row in data.iterrows():
        pitch.arrows(
            row[x_col],
            row[y_col],
            row["end_x"],
            row["end_y"],
            ax=ax,
            width=1.2,
            headwidth=2.5,
            headlength=2.5,
            color="#ffdd57",
            alpha=0.9,
        )

    ax.set_title(title, color="white", fontsize=12, fontweight="bold")
    st.pyplot(fig)


def draw_touch_tackle_map(df: pd.DataFrame, title: str):
    x_col, y_col = get_xy_columns(df)
    if not x_col or not y_col:
        st.warning("No usable x/y columns for touch & tackle map.")
        return

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#333f37", line_color="white")
    fig, ax = pitch.draw(figsize=(6, 8))

    if "touch" in df.columns:
        touches = df[df["touch"].fillna(0) > 0]
        ax.scatter(
            touches[x_col],
            touches[y_col],
            s=18,
            color=FOOTBALL_GREEN,
            alpha=0.8,
            label="Touches",
        )

    if "tackle" in df.columns:
        tackles = df[df["tackle"].fillna(0) > 0]
        ax.scatter(
            tackles[x_col],
            tackles[y_col],
            s=32,
            marker="x",
            color="#e11d48",
            alpha=0.9,
            label="Tackles",
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    ax.set_title(title, color="white", fontsize=12, fontweight="bold")
    st.pyplot(fig)


# =======================================
# HEADER
# =======================================

def render_header():
    st.markdown(
        '<p style="font-size:0.85rem; color:#6b7280; text-transform:uppercase; '
        'letter-spacing:0.20em;">FC Barcelona • 2024-25 La Liga</p>',
        unsafe_allow_html=True,
    )

    col_img, col_info = st.columns([1, 2], gap="large")

    with col_img:
        if os.path.exists("pedri.png"):
            try:
                img64 = load_image_base64("pedri.png")
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:center; align-items:center;">
                        <img src="data:image/png;base64,{img64}" class="player-image" />
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.warning(f"Could not load pedri.png: {e}")
        else:
            st.markdown(
                """
                <div style="display:flex; justify-content:center; align-items:center;">
                    <div style="
                        width:160px; height:160px; border-radius:50%;
                        background:linear-gradient(135deg,#22c55e,#16a34a);
                        display:flex; align-items:center; justify-content:center;
                        color:white; font-size:3rem; font-weight:800;
                        box-shadow:0 12px 35px rgba(15,23,42,0.25);
                    ">
                        8
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_info:
        st.markdown(
            """
            <div class="hero-card">
                <div class="hero-name">Pedro González López</div>
                <div class="hero-subtitle">
                    FC Barcelona • Midfielder • #8
                </div>
                <div style="margin-bottom:0.6rem;">
                    <span class="hero-tag">Positional Play Engine</span>
                    <span class="hero-tag">Left Half-Space Creator</span>
                    <span class="hero-tag">2024-25 La Liga</span>
                </div>
                <div style="font-size:0.9rem; color:#4b5563;">
                    Premium spatial-intelligence dashboard for match-by-match
                    heatmap analysis, role detection, and seasonal benchmarking
                    of Pedri’s influence on Barcelona’s positional structure.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-title">2024-25 Season Snapshot (Static)</div>
        <div class="section-caption">
            Fixed KPIs so the dashboard already feels rich and analytical,
            even before loading any match data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(6)

    def metric(col, label, value, sub=None):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    {'<div class="metric-sub">'+sub+'</div>' if sub else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

    metric(cols[0], "Matches Played", "24", "All competitions 24/25")
    metric(cols[1], "La Liga Goals", "6", "Late box arrivals & shots")
    metric(cols[2], "Assists", "8", "Final ball & secondary assists")
    metric(cols[3], "Avg. xT / 90", "0.19", "Expected Threat impact")
    metric(cols[4], "Prog Actions / 90", "17.4", "Carries & passes")
    metric(cols[5], "Pressing / 90", "9.1", "Out-of-possession work")


# =======================================
# MATCH ANALYZER
# =======================================

def render_match_analyzer():
    st.markdown(
        """
        <div class="section-title">Match Analyzer</div>
        <div class="section-caption">
            Upload a single-match CSV (event or tracking data). The app will auto-detect
            the schema and generate tailored KPIs, visuals, and AI-tactical insights.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload a Match CSV",
        type=["csv"],
        help="Use pedri_events_matchXX.csv or pedri_tracking_matchXX.csv from your generator.",
    )

    if uploaded is None:
        st.info("Upload a match file to start the analysis.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
        return

    cols = set(df.columns)
    event_sig = {"event_type", "end_x", "end_y"}
    tracking_sig = {"speed_m_per_s", "touch", "in_possession"}

    mode = None
    if event_sig.issubset(cols):
        mode = "event"
    elif tracking_sig.issubset(cols):
        mode = "tracking"

    st.markdown(f"**Detected data type:** `{mode if mode else 'unknown'}`")

    if mode is None:
        st.error("Unknown schema. Check your columns.")
        st.dataframe(df.head())
        return

    # EVENT MODE
    if mode == "event":
        kpis = compute_event_kpis(df)
        labels = [
            "Total Actions",
            "Avg. xT Impact",
            "Pass Accuracy (%)",
            "Progressive Actions",
            "Ball Recoveries",
            "Dribbles",
        ]
        kpi_cols = st.columns(6)
        for col, label in zip(kpi_cols, labels):
            val = kpis.get(label)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                disp = "–"
            else:
                disp = f"{val:.2f}" if isinstance(val, float) else str(val)
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{disp}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        tab1, tab2, tab3 = st.tabs(["Heatmap", "Passing Network", "Ball Progression"])

        with tab1:
            draw_event_heatmap(df, "Pedri – On-ball Action Density")
        with tab2:
            draw_passing_network(df, "Pedri – Completed Passing Network")
        with tab3:
            draw_ball_progression(df, "Pedri – High-Value Ball Progression")

        insight_text = ai_insight_event(df, kpis)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">AI Match Insight</div>
                <div class="insight-body">
                    {insight_text.replace("\n", "<br/>")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # TRACKING MODE
    elif mode == "tracking":
        kpis = compute_tracking_kpis(df)
        labels = [
            "Total Touches",
            "Total Passes",
            "Total Tackles",
            "Top Speed (m/s)",
            "Time in Possession (min)",
        ]
        kpi_cols = st.columns(5)
        for col, label in zip(kpi_cols, labels):
            val = kpis.get(label)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                disp = "–"
            else:
                disp = f"{val:.2f}" if isinstance(val, float) else str(val)
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{disp}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        tab1, tab2 = st.tabs(["Heatmap", "Touch & Tackle Map"])

        with tab1:
            draw_event_heatmap(df, "Pedri – Movement Heatmap (Tracking)")
        with tab2:
            draw_touch_tackle_map(df, "Pedri – Touches & Tackles")

        insight_text = ai_insight_tracking(df, kpis)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">AI Tracking Insight</div>
                <div class="insight-body">
                    {insight_text.replace("\n", "<br/>")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =======================================
# SEASON COMPARISON
# =======================================

def render_season_comparison():
    st.markdown(
        """
        <div class="section-title">Season Comparison</div>
        <div class="section-caption">
            Benchmark Pedri against other elite midfielders in terms of xT, action volume,
            and tactical role split (pivot vs creator).
        </div>
        """,
        unsafe_allow_html=True,
    )

    csv_path = "player_comparison_data.csv"
    if not os.path.exists(csv_path):
        st.error("`player_comparison_data.csv` not found in the app directory.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Could not read player_comparison_data.csv: {e}")
        return

    if "player" not in df.columns:
        st.error("Comparison file must contain a `player` column.")
        st.dataframe(df.head())
        return

    players = sorted(df["player"].unique().tolist())
    default = [p for p in players if "Pedri" in p] or players[:3]

    selected = st.multiselect(
        "Select players to compare",
        options=players,
        default=default,
    )

    if not selected:
        st.info("Select at least one player.")
        return

    df_sel = df[df["player"].isin(selected)]

    def bar_chart(metric_col: str, title: str, ylabel: str):
        if metric_col not in df_sel.columns:
            st.warning(f"Column `{metric_col}` not found in comparison CSV.")
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df_sel, x="player", y=metric_col, ax=ax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        bar_chart("avg_xt_impact", "Average xT Impact per 90", "xT / 90")
    with col2:
        bar_chart("total_actions", "Total Actions per 90", "Actions / 90")

    st.markdown("<br/>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        bar_chart("pivot_role_pct", "Tactical Role Share – Pivot", "Pivot Role (%)")
    with col4:
        bar_chart("creator_role_pct", "Tactical Role Share – Creator", "Creator Role (%)")


# =======================================
# MAIN
# =======================================

def main():
    render_header()
    tab1, tab2 = st.tabs(["🎯 Match Analyzer", "📊 Season Comparison"])
    with tab1:
        render_match_analyzer()
    with tab2:
        render_season_comparison()


if __name__ == "__main__":
    main()
