import os
import math
import random
import numpy as np
import pandas as pd

# ==============================
# CONFIGURATION
# ==============================

# Number of La Liga matches to simulate
N_MATCHES = 30   # change to 20/25/38 if you like

# Tracking frequency (Hz)
TRACKING_HZ = 2  # 2 samples per second => 90*60*2 = 10,800 rows per match

# Pitch dimensions (StatsBomb style)
PITCH_LENGTH = 120
PITCH_WIDTH = 80

# Output folder
OUTPUT_DIR = "pedri_24_25_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)


# ==============================
# HELPER FUNCTIONS
# ==============================

def sample_position(role_bias="normal"):
    """
    Return a realistic (x, y) location for Pedri based on typical behaviour.
    role_bias:
        "deep"      -> closer to own half
        "advanced"  -> closer to final third
        "wide_left" -> left halfspace
        "normal"    -> central / left interior
    """
    if role_bias == "deep":
        x = np.random.normal(loc=45, scale=12)   # own / mid third
        y = np.random.normal(loc=35, scale=10)
    elif role_bias == "advanced":
        x = np.random.normal(loc=80, scale=10)   # attacking zones
        y = np.random.normal(loc=38, scale=8)
    elif role_bias == "wide_left":
        x = np.random.normal(loc=70, scale=12)
        y = np.random.normal(loc=25, scale=10)
    else:  # normal interior
        x = np.random.normal(loc=65, scale=12)
        y = np.random.normal(loc=38, scale=9)

    x = float(np.clip(x, 0, PITCH_LENGTH))
    y = float(np.clip(y, 0, PITCH_WIDTH))
    return x, y


def compute_pass_length_angle(x, y, end_x, end_y):
    dx = end_x - x
    dy = end_y - y
    length = math.sqrt(dx * dx + dy * dy)
    angle = math.degrees(math.atan2(dy, dx))
    return length, angle


def zone_from_xy(x, y):
    """
    Simple zone naming: thirds x3 * corridors x3.
    """
    # thirds in x
    if x < 40:
        third = "defensive"
    elif x < 80:
        third = "middle"
    else:
        third = "attacking"

    # corridors in y
    if y < PITCH_WIDTH / 3:
        corridor = "left"
    elif y < 2 * PITCH_WIDTH / 3:
        corridor = "central"
    else:
        corridor = "right"

    if corridor == "left" and third in ["middle", "attacking"]:
        return f"left_halfspace_{third}"
    if corridor == "central" and third in ["middle", "attacking"]:
        return f"central_{third}"
    if corridor == "right" and third in ["middle", "attacking"]:
        return f"right_halfspace_{third}"
    return f"{corridor}_{third}"


def xt_from_position(x, y):
    """
    Very rough xT approximation based on how close x is to the goal and centrality.
    Just to create realistic gradients.
    """
    # base from x
    base = max(0, (x - 50) / 70)  # 0 in own half, up to ~1 near box

    # central bonus
    mid_band_low = PITCH_WIDTH * 0.3
    mid_band_high = PITCH_WIDTH * 0.7
    if mid_band_low <= y <= mid_band_high:
        base *= 1.1

    noise = np.random.normal(0, 0.03)
    xt = max(0, base + noise)
    return round(float(xt), 4)


def simulate_events_for_match(match_id):
    """
    Generate one match of event data for Pedri.
    """
    events = []

    # Roughly 260–330 actions per match
    n_events = int(np.random.normal(loc=290, scale=35))
    n_events = max(220, min(360, n_events))

    timestamp = 0.0
    for i in range(n_events):
        # event spacing: 10–20 seconds between on-ball actions on average
        timestamp += np.random.exponential(scale=12)
        if timestamp > 5400:
            break

        # decide event type
        event_type = random.choices(
            population=["pass", "carry", "dribble", "shot", "recovery", "duel"],
            weights=[0.58, 0.18, 0.08, 0.04, 0.07, 0.05],
            k=1,
        )[0]

        # bias roles: deeper in buildup, more advanced later
        if timestamp < 1800:          # first half hour
            role_bias = random.choices(
                ["deep", "normal", "advanced"], [0.4, 0.5, 0.1], k=1
            )[0]
        elif timestamp < 3600:
            role_bias = random.choices(
                ["deep", "normal", "advanced"], [0.25, 0.5, 0.25], k=1
            )[0]
        else:
            role_bias = random.choices(
                ["deep", "normal", "advanced"], [0.2, 0.35, 0.45], k=1
            )[0]

        x, y = sample_position(role_bias)

        # choose end location
        if event_type in ["pass", "carry", "dribble", "shot"]:
            # general forward-ish tendency
            end_x = x + np.random.normal(loc=8, scale=10)
            end_y = y + np.random.normal(loc=0, scale=8)
        else:
            # recoveries & duels roughly local
            end_x, end_y = x + np.random.normal(0, 4), y + np.random.normal(0, 4)

        end_x = float(np.clip(end_x, 0, PITCH_LENGTH))
        end_y = float(np.clip(end_y, 0, PITCH_WIDTH))

        # outcomes
        if event_type == "pass":
            outcome = random.choices(
                ["success", "fail"], weights=[0.9, 0.1], k=1
            )[0]
        elif event_type in ["dribble", "duel"]:
            outcome = random.choices(
                ["success", "fail"], weights=[0.6, 0.4], k=1
            )[0]
        elif event_type == "shot":
            outcome = random.choices(
                ["on_target", "off_target", "blocked"], weights=[0.3, 0.45, 0.25], k=1
            )[0]
        else:
            outcome = "success"

        is_complete = outcome in ["success", "on_target"]

        # progressive?
        is_progressive = (end_x - x) >= 15

        # ball recovery flag
        is_recovery = event_type == "recovery"

        body_part = random.choices(
            ["right_foot", "left_foot", "head", "chest"],
            weights=[0.6, 0.25, 0.1, 0.05],
            k=1,
        )[0]

        under_pressure = random.random() < 0.25

        xt = xt_from_position(end_x, end_y)

        pass_length, pass_angle = (0.0, 0.0)
        if event_type in ["pass", "carry", "dribble"]:
            pass_length, pass_angle = compute_pass_length_angle(x, y, end_x, end_y)

        events.append({
            "match_id": match_id,
            "timestamp": round(timestamp, 1),
            "event_type": event_type,
            "x": round(x, 2),
            "y": round(y, 2),
            "end_x": round(end_x, 2),
            "end_y": round(end_y, 2),
            "outcome": outcome,
            "xT": xt,
            "is_progressive": int(is_progressive),
            "is_recovery": int(is_recovery),
            "is_complete": int(is_complete),
            "body_part": body_part,
            "under_pressure": int(under_pressure),
            "pass_length": round(pass_length, 2),
            "pass_angle": round(pass_angle, 1),
            "team_id": "FCB",
            "player_id": "PEDRI_8",
        })

    return pd.DataFrame(events)


def simulate_tracking_for_match(match_id):
    """
    Generate one match of tracking data (downsampled, synthetic).
    """
    total_seconds = 90 * 60
    n_frames = total_seconds * TRACKING_HZ

    timestamps = np.arange(0, total_seconds, 1 / TRACKING_HZ)
    timestamps = timestamps[:n_frames]

    xs = []
    ys = []
    speeds = []
    accels = []
    distance_covered = []

    last_x, last_y = sample_position("normal")
    last_speed = 0.0
    total_dist = 0.0

    for t in timestamps:
        # choose broad zone by phase, then random walk within it
        if t < 1800:
            role_bias = random.choices(["deep", "normal"], [0.6, 0.4], k=1)[0]
        elif t < 3600:
            role_bias = random.choices(["normal", "advanced"], [0.6, 0.4], k=1)[0]
        else:
            role_bias = random.choices(["normal", "advanced", "wide_left"],
                                       [0.4, 0.4, 0.2], k=1)[0]

        target_x, target_y = sample_position(role_bias)

        # small step towards target
        alpha = np.random.uniform(0.02, 0.08)  # smooth motion
        new_x = (1 - alpha) * last_x + alpha * target_x + np.random.normal(0, 1.5)
        new_y = (1 - alpha) * last_y + alpha * target_y + np.random.normal(0, 1.5)
        new_x = float(np.clip(new_x, 0, PITCH_LENGTH))
        new_y = float(np.clip(new_y, 0, PITCH_WIDTH))

        dx = new_x - last_x
        dy = new_y - last_y
        dt = 1.0 / TRACKING_HZ
        dist = math.sqrt(dx * dx + dy * dy)
        speed = dist / dt  # m/s, but pitch is not in metres; treat as relative
        accel = (speed - last_speed) / dt

        total_dist += dist

        xs.append(round(new_x, 2))
        ys.append(round(new_y, 2))
        speeds.append(round(speed, 2))
        accels.append(round(accel, 2))
        distance_covered.append(round(total_dist, 2))

        last_x, last_y, last_speed = new_x, new_y, speed

    df = pd.DataFrame({
        "match_id": match_id,
        "timestamp": np.round(timestamps, 2),
        "x": xs,
        "y": ys,
        "speed_m_per_s": speeds,
        "accel_m_per_s2": accels,
        "distance_covered_m": distance_covered,
    })

    # high intensity & sprint flags (relative thresholds)
    df["high_intensity_run"] = (df["speed_m_per_s"] > 5.5).astype(int)
    df["sprint"] = (df["speed_m_per_s"] > 7.0).astype(int)

    # synthetic ball location & distances
    ball_x = np.clip(
        np.random.normal(loc=70, scale=25, size=len(df)), 0, PITCH_LENGTH
    )
    ball_y = np.clip(
        np.random.normal(loc=40, scale=20, size=len(df)), 0, PITCH_WIDTH
    )

    df["distance_to_ball"] = np.sqrt((df["x"] - ball_x) ** 2 + (df["y"] - ball_y) ** 2)

    # approximate goal at x=120, y=40
    df["distance_to_goal"] = np.sqrt((120 - df["x"]) ** 2 + (40 - df["y"]) ** 2)

    # touches: occasional frames when close to ball
    touch_prob = np.exp(-df["distance_to_ball"] / 8.0) * 0.3
    df["touch"] = (np.random.rand(len(df)) < touch_prob).astype(int)

    # tackles: rarer and in middle/defensive zones
    tackle_base_prob = np.where(df["x"] < 70, 0.012, 0.006)
    tackle_prob = tackle_base_prob * np.exp(-df["distance_to_ball"] / 10.0)
    df["tackle"] = (np.random.rand(len(df)) < tackle_prob).astype(int)

    # in-possession: approximate from touches smoothed over time
    in_possession = df["touch"].rolling(window=TRACKING_HZ * 4, min_periods=1).max()
    df["in_possession"] = in_possession.astype(int)

    # zone labels
    df["zone_name"] = df.apply(lambda r: zone_from_xy(r["x"], r["y"]), axis=1)

    return df


def generate_player_comparison_csv():
    players = [
        {
            "player": "Pedri",
            "avg_xt_impact": 0.19,
            "total_actions": 80,
            "pivot_role_pct": 35,
            "creator_role_pct": 65,
            "progressive_actions": 17,
            "touches_per_90": 85,
            "defensive_actions_per_90": 9,
        },
        {
            "player": "Jude Bellingham",
            "avg_xt_impact": 0.23,
            "total_actions": 75,
            "pivot_role_pct": 28,
            "creator_role_pct": 72,
            "progressive_actions": 19,
            "touches_per_90": 70,
            "defensive_actions_per_90": 7,
        },
        {
            "player": "Kevin De Bruyne",
            "avg_xt_impact": 0.26,
            "total_actions": 78,
            "pivot_role_pct": 22,
            "creator_role_pct": 78,
            "progressive_actions": 21,
            "touches_per_90": 72,
            "defensive_actions_per_90": 5,
        },
        {
            "player": "Vitinha",
            "avg_xt_impact": 0.17,
            "total_actions": 77,
            "pivot_role_pct": 40,
            "creator_role_pct": 60,
            "progressive_actions": 15,
            "touches_per_90": 82,
            "defensive_actions_per_90": 10,
        },
        {
            "player": "Federico Valverde",
            "avg_xt_impact": 0.16,
            "total_actions": 68,
            "pivot_role_pct": 32,
            "creator_role_pct": 68,
            "progressive_actions": 14,
            "touches_per_90": 65,
            "defensive_actions_per_90": 11,
        },
    ]

    df = pd.DataFrame(players)
    df.to_csv(os.path.join(OUTPUT_DIR, "player_comparison_data.csv"), index=False)
    print("Saved player_comparison_data.csv")


def main():
    print(f"Generating synthetic Pedri 24/25 La Liga data in '{OUTPUT_DIR}'")
    all_event_matches = []
    all_tracking_matches = []

    for match_idx in range(1, N_MATCHES + 1):
        match_id = f"LALIGA_24_25_M{match_idx:02d}"
        print(f"  - Simulating match {match_idx}/{N_MATCHES}: {match_id}")

        # Events
        df_events = simulate_events_for_match(match_id)
        events_path = os.path.join(
            OUTPUT_DIR, f"pedri_events_match{match_idx:02d}.csv"
        )
        df_events.to_csv(events_path, index=False)

        # Tracking
        df_tracking = simulate_tracking_for_match(match_id)
        tracking_path = os.path.join(
            OUTPUT_DIR, f"pedri_tracking_match{match_idx:02d}.csv"
        )
        df_tracking.to_csv(tracking_path, index=False)

        all_event_matches.append(df_events)
        all_tracking_matches.append(df_tracking)

    # Optionally save concatenated season-level files too
    pd.concat(all_event_matches).to_csv(
        os.path.join(OUTPUT_DIR, "pedri_events_season_all_matches.csv"),
        index=False,
    )
    pd.concat(all_tracking_matches).to_csv(
        os.path.join(OUTPUT_DIR, "pedri_tracking_season_all_matches.csv"),
        index=False,
    )

    # Season comparison file
    generate_player_comparison_csv()

    print("Done. All files generated.")


if __name__ == "__main__":
    main()
