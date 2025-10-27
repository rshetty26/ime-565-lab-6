# app.py
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

# ========== FILES ==========
DATA_PATH = "airline.csv"            # provided dataset (for encoding + stats)
MODEL_PATH = "decision_tree_airline.pickle"     # pre-trained DecisionTreeClassifier
HEADER_IMG = "airline.jpg"           # header image

# ========== UI: INTRO ==========
st.set_page_config(page_title="Airline Satisfaction Predictor", page_icon="✈️", layout="wide")

st.title("Airline Satisfaction Predictor")
st.write(
    "This app predicts whether a passenger is **Satisfied** or **Dissatisfied** "
    "based on their flight and experience details. Fill out the survey in the sidebar and click **Predict**."
)

# Header image (optional if present)
if Path(HEADER_IMG).exists():
    st.image(HEADER_IMG, use_container_width=True, caption="Your flight experience, predicted.")

# ========== CACHING HELPERS ==========
@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# ========== DATA & MODEL LOADING ==========
data_err = model_err = None
df_ref: Optional[pd.DataFrame] = None
model = None

try:
    df_ref = load_dataset(DATA_PATH)
except Exception as e:
    data_err = f"Could not load `{DATA_PATH}`: {e}"

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model_err = f"Could not load `{MODEL_PATH}`: {e}"

if data_err:
    st.error(data_err)
if model_err:
    st.error(model_err)

if (df_ref is None) or (model is None):
    st.stop()

# ========== FEATURE/ENCODING HELPERS ==========
# List your core survey fields (categorical & numeric)
CATEGORICAL_FIELDS = [
    "customer_type",          # e.g., 'Loyal Customer' / 'disloyal Customer'
    "type_of_travel",         # e.g., 'Personal Travel' / 'Business travel'
    "class"                   # e.g., 'Eco', 'Eco Plus', 'Business'
]
NUMERIC_FIELDS = [
    "age",
    "flight_distance",
    "departure_delay_in_minutes",
    "arrival_delay_in_minutes",
    # Ratings 1-5 (add/trim to match your dataset columns exactly)
    "inflight_wifi_service",
    "departure_arrival_time_convenient",
    "ease_of_online_booking",
    "gate_location",
    "food_and_drink",
    "online_boarding",
    "seat_comfort",
    "inflight_entertainment",
    "on-board_service",
    "leg_room_service",
    "baggage_handling",
    "checkin_service",
    "inflight_service",
    "cleanliness",
    "online_support",
]

# Some datasets use underscores; others use slashes/case variants.
# We'll dynamically detect and map common aliases if needed.
ALIAS_MAP: Dict[str, List[str]] = {
    "customer_type": ["Customer Type", "customer_type"],
    "type_of_travel": ["Type of Travel", "type_of_travel"],
    "class": ["Class", "class"],
    "age": ["Age", "age"],
    "flight_distance": ["Flight Distance", "flight_distance"],
    "departure_delay_in_minutes": ["Departure Delay in Minutes", "departure_delay_in_minutes"],
    "arrival_delay_in_minutes": ["Arrival Delay in Minutes", "arrival_delay_in_minutes"],
    "inflight_wifi_service": ["Inflight wifi service", "inflight_wifi_service"],
    "departure_arrival_time_convenient": ["Departure/Arrival time convenient", "departure_arrival_time_convenient"],
    "ease_of_online_booking": ["Ease of Online booking", "ease_of_online_booking"],
    "gate_location": ["Gate location", "gate_location"],
    "food_and_drink": ["Food and drink", "food_and_drink"],
    "online_boarding": ["Online boarding", "online_boarding"],
    "seat_comfort": ["Seat comfort", "seat_comfort"],
    "inflight_entertainment": ["Inflight entertainment", "inflight_entertainment"],
    "on-board_service": ["On-board service", "on-board_service", "On-board Service"],
    "leg_room_service": ["Leg room service", "leg_room_service"],
    "baggage_handling": ["Baggage handling", "baggage_handling"],
    "checkin_service": ["Checkin service", "checkin_service"],
    "inflight_service": ["Inflight service", "inflight_service"],
    "cleanliness": ["Cleanliness", "cleanliness"],
    # target label (not used in inference form, but for stats sanity)
    "satisfaction": ["satisfaction", "Satisfaction"]
}

def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create canonical, snake-case-like columns for consistent processing."""
    out = df.copy()
    mapping = {}
    for canonical, aliases in ALIAS_MAP.items():
        existing = first_existing_column(out, aliases)
        if existing and existing != canonical:
            mapping[existing] = canonical
    if mapping:
        out = out.rename(columns=mapping)
    return out

df_ref = canonicalize_columns(df_ref)

def build_dummy_template(df: pd.DataFrame) -> List[str]:
    base_cols = CATEGORICAL_FIELDS + NUMERIC_FIELDS
    exist_cols = [c for c in base_cols if c in df.columns]
    template = pd.get_dummies(df[exist_cols], dtype=int)
    return template.columns.tolist()

TEMPLATE_COLUMNS = build_dummy_template(df_ref)

def get_expected_feature_names(model_obj) -> Optional[List[str]]:
    names = getattr(model_obj, "feature_names_in_", None)
    if names is not None:
        return list(names)
    return None

EXPECTED_FEATURES = get_expected_feature_names(model)

def encode_and_align(df_input_row: pd.DataFrame) -> pd.DataFrame:
    # dummy-encode the single row
    enc = pd.get_dummies(df_input_row[CATEGORICAL_FIELDS + NUMERIC_FIELDS], dtype=int)
    # choose expected column list
    expected = EXPECTED_FEATURES if EXPECTED_FEATURES else TEMPLATE_COLUMNS
    # align and fill missing with 0
    enc = enc.reindex(columns=expected, fill_value=0)
    # if numeric columns got lost, ensure they exist
    for n in NUMERIC_FIELDS:
        if n in enc.columns and enc[n].isna().any():
            enc[n] = enc[n].fillna(0)
    return enc

# ========== SIDEBAR SURVEY ==========
with st.sidebar:
    st.header("🧩 Survey Form")

    # Pull category options from dataset when possible; otherwise provide safe defaults.
    def cat_options(col_name: str, fallback: List[str]) -> List[str]:
        col = first_existing_column(df_ref, ALIAS_MAP[col_name])
        if (col is not None) and (col in df_ref.columns):
            vals = df_ref[col].dropna().unique().tolist()
            # keep stable order: sort by frequency then value
            order = df_ref[col].value_counts().index.tolist()
            vals_sorted = [v for v in order if v in vals]
            return vals_sorted
        return fallback

    customer_type_opts = cat_options("customer_type", ["Loyal Customer", "Disloyal Customer"])
    travel_type_opts   = cat_options("type_of_travel", ["Personal Travel", "Business travel"])
    class_opts         = cat_options("class", ["Eco", "Eco Plus", "Business"])

    with st.form("survey_form", clear_on_submit=False):
        st.subheader("Customer Details")
        customer_type = st.selectbox("Customer Type", customer_type_opts)
        type_of_travel = st.selectbox("Type of Travel", travel_type_opts)
        cls = st.selectbox("Class", class_opts)
        age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)

        st.subheader("Flight Details")
        flight_distance = st.number_input("Flight Distance", min_value=0, value=800, step=10)
        dep_delay = st.number_input("Departure Delay (minutes)", min_value=0, value=5, step=1)
        arr_delay = st.number_input("Arrival Delay (minutes)", min_value=0, value=5, step=1)

        st.subheader("Experience Ratings (1-5)")
        def rating(label, default):
            return st.slider(label, 1, 5, default, step=1)

        inflight_wifi_service                 = rating("Inflight WiFi Service", 3)
        dep_arr_time_convenient               = rating("Departure/Arrival time convenient", 3)
        ease_of_online_booking                = rating("Ease of Online booking", 3)
        gate_location                         = rating("Gate location", 3)
        food_and_drink                        = rating("Food and drink", 3)
        online_boarding                       = rating("Online boarding", 3)
        seat_comfort                          = rating("Seat comfort", 3)
        inflight_entertainment                = rating("Inflight entertainment", 3)
        on_board_service                      = rating("On-board service", 3)
        leg_room_service                      = rating("Leg room service", 3)
        baggage_handling                      = rating("Baggage handling", 3)
        checkin_service                       = rating("Checkin service", 3)
        inflight_service                      = rating("Inflight service", 3)
        cleanliness                           = rating("Cleanliness", 4)
        online_support                        = rating("Online Support", 3)

        submitted = st.form_submit_button("Predict")

# ========== PREDICTION LOGIC ==========
def validate_form() -> Tuple[bool, str]:
    # Add any guardrails you need (e.g., min/max age already enforced by inputs)
    # Here we just ensure required fields are present
    return True, ""

def make_input_row() -> pd.DataFrame:
    row = {
        "customer_type": customer_type,
        "type_of_travel": type_of_travel,
        "class": cls,
        "age": int(age),
        "flight_distance": int(flight_distance),
        "departure_delay_in_minutes": int(dep_delay),
        "arrival_delay_in_minutes": int(arr_delay),
        "inflight_wifi_service": int(inflight_wifi_service),
        "departure_arrival_time_convenient": int(dep_arr_time_convenient),
        "ease_of_online_booking": int(ease_of_online_booking),
        "gate_location": int(gate_location),
        "food_and_drink": int(food_and_drink),
        "online_boarding": int(online_boarding),
        "seat_comfort": int(seat_comfort),
        "inflight_entertainment": int(inflight_entertainment),
        "on-board_service": int(on_board_service),
        "leg_room_service": int(leg_room_service),
        "baggage_handling": int(baggage_handling),
        "online_support": int(online_support),
        "checkin_service": int(checkin_service),
        "inflight_service": int(inflight_service),
        "cleanliness": int(cleanliness),
    }
    return pd.DataFrame([row])

def predict_and_confidence(X_enc: pd.DataFrame) -> Tuple[str, float]:
    """
    Returns (predicted_label, confidence_percent).
    """
    # predict class
    y_pred = model.predict(X_enc)
    label = str(y_pred[0])

    # predict probabilities if available
    conf_pct = None
    proba = model.predict_proba(X_enc)
    classes = getattr(model, "classes_", None)
    if classes is not None:
        # probability of the predicted class
        idx = int(np.argmax(proba[0]))
        conf_pct = float(proba[0][idx] * 100.0)
    return label, conf_pct

# ========== DEMOGRAPHIC COMPARISON ==========
def demographic_summary(df: pd.DataFrame) -> Dict[str, Any]:
    d = {}
    dfc = canonicalize_columns(df)
    # Age
    if "age" in dfc.columns:
        d["age_mean"] = float(dfc["age"].dropna().mean())
        d["age_median"] = float(dfc["age"].dropna().median())
    # Travel type distribution
    tcol = first_existing_column(dfc, ALIAS_MAP["type_of_travel"])
    if tcol:
        t_counts = dfc[tcol].value_counts(normalize=True) * 100
        d["type_dist"] = t_counts.to_dict()
    # Class distribution
    ccol = first_existing_column(dfc, ALIAS_MAP["class"])
    if ccol:
        c_counts = dfc[ccol].value_counts(normalize=True) * 100
        d["class_dist"] = c_counts.to_dict()
    return d

demo_stats = demographic_summary(df_ref)

# ========== MAIN PANEL OUTPUT ==========
if submitted:
    ok, msg = validate_form()
    if not ok:
        st.error(msg)
    else:
        user_df = make_input_row()
        # Encode + align to model features
        X_enc = encode_and_align(user_df)

        try:
            pred_label, conf_pct = predict_and_confidence(X_enc)
            # Normalize labels if needed
            # e.g., some datasets have "satisfied"/"neutral or dissatisfied"
            # We'll prettify common values:
            pretty = {
                "satisfied": "Satisfied",
                "neutral or dissatisfied": "Dissatisfied",
                "dissatisfied": "Dissatisfied",
                "Satisfied": "Satisfied",
                "Dissatisfied": "Dissatisfied"
            }
            pred_display = pretty.get(pred_label, pred_label)

            if pred_display.lower() == "satisfied":
                st.success(f"Prediction: {pred_display}")
            else:
                st.error(f"Prediction: {pred_display}")
            st.metric("Confidence", f"{conf_pct:.1f}%")

            st.markdown("### Demographic Comparison")
            colA, colB, colC = st.columns(3)

            # Age vs dataset
            with colA:
                if "age_mean" in demo_stats and "age_median" in demo_stats:
                    st.write(f"**Your Age:** {int(age)}")
                    st.write(f"**Dataset Mean Age:** {demo_stats['age_mean']:.1f}")
                    st.write(f"**Dataset Median Age:** {demo_stats['age_median']:.1f}")
                else:
                    st.write("Age statistics unavailable in dataset.")

            # Travel type distribution
            with colB:
                st.write(f"**Your Type:** {type_of_travel}")
                st.write("**Type of Travel (Dataset %):**")
                dist = demo_stats.get("type_dist", {})
                if dist:
                    lines = [f"- {k}: {v:.1f}%" for k, v in dist.items()]
                    st.markdown("\n".join(lines))
                else:
                    st.write("Not available.")

            # Class distribution
            with colC:
                st.write(f"**Your Class:** {cls}")
                st.write("**Class (Dataset %):**")
                cdist = demo_stats.get("class_dist", {})
                if cdist:
                    lines = [f"- {k}: {v:.1f}%" for k, v in cdist.items()]
                    st.markdown("\n".join(lines))
                else:
                    st.write("Not available.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("ℹ️ Please fill out the survey form in the sidebar and click **Predict** to see the satisfaction prediction.")
