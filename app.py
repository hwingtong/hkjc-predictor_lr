import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import statsmodels.api as sm

# --- 1. JOCKEY DATA ---
JOCKEY_WIN_RATES = {
    "A Atzeni": 0.316860, "A Badel": 0.265082, "A Hamelin": 0.198198, "A Pouchin": 0.135593,
    "B Avdulla": 0.297398, "B Shinn": 0.117647, "B Thompson": 0.182266, "C Keane": 0.500000,
    "C L Chau": 0.253304, "C Lemaire": 0.250000, "C Soumillon": 0.250000, "C Williams": 0.380952,
    "C Y Ho": 0.328084, "D B McMonagle": 0.226415, "D McDonogh": 0.052632, "D Probert": 0.070000,
    "E C W Wong": 0.268882, "H Bentley": 0.240602, "H Bowman": 0.351156, "H Doyle": 0.250000,
    "H T Mo": 0.133929, "J McDonald": 0.422619, "J Melham": 0.333333, "J Moreira": 0.428571,
    "J Orman": 0.252660, "K C Leung": 0.244306, "K De Melo": 0.185792, "K H Chan": 0.285714,
    "K Teetan": 0.244413, "L Ferraris": 0.276051, "L Hewitson": 0.258845, "M Barzalona": 0.266667,
    "M Chadwick": 0.251712, "M F Poon": 0.244147, "M Guyon": 0.301887, "M L Yeung": 0.189956,
    "P N Wong": 0.223602, "R King": 0.142857, "R Kingscote": 0.213439, "R Moore": 0.437500,
    "T Marquand": 0.090909, "U Rispoli": 0.083333, "W Buick": 0.500000, "Y Kawada": 0.000000,
    "Y L Chung": 0.165254, "Z Purton": 0.509271
}
MEAN_WIN_RATE = 0.27509867856529946

# --- 2. LBW HELPERS ---
conversion_dict = {'SH': 0.05, 'HD': 0.1, 'NK': 0.2, 'N': 0.05}

def get_lbw_input(label_id):
    st.write(f"**{label_id}**")
    mode = st.radio("Type", ["Small Margins", "Lengths"], key=f"mode_{label_id}", horizontal=True)
    if mode == "Small Margins":
        choice = st.selectbox("Select", ["N", "SH", "HD", "NK"], key=f"small_{label_id}")
        return conversion_dict[choice]
    else:
        col1, col2 = st.columns(2)
        with col1: w = st.number_input("Whole", 0, 80, 0, key=f"w_{label_id}")
        with col2:
            f_map = {"0": 0.0, "1/4": 0.25, "1/2": 0.5, "3/4": 0.75}
            f = st.selectbox("Frac", list(f_map.keys()), key=f"f_{label_id}")
            return float(w) + f_map[f]

# --- 3. UI & LOADING ---
st.set_page_config(page_title="HKJC Prediction", layout="wide")
st.title("🏇 HKJC Horse Racing Online Predictor")

model_path = 'lasso_lr_model.pkl'
scaler_path = 'scaler.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    st.subheader("1. Race & Jockey Info")
    c1, c2 = st.columns(2)
    with c1:
        distance = st.selectbox("Distance", [1000, 1200, 1400, 1600, 1650, 1800, 2000, 2200, 2400])
        actual_weight = st.number_input("Weight", value=120.0)
        j_name = st.selectbox("Jockey", options=sorted(list(JOCKEY_WIN_RATES.keys())) + ["Others"])
        j_rate = JOCKEY_WIN_RATES.get(j_name, MEAN_WIN_RATE)
        st.caption(f"Rate: {j_rate:.4f}")
    with c2:
        days_since = st.number_input("Days Since", value=14)
        barrier = st.slider("Barrier", 1, 14, 7)
        class_move = st.selectbox("Class Change", options=[-1, 0, 1])

    st.markdown("---")
    st.subheader("2. Past 3 Runs")
    p1, p2, p3 = st.columns(3)
    with p1: lbw1 = get_lbw_input("Race 1"); res1 = st.checkbox("Top 3?", key="res1")
    with p2: lbw2 = get_lbw_input("Race 2"); res2 = st.checkbox("Top 3?", key="res2")
    with p3: lbw3 = get_lbw_input("Race 3"); res3 = st.checkbox("Top 3?", key="res3")

    if st.button("Predict"):
        # Engineering
        avg_lbw = (lbw1 + lbw2 + lbw3) / 3.0
        calc_p3r_dist = (sum([res1, res2, res3]) / 3.0) * distance
        calc_rel_weight_dist = (actual_weight / 120.0) * distance

        # Data creation
        data = {
            'P3R_Top3_Pct_x_Dist': float(calc_p3r_dist),
            'P3R_Avg_LBW': float(avg_lbw),
            'Days_Since_Last_Run': float(days_since),
            'Rel_Weight_x_Dist': float(calc_rel_weight_dist),
            'Barrier_Rank': float(barrier),
            'Jockey_Strike_Rate_Sea': float(j_rate),
            'Class_Change': float(class_move)
        }
        
        feature_order = ['P3R_Top3_Pct_x_Dist', 'P3R_Avg_LBW', 'Days_Since_Last_Run', 
                         'Rel_Weight_x_Dist', 'Barrier_Rank', 'Jockey_Strike_Rate_Sea', 'Class_Change']
        
        input_df = pd.DataFrame([data])[feature_order]

        # ---------------------------------------------------------
        # THE FIX: SCALE THEN PREDICT
        # ---------------------------------------------------------
        input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_order)
        input_df_final = sm.add_constant(input_df_scaled, has_constant='add')

        try:
            prob = model.predict(input_df_final).iloc[0]
            
            if prob >= 0.5:
                st.success(f"PROBABLE TOP 3 (Confidence: {prob:.2%})")
            else:
                st.warning(f"OUTSIDE TOP 3 (Confidence: {1-prob:.2%})")
            
            with st.expander("Show Scaled Metrics"):
                st.write("Scaled Input Data:", input_df_scaled)
                z = (model.params.values * input_df_final.values).sum()
                st.write(f"New Scaled Z-Value: {z:.4f}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

else:
    st.error("Please upload 'logistic_regression_model.pkl' AND 'scaler.pkl' to the repo.")
