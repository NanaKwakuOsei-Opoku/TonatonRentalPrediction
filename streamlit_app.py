import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1) APP CONFIG

st.set_page_config(
    page_title="Ghana Rental Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH  = "house_rentals.csv"
MODEL_PATH = "ensemble_model.pkl"


# 2) DATA LOADING & PREPROCESSING

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH).rename(columns=str.strip)
    df["region"] = df["region"].str.replace(r"\s+[Rr]egion$", "", regex=True)
    df["parking_space"] = df["parking_space"].map({"Yes":1,"No":0}).fillna(0)
    df["is_furnished"]  = df["is_furnished"].map({
        "Furnished":1, "Unfurnished":0, "Semi-Furnished":0.5
    }).fillna(0)
    df["amenities_count"] = (
        df["amenities"].fillna("").apply(lambda s: len([a for a in s.split(",") if a.strip()]))
    )
    df["region_demand"] = (
        df.groupby("region")["price"]
          .transform(lambda s: s.clip(s.quantile(0.05), s.quantile(0.95)).median())
    )
    df["density"] = df.groupby(["lat","lng"])["price"].transform("count")
    low, high = df["price"].quantile([0.01,0.99])
    df = df[(df.price>=low)&(df.price<=high)].reset_index(drop=True)
    df["log_price"] = np.log1p(df["price"])
    return df

df = load_data()
ALL_AMENITIES  = sorted({a.strip() for s in df["amenities"].fillna("") for a in s.split(",") if a.strip()})
ALL_LOCALITIES = sorted(df["locality"].dropna().unique())


# 3) FEATURE ENGINEERING

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    def fit(self, X, y=None):
        self.enc.fit(X[["category","condition","locality"]])
        return self
    def transform(self, X):
        X = X.copy()
        X["room_ratio"]   = X["bedrooms"] / (X["bathrooms"] + 0.1)
        X["total_rooms"]  = X["bedrooms"] + X["bathrooms"]
        X["bed_sq"]       = X["bedrooms"]**2
        X["bath_sq"]      = X["bathrooms"]**2
        X["bed_bath_int"] = X["bedrooms"] * X["bathrooms"]
        mat = self.enc.transform(X[["category","condition","locality"]])
        cols = self.enc.get_feature_names_out()
        cat_df = pd.DataFrame(mat, columns=cols, index=X.index)
        num = X[[
            "bedrooms","bathrooms","parking_space","is_furnished",
            "amenities_count","region_demand","density",
            "room_ratio","total_rooms","bed_sq","bath_sq","bed_bath_int"
        ]]
        return pd.concat([num, cat_df], axis=1)


# 4) MODEL TRAINING / LOADING

@st.cache_data(show_spinner=False)
def train_model(df):
    X = df.drop(columns=["price","log_price","amenities"], errors="ignore")
    y = df["log_price"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    lgb = LGBMRegressor(max_depth=6, n_estimators=100, random_state=42)
    cat = CatBoostRegressor(depth=6, iterations=100,
                            learning_rate=0.1, verbose=0, random_state=42)
    stack = StackingRegressor(
        estimators=[("lgbm",lgb),("catb",cat)],
        final_estimator=Ridge()
    )

    pipe = Pipeline([
        ("feat",  FeatureEngineer()),
        ("scale", RobustScaler()),
        ("model", stack),
    ])
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    st.session_state["rmse"] = np.sqrt(mean_squared_error(yte, preds))
    st.session_state["r2"]   = r2_score(yte, preds)
    joblib.dump(pipe, MODEL_PATH)
    return pipe

def get_model(df):
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except:
            os.remove(MODEL_PATH)
    return train_model(df)

model = get_model(df)

# 5) STREAMLIT UI

def main():
    st.markdown("## 🏘️ Ghana Rental Price Predictor")

    # show RMSE & R² if available
    if st.session_state.get("rmse") is not None:
        rmse_col, r2_col = st.columns(2)
        rmse_col.metric("RMSE (log)", f"{st.session_state['rmse']:.3f}")
        r2_col.metric("R²",           f"{st.session_state['r2']:.3f}")

    # sidebar inputs
    with st.sidebar:
        st.header("🔎 Property Details")
        region   = st.selectbox("Region",   df["region"].unique())
        locality = st.selectbox("Locality", ALL_LOCALITIES)

        region_df = df[(df.region==region)&(df.locality==locality)]
        st.write(f"• {len(region_df):,} listings in {region} / {locality}")
        can_run = len(region_df) >= 20
        if not can_run:
            st.warning("⚠️ Data sparse—predictions may vary.")

        bedrooms  = st.slider("Bedrooms",    0,5,2)
        bathrooms = st.slider("Bathrooms",   0,4,1)
        category  = st.selectbox("Category",  df["category"].unique())
        condition = st.selectbox("Condition", df["condition"].unique())
        parking   = st.selectbox("Parking?",  ["Yes","No"])
        furnish   = st.selectbox("Furnishing", ["Furnished","Unfurnished","Semi-Furnished"])
        amenities = st.multiselect("Amenities", ALL_AMENITIES)

        if st.button("Predict", disabled=not can_run):
            # prepare input & predict
            wins = region_df["price"].clip(
                lower=region_df["price"].quantile(0.05),
                upper=region_df["price"].quantile(0.95)
            ).median()
            dens = region_df.groupby(["lat","lng"]).size().mean()
            inp = {
                "bedrooms":       bedrooms,
                "bathrooms":      bathrooms,
                "parking_space":  1 if parking=="Yes" else 0,
                "is_furnished":   {"Furnished":1,"Unfurnished":0,"Semi-Furnished":0.5}[furnish],
                "amenities_count": len(amenities),
                "region_demand":  wins,
                "density":        dens,
                "category":       category,
                "condition":      condition,
                "locality":       locality,
                "floor_area":     region_df["floor_area"].median()
            }
            df_in = pd.DataFrame([inp])
            lp    = model.predict(df_in)[0]
            price = np.expm1(lp)

            st.session_state["pred"] = {
                "price":     price,
                "filters":   inp,
                "region_df": region_df
            }

    # once we have a prediction, display side-by-side
    if "pred" in st.session_state:
        pred = st.session_state["pred"]
        left, right = st.columns([1,2])

        # price card 
        with left:
            st.markdown(f"""
                <div style="
                  background:#1f1f29; padding:20px; border-radius:8px;
                  width:100%;
                ">
                  <h3 style="color:#00DAC6; margin:0;">Rent Estimate (₵/mo)</h3>
                  <h1 style="color:white; margin:0;font-size:2.5rem;">
                    ₵ {pred['price']:,.0f}
                  </h1>
                  <p style="color:#888; font-size:0.9rem; margin-top:4px;">
                    per month
                  </p>
                </div>
            """, unsafe_allow_html=True)

        # heatmap + **clean legend** 
        with right:
            st.markdown(
                """
                **🔵 Blue pins** = historical matching listings  
                **⭐️ Green star** = predicted location (centroid)
                """
            )

            # filter exact matches (fallback if too few)
            f = pred["filters"]
            mask = (
                (df.region         == region)             &
                (df.locality       == f["locality"])      &
                (df.category       == f["category"])      &
                (df.condition      == f["condition"])     &
                (df.bedrooms       == f["bedrooms"])      &
                (df.bathrooms      == f["bathrooms"])     &
                (df.parking_space  == f["parking_space"]) &
                (df.is_furnished   == f["is_furnished"])  &
                (df.amenities_count== f["amenities_count"])
            )
            heat_df = df[mask]
            if len(heat_df) < 10:
                heat_df = pred["region_df"]
                st.info("Showing all listings in this locality instead (too few exact matches).")

            # compute centroid for green ★
            centroid_lat = heat_df["lat"].mean()
            centroid_lng = heat_df["lng"].mean()

            # build the map
            m = folium.Map(
                location=(centroid_lat, centroid_lng),
                zoom_start=11,
                tiles="cartodbpositron",
                control_scale=True
            )
            cap  = heat_df["log_price"].quantile(0.95)
            data = heat_df[["lat","lng","log_price"]].clip(upper=cap).values.tolist()
            HeatMap(
                data,
                min_opacity=0.3,
                radius=8,
                gradient={"0.2":"white","0.5":"purple","0.8":"red"}
            ).add_to(m)

            # blue pins
            for _, r in heat_df.iterrows():
                folium.Marker(
                    location=(r.lat, r.lng),
                    icon=folium.Icon(color="blue", icon="info-sign"),
                    popup=f"₵ {r.price:,.0f}/mo"
                ).add_to(m)

            # green star
            folium.Marker(
                location=(centroid_lat, centroid_lng),
                icon=folium.Icon(color="green", icon="star"),
                popup=f"<b>Predicted:</b> ₵ {pred['price']:,.0f}/mo"
            ).add_to(m)

            # disable reruns on pan/zoom
            st_folium(
                m,
                width=800,
                height=500,
                returned_objects=[]  # stops map events from triggering reruns
            )

if __name__ == "__main__":
    main()

