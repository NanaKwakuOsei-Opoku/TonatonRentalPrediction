
# 🏡 Ghana Rental Price Predictor 🇬🇭

Welcome to the **Ghana Rental Price Predictor** — a Streamlit-powered dashboard that uses **machine learning** 🧠 and **geospatial visualization** 🗺️ to estimate **fair rental prices** across Ghana!

This tool helps renters, landlords, and real estate analysts get **transparent**, **data-driven** price estimates based on property features like bedrooms, bathrooms, amenities, furnishing, and location.

---

## 🚀 Features

* 🎯 **Accurate ML-based predictions** using an ensemble of LightGBM, CatBoost, and Ridge Regression
* 📊 Log-transformed price modeling for stable, skew-resistant learning
* 🌍 Interactive **Folium heatmap** to visualize historical listings and predicted prices
* 📌 Dynamic input sliders and selectors for user-defined property features
* 📈 Real-time RMSE and R² performance display
* 🧩 Intelligent handling of sparse data regions with feedback

---

## 🛠️ Tech Stack

| Component     | Library / Tool                             |
| ------------- | ------------------------------------------ |
| UI Framework  | `Streamlit`                                |
| Maps          | `folium`, `streamlit_folium`               |
| ML Models     | `LightGBM`, `CatBoost`, `Ridge Regression` |
| Ensemble      | `StackingRegressor` (via `scikit-learn`)   |
| Data Handling | `Pandas`, `NumPy`                          |
| Deployment    | Localhost via Streamlit or Docker-ready    |

---

## 📦 Installation & Setup

1. **Clone this repository**

   ```bash
   git clone https://github.com/yourusername/ghana-rental-predictor.git
   cd ghana-rental-predictor
   ```

2. **Install dependencies**
   Create a virtual environment and install packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your dataset**
   Drop the `house_rentals.csv` file (scraped from Tonaton.com) in the root directory.

4. **Launch the app** 🚀

   ```bash
   streamlit run app.py
   ```

---

## 🧠 How It Works

1. **Preprocessing**:

   * Cleans and transforms categorical and numerical features
   * Derives new features like amenity count, region demand, density, and interaction terms

2. **Modeling Pipeline**:

   * Custom transformer (`FeatureEngineer`) handles feature construction
   * `RobustScaler` minimizes the effect of outliers
   * **Stacking ensemble**:

     * `LGBMRegressor` and `CatBoostRegressor` as base models
     * `Ridge Regression` as a meta-learner

3. **Prediction Interface**:

   * Enter your property features in the sidebar
   * Hit "Predict" and see the estimated rent 💵
   * View nearby listings and the prediction location on a Folium map 🔍

---

## 📈 Performance

| Model        | R² Score | RMSE (log) |
| ------------ | -------- | ---------- |
| Ridge        | 0.60     | 0.32       |
| LightGBM     | 0.84     | 0.21       |
| CatBoost     | 0.85     | 0.20       |
| **Ensemble** | **0.86** | **0.19**   |

✅ **The ensemble outperformed all individual models**, capturing both linear and nonlinear patterns while maintaining generalizability.

---

## 🧭 Example Use Case

> 💬 *“I’m looking to rent out my 3-bedroom furnished apartment in East Legon. What’s a fair price?”*

* Input:

  * Region: Greater Accra
  * Locality: East Legon
  * Bedrooms: 3
  * Bathrooms: 2
  * Furnishing: Furnished
* 📈 Result: Rent Estimate = ₵5,300/month
* 📍 See the green ★ and surrounding blue markers for context!

---

## 🛡️ Disclaimers

* 📅 The dataset is from 2023 — retraining quarterly is encouraged.
* 🏘️ Results may vary in sparsely listed localities.
* 🌐 Source: Publicly scraped from Tonaton.com

---

## 🤝 Contributing

Pull requests welcome! If you’d like to:

* Improve the ML model (e.g., add XGBoost or deep learning)
* Add satellite-based features or APIs
* Connect to live listing APIs
  Then fork and fire away!

---

## 📄 License

MIT License © 2025 \[NanaKwakuOsei-Opoku]

