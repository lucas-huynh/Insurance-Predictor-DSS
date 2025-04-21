import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import uuid
from datetime import datetime
import seaborn as sns

# Define a dictionary with username-password mapping
USER_CREDENTIALS = {
    "lucas": "his2025_lh",
    "joanna": "his2025_jc",
    "anshika": "his2025_as",
    "aditya": "his2025_ad",
    "xinyu": "his2025_xy",
    "rema": "his2025_rp"
}

# Function for login
def login():
    # Title for login screen
    st.title("Welcome to Clinical DSS")
    st.markdown("Please enter your username and password to access the system:")

    # Custom styling
    st.markdown("""
        <style>
        .login-container {
            background-color: #f0f0f0;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .login-title {
            font-size: 24px;
            color: #333;
        }
        .login-button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .login-button:hover {
            background-color: #45a049;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create a container for the login form
    with st.container():
        # Username input
        username = st.text_input("Enter Username", key="username_input")
        # Password input
        password = st.text_input("Enter Password", type="password", key="password_input")

        # Login button
        if st.button("Login", key="login_button"):
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                # Update session state only after login button is pressed and credentials are validated
                st.session_state['authenticated'] = True
                st.session_state['username'] = username  # Store username in session state
                st.success("Login successful! Redirecting...")
                st.rerun()  # Automatically rerun the script after login
            else:
                st.session_state['authenticated'] = False
                st.error("Invalid username or password, please try again.")

# Check if the user is authenticated
if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
    login()  # Show login page if not authenticated
else:
    # Only allow access to the app after successful login
    # Load model
    model = joblib.load("best_gradient_boosting_model.pkl")

    # ------------------------
    # Region mapping by state
    # ------------------------
    state_to_region = {
        'ME': 'northeast', 'NH': 'northeast', 'VT': 'northeast', 'MA': 'northeast',
        'RI': 'northeast', 'CT': 'northeast', 'NY': 'northeast', 'NJ': 'northeast',
        'PA': 'northeast', 'DE': 'southeast', 'MD': 'southeast', 'VA': 'southeast',
        'WV': 'southeast', 'NC': 'southeast', 'SC': 'southeast', 'GA': 'southeast',
        'FL': 'southeast', 'KY': 'southeast', 'TN': 'southeast', 'AL': 'southeast',
        'MS': 'southeast', 'AR': 'southeast', 'LA': 'southeast', 'WA': 'northwest',
        'OR': 'northwest', 'ID': 'northwest', 'MT': 'northwest', 'WY': 'northwest',
        'AK': 'northwest', 'CA': 'southwest', 'NV': 'southwest', 'UT': 'southwest',
        'AZ': 'southwest', 'NM': 'southwest', 'CO': 'southwest', 'TX': 'southwest',
        'OK': 'southwest'
    }

    # ------------------------
    # Navigation
    # ------------------------
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["💻 Prediction", "📊 Analytics Dashboard", "📄 Patient Records"])

    data_path = "healthinsurancedatabase.csv"
    if os.path.exists(data_path):
        df_main = pd.read_csv(data_path)

        # Assign patient_id to rows missing it
        if "patient_id" not in df_main.columns:
            df_main["patient_id"] = [str(uuid.uuid4()) for _ in range(len(df_main))]
        else:
            df_main["patient_id"] = df_main["patient_id"].fillna("").apply(
                lambda x: str(uuid.uuid4()) if x == "" else x
            )

        # Add timestamp if missing
        if "timestamp" not in df_main.columns:
            df_main["timestamp"] = datetime.now().isoformat()
        else:
            df_main["timestamp"] = df_main["timestamp"].fillna(datetime.now().isoformat())

        # Recompute age_group based on current age values
        if "age" in df_main.columns:
            df_main["age_group"] = pd.cut(df_main["age"], bins=[0, 30, 50, 100], labels=["<30", "30-50", "50+"])

        # Move patient_id to the first column
        columns = df_main.columns.tolist()
        if "patient_id" in columns:
            columns.insert(0, columns.pop(columns.index("patient_id")))
            df_main = df_main[columns]

        df_main.to_csv(data_path, index=False)

    else:
        df_main = pd.DataFrame()

    # ------------------------
    # Prediction Page
    # ------------------------
    if page == "💻 Prediction":
        st.title("🩺 Clinical DSS: Insurance Cost Estimator")

        with st.sidebar.form("patient_form"):
            st.header("Enter Patient Info")
            age = st.slider("Age", 18, 100, 35)
            sex = st.selectbox("Sex", ["male", "female"])
            bmi = st.slider("BMI", 10.0, 50.0, 25.0)
            children = st.slider("Number of Children", 0, 5, 0)
            smoker = st.selectbox("Smoker?", ["yes", "no"])
            state_input = st.selectbox("State of Residence", sorted(state_to_region.keys()))
            region = state_to_region.get(state_input, 'southeast')
            submit = st.form_submit_button("Predict")

        def prepare_input(age, sex, bmi, children, smoker, region):
            input_dict = {
                'age': age,
                'bmi': bmi,
                'children': children,
                'sqrt_children': np.sqrt(children),
                'sex_male': 1 if sex == 'male' else 0,
                'sex_encoded': 1 if sex == 'male' else 0,
                'smoker_yes': 1 if smoker == 'yes' else 0,
                'smoker_encoded': 1 if smoker == 'yes' else 0,
                'region_northwest': 1 if region == 'northwest' else 0,
                'region_southeast': 1 if region == 'southeast' else 0,
                'region_southwest': 1 if region == 'southwest' else 0
            }
            input_df = pd.DataFrame([input_dict])
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
            return input_df

        if submit:
            X_new = prepare_input(age, sex, bmi, children, smoker, region)
            pred_log = model.predict(X_new)[0]
            pred = np.expm1(pred_log)

            st.success(f"💰 Estimated Insurance Charges: **${pred:,.2f}**")

            explainer = shap.Explainer(model)
            shap_values = explainer(X_new)

            st.subheader("SHAP Explanation")
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(plt.gcf())

            with st.expander("📘 How to Interpret This Chart", expanded=True):
                st.markdown("""
                - **What this shows:** The model starts with an average baseline cost and adjusts it based on the patient's individual features.
                - **Red bars** increase the predicted cost, while **blue bars** decrease it.
                - For example, being a smoker significantly increases insurance costs, while having a normal BMI or being younger helps reduce it.
                - The total effect of these features gives the final prediction shown above.
                
                **Note:** This chart uses SHAP (SHapley Additive exPlanations) — a method for explaining how much each feature influenced the model's output for this patient.
                """)

            top_features = shap_values[0].values
            feature_names = shap_values[0].feature_names
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'shap_value': top_features,
                'abs_val': np.abs(top_features)
            }).sort_values(by='abs_val', ascending=False)

            top_contributors = shap_df.head(3)
            explanation = []
            for _, row in top_contributors.iterrows():
                direction = "increased" if row['shap_value'] > 0 else "decreased"
                explanation.append(f"**{row['feature']}** {direction} the predicted cost")

            summary = (
                "This patient's predicted cost is influenced most by: "
                + ", ".join(explanation) + "."
            )

            st.markdown("### 🧠 Model Interpretation Summary")
            st.info(summary)

            patient_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            X_new["patient_id"] = patient_id
            X_new["sex"] = sex 
            X_new["charges"] = pred
            X_new["region"] = region
            X_new["smoker"] = smoker
            X_new["age_group"] = pd.cut([age], bins=[0, 30, 50, 100], labels=["<30", "30-50", "50+"])[0]
            X_new["timestamp"] = timestamp

            for i, name in enumerate(shap_values[0].feature_names):
                X_new[f"shap_{name}"] = shap_values[0].values[i]

            df_main = pd.concat([df_main, X_new], ignore_index=True)
            df_main.to_csv(data_path, index=False)

    # ------------------------
    # Analytics Dashboard
    # ------------------------
    elif page == "📊 Analytics Dashboard":
        st.title("📊 Analytics Dashboard")

        if df_main.empty:
            st.warning("No patient data available yet. Run some predictions first!")
            st.stop()

        # Ensure timestamp is in datetime format
        df_main["timestamp"] = pd.to_datetime(df_main["timestamp"], errors='coerce')

        # Remove rows where timestamp conversion failed
        df_main = df_main.dropna(subset=["timestamp"])


        with st.sidebar:
            st.subheader("Filter Patients")
            selected_region = st.multiselect("Region", df_main["region"].dropna().unique(), default=df_main["region"].unique())
            selected_smoker = st.multiselect("Smoker", df_main["smoker"].dropna().unique(), default=df_main["smoker"].unique())
            age_options = df_main["age_group"].dropna().astype(str).unique().tolist()
            selected_age_group = st.multiselect("Age Group", age_options, default=age_options)

        df_filtered = df_main[
            df_main["region"].isin(selected_region) &
            df_main["smoker"].isin(selected_smoker) &
            df_main["age_group"].isin(selected_age_group)
        ]

        st.metric("\U0001F4C8 Average Charge", f"${df_filtered['charges'].mean():,.2f}")
        st.metric("\U0001F465 Total Patients", len(df_filtered))

        # Boxplot: Distribution of Charges by Age Group, Region, and Smoker
        st.subheader("📦 Distribution of Charges by Age Group and Smoking Status")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df_filtered, x="age_group", y="charges", hue="smoker", ax=ax)
        ax.set_title("Distribution of Charges by Age Group and Smoking Status")
        st.pyplot(fig)

        # Boxplot: BMI by Region (with Smoking)
        st.subheader("🍏 BMI Distribution by Region and Smoking Status")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df_filtered, x="region", y="bmi", hue="smoker", ax=ax)
        ax.set_title("BMI Distribution by Region and Smoking Status")
        st.pyplot(fig)


        # Bar Chart: Average Charges by Region
        st.subheader("🌍 Average Charges by Region")
        region_avg_charges = df_filtered.groupby("region")["charges"].mean().sort_values()
        st.bar_chart(region_avg_charges)

        # Bar Chart: Average Charges by Smoking Status
        st.subheader("🚬 Avg Charges by Smoking Status")
        smoker_avg = df_filtered.groupby("smoker")["charges"].mean()
        st.bar_chart(smoker_avg)

        # Heatmap of Correlations Between Numeric Variables
        st.subheader("🔗 Feature Correlation Heatmap")
        numeric = df_filtered[["age", "bmi", "children", "charges"]]
        corr = numeric.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Pairplot: Plot pairwise relationships in a dataset
        st.subheader("🔍 Pairplot of Features")
        sns.pairplot(df_filtered[["charges", "age", "bmi", "children", "smoker"]], hue="smoker", palette="coolwarm")
        st.pyplot(plt.gcf())

        # Pie Chart: Region Distribution of Patients
        st.subheader("🗺️ Patient Distribution by Region")
        region_counts = df_filtered["region"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(region_counts, labels=region_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

        # Trend Over Time (if timestamps are present)
        st.subheader("📈 Avg Charges Over Time")
        df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"])
        df_time = df_filtered.set_index("timestamp").resample("M")["charges"].mean()
        st.line_chart(df_time)

    # ------------------------
    # Patient Records Page
    # ------------------------
    elif page == "📄 Patient Records":
        st.title("📄 Patient Prediction Records")

        if df_main.empty:
            st.warning("No patient data available yet.")
            st.stop()

        with st.sidebar:
            st.subheader("Manage Patients")

            search_term = st.text_input("Search by Patient ID or Region")
            if search_term:
                df_main = df_main[df_main["patient_id"].astype(str).str.contains(search_term) |
                                df_main["region"].astype(str).str.contains(search_term)]

            st.markdown("### Add New Patient Manually")
            manual_age = st.slider("Age", 18, 100, 40, key="add")
            manual_bmi = st.slider("BMI", 10.0, 50.0, 30.0)
            manual_children = st.slider("Children", 0, 5, 1)
            manual_sex = st.selectbox("Sex", ["male", "female"], key="add_sex")
            manual_smoker = st.selectbox("Smoker", ["yes", "no"], key="add_smoker")
            manual_state = st.selectbox("State", sorted(state_to_region.keys()), key="add_state")
            manual_region = state_to_region.get(manual_state, 'southeast')

            # Add a new field for premium
            manual_premium = st.number_input("Premium", min_value=0.0, max_value=100000.0, step=100.0, value=5000.0)


            if st.button("Add Patient"):
                patient_id = str(uuid.uuid4())
                timestamp = datetime.now().isoformat()
                row = {
                    'age': manual_age,
                    'bmi': manual_bmi,
                    'children': manual_children,
                    'sex': manual_sex,
                    'smoker': manual_smoker,
                    'region': manual_region,
                    'patient_id': patient_id,
                    'timestamp': timestamp,
                    'age_group': pd.cut([manual_age], bins=[0, 30, 50, 100], labels=["<30", "30-50", "50+"])[0]
                }
                df_main = pd.concat([df_main, pd.DataFrame([row])], ignore_index=True)
                df_main.to_csv(data_path, index=False)
                st.success("Patient added!")

            if "patient_id" in df_main.columns:
                delete_id = st.selectbox("Select Patient ID to Remove", df_main["patient_id"])
                if st.button("Remove Patient"):
                    df_main = df_main[df_main["patient_id"] != delete_id]
                    df_main.to_csv(data_path, index=False)
                    st.success("Patient removed.")

        # Hide ML-specific columns (e.g., shap, encoded features)
        df_display = df_main[["patient_id", "age", "sex", "bmi", "children", "charges", "region", "smoker", "age_group", "timestamp"]]
        st.dataframe(df_display.reset_index(drop=True))
