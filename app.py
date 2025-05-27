from fastapi import FastAPI, BackgroundTasks, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import psycopg2
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = joblib.load("fraud_model.pkl")
feature_names = joblib.load("model_features.pkl")
scaler = joblib.load("scaler.pkl")
count = 0

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="fraud_db",
        user="postgres",
        password="Frox@236",
        port=5432
    )

@app.get("/", response_class=HTMLResponse)
def show_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict_ui/", response_class=HTMLResponse)
def predict_ui(
    request: Request,
    background_tasks: BackgroundTasks,
    step: int = Form(...),
    amount: float = Form(...),
    gender: str = Form(...),
    age_group: str = Form(...),
    category: str = Form(...)
):
    global count
    try:
        scaled_amount_df = pd.DataFrame({"amount": [amount]})
        scaled_amount = scaler.transform(scaled_amount_df)[0][0]

        row_dict = {col: 0 for col in feature_names}
        row_dict["step"] = step
        row_dict["amount"] = scaled_amount
        row_dict[f"gender_{gender}"] = 1
        row_dict[f"age_{age_group}"] = 1
        row_dict[f"category_{category}"] = 1

        input_df = pd.DataFrame([row_dict])
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        prediction = model.predict(input_df)[0]

        conn = get_db_connection()
        cur = conn.cursor()

        insert_query = """
            INSERT INTO fraud_table (
                step, amount,
                age_26_35, age_36_45, age_46_55, age_56_65,
                age_lte_18, age_gt_65, age_unknown,
                gender_female, gender_male, gender_unknown,
                category_es_contents, category_es_fashion, category_es_food,
                category_es_health, category_es_home, category_es_hotelservices,
                category_es_hyper, category_es_leisure, category_es_otherservices,
                category_es_sportsandtoys, category_es_tech, category_es_transportation,
                category_es_travel, category_es_wellnessandbeauty,
                prediction
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            step, scaled_amount,
            row_dict.get("age_26_35", 0), row_dict.get("age_36_45", 0), row_dict.get("age_46_55", 0), row_dict.get("age_56_65", 0),
            row_dict.get("age_lte_18", 0), row_dict.get("age_gt_65", 0), row_dict.get("age_unknown", 0),
            row_dict.get("gender_female", 0), row_dict.get("gender_male", 0), row_dict.get("gender_unknown", 0),
            row_dict.get("category_es_contents", 0), row_dict.get("category_es_fashion", 0), row_dict.get("category_es_food", 0),
            row_dict.get("category_es_health", 0), row_dict.get("category_es_home", 0), row_dict.get("category_es_hotelservices", 0),
            row_dict.get("category_es_hyper", 0), row_dict.get("category_es_leisure", 0), row_dict.get("category_es_otherservices", 0),
            row_dict.get("category_es_sportsandtoys", 0), row_dict.get("category_es_tech", 0), row_dict.get("category_es_transportation", 0),
            row_dict.get("category_es_travel", 0), row_dict.get("category_es_wellnessandbeauty", 0),
            int(prediction)
        )

        cur.execute(insert_query, values)
        conn.commit()
        cur.close()
        conn.close()

        count += 1
        background_tasks.add_task(check_and_retrain)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "prediction": "Fraud" if prediction == 1 else "Not Fraud"
        })

    except Exception as e:
        print("\U0001F525 Internal Error:", str(e))
        return HTMLResponse(content="Internal Server Error", status_code=500)

def check_and_retrain():
    global count
    try:
        engine = create_engine("postgresql+psycopg2://postgres:Frox%40236@localhost:5432/fraud_db")
        df = pd.read_sql("SELECT * FROM fraud_table", engine)
        df.fillna(0, inplace=True)

        if count % 15 != 0:
            print("⏳ Not retraining yet. Count:", count)
            return

        y = df["prediction"]
        X_raw = df.drop(columns=["prediction"], errors='ignore')

        X = X_raw.copy()
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]

        new_model = RandomForestClassifier()
        new_model.fit(X, y)
        y_proba = new_model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_proba)

        old_auc = joblib.load("fraud_model_auc.pkl") if os.path.exists("fraud_model_auc.pkl") else 0
        if auc > old_auc:
            joblib.dump(new_model, "fraud_model.pkl")
            joblib.dump(auc, "fraud_model_auc.pkl")
            print(f"✅ Model updated: AUC {old_auc:.4f} → {auc:.4f}")
        else:
            print(f"❌ AUC not improved: {old_auc:.4f} ≥ {auc:.4f}")

    except Exception as e:
        print("⚠️ Retrain Error:", str(e))