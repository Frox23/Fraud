import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import time

start = time.time()
df = pd.read_csv("final_data.csv")
df.rename(columns={
    "age_26-35": "age_26_35",
    "age_36-45": "age_36_45",
    "age_46-55": "age_46_55",
    "age_56-65": "age_56_65",
    "fraud": "prediction"
}, inplace=True)

df.drop(columns=["merchant"], inplace=True)
df.fillna(0, inplace=True)
df = df.astype(float)

conn = psycopg2.connect(
    host="localhost",
    database="fraud_db",
    user="postgres",
    password="Frox@236",
    port=5432
)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS fraud_table;")
cur.execute("""
    CREATE TABLE fraud_table (
        step INTEGER,
        amount FLOAT,
        age_26_35 INTEGER,
        age_36_45 INTEGER,
        age_46_55 INTEGER,
        age_56_65 INTEGER,
        age_lte_18 INTEGER,
        age_gt_65 INTEGER,
        age_unknown INTEGER,
        gender_female INTEGER,
        gender_male INTEGER,
        gender_unknown INTEGER,
        category_es_contents INTEGER,
        category_es_fashion INTEGER,
        category_es_food INTEGER,
        category_es_health INTEGER,
        category_es_home INTEGER,
        category_es_hotelservices INTEGER,
        category_es_hyper INTEGER,
        category_es_leisure INTEGER,
        category_es_otherservices INTEGER,
        category_es_sportsandtoys INTEGER,
        category_es_tech INTEGER,
        category_es_transportation INTEGER,
        category_es_travel INTEGER,
        category_es_wellnessandbeauty INTEGER,
        prediction INTEGER
    );
""")
print("✅ Recreated 'fraud_table'")

rows_to_insert = []
for _, row in df.iterrows():
    rows_to_insert.append((
        int(row['step']),
        float(row['amount']),
        int(row['age_26_35']),
        int(row['age_36_45']),
        int(row['age_46_55']),
        int(row['age_56_65']),
        int(row['age_<= 18']),
        int(row['age_> 65']),
        int(row['age_Unknown']),
        int(row['gender_Female']),
        int(row['gender_Male']),
        int(row['gender_Unknown']),
        int(row.get('category_es_contents', 0)),
        int(row.get('category_es_fashion', 0)),
        int(row.get('category_es_food', 0)),
        int(row.get('category_es_health', 0)),
        int(row.get('category_es_home', 0)),
        int(row.get('category_es_hotelservices', 0)),
        int(row.get('category_es_hyper', 0)),
        int(row.get('category_es_leisure', 0)),
        int(row.get('category_es_otherservices', 0)),
        int(row.get('category_es_sportsandtoys', 0)),
        int(row.get('category_es_tech', 0)),
        int(row.get('category_es_transportation', 0)),
        int(row.get('category_es_travel', 0)),
        int(row.get('category_es_wellnessandbeauty', 0)),
        int(row['prediction'])
    ))

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
execute_batch(cur, insert_query, rows_to_insert, page_size=1000)
conn.commit()
cur.close()
conn.close()

print(f"✅ Inserted {len(rows_to_insert)} rows into 'fraud_table' in {round(time.time() - start, 2)} seconds.")
