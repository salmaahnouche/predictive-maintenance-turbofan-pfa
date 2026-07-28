import pandas as pd
import numpy as np
import joblib

scaler = joblib.load(r"C:\Users\HP\Desktop\pfa_proj\scaler_cloud.pkl")

drop = ['unit_number','time_cycles','setting_3',
        'sensor_1','sensor_5','sensor_6',
        'sensor_10','sensor_16','sensor_18','sensor_19']

df = pd.read_csv(r"C:\Users\HP\Desktop\pfa_proj\test_FD003_moteur_1.csv")
features = [c for c in df.columns if c not in drop]
df[features] = scaler.transform(df[features])

print(f"Moteur 1 — nb lignes CSV : {len(df)}")
print(f"RUL reel au dernier cycle : 44")
print(f"RUL reel au cycle 1 du CSV : {len(df) + 44 - 1}")

TIME_STEPS = 30
X = []
for i in range(TIME_STEPS, len(df)+1):
    X.append(df[features].values[i-TIME_STEPS:i])
X = np.array(X)

rul_reel = np.arange(len(df) + 44 - 1, 44 - 1, -1)

print(f"\n{'Cycle':>6}  {'RUL reel':>10}")
for i in [0, 10, 20, 30, 50, 70, len(X)-1]:
    if i < len(rul_reel):
        print(f"{i+TIME_STEPS:>6}  {rul_reel[i]:>10.0f}")