import sqlite3
conn = sqlite3.connect('historique.db')
rows = conn.execute(
    "SELECT unit_number, MIN(received_at), MAX(received_at), COUNT(*) "
    "FROM sensor_history GROUP BY unit_number"
).fetchall()
for r in rows:
    print(r)