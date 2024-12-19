import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import kagglehub

# Download latest version
path = kagglehub.dataset_download("indrajuliansyahputra/premier-league-player-stats-2324")

print("Path to dataset files:", path)

player_overview_csv = pd.read_csv(path + "/player_overview.csv")

label_encoder = LabelEncoder()

player_overview_csv = player_overview_csv.drop('Name', axis=1)
player_overview_csv = player_overview_csv.drop('Club', axis=1)
player_overview_csv = player_overview_csv.drop('Clean sheets', axis=1)
player_overview_csv = player_overview_csv.drop('Facebook', axis=1)

player_overview_csv['Height'] = player_overview_csv['Height'].str.replace('cm', '').astype(float)

position_mapping = {'Goalkeeper': 1, 'Defender': 2, 'Midfielder': 3, 'Forward': 4}
player_overview_csv['Position'] = player_overview_csv['Position'].replace(position_mapping).astype('int64')
player_overview_csv['Nationality'] = label_encoder.fit_transform(player_overview_csv['Nationality']).astype('int64')
player_overview_csv['Height'].fillna(player_overview_csv['Height'].mean(), inplace=True)
player_overview_csv['Assists'] = player_overview_csv['Assists'].fillna(0)
player_overview_csv['Goals'] = player_overview_csv['Goals'].fillna(0)
player_overview_csv['Appearances'] = player_overview_csv['Appearances'].fillna(0)

def calculate_age(dob):
    birth_date_str = dob.split(' ')[0]
    birth_date = datetime.strptime(birth_date_str, '%d/%m/%Y')
        
    today = datetime.now()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

player_overview_csv['Age'] = player_overview_csv['Date of Birth'].apply(calculate_age)

player_overview_csv = player_overview_csv.drop('Date of Birth', axis=1)

player_overview_csv = player_overview_csv.loc[player_overview_csv['Appearances'] < 300]

features = [
    'Nationality', 
    'Height', 
    'Age', 
    'Position', 
    'Goals', 
    'Assists'
    ]


X = player_overview_csv[features]
y = player_overview_csv['Appearances']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardizálás
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train) # Illesztés és transzformálás a tanító adatokon
X_test_scaled = scaler.transform(X_test) # Transzformálás a teszt adatokon a tanító adatokon tanult scalerrel

model = RandomForestRegressor(n_estimators=200, min_samples_split= 5, min_samples_leaf=2, max_depth= None, random_state= 42)

# 6. Modell illesztése a standardizált tanító adatokra
model.fit(X_train, y_train)

# 7. Predikciók a standardizált teszt adatokon
y_pred = model.predict(X_test)

st.title("Player Appearances Prediction")
st.write("Adja meg a játékos adatait, hogy megkapja a predikált \"Appearances\" értéket!")

# 3. Felhasználói bemenet
st.sidebar.header("Input Data")
nationality = st.sidebar.slider("Nationality (kód)", 0, 10, 5)
height = st.sidebar.slider("Height (cm)", 150, 210, 180)
age = st.sidebar.slider("Age", 18, 40, 25)
position = st.sidebar.slider("Position (kód)", 1, 4, 2)
goals = st.sidebar.slider("Goals", 0, 30, 5)
assists = st.sidebar.slider("Assists", 0, 20, 3)
# clean_sheets = st.sidebar.slider("Clean Sheets", 0, 10, 2)

# 4. Adatok előkészítése
input_data = np.array([[nationality, height, age, position, goals, assists]])
#scaler = RobustScaler()
#scaled_input = scaler.fit_transform(input_data)

# 5. Modell betöltése és predikció
predicted_appearances = model.predict(input_data)

# 6. Predikció megjelenítése
st.subheader("Predicted Appearances")
st.write(f"A predikált meccsszám: **{predicted_appearances[0]:.2f}**")
