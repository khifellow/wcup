import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="World Cup 2026 Prediction Dashboard", layout="centered")

# 1. LOAD DATA & TRAIN MODELS (Cached for 3 Hours)
@st.cache_data(ttl=10800)
def process_data_and_train(file_path="results.xlsx"):
    if not os.path.exists(file_path):
        return None, None, None, time.time(), f"File '{file_path}' not found in the running directory."
    
    try:
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_excel(file_path)
            
        if df.empty or len(df.columns) < 3:
            return None, None, None, time.time(), "The dataset file appears to be empty."

        # Robust mapping for truncated column views
        column_mapping = {}
        for col in df.columns:
            col_clean = str(col).lower().strip()
            if 'date' in col_clean: column_mapping[col] = 'date'
            elif 'home_tea' in col_clean or 'home_tean' in col_clean: column_mapping[col] = 'home_team'
            elif 'away_tea' in col_clean or 'away_tean' in col_clean: column_mapping[col] = 'away_team'
            elif 'home_sco' in col_clean: column_mapping[col] = 'home_score'
            elif 'away_sco' in col_clean: column_mapping[col] = 'away_score'
            elif 'tournam' in col_clean: column_mapping[col] = 'tournament'
            elif 'city' in col_clean: column_mapping[col] = 'city'
            elif 'country' in col_clean: column_mapping[col] = 'country'
            elif 'id' == col_clean: column_mapping[col] = 'id'

        df = df.rename(columns=column_mapping)
        
        required = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'tournament']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return None, None, None, time.time(), f"Missing required columns: {missing}"

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'home_team', 'away_team']).copy()
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0)
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0)
        
        if 'id' in df.columns:
            df = df.sort_values(by=['date', 'id']).reset_index(drop=True)
        else:
            df = df.sort_values(by='date').reset_index(drop=True)
            
    except Exception as e:
        return None, None, None, time.time(), f"Data error: {str(e)}"

    df['outcome'] = np.where(df['home_score'] > df['away_score'], 1, 
                             np.where(df['home_score'] < df['away_score'], -1, 0))

    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    team_history = {team: [] for team in all_teams}
    
    features = []
    targets = []

    for idx, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        
        def get_weighted_form(team_matches):
            if not team_matches: return 0.0
            recent = team_matches[-10:]
            weights = np.linspace(0.1, 1.0, len(recent))
            return np.sum(np.array(recent) * weights) / np.sum(weights)

        home_form = get_weighted_form(team_history.get(home, []))
        away_form = get_weighted_form(team_history.get(away, []))
        form_diff = home_form - away_form
        is_world_cup = 1 if row['tournament'] == 'FIFA World Cup' else 0

        if len(team_history.get(home, [])) >= 3 and len(team_history.get(away, [])) >= 3:
            features.append([form_diff, is_world_cup])
            targets.append(1 if row['outcome'] == 1 else 0)

        home_pts = 3 if row['outcome'] == 1 else (1 if row['outcome'] == 0 else 0)
        away_pts = 3 if row['outcome'] == -1 else (1 if row['outcome'] == 0 else 0)
        
        if home in team_history: team_history[home].append(home_pts)
        if away in team_history: team_history[away].append(away_pts)

    X = np.array(features)
    y = np.array(targets)
    
    if len(X) == 0:
        return None, None, None, time.time(), "Insufficient training records found."
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf2 = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', clf1), ('gb', clf2)], voting='soft')
    ensemble.fit(X_scaled, y)
    
    return ensemble, scaler, team_history, time.time(), None

# Execute pipeline load
model, scaler, team_history, cache_timestamp, error_message = process_data_and_train()

st.title("🏆 World Cup 2026 Predictions")
st.markdown("Simplified expert matching recommendations driven by machine learning form analysis.")

if error_message:
    st.error(error_message)
else:
    # Full incoming schedule matching official World Cup 2026 calendar targets
    fixtures = [
        {"date": "Monday, June 15", "home": "Spain", "away": "Cabo Verde", "group": "Group H"},
        {"date": "Monday, June 15", "home": "Belgium", "away": "Egypt", "group": "Group G"},
        {"date": "Monday, June 15", "home": "Saudi Arabia", "away": "Uruguay", "group": "Group H"},
        {"date": "Monday, June 15", "home": "Iran", "away": "New Zealand", "group": "Group G"},
        {"date": "Tuesday, June 16", "home": "France", "away": "Senegal", "group": "Group I"},
        {"date": "Tuesday, June 16", "home": "Iraq", "away": "Norway", "group": "Group I"},
        {"date": "Tuesday, June 16", "home": "Argentina", "away": "Algeria", "group": "Group J"},
        {"date": "Tuesday, June 16", "home": "Austria", "away": "Jordan", "group": "Group J"},
        {"date": "Wednesday, June 17", "home": "Portugal", "away": "DR Congo", "group": "Group K"},
        {"date": "Wednesday, June 17", "home": "England", "away": "Croatia", "group": "Group L"},
        {"date": "Wednesday, June 17", "home": "Ghana", "away": "Panama", "group": "Group L"},
        {"date": "Wednesday, June 17", "home": "Uzbekistan", "away": "Colombia", "group": "Group K"},
        {"date": "Thursday, June 18", "home": "Czech Republic", "away": "South Africa", "group": "Group A"},
        {"date": "Thursday, June 18", "home": "Switzerland", "away": "Bosnia and Herzegovina", "group": "Group B"},
        {"date": "Thursday, June 18", "home": "Canada", "away": "Qatar", "group": "Group B"},
        {"date": "Thursday, June 18", "home": "Mexico", "away": "South Korea", "group": "Group A"},
        {"date": "Friday, June 19", "home": "USA", "away": "Australia", "group": "Group D"},
        {"date": "Friday, June 19", "home": "Scotland", "away": "Morocco", "group": "Group C"},
        {"date": "Friday, June 19", "home": "Türkiye", "away": "Paraguay", "group": "Group D"},
        {"date": "Friday, June 19", "home": "Brazil", "away": "Haiti", "group": "Group C"},
        {"date": "Saturday, June 20", "home": "Germany", "away": "Ivory Coast", "group": "Group E"},
        {"date": "Saturday, June 20", "home": "Ecuador", "away": "Curaçao", "group": "Group E"},
        {"date": "Saturday, June 20", "home": "Netherlands", "away": "Sweden", "group": "Group F"},
        {"date": "Saturday, June 20", "home": "Tunisia", "away": "Japan", "group": "Group F"},
        {"date": "Sunday, June 21", "home": "Spain", "away": "Saudi Arabia", "group": "Group H"},
        {"date": "Sunday, June 21", "home": "Belgium", "away": "Iran", "group": "Group G"},
        {"date": "Sunday, June 21", "home": "Uruguay", "away": "Cabo Verde", "group": "Group H"},
        {"date": "Sunday, June 21", "home": "New Zealand", "away": "Egypt", "group": "Group G"},
        {"date": "Monday, June 22", "home": "Argentina", "away": "Austria", "group": "Group J"},
        {"date": "Monday, June 22", "home": "France", "away": "Iraq", "group": "Group I"},
        {"date": "Monday, June 22", "home": "Norway", "away": "Senegal", "group": "Group I"},
        {"date": "Monday, June 22", "home": "Jordan", "away": "Algeria", "group": "Group J"}
    ]

    # Clean UI Navigation to look up matches by fixture date
    unique_dates = sorted(list(set([f["date"] for f in fixtures])), key=lambda d: time.strptime(d.split(", ")[1], "%B %d"))
    selected_date = st.selectbox("📅 Filter Match Schedule by Date", unique_dates)

    st.markdown(f"### Fixtures for {selected_date}")
    
    for match in fixtures:
        if match["date"] != selected_date:
            continue
            
        home, away, group = match["home"], match["away"], match["group"]
        
        def extract_current_form(team_matches):
            if not team_matches: return 0.0
            recent = team_matches[-10:]
            weights = np.linspace(0.1, 1.0, len(recent))
            return np.sum(np.array(recent) * weights) / np.sum(weights)

        home_f = extract_current_form(team_history.get(home, []))
        away_f = extract_current_form(team_history.get(away, []))
        form_diff = home_f - away_f
        
        input_data = np.array([[form_diff, 1]])
        input_scaled = scaler.transform(input_data)
        
        home_win_prob = model.predict_proba(input_scaled)[0][1]
        draw_prob = (1.0 - home_win_prob) * 0.32
        away_win_prob = (1.0 - home_win_prob) * 0.68

        # Simplify to recommendation and confidence level assignment
        if home_win_prob > away_win_prob and home_win_prob > draw_prob:
            recommendation = f"🏆 Expect Win for **{home}**"
            confidence_score = home_win_prob
        elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
            recommendation = f"🏆 Expect Win for **{away}**"
            confidence_score = away_win_prob
        else:
            recommendation = "⚖️ Neutral / High Draw Risk"
            confidence_score = draw_prob

        # Categorize confidence metrics
        if confidence_score >= 0.62:
            confidence_badge = "🟢 High Confidence"
        elif confidence_score >= 0.46:
            confidence_badge = "🟡 Medium Confidence"
        else:
            confidence_badge = "🔴 Low Confidence"

        # Simpler display component cards
        with st.container():
            st.markdown(f"#### {home} vs {away}  `{group}`")
            col1, col2 = st.columns(2)
            col1.markdown(f"**Recommendation:** {recommendation}")
            col2.markdown(f"**Confidence Level:** {confidence_badge}")
            st.markdown("---")

    # Reset cache trigger layout footer
    if st.sidebar.button("🔄 Force Re-train Models"):
        st.cache_data.clear()
        st.rerun()