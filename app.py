from pathlib import Path
from PIL import Image
import streamlit as st
import pandas as pd

from src.prediction_engine import predict_game

import base64
from io import BytesIO

def image_to_base64(img):

    buffer = BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

NBA_LOGO = Image.open(
    "assets/nba_playoffs_logo.webp"
)

st.set_page_config(

    page_title="NBA Analytics Platform",

    page_icon=NBA_LOGO,

    layout="wide",

    initial_sidebar_state="expanded"

)


# ==========================================================
# GLOBAL PATHS
# ==========================================================

LOGO_FOLDER = Path("assets/logos")


# ==========================================================
# IMAGE UTILITIES
# ==========================================================

def image_to_base64(image):

    """
    Convert PIL Image to Base64.
    Used later for custom HTML components.
    """

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode()


# ==========================================================
# TEAM LOGO
# ==========================================================

def load_logo(team_name):

    """
    Load team logo from assets/logos.
    """

    logo_path = LOGO_FOLDER / f"{team_name}.png"

    if logo_path.exists():

        return Image.open(logo_path)

    return None


# ==========================================================
# DISPLAY TEAM LOGO
# ==========================================================

def show_logo(team_name):

    """
    Display selected team logo.
    """

    logo = load_logo(team_name)

    if logo is None:

        return

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.image(

            logo,

            width=150

        )


# ==========================================================
# SECTION DIVIDER
# ==========================================================

def section(title):

    """
    Creates a consistent section heading.
    """

    st.divider()

    st.subheader(title)


# ==========================================================
# HERO TITLE
# ==========================================================

def hero_title(text):

    st.markdown(

        f"""
        <h1 style="
            text-align:center;
            font-size:50px;
            margin-bottom:0px;">
            {text}
        </h1>
        """,

        unsafe_allow_html=True

    )


# ==========================================================
# HERO SUBTITLE
# ==========================================================

def hero_subtitle(text):

    st.markdown(

        f"""
        <h3 style="
            text-align:center;
            color:gray;">
            {text}
        </h3>
        """,

        unsafe_allow_html=True

    )


# ==========================================================
# HERO METRIC
# ==========================================================

def hero_metric(value):

    st.markdown(

        f"""
        <h1 style="
            text-align:center;
            color:#4CAF50;
            font-size:60px;
            margin-top:-5px;
            margin-bottom:-10px;">
            {value}
        </h1>
        """,

        unsafe_allow_html=True

    )


# ==========================================================
# CONFIDENCE BADGE
# ==========================================================

def confidence_badge(confidence):

    colors = {

        "Very High": "#00C853",
        "High": "#43A047",
        "Moderate": "#FB8C00",
        "Toss-Up": "#E53935"

    }

    badge_color = colors.get(confidence, "#616161")

    st.markdown(
        f"""
<div style="text-align:center; margin-top:20px; margin-bottom:20px;">
    <span style="
        background:{badge_color};
        color:white;
        padding:10px 25px;
        border-radius:30px;
        font-size:18px;
        font-weight:bold;">
        {confidence.upper()} CONFIDENCE
    </span>
</div>
""",
        unsafe_allow_html=True
    )


# ==========================================================
# PROGRESS BAR
# ==========================================================

def probability_bar(team_name, probability):

    st.markdown(f"### {team_name}")

    st.progress(probability / 100)

    st.caption(

        f"{probability:.2f}% chance of winning"

    )
               


# ==========================================================
# HEADER
# ==========================================================

header_logo, header_text = st.columns([1, 7])

with header_logo:

    st.image(
        NBA_LOGO,
        width=95
    )

with header_text:

    st.title("NBA  Playoff Analytics Platform")

    st.caption(
        "Predict NBA Playoff Matchups using Machine Learning and Advanced Basketball Analytics."
    )

section("Matchup Selection")


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("Prediction Settings")

    model_name = st.selectbox(

        "Select Model",

        [

            "Logistic Regression",
            "Random Forest",
            "XGBoost"

        ]

    )

    st.divider()

    st.header("Features Used")

    st.markdown("""

-  Elo Rating
-  Net Rating
-  Offensive Rating
-  Defensive Rating
-  True Shooting %
-  Effective FG %
-  PIE
-  Last 10 Games
-  Win Streak
-  Rest Days

""")

    st.divider()

    st.header("Tech Stack")

    st.markdown("""

- Python
- Streamlit
- Scikit-Learn
- XGBoost
- NBA API
- Pandas

""")


# ==========================================================
# NBA TEAMS
# ==========================================================

teams = sorted([

    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "Los Angeles Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards"

])


# ==========================================================
# TEAM SELECTION
# ==========================================================

left_col, right_col = st.columns(2)

with left_col:

    st.markdown("##  Home Team")

    home_team = st.selectbox(

        "Select Home Team",

        teams,

        label_visibility="collapsed",

        key="home_team"

    )

    show_logo(home_team)


with right_col:

    st.markdown("##  Away Team")

    away_team = st.selectbox(

        "Select Away Team",

        teams,

        index=1,

        label_visibility="collapsed",

        key="away_team"

    )

    show_logo(away_team)


# ==========================================================
# ANALYZE BUTTON
# ==========================================================

section("Prediction")

predict = st.button(

    " Analyze Matchup",

    type="primary",

    use_container_width=True

)


# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    try:

        result = predict_game(

            home_team,

            away_team,

            model_name

        )

        prediction = result["prediction"]

        winner = prediction["winner"]

        home_prob = prediction["home_probability"]

        away_prob = prediction["away_probability"]

        confidence = prediction["confidence"]

        home_stats = result["home_stats"]

        away_stats = result["away_stats"]

        winner_probability = max(

            home_prob,

            away_prob

        )

         # ==========================================================
        # WINNER HERO
        # ==========================================================

        section("Prediction Result")

        winner_logo = load_logo(winner)

        left, center, right = st.columns([2.5,1,2.5])

        with center:

            if winner_logo is not None:

                st.image(

                    winner_logo,

                    width=140

                )

        hero_title(

            winner.upper()

        )

        hero_metric(

            f"{winner_probability:.2f}%"

        )

        hero_subtitle(

            "Win Probability"

        )

        confidence_badge(

            confidence

        )


        # ==========================================================
        # GAME PROJECTION
        # ==========================================================

        section("Game Projection")

        probability_bar(

            f"🏠 {home_team}",

            home_prob

        )

        st.write("")

        probability_bar(

            f"✈️ {away_team}",

            away_prob

        )


        # ==========================================================
        # ADVANCED TEAM ANALYTICS
        # ==========================================================

        section("Advanced Team Analytics")

        comparison = pd.DataFrame({

            "Metric":[

                "ELO",

                "Net Rating",

                "Off Rating",

                "Def Rating",

                "True Shooting",

                "Effective FG",

                "PIE",

                "Last 10",

                "Win Streak",

                "Rest Days"

            ],

            home_team:[

                round(home_stats["elo_rating"],1),

                round(home_stats["net_rating"],1),

                round(home_stats["off_rating"],1),

                round(home_stats["def_rating"],1),

                round(home_stats["ts_pct"]*100,1),

                round(home_stats["efg_pct"]*100,1),

                round(home_stats["pie"],3),

                round(home_stats["last10"]*100,1),

                home_stats["win_streak"],

                home_stats["rest_days"]

            ],

            away_team:[

                round(away_stats["elo_rating"],1),

                round(away_stats["net_rating"],1),

                round(away_stats["off_rating"],1),

                round(away_stats["def_rating"],1),

                round(away_stats["ts_pct"]*100,1),

                round(away_stats["efg_pct"]*100,1),

                round(away_stats["pie"],3),

                round(away_stats["last10"]*100,1),

                away_stats["win_streak"],

                away_stats["rest_days"]

            ]

        })

        st.dataframe(

            comparison,

            use_container_width=True,

            hide_index=True

        )

        # ==========================================================
        # MODEL INSIGHTS
        # ==========================================================

        section("Model Insights")

        insights = []

        if home_stats["elo_rating"] > away_stats["elo_rating"]:

            insights.append(

                f"🟢 {home_team} owns the stronger Elo Rating."

            )

        else:

            insights.append(

                f"🟢 {away_team} owns the stronger Elo Rating."

            )

        if home_stats["net_rating"] > away_stats["net_rating"]:

            insights.append(

                f"🟢 {home_team} has the better Net Rating."

            )

        else:

            insights.append(

                f"🟢 {away_team} has the better Net Rating."

            )

        if home_stats["off_rating"] > away_stats["off_rating"]:

            insights.append(

                f"🟢 {home_team} has the stronger offense."

            )

        else:

            insights.append(

                f"🟢 {away_team} has the stronger offense."

            )

        if home_stats["def_rating"] < away_stats["def_rating"]:

            insights.append(

                f"🟢 {home_team} has the better defense."

            )

        else:

            insights.append(

                f"🟢 {away_team} has the better defense."

            )

        if home_stats["ts_pct"] > away_stats["ts_pct"]:

            insights.append(

                f"🟢 {home_team} shoots more efficiently."

            )

        else:

            insights.append(

                f"🟢 {away_team} shoots more efficiently."

            )

        if home_stats["rest_days"] == away_stats["rest_days"]:

            insights.append(

                "⚪ Both teams have equal rest."

            )

        elif home_stats["rest_days"] > away_stats["rest_days"]:

            insights.append(

                f"⚪ {home_team} has more rest."

            )

        else:

            insights.append(

                f"⚪ {away_team} has more rest."

            )

        for item in insights:

            st.markdown(

                f"- {item}"

            )


    except ValueError as e:

        st.error(

            str(e)

        )

    except Exception as e:

        st.error(

            f"Unexpected Error: {e}"

        )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:15px;">
        NBA Playoff Predictor • Built by <b>Aditya Singh</b> • Powered by Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)