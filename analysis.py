import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway, chi2_contingency, pearsonr

# =========================================================
# 1. LOAD DATA
# =========================================================

survey = pd.read_csv("survey.csv")
spotify = pd.read_csv("dataset.csv")

# =========================================================
# 2. CLEAN SURVEY DATA
# =========================================================

# Rename survey columns
survey.columns = [
    "timestamp",
    "study_hours",
    "focus",
    "music",
    "genre_tr"
]

# Drop timestamp
survey = survey.drop(columns=["timestamp"])

# Strip spaces
for col in survey.columns:
    survey[col] = survey[col].astype(str).str.strip()

# Focus to numeric
survey["focus"] = pd.to_numeric(survey["focus"], errors="coerce")

# Keep only valid music answers
survey = survey[survey["music"].isin(["Evet", "Hayır"])]

# Fix empty genre values
survey["genre_tr"] = survey["genre_tr"].replace(["nan", "NaN", "", "None"], np.nan)

# Standardize Turkish survey genre values
def clean_survey_genre(g):
    if pd.isna(g):
        return np.nan

    g = str(g).strip().lower()

    if g in ["klasik"]:
        return "Classical"
    elif g in ["lo-fi", "lofi"]:
        return "Lo-fi"
    elif g in ["jazz"]:
        return "Jazz"
    elif g in ["pop", "indie", "r&b"]:
        return "Pop"
    elif g in ["rap / hip-hop", "rap/hip-hop", "rap", "hip-hop", "hip hop"]:
        return "Rap / Hip-hop"
    elif g in ["rock", "metal", "alternative rock"]:
        return "Rock"
    elif g in ["elektronik", "electronic", "edm", "techno", "house"]:
        return "Electronic"
    else:
        return "Other"

survey["genre"] = survey["genre_tr"].apply(clean_survey_genre)

# Ordered study hours
study_order = ["0-2 saat", "2-4 saat", "4-6 saat", "6+ saat"]
survey["study_hours"] = pd.Categorical(
    survey["study_hours"],
    categories=study_order,
    ordered=True
)

# Numeric midpoint version for optional numeric analysis
study_hours_map = {
    "0-2 saat": 1,
    "2-4 saat": 3,
    "4-6 saat": 5,
    "6+ saat": 6
}
survey["study_hours_num"] = survey["study_hours"].map(study_hours_map)

# Drop rows with missing focus
survey = survey.dropna(subset=["focus"])

print("Survey shape:", survey.shape)
print("\nSurvey preview:")
print(survey.head())

# =========================================================
# 3. CLEAN SPOTIFY DATA
# =========================================================

spotify = spotify[[
    "track_genre",
    "energy",
    "danceability",
    "tempo",
    "valence",
    "acousticness",
    "instrumentalness"
]].copy()

def map_spotify_genre(g):
    g = str(g).lower().strip()

    # Classical
    if (
        "classical" in g
        or "piano" in g
        or "opera" in g
        or "baroque" in g
        or "orchestra" in g
        or "romantic" in g
    ):
        return "Classical"

    # Lo-fi
    elif "lofi" in g or "lo-fi" in g:
        return "Lo-fi"

    # Jazz
    elif "jazz" in g or "blues" in g or "swing" in g:
        return "Jazz"

    # Rap / Hip-hop
    elif (
        "hip hop" in g
        or "hip-hop" in g
        or "rap" in g
        or "trap" in g
    ):
        return "Rap / Hip-hop"

    # Rock
    elif (
        "rock" in g
        or "metal" in g
        or "punk" in g
        or "grunge" in g
        or "alternative" in g
    ):
        return "Rock"

    # Electronic
    elif (
        "edm" in g
        or "electronic" in g
        or "techno" in g
        or "house" in g
        or "dance" in g
        or "trance" in g
        or "dubstep" in g
    ):
        return "Electronic"

    # Pop
    elif (
        "pop" in g
        or "indie" in g
        or "r&b" in g
        or "soul" in g
    ):
        return "Pop"

    else:
        return "Other"

spotify["genre"] = spotify["track_genre"].apply(map_spotify_genre)

spotify_grouped = spotify.groupby("genre")[[
    "energy",
    "danceability",
    "tempo",
    "valence",
    "acousticness",
    "instrumentalness"
]].mean().reset_index()

print("\nSpotify grouped features:")
print(spotify_grouped)

# =========================================================
# 4. MERGE SURVEY + SPOTIFY
# =========================================================

survey_music = survey[survey["music"] == "Evet"].copy()
survey_music = survey_music.merge(spotify_grouped, on="genre", how="left")

print("\nMerged survey + spotify preview:")
print(survey_music.head())

# Check unmatched genres after merge
unmatched = survey_music[survey_music["energy"].isna()]["genre"].dropna().unique()
print("\nUnmatched genres after merge:")
print(unmatched)

# =========================================================
# 5. EDA
# =========================================================

print("\n================ EDA ================\n")

print("Music listening distribution:")
print(survey["music"].value_counts())

print("\nStudy hours distribution:")
print(survey["study_hours"].value_counts().sort_index())

print("\nFocus distribution:")
print(survey["focus"].value_counts().sort_index())

print("\nAverage focus overall:")
print(round(survey["focus"].mean(), 3))

print("\nAverage focus by music listening:")
print(survey.groupby("music")["focus"].mean())

print("\nAverage focus by study hours:")
print(survey.groupby("study_hours", observed=False)["focus"].mean())

print("\nAverage focus by genre (music listeners only):")
print(survey_music.groupby("genre")["focus"].mean().sort_values(ascending=False))

# -------------------------
# EDA PLOTS
# -------------------------

# 1. Music listening distribution
survey["music"].value_counts().plot(kind="bar", figsize=(6, 4))
plt.title("Music Listening Distribution")
plt.xlabel("Listening to Music While Studying")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 2. Study hours distribution
survey["study_hours"].value_counts().sort_index().plot(kind="bar", figsize=(6, 4))
plt.title("Study Hours Distribution")
plt.xlabel("Study Hours")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. Average focus by music
survey.groupby("music")["focus"].mean().plot(kind="bar", figsize=(6, 4))
plt.title("Average Focus: Music vs No Music")
plt.xlabel("Listening to Music While Studying")
plt.ylabel("Average Focus")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 4. Average focus by study hours
survey.groupby("study_hours", observed=False)["focus"].mean().plot(kind="bar", figsize=(6, 4))
plt.title("Average Focus by Study Hours")
plt.xlabel("Study Hours")
plt.ylabel("Average Focus")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 5. Average focus by genre
survey_music.groupby("genre")["focus"].mean().sort_values(ascending=False).plot(kind="bar", figsize=(8, 4))
plt.title("Average Focus by Music Genre")
plt.xlabel("Genre")
plt.ylabel("Average Focus")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Spotify audio features by genre
spotify_grouped.set_index("genre")[["energy", "valence", "acousticness", "instrumentalness"]].plot(
    kind="bar", figsize=(10, 5)
)
plt.title("Spotify Audio Features by Genre")
plt.ylabel("Average Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Tempo separately because scale is much larger
spotify_grouped.set_index("genre")["tempo"].plot(kind="bar", figsize=(8, 4))
plt.title("Average Tempo by Genre")
plt.ylabel("Tempo")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =========================================================
# 6. HYPOTHESIS TESTS
# =========================================================

print("\n================ HYPOTHESIS TESTS ================\n")

# -------------------------------------------------
# H1: Music vs No Music on Focus
# -------------------------------------------------
focus_music = survey.loc[survey["music"] == "Evet", "focus"]
focus_no_music = survey.loc[survey["music"] == "Hayır", "focus"]

t_stat_1, p_value_1 = ttest_ind(focus_music, focus_no_music, nan_policy="omit")

print("H1: Music vs No Music on Focus")
print("t-statistic:", round(t_stat_1, 4))
print("p-value:", round(p_value_1, 6))
if p_value_1 < 0.05:
    print("Result: Reject H0 -> Significant difference in focus.\n")
else:
    print("Result: Fail to reject H0 -> No significant difference in focus.\n")

# -------------------------------------------------
# H2: Genre vs Focus
# -------------------------------------------------
genre_groups = []
for g in survey_music["genre"].dropna().unique():
    vals = survey_music.loc[survey_music["genre"] == g, "focus"].dropna()
    if len(vals) >= 2:
        genre_groups.append(vals)

f_stat_2, p_value_2 = f_oneway(*genre_groups)

print("H2: Genre vs Focus")
print("F-statistic:", round(f_stat_2, 4))
print("p-value:", round(p_value_2, 6))
if p_value_2 < 0.05:
    print("Result: Reject H0 -> Focus differs significantly by genre.\n")
else:
    print("Result: Fail to reject H0 -> No significant genre effect on focus.\n")

# -------------------------------------------------
# H3: Study Duration vs Focus
# -------------------------------------------------
hours_groups = []
for h in study_order:
    vals = survey.loc[survey["study_hours"] == h, "focus"].dropna()
    if len(vals) >= 2:
        hours_groups.append(vals)

f_stat_3, p_value_3 = f_oneway(*hours_groups)

print("H3: Study Duration vs Focus")
print("F-statistic:", round(f_stat_3, 4))
print("p-value:", round(p_value_3, 6))
if p_value_3 < 0.05:
    print("Result: Reject H0 -> Focus differs significantly across study-hour groups.\n")
else:
    print("Result: Fail to reject H0 -> No significant effect of study duration on focus.\n")

# -------------------------------------------------
# H4: Spotify Audio Features vs Focus
# -------------------------------------------------
print("H4: Spotify Audio Features vs Focus")
for feature in ["energy", "danceability", "tempo", "valence", "acousticness", "instrumentalness"]:
    temp_df = survey_music.dropna(subset=[feature, "focus"])
    r, p = pearsonr(temp_df[feature], temp_df["focus"])
    print(f"{feature}: r = {round(r,4)}, p-value = {round(p,6)}")
print()

# -------------------------------------------------
# H5: Music vs Study Duration
# -------------------------------------------------
contingency = pd.crosstab(survey["music"], survey["study_hours"])
chi2_stat_5, p_value_5, dof_5, expected_5 = chi2_contingency(contingency)

print("H5: Music vs Study Duration")
print("Chi-square statistic:", round(chi2_stat_5, 4))
print("p-value:", round(p_value_5, 6))
if p_value_5 < 0.05:
    print("Result: Reject H0 -> Music listening and study duration are significantly related.\n")
else:
    print("Result: Fail to reject H0 -> No significant relationship between music listening and study duration.\n")

print("Contingency Table:")
print(contingency)
