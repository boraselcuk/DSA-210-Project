import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 1. LOAD DATA
# =========================================================

survey = pd.read_csv("survey.csv")
spotify = pd.read_csv("dataset.csv")

# =========================================================
# 2. CLEAN SURVEY DATA
# =========================================================

survey.columns = ["timestamp", "study_hours", "focus", "music", "genre_tr"]
survey = survey.drop(columns=["timestamp"])

for col in survey.columns:
    survey[col] = survey[col].astype(str).str.strip()

survey["focus"] = pd.to_numeric(survey["focus"], errors="coerce")
survey = survey[survey["music"].isin(["Evet", "Hayır"])]
survey["genre_tr"] = survey["genre_tr"].replace(["nan", "NaN", "", "None"], np.nan)

def clean_survey_genre(g):
    if pd.isna(g):
        return np.nan
    g = str(g).strip().lower()
    if g in ["klasik"]:
        return "Classical"
    elif g in ["lo-fi", "lofi", "study"]:
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

study_hours_map = {
    "0-2 saat": 1,
    "2-4 saat": 3,
    "4-6 saat": 5,
    "6+ saat": 6
}
survey["study_hours_num"] = survey["study_hours"].map(study_hours_map)
survey["music_bin"] = (survey["music"] == "Evet").astype(int)
survey = survey.dropna(subset=["focus"])

# =========================================================
# 3. CLEAN SPOTIFY DATA
# =========================================================

spotify = spotify[[
    "track_genre", "energy", "danceability", "tempo",
    "valence", "acousticness", "instrumentalness"
]].copy()

def map_spotify_genre(g):
    g = str(g).lower().strip()
    if any(x in g for x in ["classical", "piano", "opera", "baroque", "orchestra", "romantic"]):
        return "Classical"
    elif "lofi" in g or "lo-fi" in g or "study" in g:
        return "Lo-fi"
    elif "jazz" in g or "blues" in g or "swing" in g:
        return "Jazz"
    elif "hip hop" in g or "hip-hop" in g or "rap" in g or "trap" in g:
        return "Rap / Hip-hop"
    elif "rock" in g or "metal" in g or "punk" in g or "grunge" in g or "alternative" in g:
        return "Rock"
    elif "edm" in g or "electronic" in g or "techno" in g or "house" in g or "dance" in g or "trance" in g or "dubstep" in g:
        return "Electronic"
    elif "pop" in g or "indie" in g or "r&b" in g or "soul" in g:
        return "Pop"
    else:
        return "Other"

spotify["genre"] = spotify["track_genre"].apply(map_spotify_genre)
spotify_grouped = spotify.groupby("genre")[[
    "energy", "danceability", "tempo", "valence", "acousticness", "instrumentalness"
]].mean().reset_index()

# =========================================================
# 4. MERGE
# =========================================================

survey_full = survey.merge(spotify_grouped, on="genre", how="left")

# =========================================================
# 5. CLASSIFICATION
# =========================================================

features = [
    "study_hours_num", "music_bin",
    "energy", "danceability", "tempo",
    "valence", "acousticness", "instrumentalness"
]

df_clf = survey_full.dropna(subset=features + ["focus"]).copy()
X = df_clf[features]
y = df_clf["focus"].astype(int)

print("Classification dataset size:", len(df_clf))
print("Class distribution:")
print(y.value_counts().sort_index())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
cv_rf = cross_val_score(rf, X, y, cv=5, scoring="accuracy").mean()

print(f"\nRandom Forest")
print(f"Test Accuracy:  {acc_rf:.4f}")
print(f"CV Accuracy:    {cv_rf:.4f}")
print(classification_report(y_test, y_pred_rf))

# --- Logistic Regression ---
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
acc_lr = accuracy_score(y_test, y_pred_lr)
cv_lr = cross_val_score(lr, scaler.fit_transform(X), y, cv=5, scoring="accuracy").mean()

print(f"\nLogistic Regression")
print(f"Test Accuracy:  {acc_lr:.4f}")
print(f"CV Accuracy:    {cv_lr:.4f}")
print(classification_report(y_test, y_pred_lr))

# --- Feature Importances ---
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importances (Random Forest):")
print(importances.round(4))

# =========================================================
# 6. CLUSTERING (K-MEANS)
# =========================================================

cluster_features = ["study_hours_num", "music_bin", "focus"]
df_clust = survey_full[cluster_features].copy()
df_clust["study_hours_num"] = pd.to_numeric(df_clust["study_hours_num"], errors="coerce")
df_clust = df_clust.dropna()

scaler2 = StandardScaler()
X_scaled = scaler2.fit_transform(df_clust)

# Elbow method
inertias = []
for k in range(1, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

print("\nElbow Method Inertias:")
for k, inertia in enumerate(inertias, 1):
    print(f"  k={k}: {inertia:.2f}")

# K=3
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clust["cluster"] = km3.fit_predict(X_scaled)

print("\nCluster Distribution:")
print(df_clust["cluster"].value_counts())

print("\nCluster Means:")
print(df_clust.groupby("cluster")[["study_hours_num", "music_bin", "focus"]].mean().round(3))

# =========================================================
# 7. PLOTS
# =========================================================

# 08: Feature importances
importances.sort_values().plot(kind="barh", figsize=(8, 4), color="#4C72B0")
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 09: Model comparison
models = ["Random Forest", "Logistic Regression"]
cv_scores = [cv_rf, cv_lr]
test_scores = [acc_rf, acc_lr]
x = np.arange(len(models))
fig, ax = plt.subplots(figsize=(6, 4))
bars1 = ax.bar(x - 0.2, cv_scores, 0.35, label="CV Accuracy", color="#4C72B0")
bars2 = ax.bar(x + 0.2, test_scores, 0.35, label="Test Accuracy", color="#DD8452")
ax.set_title("Model Comparison: CV vs Test Accuracy")
ax.set_ylabel("Accuracy")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1)
ax.legend()
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=10)
plt.tight_layout()
plt.show()

# 10: Confusion matrix RF
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_rf)
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix (Random Forest)")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
labels = sorted(y.unique())
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=11)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

# 11: Elbow method
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, 8), inertias, "bo-", linewidth=2, markersize=8)
ax.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="k=3 (chosen)")
ax.set_title("Elbow Method for Optimal k")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia")
ax.legend()
plt.tight_layout()
plt.show()

# 12: Cluster scatter
fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#4C72B0", "#DD8452", "#55A868"]
for c in [0, 1, 2]:
    mask = df_clust["cluster"] == c
    ax.scatter(df_clust.loc[mask, "study_hours_num"], df_clust.loc[mask, "focus"],
               c=colors[c], label=f"Cluster {c}", alpha=0.7, s=60)
ax.set_title("K-Means Clustering (k=3)")
ax.set_xlabel("Study Hours (numeric)")
ax.set_ylabel("Focus Level")
ax.legend()
plt.tight_layout()
plt.show()
