#2026
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import StringIO

#Data
raw = """player,rounds,rounds_counted,sg_putt,sg_arg,sg_app,sg_ott,sg_t2g,sg_ball_striking,sg_total
Scottie Scheffler,32,30,0.68,0.74,0.61,0.87,1.48,2.23,2.99
Rory McIlroy,18,18,0.17,0.71,1.03,1.05,2.08,2.79,2.96
Collin Morikawa,26,26,0.29,0.23,1.37,0.56,1.93,2.16,2.45
Cameron Young,32,31,0.30,0.28,0.82,0.91,1.73,2.00,2.29
Matt Fitzpatrick,36,34,0.12,0.45,1.04,0.77,1.81,2.26,2.28
Xander Schauffele,34,33,0.36,0.03,1.02,0.76,1.78,1.80,2.12
Jake Knapp,34,33,0.82,0.09,0.61,0.56,1.17,1.25,2.10
Jacob Bridgeman,40,38,1.12,-0.05,0.69,0.38,1.07,1.02,2.09
Russell Henley,32,30,0.59,0.34,0.67,0.35,1.03,1.36,1.96
Ludvig Aberg,32,29,0.37,0.37,0.85,0.61,1.47,1.84,1.93
Hideki Matsuyama,36,35,0.40,0.45,0.91,0.04,0.95,1.40,1.90
Si Woo Kim,44,41,-0.11,0.23,0.91,0.71,1.62,1.85,1.85
Justin Rose,23,20,0.27,0.10,1.17,0.16,1.32,1.43,1.80
Akshay Bhatia,31,28,0.96,0.02,0.96,0.03,1.00,1.02,1.75
Chris Gotterup,38,37,0.14,0.25,0.60,0.76,1.37,1.62,1.74
Tommy Fleetwood,28,28,0.11,0.46,0.54,0.58,1.12,1.58,1.69
Min Woo Lee,34,32,0.28,0.34,0.29,0.74,1.03,1.37,1.68
Robert MacIntyre,34,32,0.93,0.22,-0.39,0.85,0.46,0.68,1.67
Maverick McNealy,38,37,0.57,0.37,0.13,0.50,0.63,1.00,1.64
Patrick Cantlay,32,29,-0.12,0.59,0.60,0.53,1.13,1.72,1.58"""

df = pd.read_csv(StringIO(raw))

# Augusta - OTT + approach dominant
# Aronimink - approach + accuracy dominant
# Shinnecock - approach + patience, penalises OTT aggression
# Royal Birkdale - putting + OTT wind game

course_weights = {
    "Masters": {
        "sg_putt": 0.20,
        "sg_arg":  0.15,
        "sg_app":  0.30,
        "sg_ott":  0.35,
    },
    "PGA Championship": {
        "sg_putt": 0.20,
        "sg_arg":  0.15,
        "sg_app":  0.40,
        "sg_ott":  0.25,
    },
    "US Open": {
        "sg_putt": 0.25,
        "sg_arg":  0.20,
        "sg_app":  0.35,
        "sg_ott":  0.20,
    },
    "The Open": {
        "sg_putt": 0.30,
        "sg_arg":  0.20,
        "sg_app":  0.20,
        "sg_ott":  0.30,
    },
}
#Probabilities
def compute_win_probs(df, weights):
    score = (
        weights["sg_putt"] * df["sg_putt"] +
        weights["sg_arg"]  * df["sg_arg"]  +
        weights["sg_app"]  * df["sg_app"]  +
        weights["sg_ott"]  * df["sg_ott"]
    )
    exp_score = np.exp(score * 3)
    prob = exp_score / exp_score.sum()
    return score, prob

all_results = {}
for major, weights in course_weights.items():
    score, prob = compute_win_probs(df, weights)
    result_df = df[["player"]].copy()
    result_df["weighted_sg"] = score.values
    result_df["win_prob"]    = prob.values
    result_df = result_df.sort_values("win_prob", ascending=False).reset_index(drop=True)
    result_df["rank"]        = result_df.index + 1
    all_results[major]       = result_df

#Show predicitons
print("=" * 65)
print("  2026 GOLF MAJOR PREDICTIONS — BASED ON 2026 DATAGOLF SG DATA")
print("=" * 65)

for major, res in all_results.items():
    print(f"\n{'─' * 65}")
    print(f"  {major}")
    print(f"{'─' * 65}")
    print(f"  {'Rank':<6} {'Player':<22} {'Weighted SG':>12} {'Win Prob':>10}")
    print(f"  {'----':<6} {'------':<22} {'-----------':>12} {'--------':>10}")
    for _, row in res.head(10).iterrows():
        marker = ""
        print(f"  {int(row['rank']):<6} {row['player']:<22} {row['weighted_sg']:>12.3f} {row['win_prob']:>9.1%}{marker}")

#Graph
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("2026 Golf Major Predictions\nBased on 2026 DataGolf Strokes Gained Data",
             fontsize=15, fontweight="bold", y=0.98)

subtitles = [
    "Augusta National",
    "Aronimink GC",
    "Shinnecock Hills",
    "Royal Birkdale",
]

colors_map = {
    "Scottie Scheffler": "#185FA5",
    "Rory McIlroy":      "#1D9E75",
    "Collin Morikawa":   "#534AB7",
    "Cameron Young":     "#D85A30",
    "Matt Fitzpatrick":  "#BA7517",
}
default_color = "#888780"

for ax, major, subtitle in zip(axes.flat, all_results.keys(), subtitles):
    res        = all_results[major].head(8)
    labels     = res["player"].str.split().str[-1].tolist()
    probs      = (res["win_prob"] * 100).tolist()
    bar_colors = [colors_map.get(p, default_color) for p in res["player"]]

    bars = ax.barh(labels[::-1], probs[::-1], color=bar_colors[::-1],
                   height=0.6, edgecolor="white", linewidth=0.5)

    for bar, p in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", ha="left", fontsize=9)

    ax.set_title(f"{major}\n{subtitle}", fontsize=10, pad=8)
    ax.set_xlabel("Win probability (%)", fontsize=9)
    ax.set_xlim(0, max(probs) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

legend_patches = [mpatches.Patch(color=c, label=p.split()[-1])
                  for p, c in colors_map.items()]
fig.legend(handles=legend_patches, loc="lower center", ncol=5,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig("major_predictions.png", dpi=150, bbox_inches="tight")
plt.show(block=True)
print("\nChart saved as major_predictions.png")

#Correlation
print(f"\n{'=' * 65}")
print("  CORRELATION: SG CATEGORIES vs SG TOTAL")
print(f"{'=' * 65}")

sg_cols = ["sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_ball_striking", "sg_t2g"]
for col in sg_cols:
    corr = df[col].corr(df["sg_total"])
    bar  = "█" * int(abs(corr) * 20)
    print(f"  {col:<20} r = {corr:+.3f}  {bar}")









#import pandas as pd
#from io import StringIO
#from sklearn.linear_model import LinearRegression

#X = df[["sg_putt", "sg_ball_striking"]]
#y = df["sg_total"]

#model = LinearRegression().fit(X, y)
#print("Coefficients:")
#for name, coef in zip(X.columns, model.coef_):
    #print(f"  {name}: {coef:.3f}")
#print(f"R²: {model.score(X, y):.3f}")









#2025
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import StringIO

#Data
raw = """player,rounds,rounds_counted,sg_putt,sg_arg,sg_app,sg_ott,sg_t2g,sg_ball_striking,sg_total
Scottie Scheffler,80,80,0.62,0.40,1.49,0.79,2.29,2.69,3.30
Rory McIlroy,58,58,0.70,0.24,0.53,0.81,1.34,1.59,2.29
Tommy Fleetwood,74,74,0.51,0.40,0.84,0.41,1.26,1.66,2.17
Russell Henley,72,72,0.36,0.49,0.80,0.22,1.02,1.52,1.88
J.J. Spaun,91,84,0.25,0.02,0.84,0.42,1.26,1.28,1.60
Ben Griffin,104,97,0.47,0.23,0.59,0.22,0.81,1.04,1.57
Justin Thomas,80,78,0.47,0.41,0.52,0.08,0.60,1.01,1.51
Patrick Cantlay,74,72,0.27,0.07,0.76,0.41,1.17,1.23,1.51
Sepp Straka,76,74,0.25,-0.03,0.88,0.33,1.21,1.18,1.49
Keegan Bradley,80,79,-0.01,0.55,0.46,0.45,0.92,1.47,1.45
Collin Morikawa,76,72,-0.14,0.02,0.97,0.54,1.52,1.54,1.41
Xander Schauffele,64,60,0.01,0.10,0.83,0.31,1.14,1.23,1.41
Robert MacIntyre,84,84,0.32,0.24,0.40,0.43,0.83,1.07,1.39
Viktor Hovland,65,65,0.14,-0.06,1.12,0.15,1.26,1.20,1.35
Sam Burns,94,92,0.99,0.04,0.03,0.27,0.30,0.34,1.32
Corey Conners,81,81,0.23,-0.03,0.59,0.51,1.11,1.08,1.31
Ludvig Aberg,69,68,0.23,0.02,0.29,0.66,0.95,0.97,1.29
Harry Hall,94,88,0.96,0.40,0.04,-0.01,0.03,0.43,1.28
Hideki Matsuyama,86,81,0.10,0.52,0.72,-0.03,0.69,1.20,1.27
Jordan Spieth,68,68,0.30,0.24,0.26,0.45,0.71,0.95,1.25"""

df = pd.read_csv(StringIO(raw))

# Augusta - OTT + approach dominant
# Quail Hollow - approach + ball striking dominant
# Oakmont - approach + patience, penalises OTT aggression
# Royal Portrush - putting + OTT wind game

course_weights = {
    "Masters": {
        "sg_putt": 0.20,
        "sg_arg":  0.15,
        "sg_app":  0.30,
        "sg_ott":  0.35,
    },
    "PGA Championship": {
        "sg_putt": 0.20,
        "sg_arg":  0.15,
        "sg_app":  0.35,
        "sg_ott":  0.30,
    },
    "US Open": {
        "sg_putt": 0.25,
        "sg_arg":  0.20,
        "sg_app":  0.35,
        "sg_ott":  0.20,
    },
    "The Open": {
        "sg_putt": 0.30,
        "sg_arg":  0.20,
        "sg_app":  0.20,
        "sg_ott":  0.30,
    },
}

#Probabilities
def compute_win_probs(df, weights):
    score = (
        weights["sg_putt"] * df["sg_putt"] +
        weights["sg_arg"]  * df["sg_arg"]  +
        weights["sg_app"]  * df["sg_app"]  +
        weights["sg_ott"]  * df["sg_ott"]
    )
    exp_score = np.exp(score * 3)
    prob = exp_score / exp_score.sum()
    return score, prob

all_results = {}
for major, weights in course_weights.items():
    score, prob = compute_win_probs(df, weights)
    result_df = df[["player"]].copy()
    result_df["weighted_sg"] = score.values
    result_df["win_prob"]    = prob.values
    result_df = result_df.sort_values("win_prob", ascending=False).reset_index(drop=True)
    result_df["rank"]        = result_df.index + 1
    all_results[major]       = result_df

#Show predictions
print("=" * 65)
print("  2025 GOLF MAJOR PREDICTIONS — BASED ON 2025 DATAGOLF SG DATA")
print("=" * 65)

for major, res in all_results.items():
    print(f"\n{'─' * 65}")
    print(f"  {major}")
    print(f"{'─' * 65}")
    print(f"  {'Rank':<6} {'Player':<22} {'Weighted SG':>12} {'Win Prob':>10}")
    print(f"  {'----':<6} {'------':<22} {'-----------':>12} {'--------':>10}")
    for _, row in res.head(10).iterrows():
        marker = ""
        print(f"  {int(row['rank']):<6} {row['player']:<22} {row['weighted_sg']:>12.3f} {row['win_prob']:>9.1%}{marker}")

#Graph
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("2025 Golf Major Predictions\nBased on 2025 DataGolf Strokes Gained Data",
             fontsize=15, fontweight="bold", y=0.98)

subtitles = [
    "Augusta National",
    "Quail Hollow",
    "Oakmont",
    "Royal Portrush",
]

colors_map = {
    "Scottie Scheffler": "#185FA5",
    "Rory McIlroy":      "#1D9E75",
    "Tommy Fleetwood":   "#534AB7",
    "Russell Henley":    "#D85A30",
    "J.J. Spaun":        "#BA7517",
}
default_color = "#888780"

for ax, major, subtitle in zip(axes.flat, all_results.keys(), subtitles):
    res        = all_results[major].head(8)
    labels     = res["player"].str.split().str[-1].tolist()
    probs      = (res["win_prob"] * 100).tolist()
    bar_colors = [colors_map.get(p, default_color) for p in res["player"]]

    bars = ax.barh(labels[::-1], probs[::-1], color=bar_colors[::-1],
                   height=0.6, edgecolor="white", linewidth=0.5)

    for bar, p in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", ha="left", fontsize=9)

    ax.set_title(f"{major}\n{subtitle}", fontsize=10, pad=8)
    ax.set_xlabel("Win probability (%)", fontsize=9)
    ax.set_xlim(0, max(probs) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

legend_patches = [mpatches.Patch(color=c, label=p.split()[-1])
                  for p, c in colors_map.items()]
fig.legend(handles=legend_patches, loc="lower center", ncol=5,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig("major_predictions_2025.png", dpi=150, bbox_inches="tight")
plt.show(block=True)
print("\nChart saved as major_predictions_2025.png")

#Correlation
print(f"\n{'=' * 65}")
print("  CORRELATION: SG CATEGORIES vs SG TOTAL")
print(f"{'=' * 65}")

sg_cols = ["sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_ball_striking", "sg_t2g"]
for col in sg_cols:
    corr = df[col].corr(df["sg_total"])
    bar  = "█" * int(abs(corr) * 20)
    print(f"  {col:<20} r = {corr:+.3f}  {bar}")









import pandas as pd
from io import StringIO
from sklearn.linear_model import LinearRegression

X = df[["sg_putt", "sg_ball_striking"]]
y = df["sg_total"]

model = LinearRegression().fit(X, y)
print("Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.3f}")
print(f"R²: {model.score(X, y):.3f}")