import matplotlib.pyplot as plt
import pandas as pd

# Creating the dataset
data = {
    "Methodology": [
        "Hybrid ML Ensemble", 
        "R-CSBD Algorithm", 
        "Pairwise Fixed Effects", 
        "Real-time SNA approach"
    ],
    "Dataset": [
        "Synthetic Auction", 
        "TBAuctions (small sets)", 
        "Ukraine Procurement", 
        "Synthetic & eBay data"
    ],
    "Accuracy": [97.72, 92.5, 88.0, 94.0],
    "Precision": [0.98, 0.90, 0.85, 0.92],
    "Recall": [0.97, 0.88, 0.82, 0.91],
    "ROCAUC": [0.999, 0.91, 0.74, 0.93]
}

df = pd.DataFrame(data)

# Plot grouped bar chart
metrics = ["Accuracy", "Precision", "Recall", "ROCAUC"]
x = range(len(df))

fig, ax = plt.subplots(figsize=(12, 7))

bar_width = 0.2

for i, metric in enumerate(metrics):
    ax.bar([p + bar_width*i for p in x], df[metric], width=bar_width, label=metric)

# Labels and styling
ax.set_xticks([p + bar_width*1.5 for p in x])
ax.set_xticklabels(df["Methodology"], rotation=20, ha="right")
ax.set_ylabel("Performance Score")
ax.set_title("Comparison of ML Approaches for Auction Fraud Detection")
ax.legend()

plt.tight_layout()
plt.show()
