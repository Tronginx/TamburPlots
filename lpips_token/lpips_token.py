import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("/Users/tron/RealTron/UIUC/25Spring/CS538/Project/pretty-plots/lpips_token/GenStream.csv")

# Step 1: Classify loss rate (0% and 20% ± 5%)
def classify_loss_rate(x):
    if abs(x) < 0.01:
        return "0%"
    elif abs(x - 0.20) <= 0.05:
        return "20%"
    return None

df["loss_rate"] = df["avg_packet_loss"].apply(classify_loss_rate)
df = df.dropna(subset=["loss_rate"])

# Step 2: Find token numbers available in both 0% and 20% loss
token_20_set = set(df[df["loss_rate"] == "20%"]["n_codes"])
token_0_set = set(df[df["loss_rate"] == "0%"]["n_codes"])
token_common = sorted(token_0_set.union(token_20_set))  # include both

# Step 3: Pick 8 from original selection excluding 384
token_counts = df["n_codes"].value_counts().sort_index()
token_values_sorted = sorted(token_counts.index.unique())

# Pick 9 representative tokens evenly spaced
all_tokens = [token_values_sorted[i] for i in 
              np.linspace(0, len(token_values_sorted)-1, 9, dtype=int)]

# Remove 384 if it's in the list and doesn't have 20% loss
if 384 in all_tokens and 384 not in token_20_set:
    all_tokens.remove(384)

# Keep 8 from the original selection
selected_tokens = all_tokens[:8]

# Add the lowest available token with 20% loss data
lowest_with_20 = min(token_20_set)
if lowest_with_20 not in selected_tokens:
    selected_tokens.insert(0, lowest_with_20)

print("Final selected token numbers:", selected_tokens)

# Step 4: Filter dataset to only include selected token numbers
df_filtered = df[df["n_codes"].isin(selected_tokens)]

# Step 5: Group by token and loss rate
grouped = df_filtered.groupby(["n_codes", "loss_rate"])["lpips"].mean().reset_index()

# Step 6: Prepare plotting data
selected_tokens_sorted = sorted(set(grouped["n_codes"]))
bar_width = 0.35
x = np.arange(len(selected_tokens_sorted))

lpips_0 = []
lpips_20 = []

for token_num in selected_tokens_sorted:
    group = grouped[grouped["n_codes"] == token_num]
    val_0 = group[group["loss_rate"] == "0%"]["lpips"].mean()
    val_20 = group[group["loss_rate"] == "20%"]["lpips"].mean()
    lpips_0.append(val_0)
    lpips_20.append(val_20)

# Step 7: Plot
fig, ax = plt.subplots(figsize=(15, 8))

rects1 = ax.bar(x - bar_width/2, lpips_0, bar_width, label='0% Packet Loss', color='royalblue')
rects2 = ax.bar(x + bar_width/2, lpips_20, bar_width, label='20% Packet Loss', color='sandybrown')

ax.set_xlabel('Token Number', fontsize=20, labelpad=10)
ax.set_ylabel('LPIPS', fontsize=20, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels([int(t) for t in selected_tokens_sorted], rotation=45, ha="right", fontsize=18)
ax.tick_params(axis='y', labelsize=16)
ax.legend(fontsize=20)

# Add value labels on top of each bar
ax.bar_label(rects1, padding=3, fmt='%.4f', fontsize=11)
ax.bar_label(rects2, padding=3, fmt='%.4f', fontsize=11)

# Aesthetic improvements
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%.4f'))

fig.tight_layout()
plt.savefig("lpips_vs_token_number_filtered.png", dpi=300, bbox_inches='tight')
plt.savefig("lpips_vs_token_number_filtered.pdf", dpi=300, bbox_inches='tight')
plt.show()
