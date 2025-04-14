import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Paths
results_dir = Path("explainations")
plots_dir = Path("explainations\plots_alt")
plots_dir.mkdir(exist_ok=True, parents=True)

# Load explanations with error handling
try:
    with open(results_dir / "lime_explanations.pkl", "rb") as f:
        lime_explanations = pickle.load(f)
    print(f"Loaded {len(lime_explanations)} LIME explanations")
except FileNotFoundError:
    print(f"Error: lime_explanations.pkl not found in {results_dir}")
    lime_explanations = []
except Exception as e:
    print(f"Error loading LIME explanations: {e}")
    lime_explanations = []

try:
    with open(results_dir / "ig_explanations.pkl", "rb") as f:
        ig_explanations = pickle.load(f)
    print(f"Loaded {len(ig_explanations)} IG explanations")
except FileNotFoundError:
    print(f"Error: ig_explanations.pkl not found in {results_dir}")
    ig_explanations = []
except Exception as e:
    print(f"Error loading IG explanations: {e}")
    ig_explanations = []

# Compute average scores per instance
instance_scores = defaultdict(list)  # {idx: [lime_score, ig_score]}

for exp in lime_explanations:
    try:
        idx = exp["idx"]
        if isinstance(exp.get("explanation"), list) and exp["explanation"]:
            score = sum(abs(score) for _, score in exp["explanation"])
            instance_scores[idx].append(("LIME", score))
    except (KeyError, TypeError) as e:
        print(f"Skipping LIME instance {exp.get('idx', 'unknown')}: {e}")

for exp in ig_explanations:
    try:
        idx = exp["idx"]
        if isinstance(exp.get("explanation"), list) and exp["explanation"]:
            score = sum(abs(score) for _, score in exp["explanation"])
            instance_scores[idx].append(("IG", score))
    except (KeyError, TypeError) as e:
        print(f"Skipping IG instance {exp.get('idx', 'unknown')}: {e}")

# Print average scores per instance
print("\nAverage Scores per Instance:")
for idx, scores in sorted(instance_scores.items()):
    lime_score = next((s for m, s in scores if m == "LIME"), None)
    ig_score = next((s for m, s in scores if m == "IG"), None)
    if lime_score is not None and ig_score is not None:
        avg_score = (lime_score + ig_score) / 2
        print(f"Instance {idx}: LIME={lime_score:.4f}, IG={ig_score:.4f}, Average={avg_score:.4f}")
    elif lime_score is not None:
        print(f"Instance {idx}: LIME={lime_score:.4f}, IG=None")
    elif ig_score is not None:
        print(f"Instance {idx}: LIME=None, IG={ig_score:.4f}")

# Compute average feature weights
lime_feature_scores = defaultdict(list)  # {feature: [scores]}
ig_feature_scores = defaultdict(list)

for exp in lime_explanations:
    try:
        if isinstance(exp.get("explanation"), list):
            for feature, score in exp["explanation"]:
                lime_feature_scores[feature].append(score)
    except (KeyError, TypeError) as e:
        print(f"Skipping LIME feature processing for instance {exp.get('idx', 'unknown')}: {e}")

for exp in ig_explanations:
    try:
        if isinstance(exp.get("explanation"), list):
            for feature, score in exp["explanation"]:
                ig_feature_scores[feature].append(score)
    except (KeyError, TypeError) as e:
        print(f"Skipping IG feature processing for instance {exp.get('idx', 'unknown')}: {e}")

# Calculate mean feature weights
features = sorted(set(lime_feature_scores.keys()) | set(ig_feature_scores.keys()))
lime_means = [np.mean(lime_feature_scores[f]) if f in lime_feature_scores else 0 for f in features]
ig_means = [np.mean(ig_feature_scores[f]) if f in ig_feature_scores else 0 for f in features]

# Print average feature weights
print("\nAverage Feature Weights:")
for f, lime_w, ig_w in zip(features, lime_means, ig_means):
    print(f"Feature {f}: LIME={lime_w:.4f}, IG={ig_w:.4f}")

# Plot average feature weights for LIME (horizontal)
plt.figure(figsize=(10, 8))
plt.barh(range(len(features)), lime_means, color=plt.cm.viridis(0.3), edgecolor='black', height=0.4, label='LIME')
plt.yticks(range(len(features)), features, fontsize=10)
plt.xlabel("Average Importance", fontsize=12)
plt.title("Average LIME Feature Importance", fontsize=14)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
save_path = plots_dir / "lime_average_feature_importance.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved LIME average plot to {save_path}")
plt.close()

# Plot average feature weights for IG (horizontal)
plt.figure(figsize=(10, 8))
plt.barh(range(len(features)), ig_means, color=plt.cm.plasma(0.7), edgecolor='black', height=0.4, label='IG')
plt.yticks(range(len(features)), features, fontsize=10)
plt.xlabel("Average Importance", fontsize=12)
plt.title("Average Integrated Gradients Feature Importance", fontsize=14)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
save_path = plots_dir / "ig_average_feature_importance.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved IG average plot to {save_path}")
plt.close()

# Original plotting function (unchanged)
def plot_explanation(explanation, title, filename):
    try:
        if not isinstance(explanation.get("explanation"), list) or not explanation["explanation"]:
            print(f"Skipping plot for {filename}: Invalid or empty explanation data")
            return
        features, scores = zip(*explanation["explanation"])
        if not features or not scores:
            print(f"Skipping plot for {filename}: No features or scores")
            return
        plt.figure(figsize=(8, 10))
        plt.bar(range(len(features)), scores, color=plt.cm.magma(np.linspace(0, 1, len(features))),
                edgecolor='black', linewidth=1)
        plt.title(title, fontsize=14)
        plt.ylabel("Importance", fontsize=12)
        plt.xlabel("Features", fontsize=12)
        plt.xticks(range(len(features)), features, rotation=45, ha='right', fontsize=10)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_path = plots_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Error generating plot for {filename}: {e}")

# Generate individual plots
for exp in lime_explanations:
    try:
        idx = exp["idx"]
        label = exp["label"]
        title = f"LIME Explanation for Instance {idx}, Label {label}"
        filename = f"lime_instance_{idx}_label_{label}.png"
        plot_explanation(exp, title, filename)
    except KeyError as e:
        print(f"Skipping LIME explanation: Missing key {e}")
    except Exception as e:
        print(f"Error processing LIME explanation: {e}")

for exp in ig_explanations:
    try:
        idx = exp["idx"]
        label = exp["label"]
        title = f"Integrated Gradients for Instance {idx}, Label {label}"
        filename = f"ig_instance_{idx}_label_{label}.png"
        plot_explanation(exp, title, filename)
    except KeyError as e:
        print(f"Skipping IG explanation: Missing key {e}")
    except Exception as e:
        print(f"Error processing IG explanation: {e}")

print(f"Completed. Plots should be saved to {plots_dir}")