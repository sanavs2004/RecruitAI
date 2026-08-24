import json

with open("training_report.json", "r") as f:
    report = json.load(f)

precision = report["precision"]
recall = report["recall"]
f1 = report["f1_score"]
support = report["support"]

total_samples = report["test_samples"]

print("\n              precision    recall  f1-score   support\n")

# Print each class
for i, cls in enumerate(precision.keys()):
    p = precision[cls] / 100
    r = recall[cls] / 100
    f = f1[cls] / 100
    s = support[cls]

    print(f"{i:^12}{p:>10.2f}{r:>10.2f}{f:>10.2f}{s:>10}")

# Accuracy
accuracy = report["ensemble_accuracy"] / 100
print(f"\n    accuracy{'':>27}{accuracy:.2f}{total_samples:>10}")

# Macro average
macro_p = sum(precision.values()) / (len(precision) * 100)
macro_r = sum(recall.values()) / (len(recall) * 100)
macro_f = sum(f1.values()) / (len(f1) * 100)

print(f"   macro avg{macro_p:>10.2f}{macro_r:>10.2f}{macro_f:>10.2f}{total_samples:>10}")

# Weighted average
total = sum(support.values())
weighted_p = sum((precision[k]/100)*support[k] for k in support) / total
weighted_r = sum((recall[k]/100)*support[k] for k in support) / total
weighted_f = sum((f1[k]/100)*support[k] for k in support) / total

print(f"weighted avg{weighted_p:>10.2f}{weighted_r:>10.2f}{weighted_f:>10.2f}{total_samples:>10}")