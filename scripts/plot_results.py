import json
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/figures", exist_ok=True)

# Load both JSON results
teacher_json = "results/teacher_legalbert_inlegalner_test.json"
student_json = "results/student_kd_distilbert_inlegalner_test.json"

with open(teacher_json, "r") as f:
    teacher = json.load(f)

with open(student_json, "r") as f:
    student = json.load(f)

teacher_f1 = teacher["test_metrics"]["test_f1"]
student_f1 = student["test_metrics"]["test_f1"]

# ---------- 1. F1 Score Bar Chart ----------
plt.figure(figsize=(6, 4))
plt.bar(["LegalBERT (Teacher)", "KD-Student (DistilBERT)"],
        [teacher_f1, student_f1],
        color=["#4c72b0", "#55a868"])
plt.ylabel("F1 Score")
plt.title("InLegalNER – Teacher vs Student Performance")
plt.ylim(0, 1.0)
plt.savefig("outputs/figures/f1_comparison.png", dpi=200)
plt.close()

# ---------- 2. Parameter Comparison ----------
teacher_params = 108_000_000
student_params = 66_400_000

plt.figure(figsize=(6, 4))
plt.bar(["Teacher", "Student"], [teacher_params, student_params],
        color=["#4c72b0", "#c44e52"])
plt.ylabel("Number of Parameters")
plt.title("Model Size Comparison")
plt.savefig("outputs/figures/param_comparison.png", dpi=200)
plt.close()

print("Saved figures to outputs/figures/")
