"""patch_train.py — fixes train_model.py in-place"""
import re

with open('train_model.py', 'r') as f:
    content = f.read()

# Fix 1: Remove respiratory, add yellow_fever
content = content.replace(
    '"meningitis", "lassa_fever", "diarrhoeal", "respiratory"',
    '"meningitis", "lassa_fever", "diarrhoeal", "yellow_fever"'
)

# Fix 2: Replace year-based split with week-based split
old_split = '    train_mask = df["epi_year"] < split_year\n    test_mask  = df["epi_year"] >= split_year'
new_split = '    weeks = sorted(df["epi_week"].unique())\n    split_week = weeks[int(len(weeks) * 0.70)]\n    train_mask = df["epi_week"] < split_week\n    test_mask  = df["epi_week"] >= split_week'
content = content.replace(old_split, new_split)

with open('train_model.py', 'w') as f:
    f.write(content)

if '"yellow_fever"' in content:
    print("Fix 1 OK: yellow_fever added")
else:
    print("Fix 1 FAILED")

if 'split_week' in content:
    print("Fix 2 OK: week-based split applied")
else:
    print("Fix 2 FAILED")

print("Done - run: python train_model.py")
