import os
from pathlib import Path

# Paths to your label directories
label_dirs = [
    r"C:\Users\tanis\OneDrive\Documents\dataset\IndianBankNotes\train\labels",
    r"C:\Users\tanis\OneDrive\Documents\dataset\IndianBankNotes\valid\labels"
]

# Mapping filename keywords to their class ID based on data.yaml
class_mapping = {
    "INDIA10NEW": 0,
    "INDIA10OLD": 1,
    "INDIA20": 2,
    "INDIA50NEW": 3,
    "INDIA50OLD": 4,
    "INDIA100NEW": 5,
    "INDIA100OLD": 6,
    "INDIA200": 7,
    "INDIA500": 8,
    "INDIA2000": 9,
}

fixed_count = 0

for label_dir in label_dirs:
    p = Path(label_dir)
    if not p.exists():
        print(f"Directory not found: {label_dir}")
        continue

    for txt_file in p.glob("*.txt"):
        filename = txt_file.name.upper()
        
        # Determine class ID from filename
        target_class = None
        for key, class_id in class_mapping.items():
            if key in filename:
                target_class = class_id
                break
                
        if target_class is None:
            continue

        with open(txt_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            # Replace -1 (or any corrupt class id) with the correct target class ID
            parts[0] = str(target_class)
            new_lines.append(" ".join(parts) + "\n")

        with open(txt_file, "w") as f:
            f.writelines(new_lines)

        fixed_count += 1

print(f"Successfully fixed {fixed_count} label files!")