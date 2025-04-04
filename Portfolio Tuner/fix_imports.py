import os
import re

# Root folder where App/ lives
ROOT_DIR = os.path.join(os.getcwd(), "Portfolio Tuner", "App")

# Regex pattern to find 'from plots import' or 'import plots'
IMPORT_PATTERN = re.compile(r'^(from|import)\s+plots(\b|\.|$)', re.MULTILINE)

def fix_imports_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace 'from plots' with 'from utils.plots'
    fixed_content = IMPORT_PATTERN.sub(lambda m: m.group(1) + ' utils.plots', content)

    if content != fixed_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        print(f"✅ Fixed imports in: {filepath}")

def scan_and_fix(directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                fix_imports_in_file(filepath)

if __name__ == "__main__":
    print("🔍 Scanning and fixing import paths...")
    scan_and_fix(ROOT_DIR)
    print("✅ Done.")
