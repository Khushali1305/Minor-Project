from datasets import load_dataset
import json, os

os.makedirs("transfer_data", exist_ok=True)

# 1. TSAR 2025
print("Downloading TSAR 2025...")
tsar = load_dataset("cardiffnlp/TSAR2025_SharedTask_RCTS_Trial-Data")
tsar["train"].to_json("transfer_data/tsar2025.json")
print(f"  TSAR 2025: {len(tsar['train'])} rows saved")

# 2. OneStopEnglish
print("Downloading OneStopEnglish...")
ose = load_dataset("SetFit/onestop_english")
ose["train"].to_json("transfer_data/onestop_english.json")
print(f"  OneStopEnglish: {len(ose['train'])} rows saved")

# 3. ASSET
print("Downloading ASSET...")
asset_ratings = load_dataset("facebook/asset", "ratings")
asset_ratings["full"].to_json("transfer_data/asset_ratings.json")
asset_simplification = load_dataset("facebook/asset", "simplification")
asset_simplification["test"].to_json("transfer_data/asset_simplification_test.json")
asset_simplification["validation"].to_json("transfer_data/asset_simplification_val.json")
print(f"  ASSET ratings: {len(asset_ratings['full'])} rows")
print(f"  ASSET simplification test: {len(asset_simplification['test'])} rows")
print(f"  ASSET simplification val: {len(asset_simplification['validation'])} rows")

print("\nAll saved to ./transfer_data/")