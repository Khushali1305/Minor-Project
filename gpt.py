import requests

api_key = ""  # paste your other account's API key here

headers = {"Authorization": f"Bearer {api_key}"}

# Check credit balance (works for prepaid credits)
resp = requests.get(
    "https://api.openai.com/dashboard/billing/credit_grants",
    headers=headers
)

if resp.status_code == 200:
    data = resp.json()
    total = data.get("total_granted", 0)
    used  = data.get("total_used", 0)
    remaining = data.get("total_available", 0)
    print(f"Total granted : ${total:.2f}")
    print(f"Total used    : ${used:.2f}")
    print(f"Remaining     : ${remaining:.2f}")
else:
    print(f"Error {resp.status_code}: {resp.text}")