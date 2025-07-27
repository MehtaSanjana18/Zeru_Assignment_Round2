# Wallet Input
import pandas as pd


wallets_url = "https://docs.google.com/spreadsheets/d/1ZzaeMgNYnxvriYYpe8PE7uMEblTI0GV5GIVUnsP-sBs/export?format=csv"
wallet_df = pd.read_csv(wallets_url)
wallets = wallet_df['wallet_id'].str.lower().tolist()
wallets

#Feetch Comppound V2 Data
import requests
from tqdm import tqdm

GRAPH_URL = "https://api.thegraph.com/subgraphs/name/graphprotocol/compound-v2"

def run_query(query):
    response = requests.post(GRAPH_URL, json={'query': query})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Query failed. {response.text}")

def get_account_data(wallet):
    wallet = wallet.lower()
    query = f"""
    {{
      account(id: "{wallet}") {{
        id
        tokens {{
          symbol
          supplyBalanceUnderlying
          borrowBalanceUnderlying
        }}
        supplyCount
        borrowCount
        liquidationCount
      }}
    }}
    """
    return run_query(query)


# Process And Extract Features
import numpy as np

results = []

for wallet in tqdm(wallets):
    try:
        data = get_account_data(wallet)
        account = data.get("data", {}).get("account", {})

        if not account:
            continue

        tokens = account.get("tokens", [])
        total_supplied = sum(float(t.get("supplyBalanceUnderlying", 0)) for t in tokens)
        total_borrowed = sum(float(t.get("borrowBalanceUnderlying", 0)) for t in tokens)

        result = {
            "wallet_id": wallet,
            "total_supplied": total_supplied,
            "total_borrowed": total_borrowed,
            "borrow_to_supply_ratio": total_borrowed / total_supplied if total_supplied > 0 else 0,
            "supply_count": account.get("supplyCount", 0),
            "borrow_count": account.get("borrowCount", 0),
            "liquidation_count": account.get("liquidationCount", 0),
        }
        results.append(result)

    except Exception as e:
        print(f"Failed for {wallet}: {e}")

# Feature Engineering and normalization
features_df = pd.DataFrame(results)

# Fill missing
features_df.fillna(0, inplace=True)

# Normalize
def minmax(col):
    return (col - col.min()) / (col.max() - col.min()) if col.max() > col.min() else col

features_df["norm_borrow_to_supply"] = minmax(features_df["borrow_to_supply_ratio"])
features_df["norm_liquidations"] = minmax(features_df["liquidation_count"])
features_df["norm_supply"] = minmax(features_df["total_supplied"])
features_df["norm_borrow"] = minmax(features_df["total_borrowed"])

# Scoring Model
# Wallet Age as feature
from datetime import datetime

def get_wallet_age(wallet):
    url = f'https://api.etherscan.io/api?module=account&action=txlist&address={wallet}&startblock=0&endblock=99999999&sort=asc&apikey=YOUR_API_KEY'
    r = requests.get(url)
    data = r.json()
    if data["result"]:
        first_tx_time = int(data["result"][0]["timeStamp"])
        wallet_age_days = (datetime.now() - datetime.fromtimestamp(first_tx_time)).days
        return wallet_age_days
    return 0

features_df["wallet_age_days"] = features_df["wallet_id"].apply(get_wallet_age)
features_df["norm_wallet_age"] = minmax(features_df["wallet_age_days"])

# Asset Diversity
def get_asset_count(wallet_data):
    return len(wallet_data.get("tokens", []))

features_df["asset_count"] = features_df["wallet_id"].apply(lambda w: get_asset_count(get_account_data(w)["data"]["account"]))
features_df["norm_asset_count"] = minmax(features_df["asset_count"])

#Repayment Ratio
# Use borrow + repay txs from Compound tx history
repayment_ratio = total_repaid / total_borrowed

# Activity Score
activity_score = supply_count + borrow_count + repay_count
features_df["norm_activity"] = minmax(activity_score)

# Final Risk Score
# Weight dictionary (optional)
weights = {
    "borrow_to_supply_ratio": 0.25,
    "liquidation_count": 0.20,
    "total_borrowed": 0.10,
    "total_supplied": 0.10,
    "wallet_age_days": 0.10,
    "asset_count": 0.10,
    "activity_score": 0.05,
}

# Risk score calculation (higher = more risk)
features_df["risk_score"] = (
    weights["borrow_to_supply_ratio"] * features_df["norm_borrow_to_supply"] +
    weights["liquidation_count"] * features_df["norm_liquidations"] +
    weights["total_borrowed"] * features_df["norm_borrow"] -
    weights["total_supplied"] * features_df["norm_supply"] -
    weights["wallet_age_days"] * features_df["norm_wallet_age"] -
    weights["asset_count"] * features_df["norm_asset_count"] -
    weights["activity_score"] * features_df["norm_activity_score"]
)

# Normalize risk score to 0–1
features_df["normalized_risk"] = (
    (features_df["risk_score"] - features_df["risk_score"].min()) / 
    (features_df["risk_score"].max() - features_df["risk_score"].min())
)

# Convert to score out of 1000 (higher = safer)
features_df["score"] = ((1 - features_df["normalized_risk"]) * 1000).round().astype(int)

# Output CSV
final_df = features_df[["wallet_id", "score"]].copy()
final_df["score"] = final_df["score"].round(0).astype(int)
final_df.to_csv("wallet_scores.csv", index=False)
