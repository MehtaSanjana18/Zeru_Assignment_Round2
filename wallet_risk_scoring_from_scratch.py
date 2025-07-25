# Wallet Input
import pandas as pd


wallets_url = "https://docs.google.com/spreadsheets/d/1ZzaeMgNYnxvriYYpe8PE7uMEblTI0GV5GIVUnsP-sBs/export?format=csv"
wallet_df = pd.read_csv(wallets_url)
wallets = wallet_df['wallet_id'].str.lower().tolist()
wallets

