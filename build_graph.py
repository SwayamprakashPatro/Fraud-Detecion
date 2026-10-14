import mysql.connector as mysql
import networkx as nx
import pickle
import pandas as pd
import os

# Database connection parameters - ensure they match your setup
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "fraud_detection_gnn",
}

def get_dataframe(query: str) -> pd.DataFrame:
    """Execute a SQL query via mysql.connector and return a pandas DataFrame."""
    conn = mysql.connect(**DB_CONFIG)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def build_graph() -> nx.MultiDiGraph:
    """Construct a heterogeneous graph for fraud detection.
    Nodes:
        - accounts (sender & receiver)
        - customers (linked to accounts)
        - devices
        - merchants
        - upi accounts
    Edges:
        - transaction edges from sender account to receiver account
        - ownership edges linking accounts to customers
        - device usage edges linking transactions to devices
        - merchant edges linking transactions to merchants
        - upi usage edges linking transactions to UPI accounts
    """
    G = nx.MultiDiGraph()

    # Load the main transaction table with enriched fields
    transaction_query = """
    SELECT
        t.Transaction_ID,
        t.Amount,
        t.Channel,
        t.Transaction_Timestamp,
        t.Is_International,
        t.Is_Reversal,
        t.Failure_Count,
        sa.Account_ID AS Sender_Account,
        ra.Account_ID AS Receiver_Account,
        d.Device_ID,
        m.Merchant_ID,
        u.UPI_ID,
        f.Fraud_Type,
        COALESCE(f.Label, 0) AS Label
    FROM TRANSACTION t
    LEFT JOIN ACCOUNT sa ON t.Sender_Account_ID = sa.Account_ID
    LEFT JOIN ACCOUNT ra ON t.Receiver_Account_ID = ra.Account_ID
    LEFT JOIN DEVICE d ON t.Device_ID = d.Device_ID
    LEFT JOIN MERCHANT m ON t.Merchant_ID = m.Merchant_ID
    LEFT JOIN UPI_ACCOUNT u ON t.UPI_ID = u.UPI_ID
    LEFT JOIN FRAUD_CASE f ON t.Transaction_ID = f.Transaction_ID;
    """
    tx_df = get_dataframe(transaction_query)

    # Add nodes for accounts
    for acct_id in pd.concat([tx_df['Sender_Account'], tx_df['Receiver_Account']]).unique():
        if pd.isna(acct_id):
            continue
        G.add_node(f"account_{int(acct_id)}", type="account")

    # Add nodes for devices
    for dev_id in tx_df['Device_ID'].dropna().unique():
        G.add_node(f"device_{int(dev_id)}", type="device")

    # Add nodes for merchants
    for merch_id in tx_df['Merchant_ID'].dropna().unique():
        G.add_node(f"merchant_{int(merch_id)}", type="merchant")

    # Add nodes for UPI accounts
    for upi_id in tx_df['UPI_ID'].dropna().unique():
        G.add_node(f"upi_{int(upi_id)}", type="upi")

    # Transaction edges (sender -> receiver)
    for _, row in tx_df.iterrows():
        sender = row['Sender_Account']
        receiver = row['Receiver_Account']
        if pd.notna(sender) and pd.notna(receiver):
            G.add_edge(
                f"account_{int(sender)}",
                f"account_{int(receiver)}",
                key=row['Transaction_ID'],
                type="transaction",
                amount=row['Amount'],
                timestamp=row['Transaction_Timestamp'],
                channel=row['Channel'],
                label=row['Label'],
                fraud_type=row['Fraud_Type'] if pd.notna(row['Fraud_Type']) else None,
            )
        # Device usage edge
        if pd.notna(row['Device_ID']):
            G.add_edge(
                f"account_{int(sender)}" if pd.notna(sender) else f"account_{int(receiver)}",
                f"device_{int(row['Device_ID'])}",
                key=row['Transaction_ID'],
                type="used_device",
                timestamp=row['Transaction_Timestamp'],
            )
        # Merchant edge
        if pd.notna(row['Merchant_ID']):
            G.add_edge(
                f"account_{int(sender)}" if pd.notna(sender) else f"account_{int(receiver)}",
                f"merchant_{int(row['Merchant_ID'])}",
                key=row['Transaction_ID'],
                type="merchant",
                timestamp=row['Transaction_Timestamp'],
            )
        # UPI edge
        if pd.notna(row['UPI_ID']):
            G.add_edge(
                f"account_{int(sender)}" if pd.notna(sender) else f"account_{int(receiver)}",
                f"upi_{int(row['UPI_ID'])}",
                key=row['Transaction_ID'],
                type="upi_used",
                timestamp=row['Transaction_Timestamp'],
            )

    return G

def save_graph(G: nx.MultiDiGraph, path: str = "graph.gpickle"):
    """Save the graph to disk using pickle format."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Graph saved to {path}")

if __name__ == "__main__":
    graph = build_graph()
    save_graph(graph, "c:/GML project/graph.gpickle")
