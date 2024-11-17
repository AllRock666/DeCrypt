import csv
from collections import defaultdict
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np

# Helper function to parse dates from the CSV
def parse_date(date_string):
    try:
        return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            return datetime.strptime(date_string, '%d-%m-%Y %H:%M')
        except ValueError:
            print(f"Unable to parse date: {date_string}")
            return None

# Function to read and analyze transactions
def analyze_transactions(file_path):
    transactions = defaultdict(list)

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) != 5:
                print(f"Skipping malformed row: {row}")
                continue
            user_id, transaction_id, date, transaction_type, amount = row
            parsed_date = parse_date(date)
            if parsed_date is None:
                continue
            transactions[user_id].append({
                'transaction_id': transaction_id,
                'date': parsed_date,
                'type': transaction_type,
                'amount': float(amount)
            })

    return transactions

# Preprocessing the data to create features
def preprocess_data(transactions):
    features = []
    labels = []
    user_transaction_ids = []
    user_ids = []  # Track user IDs for later output
    
    for user_id, user_transactions in transactions.items():
        user_transactions.sort(key=lambda x: x['date'])

        num_transactions = len(user_transactions)
        total_amount = sum(t['amount'] for t in user_transactions)
        avg_transaction_amount = total_amount / num_transactions if num_transactions > 0 else 0
        
        num_deposits = sum(1 for t in user_transactions if t['type'] == 'deposit')
        num_withdrawals = sum(1 for t in user_transactions if t['type'] == 'withdraw')
        
        time_diffs = [(user_transactions[i+1]['date'] - user_transactions[i]['date']).total_seconds() 
                      for i in range(len(user_transactions) - 1)]
        avg_time_diff = np.mean(time_diffs) if time_diffs else 0
        
        features.append([
            num_transactions, total_amount, avg_transaction_amount, 
            num_deposits, num_withdrawals, avg_time_diff
        ])
        
        # Collect transaction IDs to associate with each user's features
        transaction_ids = [t['transaction_id'] for t in user_transactions]
        user_transaction_ids.append(transaction_ids)
        user_ids.append(user_id)  # Collect user IDs
        
        # Simulate labels: Assume 10% users are suspicious
        label = 1 if random.random() < 0.1 else 0
        labels.append(label)
    
    return np.array(features), np.array(labels), user_transaction_ids, user_ids

# Creating a graph of user activities
def create_graph(user_activities, user_transaction_ids, predictions):
    G = nx.Graph()
    
    for idx, (user, activities) in enumerate(user_activities.items()):
        G.add_node(user, node_type='user')
        for activity in activities:
            activity_node = f"{activity[:20]}..."  # Truncate long activity descriptions
            G.add_node(activity_node, node_type='activity')
            G.add_edge(user, activity_node)

        # Attach transaction IDs to user nodes
        transaction_ids = user_transaction_ids[idx]
        for tx_id in transaction_ids:
            G.add_node(tx_id, node_type='transaction')
            G.add_edge(user, tx_id)
    
    return G

# Visualize the graph and save the image
def visualize_graph_with_predictions(G, predictions, user_ids, num_transactions):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))

    user_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'user']
    activity_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'activity']
    transaction_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'transaction']
    
    user_colors = ['red' if predictions[int(node)] == 1 else 'lightblue' for node in user_nodes]

    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color=user_colors, node_size=500, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=activity_nodes, node_color='lightgreen', node_size=1000, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=transaction_nodes, node_color='orange', node_size=300, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
    
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Suspicious Activity Network with XGBoost Predictions and Transaction IDs")
    
    # Create a table to show user transaction info and fraud status
    table_data = [["User ID", "Number of Transactions", "Fraudulent (Yes/No)"]]
    for user_id, num_tx, pred in zip(user_ids, num_transactions, predictions):
        fraud_status = "Yes" if pred == 1 else "No"
        table_data.append([user_id, num_tx, fraud_status])
    
    # Create the table and position it on the plot
    table = plt.table(cellText=table_data, colLabels=None, loc='bottom', cellLoc='center', bbox=[0.1, -0.4, 0.8, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Adjust for table space
    plt.savefig('suspicious_activity_network_with_table.png', dpi=300, bbox_inches='tight')
    print("Graph visualization with table saved as 'suspicious_activity_network_with_table.png'")

# Function to print the transaction table to the console
def print_transaction_table(user_ids, num_transactions, predictions):
    print(f"{'User ID':<20} {'Number of Transactions':<25} {'Fraudulent (Yes/No)':<20}")
    print("=" * 65)
    for user_id, num_tx, pred in zip(user_ids, num_transactions, predictions):
        fraud_status = "Yes" if pred == 1 else "No"
        print(f"{user_id:<20} {num_tx:<25} {fraud_status:<20}")

# Main function to execute the analysis and visualization
def main(file_path):
    transactions = analyze_transactions(file_path)

    # Preprocess data and split into training and test sets
    X, y, user_transaction_ids, user_ids = preprocess_data(transactions)

    if len(X) <= 1:
        print("Not enough data for train-test split. Using the entire dataset for both training and testing.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost model
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"XGBoost Model Accuracy: {accuracy * 100:.2f}%")

    # Display the output table with user ID, number of transactions, and fraud status
    num_transactions = [len(t) for t in user_transaction_ids]
    print_transaction_table(user_ids, num_transactions, y_pred)
    
    # Visualize the graph with predictions and transaction IDs
    user_activities = {str(i): ["Suspicious" if y_pred[i] == 1 else "Normal"] for i in range(len(X_test))}
    G = create_graph(user_activities, user_transaction_ids, y_pred)
    visualize_graph_with_predictions(G, y_pred, user_ids, num_transactions)

# Assuming the CSV data is saved in a file named 'bitcoin_transactions.csv'
main('bitcoin_transactions.csv')
