import csv
from collections import defaultdict
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Function to parse date in multiple formats
def parse_date(date_string):
    try:
        return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            return datetime.strptime(date_string, '%d-%m-%Y %H:%M')
        except ValueError:
            print(f"Unable to parse date: {date_string}")
            return None

# Analyzes transactions and returns suspicious activities and user activities
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

    suspicious_activities = []
    user_activities = defaultdict(list)

    for user_id, user_transactions in transactions.items():
        user_transactions.sort(key=lambda x: x['date'])
        
        # Check for rapid succession withdrawals
        for i in range(len(user_transactions) - 1):
            if (user_transactions[i]['type'] == 'withdraw' and 
                user_transactions[i+1]['type'] == 'withdraw' and
                (user_transactions[i+1]['date'] - user_transactions[i]['date']) < timedelta(hours=1)):
                activity = "Rapid succession withdrawals"
                suspicious_activities.append(f"{activity} for user {user_id}")
                user_activities[user_id].append(activity)
                break

        # Check for large deposits followed by immediate withdrawals
        for i in range(len(user_transactions) - 1):
            if (user_transactions[i]['type'] == 'deposit' and 
                user_transactions[i+1]['type'] == 'withdraw' and
                abs(user_transactions[i]['amount']) > 100 and
                (user_transactions[i+1]['date'] - user_transactions[i]['date']) < timedelta(hours=24)):
                activity = "Large deposit followed by quick withdrawal"
                suspicious_activities.append(f"{activity} for user {user_id}")
                user_activities[user_id].append(activity)
                break

        # Check for unusually large transactions
        if user_transactions:
            avg_transaction = sum(abs(t['amount']) for t in user_transactions) / len(user_transactions)
            for transaction in user_transactions:
                if abs(transaction['amount']) > avg_transaction * 10:
                    activity = f"Unusually large transaction: {transaction['amount']} BTC"
                    suspicious_activities.append(f"{activity} for user {user_id}")
                    user_activities[user_id].append(activity)

    return suspicious_activities, user_activities, transactions

# Preprocess transaction data for AI model training
def preprocess_data(transactions):
    data = []
    
    for user_id, user_transactions in transactions.items():
        num_transactions = len(user_transactions)
        total_amount = sum(abs(t['amount']) for t in user_transactions)
        avg_amount = total_amount / num_transactions if num_transactions else 0
        deposit_count = sum(1 for t in user_transactions if t['type'] == 'deposit')
        withdraw_count = sum(1 for t in user_transactions if t['type'] == 'withdraw')
        time_gaps = [(user_transactions[i+1]['date'] - user_transactions[i]['date']).total_seconds() for i in range(len(user_transactions) - 1)]
        avg_time_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 0
        
        # Append features for each user
        data.append({
            'user_id': user_id,
            'num_transactions': num_transactions,
            'total_amount': total_amount,
            'avg_amount': avg_amount,
            'deposit_count': deposit_count,
            'withdraw_count': withdraw_count,
            'avg_time_gap': avg_time_gap,
        })
    
    return pd.DataFrame(data)

# Create a graph from user activities
def create_graph(user_activities):
    G = nx.Graph()
    
    for user, activities in user_activities.items():
        G.add_node(user, node_type='user')
        for activity in activities:
            activity_node = f"{activity[:20]}..."  # Truncate long activity descriptions
            G.add_node(activity_node, node_type='activity')
            G.add_edge(user, activity_node)
    
    return G

# Visualize the graph, coloring suspicious users red
def visualize_graph_with_predictions(G, suspicious_users):
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(12, 8))
    
    user_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'user']
    activity_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'activity']
    
    user_colors = ['red' if user in suspicious_users else 'lightblue' for user in user_nodes]
    
    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color=user_colors, node_size=500, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=activity_nodes, node_color='lightgreen', node_size=1000, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
    
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("AI-Enhanced Suspicious Activity Network")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('ai_suspicious_activity_network.png', dpi=300, bbox_inches='tight')
    print("AI-enhanced graph visualization saved as 'ai_suspicious_activity_network.png'")

# Predict suspicious users based on AI model
def predict_suspicious_activities(new_transactions, model):
    new_data = preprocess_data(new_transactions)  # Preprocess new transactions
    new_X = new_data.drop('user_id', axis=1)
    predictions = model.predict(new_X)
    
    suspicious_users = new_data['user_id'][predictions == 1]
    return suspicious_users

# Main function to analyze transactions and integrate AI
def main(file_path):
    suspicious_activities, user_activities, transactions = analyze_transactions(file_path)

    print("Potential fraudulent activities detected:")
    for activity in suspicious_activities:
        print(f"- {activity}")

    # Preprocess data for AI model
    data = preprocess_data(transactions)
    
    # Simulate labels for the sake of demonstration (0 = non-suspicious, 1 = suspicious)
    data['label'] = [random.choice([0, 1]) for _ in range(len(data))]
    
    X = data.drop(['user_id', 'label'], axis=1)
    y = LabelEncoder().fit_transform(data['label'])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict suspicious users
    suspicious_users = predict_suspicious_activities(transactions, model)
    print(f"Suspicious users detected by AI: {list(suspicious_users)}")

    # Create and visualize the graph
    G = create_graph(user_activities)
    visualize_graph_with_predictions(G, suspicious_users)

# Assuming the CSV data is saved in a file named 'bitcoin_transactions.csv'
main('bitcoin_transactions.csv')
