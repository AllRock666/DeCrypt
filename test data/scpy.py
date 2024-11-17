import csv
from collections import defaultdict
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt

def parse_date(date_string):
    try:
        return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            return datetime.strptime(date_string, '%d-%m-%Y %H:%M')
        except ValueError:
            print(f"Unable to parse date: {date_string}")
            return None

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

    return suspicious_activities, user_activities

def create_graph(user_activities):
    G = nx.Graph()
    
    for user, activities in user_activities.items():
        G.add_node(user, node_type='user')
        for activity in activities:
            activity_node = f"{activity[:20]}..."  # Truncate long activity descriptions
            G.add_node(activity_node, node_type='activity')
            G.add_edge(user, activity_node)
    
    return G

def visualize_graph(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    
    user_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'user']
    activity_nodes = [node for node, data in G.nodes(data=True) if data['node_type'] == 'activity']
    
    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color='lightblue', node_size=500, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=activity_nodes, node_color='lightgreen', node_size=1000, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
    
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Suspicious Activity Network")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('suspicious_activity_network.png', dpi=300, bbox_inches='tight')
    print("Graph visualization saved as 'suspicious_activity_network.png'")

# Assuming the CSV data is saved in a file named 'bitcoin_transactions.csv'
suspicious_activities, user_activities = analyze_transactions('bitcoin_transactions.csv')

print("Potential fraudulent activities detected:")
for activity in suspicious_activities:
    print(f"- {activity}")

G = create_graph(user_activities)
visualize_graph(G)