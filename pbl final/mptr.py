import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TransactionSequenceDataset(Dataset):
    """Dataset class for transaction sequences"""
    def __init__(self, transactions: pd.DataFrame, max_seq_length: int = 128):
        self.transactions = transactions
        self.max_seq_length = max_seq_length
        self.address_sequences = self._prepare_sequences()
        
    def _prepare_sequences(self) -> Dict[str, List[Dict]]:
        """Convert transactions into sequences per address"""
        sequences = {}
        for address, group in self.transactions.groupby('address_id'):
            # Sort transactions by timestamp
            group = group.sort_values('timestamp')
            
            # Create sequence of transaction features
            sequence = []
            for _, tx in group.iterrows():
                tx_features = {
                    'amount': float(tx['amount']),
                    'type': 1 if tx['type'] == 1 else 0,  # deposit=1, withdraw=0
                    'timestamp': tx['timestamp'].timestamp(),
                }
                sequence.append(tx_features)
            sequences[address] = sequence
        return sequences
    
    def __len__(self) -> int:
        return len(self.address_sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        address = list(self.address_sequences.keys())[idx]
        sequence = self.address_sequences[address]
        
        # Pad or truncate sequence
        if len(sequence) > self.max_seq_length:
            sequence = sequence[-self.max_seq_length:]
        else:
            padding = [{
                'amount': 0,
                'type': 0,
                'timestamp': 0
            }] * (self.max_seq_length - len(sequence))
            sequence = sequence + padding
        
        # Convert to tensors
        amounts = torch.tensor([tx['amount'] for tx in sequence], dtype=torch.float)
        types = torch.tensor([tx['type'] for tx in sequence], dtype=torch.long)
        timestamps = torch.tensor([tx['timestamp'] for tx in sequence], dtype=torch.float)
        
        return {
            'address': address,
            'amounts': amounts,
            'types': types,
            'timestamps': timestamps,
            'attention_mask': torch.tensor([1] * len(sequence) + [0] * (self.max_seq_length - len(sequence)), dtype=torch.long)
        }

class TransactionTransformer(nn.Module):
    """Transformer model for transaction sequence analysis"""
    def __init__(self, 
                 d_model: int = 256, 
                 nhead: int = 8, 
                 num_layers: int = 4, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.embedding_amount = nn.Linear(1, d_model // 3)
        self.embedding_type = nn.Embedding(2, d_model // 3)  # deposit/withdraw
        self.embedding_time = nn.Linear(1, d_model // 3)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, amounts: torch.Tensor, types: torch.Tensor, 
                timestamps: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Create embeddings
        amount_emb = self.embedding_amount(amounts.unsqueeze(-1))
        type_emb = self.embedding_type(types)
        time_emb = self.embedding_time(timestamps.unsqueeze(-1))
        
        # Combine embeddings
        x = torch.cat([amount_emb, type_emb, time_emb], dim=-1)
        
        # Create attention mask
        attention_mask = attention_mask.bool()
        
        # Pass through transformer
        x = x.permute(1, 0, 2)  # (seq_len, batch, features)
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask)
        
        # Get sequence representation (use [CLS] token or mean)
        x = x.mean(dim=0)  # (batch, features)
        
        # Predict risk score
        risk_score = self.fc_out(x)
        return risk_score

class TransformerBitcoinDetector:
    """Enhanced Bitcoin fraud detector using transformers"""
    def __init__(self, 
                 max_seq_length: int = 128,
                 batch_size: int = 32,
                 learning_rate: float = 1e-4,
                 num_epochs: int = 10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        self.model = TransactionTransformer().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
    def prepare_data(self, df: pd.DataFrame) -> TransactionSequenceDataset:
        """Prepare transaction data for the transformer"""
        return TransactionSequenceDataset(df, self.max_seq_length)
    
    def train(self, train_dataset: TransactionSequenceDataset, 
              val_dataset: Optional[TransactionSequenceDataset] = None):
        """Train the transformer model"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                amounts = batch['amounts'].to(self.device)
                types = batch['types'].to(self.device)
                timestamps = batch['timestamps'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                self.optimizer.zero_grad()
                
                risk_scores = self.model(amounts, types, timestamps, attention_mask)
                
                # For training, we can use isolation forest scores as pseudo-labels
                # or implement a self-supervised approach
                # This is a simplified example using random labels
                pseudo_labels = torch.rand_like(risk_scores)
                
                loss = self.criterion(risk_scores, pseudo_labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, dataset: TransactionSequenceDataset) -> Dict[str, float]:
        """Predict risk scores for addresses"""
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        risk_scores = {}
        
        with torch.no_grad():
            for batch in dataloader:
                amounts = batch['amounts'].to(self.device)
                types = batch['types'].to(self.device)
                timestamps = batch['timestamps'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                addresses = batch['address']
                
                scores = self.model(amounts, types, timestamps, attention_mask)
                
                for addr, score in zip(addresses, scores.cpu().numpy()):
                    risk_scores[addr] = float(score)
        
        return risk_scores

class EnhancedBitcoinFraudDetector:
    """Combined detector using both traditional methods and transformers"""
    def __init__(self):
        self.traditional_detector = BitcoinFraudDetector()  # Your existing detector
        self.transformer_detector = TransformerBitcoinDetector()
        
    def analyze_transactions(self, filepath: str) -> Dict:
        # Load data
        df = self.traditional_detector.load_data(filepath)
        
        # Traditional analysis
        traditional_results = self.traditional_detector.analyze_transactions(filepath)
        
        # Transformer analysis
        dataset = self.transformer_detector.prepare_data(df)
        self.transformer_detector.train(dataset)
        transformer_risk_scores = self.transformer_detector.predict(dataset)
        
        # Combine results
        combined_results = self._combine_results(
            traditional_results,
            transformer_risk_scores
        )
        
        return combined_results
    
    def _combine_results(self, 
                        traditional_results: Dict, 
                        transformer_scores: Dict) -> Dict:
        """Combine results from both detection methods"""
        combined_risk_scores = {}
        
        # Combine risk scores with weighted average
        for address in traditional_results['risk_metrics']['address']:
            trad_score = traditional_results['risk_metrics']['risk_score'][
                traditional_results['risk_metrics']['address'].index(address)
            ]
            trans_score = transformer_scores.get(address, 0) * 100  # Scale to 0-100
            
            # Weight transformer scores more heavily for sequential patterns
            combined_score = 0.4 * trad_score + 0.6 * trans_score
            combined_risk_scores[address] = combined_score
        
        # Update results
        results = traditional_results.copy()
        results['transformer_risk_scores'] = transformer_scores
        results['combined_risk_scores'] = combined_risk_scores
        
        # Update suspicious addresses based on combined scores
        high_risk_threshold = 75
        results['suspicious_addresses'] = [
            addr for addr, score in combined_risk_scores.items()
            if score > high_risk_threshold
        ]
        
        return results

if __name__ == "__main__":
    # Example usage
    detector = EnhancedBitcoinFraudDetector()
    results = detector.analyze_transactions("bitcoin_transactions.csv")
    print(f"Detected {len(results['suspicious_addresses'])} suspicious addresses")