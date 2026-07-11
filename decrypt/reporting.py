import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

from .config import setup_logger, default_config

logger = setup_logger(__name__, default_config)

def generate_visualizations(df: pd.DataFrame, 
                            suspicious_addresses: List[str], 
                            risk_metrics: Dict[str, list],
                            cluster_results: Dict[str, Any],
                            G: nx.DiGraph,
                            output_dir: Path):
    """
    Generate and save visualization plots.
    """
    logger.info("Generating visualizations...")
    plt.figure(figsize=(20, 16))
    
    # 1. Risk Score Distribution
    plt.subplot(2, 3, 1)
    sns.histplot(risk_metrics['risk_score'], bins=50)
    plt.title('Risk Score Distribution')
    plt.xlabel('Risk Score')
    plt.ylabel('Count')
    
    # 2. Transaction Amounts Box Plot
    plt.subplot(2, 3, 2)
    suspicious_mask = df['address_id'].isin(suspicious_addresses)
    sns.boxplot(x=suspicious_mask, y=df['amount'].abs())
    plt.title('Transaction Amounts by Address Type')
    plt.xticks([0, 1], ['Normal', 'Suspicious'])
    plt.yscale('log')
    
    # 3. Cluster Sizes
    plt.subplot(2, 3, 3)
    cluster_sizes = pd.Series(cluster_results['cluster_sizes'])
    sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values)
    plt.title('Addresses per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Addresses')
    
    # 4. Network Graph of Suspicious Clusters (Directed)
    plt.subplot(2, 3, 4)
    suspicious_subgraph = G.subgraph(suspicious_addresses)
    if len(suspicious_subgraph.nodes) > 0:
        pos = nx.spring_layout(suspicious_subgraph, seed=42)
        nx.draw_networkx_nodes(suspicious_subgraph, pos, node_color='red', node_size=100, alpha=0.8)
        nx.draw_networkx_edges(suspicious_subgraph, pos, edge_color='gray', alpha=0.5, arrows=True)
    plt.title('Directed Network of Suspicious Addresses')
    
    # 5. Temporal Pattern Analysis
    plt.subplot(2, 3, 5)
    df_suspicious = df[df['address_id'].isin(suspicious_addresses)]
    df_normal = df[~df['address_id'].isin(suspicious_addresses)]
    
    plt.hist(df_suspicious['timestamp'].dt.hour.dropna(), bins=24, alpha=0.5, 
             density=True, label='Suspicious', color='red')
    plt.hist(df_normal['timestamp'].dt.hour.dropna(), bins=24, alpha=0.5, 
             density=True, label='Normal', color='blue')
    plt.title('Transaction Time Distribution (Hour of Day)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Density')
    plt.legend()
    
    # 6. Cluster Characteristics Heatmap
    plt.subplot(2, 3, 6)
    if cluster_results['cluster_characteristics']:
        cluster_data = []
        for cluster_id, chars in cluster_results['cluster_characteristics'].items():
            cluster_data.append({
                'Cluster': cluster_id,
                'Size': chars['size'],
                'Avg Volume': chars['avg_transaction_volume'],
                'Avg Freq': chars['avg_transaction_frequency'],
                'Avg In-Degree': chars['avg_in_degree']
            })
            
        cluster_chars = pd.DataFrame(cluster_data).set_index('Cluster')
        # Normalize
        if not cluster_chars.empty and cluster_chars.shape[0] > 1:
            # Handle divide by zero if max==min
            range_vals = cluster_chars.max() - cluster_chars.min()
            range_vals[range_vals == 0] = 1
            normalized = (cluster_chars - cluster_chars.min()) / range_vals
            sns.heatmap(normalized, annot=cluster_chars.round(2), fmt='.2f', cmap='YlOrRd')
            plt.title('Cluster Characteristics (Normalized)')
        else:
            plt.text(0.5, 0.5, 'Not enough clusters for heatmap', ha='center', va='center')
    
    plt.tight_layout()
    plot_path = output_dir / 'fraud_analysis_patterns.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualizations saved to {plot_path}")
    return plot_path

def generate_pdf_report(suspicious_addresses: List[str], 
                        risk_metrics: Dict[str, list], 
                        cluster_results: Dict[str, Any],
                        plot_path: Path,
                        output_dir: Path):
    """
    Generate a comprehensive PDF report using ReportLab.
    """
    logger.info("Generating PDF report...")
    output_path = output_dir / 'fraud_analysis_report.pdf'
    
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = styles['Title']
    heading_style = styles['Heading1']
    normal_style = styles['Normal']
    
    elements = []
    
    # Title
    elements.append(Paragraph("Bitcoin Address Fraud Analysis Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Summary
    elements.append(Paragraph("Summary Statistics", heading_style))
    summary_data = [
        ['Total Addresses Analyzed', str(len(risk_metrics['address']))],
        ['Suspicious Addresses Detected', str(len(suspicious_addresses))],
        ['Optimal Number of Clusters', str(cluster_results.get('optimal_clusters', 0))],
        ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    ]
    t = Table(summary_data, colWidths=[3*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))
    
    # High-Risk Addresses Table
    elements.append(Paragraph("High-Risk Addresses (Top 20)", heading_style))
    risk_df = pd.DataFrame(risk_metrics)
    high_risk = risk_df.sort_values(by='risk_score', ascending=False).head(20)
    
    risk_data = [['Address', 'Risk Score']]
    for _, row in high_risk.iterrows():
        risk_data.append([str(row['address']), f"{row['risk_score']:.2f}"])
        
    rt = Table(risk_data, colWidths=[4*inch, 1*inch])
    rt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkred),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,1), (-1,-1), colors.mistyrose),
    ]))
    elements.append(rt)
    elements.append(Spacer(1, 12))
    
    # Plots
    if plot_path.exists():
        elements.append(Paragraph("Visual Analysis", heading_style))
        elements.append(Image(str(plot_path), width=6*inch, height=4.8*inch))
        elements.append(Spacer(1, 12))
        
    # Build
    doc.build(elements)
    logger.info(f"PDF report saved to {output_path}")
