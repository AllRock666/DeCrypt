import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

from .config import setup_logger, default_config

logger = setup_logger(__name__, default_config)

def generate_visualizations(
    df: pd.DataFrame, 
    risk_scores_df: pd.DataFrame,
    cluster_results: Dict[str, Any],
    embedding_clusters: Optional[Dict[str, Any]],
    G: nx.DiGraph,
    output_dir: Path
) -> Path:
    """
    Generate and save comprehensive visual analysis dashboard.
    """
    logger.info("Generating visualizations...")
    fig = plt.figure(figsize=(18, 12))
    
    suspicious_mask = risk_scores_df['ensemble_risk_score'] >= 90.0
    suspicious_addrs = set(risk_scores_df[suspicious_mask].index)
    
    # 1. Risk Score Distribution (Ensemble vs IF)
    ax1 = plt.subplot(2, 3, 1)
    sns.histplot(risk_scores_df['ensemble_risk_score'], bins=30, color='#8e44ad', kde=True, label='Ensemble', ax=ax1)
    if 'if_risk_score' in risk_scores_df.columns:
        sns.histplot(risk_scores_df['if_risk_score'], bins=30, color='#3498db', alpha=0.4, label='Isolation Forest', ax=ax1)
    ax1.set_title('Risk Score Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Risk Score (0 - 100)')
    ax1.set_ylabel('Number of Addresses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Transaction Amounts Box Plot
    ax2 = plt.subplot(2, 3, 2)
    df_plot = df.copy()
    df_plot['is_suspicious'] = df_plot['address_id'].isin(suspicious_addrs)
    sns.boxplot(x='is_suspicious', y=df_plot['amount'].abs(), hue='is_suspicious', legend=False, data=df_plot, palette=['#2ecc71', '#e74c3c'], ax=ax2)
    ax2.set_title('Transaction Amounts: Normal vs High-Risk', fontsize=12, fontweight='bold')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Normal (Risk < 90)', 'High-Risk (Risk >= 90)'])
    ax2.set_yscale('log')
    ax2.set_ylabel('Transaction Amount (log scale)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cluster Sizes (KMeans vs HDBSCAN)
    ax3 = plt.subplot(2, 3, 3)
    if embedding_clusters and 'hdbscan_clusters' in embedding_clusters:
        hdb_s = pd.Series(embedding_clusters['hdbscan_clusters']).value_counts().sort_index()
        hdb_df = pd.DataFrame({'Cluster': [f"C{c}" if c != -1 else "Noise" for c in hdb_s.index], 'Count': hdb_s.values})
        sns.barplot(data=hdb_df, x='Cluster', y='Count', color='#2980b9', ax=ax3)
        ax3.set_title(f"HDBSCAN Embedding Clusters (Sil={embedding_clusters['hdbscan_silhouette']:.3f})", fontsize=12, fontweight='bold')
    else:
        cluster_sizes = pd.Series(cluster_results['cluster_sizes']).sort_index()
        sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, palette='Blues_r', ax=ax3)
        ax3.set_title('KMeans Address Clusters', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Cluster ID')
    ax3.set_ylabel('Address Count')
    ax3.grid(True, alpha=0.3)
    
    # 4. Network Graph Plot (Conditional)
    ax4 = plt.subplot(2, 3, 4)
    if G.number_of_edges() > 0:
        sub_nodes = [n for n in G.nodes() if n in suspicious_addrs]
        if not sub_nodes:
            sub_nodes = list(G.nodes())[:50]
        sub_G = G.subgraph(sub_nodes)
        pos = nx.spring_layout(sub_G, seed=42)
        nx.draw_networkx_nodes(sub_G, pos, node_color='#e74c3c', node_size=80, alpha=0.8, ax=ax4)
        nx.draw_networkx_edges(sub_G, pos, edge_color='#7f8c8d', alpha=0.6, arrows=True, ax=ax4)
        ax4.set_title(f'Heuristic Flow Subgraph ({sub_G.number_of_nodes()} nodes, {sub_G.number_of_edges()} edges)', fontsize=12, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, "No Linkable Flow Graph\n(Dataset contains isolated single-address records)", 
                 ha='center', va='center', fontsize=11, color='#7f8c8d',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7'))
        ax4.set_title('Transaction Flow Network', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 5. Temporal Patterns: Hour of Day Density
    ax5 = plt.subplot(2, 3, 5)
    df_susp = df[df['address_id'].isin(suspicious_addrs)]
    df_norm = df[~df['address_id'].isin(suspicious_addrs)]
    
    if not df_susp.empty and not df_norm.empty:
        sns.kdeplot(df_norm['timestamp'].dt.hour.dropna(), label='Normal', color='#3498db', fill=True, alpha=0.2, ax=ax5)
        sns.kdeplot(df_susp['timestamp'].dt.hour.dropna(), label='High-Risk', color='#e74c3c', fill=True, alpha=0.2, ax=ax5)
        ax5.set_xlim(0, 23)
    ax5.set_title('Hourly Activity Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Hour of Day (0 - 23)')
    ax5.set_ylabel('Density')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Cluster Characteristics Heatmap
    ax6 = plt.subplot(2, 3, 6)
    if cluster_results.get('cluster_characteristics'):
        rows = []
        for c_id, chars in cluster_results['cluster_characteristics'].items():
            rows.append({
                'Cluster': f"C{c_id}",
                'Size': chars.get('size', 0),
                'Avg Vol': chars.get('avg_transaction_volume', 0.0),
                'Avg Freq': chars.get('avg_transaction_frequency', 0.0),
                'Avg In-Deg': chars.get('avg_in_degree', 0.0)
            })
        c_df = pd.DataFrame(rows).set_index('Cluster')
        if not c_df.empty and len(c_df) > 1:
            norm_df = (c_df - c_df.min()) / (c_df.max() - c_df.min() + 1e-9)
            sns.heatmap(norm_df, annot=c_df.round(1), fmt='.1f', cmap='YlOrRd', ax=ax6, cbar=False)
            ax6.set_title('KMeans Profile Characteristics', fontsize=12, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, "Single Cluster Profile", ha='center', va='center')
    else:
        ax6.text(0.5, 0.5, "No Cluster Data Available", ha='center', va='center')
        
    plt.tight_layout()
    plot_path = output_dir / 'fraud_analysis_patterns.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Visualizations saved to {plot_path}")
    return plot_path

def generate_pdf_report(
    risk_scores_df: pd.DataFrame,
    explanations: Dict[str, List[Dict[str, Any]]],
    cluster_results: Dict[str, Any],
    embedding_clusters: Optional[Dict[str, Any]],
    plot_path: Path,
    output_dir: Path,
    benchmark_data: Optional[Dict[str, Any]] = None,
    graph_edges: int = 0
) -> Path:
    """
    Generate a rigorous, multi-page professional PDF report using ReportLab.
    """
    logger.info("Generating comprehensive PDF report...")
    output_path = output_dir / 'fraud_analysis_report.pdf'
    
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()
    
    title_style = styles['Title']
    h1_style = styles['Heading1']
    h2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    cell_style = ParagraphStyle(
        'SmallCell',
        parent=styles['Normal'],
        fontSize=8,
        leading=10
    )
    
    elements = []
    
    # Header & Title
    elements.append(Paragraph("DeCrypt: Advanced Anomaly Detection Dossier", title_style))
    elements.append(Paragraph(f"Analysis Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | Pipeline: Transformer + Graph + Isolation Forest", normal_style))
    elements.append(Spacer(1, 14))
    
    # 1. Methodology Summary
    elements.append(Paragraph("1. Methodology & Architecture Overview", h1_style))
    method_text = (
        "<b>DeCrypt</b> employs a multi-view anomaly detection architecture designed for financial and cryptocurrency transactions: "
        "<br/>• <b>Vectorized Behavioral Profiling:</b> Extracts net flow, pass-through ratio, Goh-Barabási burstiness, and Shannon amount entropy. "
        "<br/>• <b>Heuristic Flow Graph:</b> Traces temporal fund movements between withdrawals and matching deposits within a configurable time/amount window. "
        "<br/>• <b>BERT-Style Masked Event Transformer:</b> 2-layer Transformer encoder trained with 15% masked event modeling over transaction direction, binned amounts, and inter-arrival intervals. Anomaly score represents leave-one-out event surprise (NLL). "
        "<br/>• <b>Ensemble Scoring:</b> Combines Isolation Forest statistical deviance and Transformer surprise via percentile rank calibration. "
        "<br/>• <b>Dual Clustering:</b> Evaluates KMeans on tabular features alongside HDBSCAN on PCA-reduced sequence embeddings."
    )
    elements.append(Paragraph(method_text, normal_style))
    elements.append(Spacer(1, 12))
    
    # 2. Summary Metrics Table
    elements.append(Paragraph("2. Executive Summary", h1_style))
    n_total = len(risk_scores_df)
    n_high_risk = int((risk_scores_df['ensemble_risk_score'] >= 90.0).sum())
    n_insufficient = int(risk_scores_df['insufficient_history'].sum())
    
    summary_rows = [
        ['Metric', 'Value', 'Description'],
        ['Total Addresses Evaluated', str(n_total), 'Distinct blockchain/account entities'],
        ['High-Risk Addresses (Risk >= 90)', str(n_high_risk), 'Top decile flagged for investigation'],
        ['Insufficient History (<3 events)', str(n_insufficient), 'Assessed via Isolation Forest only'],
        ['Heuristic Graph Edges', str(graph_edges), 'Linkages found between withdrawals & deposits'],
        ['KMeans Clusters (Tabular)', f"k={cluster_results.get('optimal_clusters', 0)}", f"Silhouette: {cluster_results.get('silhouette_score', 0.0):.3f}"]
    ]
    if embedding_clusters:
        pref = embedding_clusters.get('preferred', 'kmeans')
        summary_rows.append([
            'Embedding Clustering (HDBSCAN)',
            f"{embedding_clusters.get('hdbscan_n_clusters', 0)} clusters",
            f"Silhouette: {embedding_clusters.get('hdbscan_silhouette', 0.0):.3f} (Preferred: {pref.upper()})"
        ])
        
    t_summary = Table(summary_rows, colWidths=[2.2*inch, 1.8*inch, 3.2*inch])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#ecf0f1')),
    ]))
    elements.append(t_summary)
    elements.append(Spacer(1, 14))
    
    # 3. Top-20 High-Risk Addresses with Top-3 Explainability Events
    elements.append(Paragraph("3. Top 20 High-Risk Entities & Event Explanations", h1_style))
    top20 = risk_scores_df.sort_values(by='ensemble_risk_score', ascending=False).head(20)
    
    table_rows = [
        ['Address', 'Ensemble Risk', 'IF Score', 'Surprise (NLL)', 'Top Surprising Events (Direction, Amt, NLL)']
    ]
    
    for addr, row in top20.iterrows():
        addr_str = str(addr)
        ens_score = f"{row['ensemble_risk_score']:.1f}"
        if_score = f"{row['if_risk_score']:.1f}"
        surp_score = f"{row['transformer_surprise']:.2f}" if not row['insufficient_history'] else "N/A (<3 txs)"
        
        # Explanations
        top_events = explanations.get(addr_str, [])
        if top_events:
            ev_strs = []
            for ev in top_events[:3]:
                ev_strs.append(f"• {ev['timestamp']} | {ev['direction'].upper()} {abs(ev['amount']):.1f} (NLL={ev['nll']})")
            ev_desc = "<br/>".join(ev_strs)
        else:
            ev_desc = "Single transaction or baseline activity."
            
        table_rows.append([
            Paragraph(f"<b>{addr_str[:16]}...</b>", cell_style),
            Paragraph(f"<b>{ens_score}</b>", cell_style),
            Paragraph(if_score, cell_style),
            Paragraph(surp_score, cell_style),
            Paragraph(ev_desc, cell_style)
        ])
        
    t_top20 = Table(table_rows, colWidths=[1.4*inch, 0.9*inch, 0.8*inch, 1.0*inch, 3.1*inch])
    t_top20.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#c0392b')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e74c3c')),
        ('VALIGN', (0,0), (-1,-1), 'TOP')
    ]))
    elements.append(t_top20)
    elements.append(Spacer(1, 14))
    
    # 4. Benchmark Results Table (if run)
    if benchmark_data and 'summary' in benchmark_data:
        elements.append(Paragraph("4. Synthetic Anomaly Benchmark Evaluation (5 Seeds)", h1_style))
        b_summary = benchmark_data['summary']
        b_rows = [
            ['Model', 'AUROC', 'AP', 'Recall@K', 'Structuring', 'Burst', 'Pass-Thru', 'Periodic', 'Spike']
        ]
        for m_name, vals in b_summary.items():
            b_rows.append([
                Paragraph(f"<b>{m_name}</b>", cell_style),
                Paragraph(f"{vals['auroc_mean']:.3f}±{vals['auroc_std']:.2f}", cell_style),
                Paragraph(f"{vals['ap_mean']:.3f}±{vals['ap_std']:.2f}", cell_style),
                Paragraph(f"{vals['recall_at_k_mean']:.3f}±{vals['recall_at_k_std']:.2f}", cell_style),
                Paragraph(f"{vals['archetype_recalls']['structuring']['mean']:.2f}", cell_style),
                Paragraph(f"{vals['archetype_recalls']['burst']['mean']:.2f}", cell_style),
                Paragraph(f"{vals['archetype_recalls']['pass_through']['mean']:.2f}", cell_style),
                Paragraph(f"{vals['archetype_recalls']['bot_periodic']['mean']:.2f}", cell_style),
                Paragraph(f"{vals['archetype_recalls']['dormant_spike']['mean']:.2f}", cell_style)
            ])
            
        t_bench = Table(b_rows, colWidths=[1.1*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.7*inch, 0.6*inch, 0.7*inch, 0.7*inch, 0.7*inch])
        t_bench.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#8e44ad')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#9b59b6')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('TOPPADDING', (0,0), (-1,-1), 3)
        ]))
        elements.append(t_bench)
        elements.append(Paragraph(f"<i>Caveat: {benchmark_data.get('caveat')}</i>", cell_style))
        elements.append(Spacer(1, 14))
        
    # 5. Visual Dashboard Image
    if plot_path.exists():
        elements.append(Paragraph("5. Visual Analytics Dashboard", h1_style))
        elements.append(Image(str(plot_path), width=7.2*inch, height=4.8*inch))
        elements.append(Spacer(1, 10))
        
    # Flow Graph Note
    if graph_edges == 0:
        elements.append(Paragraph(
            "<b>Note on Graph Analytics:</b> No heuristic flow linkages met the threshold window in this dataset. "
            "The system gracefully bypassed network centralities to avoid spurious zero-fill bias.",
            cell_style
        ))
        
    doc.build(elements)
    logger.info(f"PDF report successfully saved to {output_path}")
    return output_path
