import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def analyze_sweep_results(sweep_dir):
    """Analyze results from a parameter sweep directory"""
    
    # Load summary CSV
    summary_file = None
    for file in os.listdir(sweep_dir):
        if file.endswith('_summary.csv'):
            summary_file = os.path.join(sweep_dir, file)
            break
    
    if not summary_file:
        print(f"No summary CSV found in {sweep_dir}")
        return
    
    df = pd.read_csv(summary_file)
    
    # Filter out failed runs
    df_valid = df[df['best_val_f1'] >= 0].copy()
    
    if len(df_valid) == 0:
        print("No valid results found")
        return
    
    print(f"Analyzing {len(df_valid)} valid runs from {len(df)} total runs")
    
    # Create output directory for plots
    plots_dir = os.path.join(sweep_dir, "analysis_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Overall performance distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Parameter Sweep Results Analysis', fontsize=16)
    
    # F1 score distribution
    axes[0, 0].hist(df_valid['best_val_f1'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Best Validation F1 Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Best F1 Scores')
    axes[0, 0].axvline(df_valid['best_val_f1'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df_valid["best_val_f1"].mean():.4f}')
    axes[0, 0].legend()
    
    # AUC distribution
    axes[0, 1].hist(df_valid['final_val_auc'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Final Validation AUC')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Final AUC Scores')
    axes[0, 1].axvline(df_valid['final_val_auc'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df_valid["final_val_auc"].mean():.4f}')
    axes[0, 1].legend()
    
    # F1 vs AUC scatter
    axes[1, 0].scatter(df_valid['best_val_f1'], df_valid['final_val_auc'], alpha=0.6)
    axes[1, 0].set_xlabel('Best Validation F1 Score')
    axes[1, 0].set_ylabel('Final Validation AUC')
    axes[1, 0].set_title('F1 Score vs AUC Correlation')
    
    # Best epoch distribution
    axes[1, 1].hist(df_valid['best_epoch'], bins=range(1, int(df_valid['best_epoch'].max())+2), 
                    alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Best Epoch')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Best Epochs')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'overall_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Parameter-specific analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Parameter Impact Analysis', fontsize=16)
    
    # Weight decay impact
    for wd in sorted(df_valid['weight_decay'].unique()):
        subset = df_valid[df_valid['weight_decay'] == wd]
        axes[0, 0].scatter([wd] * len(subset), subset['best_val_f1'], alpha=0.6, label=f'WD={wd}')
    axes[0, 0].set_xlabel('Weight Decay')
    axes[0, 0].set_ylabel('Best Validation F1 Score')
    axes[0, 0].set_title('Weight Decay Impact on F1 Score')
    axes[0, 0].legend()
    
    # Entropy regularization impact
    for er in sorted(df_valid['entropy_reg_weight'].unique()):
        subset = df_valid[df_valid['entropy_reg_weight'] == er]
        axes[0, 1].scatter([er] * len(subset), subset['best_val_f1'], alpha=0.6, label=f'ER={er}')
    axes[0, 1].set_xlabel('Entropy Regularization Weight')
    axes[0, 1].set_ylabel('Best Validation F1 Score')
    axes[0, 1].set_title('Entropy Regularization Impact on F1 Score')
    axes[0, 1].legend()
    
    # Dropout rate impact
    for dr in sorted(df_valid['dropout_rate'].unique()):
        subset = df_valid[df_valid['dropout_rate'] == dr]
        axes[1, 0].scatter([dr] * len(subset), subset['best_val_f1'], alpha=0.6, label=f'DR={dr}')
    axes[1, 0].set_xlabel('Dropout Rate')
    axes[1, 0].set_ylabel('Best Validation F1 Score')
    axes[1, 0].set_title('Dropout Rate Impact on F1 Score')
    axes[1, 0].legend()
    
    # Attention dropout impact
    for adr in sorted(df_valid['attn_dropout_rate'].unique()):
        subset = df_valid[df_valid['attn_dropout_rate'] == adr]
        axes[1, 1].scatter([adr] * len(subset), subset['best_val_f1'], alpha=0.6, label=f'ADR={adr}')
    axes[1, 1].set_xlabel('Attention Dropout Rate')
    axes[1, 1].set_ylabel('Best Validation F1 Score')
    axes[1, 1].set_title('Attention Dropout Impact on F1 Score')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'parameter_impact.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of parameter combinations
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # F1 score heatmap for weight decay vs entropy regularization
    pivot_f1 = df_valid.pivot_table(
        values='best_val_f1', 
        index='weight_decay', 
        columns='entropy_reg_weight', 
        aggfunc='mean'
    )
    sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='viridis', ax=axes[0])
    axes[0].set_title('Mean F1 Score: Weight Decay vs Entropy Regularization')
    axes[0].set_xlabel('Entropy Regularization Weight')
    axes[0].set_ylabel('Weight Decay')
    
    # F1 score heatmap for dropout vs attention dropout
    pivot_f1_dropout = df_valid.pivot_table(
        values='best_val_f1', 
        index='dropout_rate', 
        columns='attn_dropout_rate', 
        aggfunc='mean'
    )
    sns.heatmap(pivot_f1_dropout, annot=True, fmt='.4f', cmap='viridis', ax=axes[1])
    axes[1].set_title('Mean F1 Score: Dropout Rate vs Attention Dropout Rate')
    axes[1].set_xlabel('Attention Dropout Rate')
    axes[1].set_ylabel('Dropout Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'parameter_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Top performing configurations
    top_configs = df_valid.nlargest(10, 'best_val_f1')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(top_configs)), top_configs['best_val_f1'])
    ax.set_yticks(range(len(top_configs)))
    ax.set_yticklabels([f"WD={row['weight_decay']}, ER={row['entropy_reg_weight']}, "
                       f"DR={row['dropout_rate']}, ADR={row['attn_dropout_rate']}" 
                       for _, row in top_configs.iterrows()])
    ax.set_xlabel('Best Validation F1 Score')
    ax.set_title('Top 10 Parameter Configurations')
    
    # Color bars by F1 score
    colors = plt.cm.viridis(top_configs['best_val_f1'] / top_configs['best_val_f1'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'top_configurations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Statistical summary
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)
    
    print(f"Total runs: {len(df)}")
    print(f"Successful runs: {len(df_valid)}")
    print(f"Failed runs: {len(df) - len(df_valid)}")
    
    print(f"\nF1 Score Statistics:")
    print(f"  Mean: {df_valid['best_val_f1'].mean():.4f}")
    print(f"  Std:  {df_valid['best_val_f1'].std():.4f}")
    print(f"  Min:  {df_valid['best_val_f1'].min():.4f}")
    print(f"  Max:  {df_valid['best_val_f1'].max():.4f}")
    
    print(f"\nAUC Score Statistics:")
    print(f"  Mean: {df_valid['final_val_auc'].mean():.4f}")
    print(f"  Std:  {df_valid['final_val_auc'].std():.4f}")
    print(f"  Min:  {df_valid['final_val_auc'].min():.4f}")
    print(f"  Max:  {df_valid['final_val_auc'].max():.4f}")
    
    # Parameter-wise statistics
    print(f"\nParameter-wise F1 Score Statistics:")
    for param in ['weight_decay', 'entropy_reg_weight', 'dropout_rate', 'attn_dropout_rate']:
        print(f"\n{param}:")
        for value in sorted(df_valid[param].unique()):
            subset = df_valid[df_valid[param] == value]
            print(f"  {value}: mean={subset['best_val_f1'].mean():.4f}, "
                  f"std={subset['best_val_f1'].std():.4f}, n={len(subset)}")
    
    # Save detailed analysis
    analysis_results = {
        'summary_stats': {
            'total_runs': len(df),
            'successful_runs': len(df_valid),
            'failed_runs': len(df) - len(df_valid),
            'f1_mean': df_valid['best_val_f1'].mean(),
            'f1_std': df_valid['best_val_f1'].std(),
            'f1_min': df_valid['best_val_f1'].min(),
            'f1_max': df_valid['best_val_f1'].max(),
            'auc_mean': df_valid['final_val_auc'].mean(),
            'auc_std': df_valid['final_val_auc'].std(),
            'auc_min': df_valid['final_val_auc'].min(),
            'auc_max': df_valid['final_val_auc'].max(),
        },
        'top_5_configurations': top_configs.head(5).to_dict('records'),
        'parameter_analysis': {}
    }
    
    for param in ['weight_decay', 'entropy_reg_weight', 'dropout_rate', 'attn_dropout_rate']:
        analysis_results['parameter_analysis'][param] = {}
        for value in sorted(df_valid[param].unique()):
            subset = df_valid[df_valid[param] == value]
            analysis_results['parameter_analysis'][param][str(value)] = {
                'mean_f1': subset['best_val_f1'].mean(),
                'std_f1': subset['best_val_f1'].std(),
                'count': len(subset)
            }
    
    with open(os.path.join(plots_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    print(f"\nAnalysis complete! Results saved to: {plots_dir}")
    
    return analysis_results

def main():
    """Main function to analyze sweep results"""
    
    # Look for sweep directories
    output_dirs = ['./parameter_sweep_output', './focused_sweep_output']
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            print(f"Found output directory: {output_dir}")
            
            # Find sweep directories
            sweep_dirs = []
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path) and (item.startswith('sweep_') or item.startswith('focused_sweep_')):
                    sweep_dirs.append(item_path)
            
            if sweep_dirs:
                print(f"Found {len(sweep_dirs)} sweep directories:")
                for sweep_dir in sweep_dirs:
                    print(f"  - {sweep_dir}")
                
                # Analyze the most recent sweep
                latest_sweep = max(sweep_dirs, key=os.path.getctime)
                print(f"\nAnalyzing latest sweep: {latest_sweep}")
                analyze_sweep_results(latest_sweep)
            else:
                print(f"No sweep directories found in {output_dir}")
        else:
            print(f"Output directory not found: {output_dir}")

if __name__ == "__main__":
    main() 