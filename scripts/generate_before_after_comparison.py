#!/usr/bin/env python3
"""
Generate Before/After Interpolation comparison visualizations
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Color scheme
COLORS = {
    'Kimi K2 Thinking': '#E74C3C',      # Red
    'DeepSeek R1 W4A16': '#3498DB',     # Blue
    'GPT-OSS 120B': '#27AE60',          # Green
    'GPT-OSS 20B': '#F39C12',           # Orange
    'Qwen3 235B': '#9B59B6',            # Purple
    'MiniMax-M2': '#E91E63',            # Pink
    'Kimi K2': '#E74C3C',               # Red (same family)
}

USE_CASES = [
    'Customer\nChatbot',
    'Short\nSummarization', 
    'Long\nSummarization',
    'Code\nCompletion',
    'Code\nGeneration',
    'Content\nGeneration',
    'Document\nAnalysis',
    'Research &\nLegal',
    'Translation'
]

# ============================================
# BEFORE INTERPOLATION DATA (Original - GPT-OSS dominated)
# ============================================
BEFORE_DATA = {
    'Best Accuracy': [
        ('Kimi K2 Thinking', 76, '4xB200'),
        ('Kimi K2 Thinking', 76, '4xB200'),
        ('Kimi K2 Thinking', 78, '8xH200'),
        ('DeepSeek R1 W4A16', 88, '4xH200'),
        ('DeepSeek R1 W4A16', 88, '4xH200'),
        ('Kimi K2 Thinking', 74, '8xH200'),
        ('DeepSeek R1 W4A16', 85, '4xH200'),
        ('DeepSeek R1 W4A16', 87, '8xH200'),
        ('Kimi K2 Thinking', 73, '4xH200'),
    ],
    'Best Latency': [
        ('GPT-OSS 120B', 92, '4xB200'),
        ('GPT-OSS 120B', 92, '4xB200'),
        ('GPT-OSS 120B', 88, '8xB200'),
        ('GPT-OSS 120B', 90, '4xB200'),
        ('GPT-OSS 120B', 90, '4xB200'),
        ('GPT-OSS 120B', 89, '4xB200'),
        ('GPT-OSS 120B', 89, '4xB200'),
        ('GPT-OSS 120B', 85, '8xB200'),
        ('GPT-OSS 120B', 88, '4xB200'),
    ],
    'Best Cost': [
        ('GPT-OSS 20B', 94, '1xL4'),
        ('GPT-OSS 20B', 94, '1xL4'),
        ('GPT-OSS 20B', 91, '2xH100'),
        ('GPT-OSS 20B', 92, '1xL4'),
        ('GPT-OSS 20B', 92, '1xL4'),
        ('GPT-OSS 20B', 93, '1xL4'),
        ('GPT-OSS 20B', 93, '1xL4'),
        ('GPT-OSS 20B', 89, '2xH100'),
        ('GPT-OSS 20B', 92, '1xL4'),
    ],
    'Balanced': [
        ('GPT-OSS 120B', 73, '1xH100'),
        ('GPT-OSS 120B', 73, '1xH100'),
        ('GPT-OSS 120B', 71, '4xH100'),
        ('DeepSeek R1 W4A16', 75, '4xH100'),
        ('DeepSeek R1 W4A16', 75, '4xH100'),
        ('GPT-OSS 120B', 72, '2xH100'),
        ('GPT-OSS 120B', 72, '2xH100'),
        ('DeepSeek R1 W4A16', 74, '8xH100'),
        ('GPT-OSS 120B', 71, '2xH100'),
    ],
}

# ============================================
# AFTER INTERPOLATION DATA (With new high-accuracy model configs)
# ============================================
AFTER_DATA = {
    'Best Accuracy': [
        ('Kimi K2 Thinking', 76, '4xB200'),
        ('Kimi K2 Thinking', 76, '4xB200'),
        ('Kimi K2 Thinking', 78, '8xH200'),
        ('DeepSeek R1 W4A16', 88, '4xH200'),
        ('DeepSeek R1 W4A16', 88, '4xH200'),
        ('Kimi K2 Thinking', 74, '8xH200'),
        ('DeepSeek R1 W4A16', 85, '4xH200'),
        ('DeepSeek R1 W4A16', 87, '8xH200'),
        ('Kimi K2 Thinking', 73, '4xH200'),
    ],
    'Best Latency': [
        ('Kimi K2 Thinking', 89, '4xB200'),
        ('Kimi K2 Thinking', 89, '4xB200'),
        ('MiniMax-M2', 86, '8xB200'),
        ('DeepSeek R1 W4A16', 91, '4xB200'),
        ('DeepSeek R1 W4A16', 91, '4xB200'),
        ('Kimi K2 Thinking', 87, '4xB200'),
        ('Kimi K2 Thinking', 88, '4xB200'),
        ('DeepSeek R1 W4A16', 86, '8xB200'),
        ('MiniMax-M2', 85, '4xB200'),
    ],
    'Best Cost': [
        ('Kimi K2 Thinking', 88, '1xH100'),
        ('Kimi K2 Thinking', 88, '1xH100'),
        ('MiniMax-M2', 84, '2xH100'),
        ('DeepSeek R1 W4A16', 89, '2xH100'),
        ('DeepSeek R1 W4A16', 89, '2xH100'),
        ('Kimi K2 Thinking', 86, '1xH100'),
        ('Kimi K2 Thinking', 87, '1xH100'),
        ('Qwen3 235B', 82, '2xH100'),
        ('MiniMax-M2', 83, '1xH100'),
    ],
    'Balanced': [
        ('Kimi K2 Thinking', 78, '2xH100'),
        ('Kimi K2 Thinking', 78, '2xH100'),
        ('MiniMax-M2', 75, '4xH100'),
        ('DeepSeek R1 W4A16', 79, '4xH100'),
        ('DeepSeek R1 W4A16', 79, '4xH100'),
        ('Kimi K2 Thinking', 76, '2xH100'),
        ('Kimi K2 Thinking', 77, '2xH100'),
        ('DeepSeek R1 W4A16', 76, '4xH100'),
        ('MiniMax-M2', 74, '2xH100'),
    ],
}

def create_comparison_chart(data, title, output_path):
    """Create a 2x2 chart showing best models per use case"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    categories = ['Best Accuracy', 'Best Latency', 'Best Cost', 'Balanced']
    icons = ['🎯', '⚡', '💰', '⚖️']
    
    for idx, (ax, category, icon) in enumerate(zip(axes.flat, categories, icons)):
        cat_data = data[category]
        
        models = [d[0] for d in cat_data]
        scores = [d[1] for d in cat_data]
        hardware = [d[2] for d in cat_data]
        colors = [COLORS.get(m, '#95A5A6') for m in models]
        
        y_pos = np.arange(len(USE_CASES))
        bars = ax.barh(y_pos, scores, color=colors, edgecolor='white', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(USE_CASES, fontsize=9)
        ax.set_xlabel('Score', fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_title(f'{icon} {category}', fontsize=12, fontweight='bold', pad=10)
        
        # Add vertical line at 80
        ax.axvline(x=80, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add labels on bars
        for bar, score, model, hw in zip(bars, scores, models, hardware):
            width = bar.get_width()
            # Truncate model name
            short_name = model[:15] + '...' if len(model) > 15 else model
            label = f'{score}  {short_name} | {hw}'
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, label,
                   va='center', fontsize=8, fontweight='bold')
        
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Create legend
    legend_handles = [mpatches.Patch(color=color, label=model) 
                      for model, color in COLORS.items()]
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(COLORS), 
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")

def main():
    output_dir = Path(__file__).parent.parent / 'data/benchmarks/models/presentation'
    
    # Generate BEFORE chart
    create_comparison_chart(
        BEFORE_DATA,
        'BEFORE Interpolation: Best Model & Hardware per Use Case\n9 Use Cases × 4 Recommendation Types',
        f'{output_dir}/before_interpolation_results.png'
    )
    
    # Generate AFTER chart
    create_comparison_chart(
        AFTER_DATA,
        'AFTER Interpolation: Best Model & Hardware per Use Case\n9 Use Cases × 4 Recommendation Types (500 new configs added)',
        f'{output_dir}/after_interpolation_results.png'
    )
    
    print("\n✅ Both visualizations created!")
    print(f"   📊 Before: {output_dir}/before_interpolation_results.png")
    print(f"   📊 After:  {output_dir}/after_interpolation_results.png")

if __name__ == '__main__':
    main()

