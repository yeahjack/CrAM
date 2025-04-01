import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dcor  # distance correlation
from matplotlib.backends.backend_pdf import PdfPages

# -------------------------------
# 读取 JSON 数据
# -------------------------------
def load_json_data(file_path):
    """从文件中读取并返回 JSON 列表。"""
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试查找JSON数组
            content = file.read()
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
            else:
                raise ValueError("无法在文件中找到有效的JSON数据")


# -------------------------------
# 提取 doc-level utility 指标
# -------------------------------
def extract_doc_level_utility(data):
    """
    仅提取 doc-level 的 utility 指标:
    - entropy_reduction
    - perplexity_reduction
    - seper_reduction (ground truth)
    - semantic_entropy_reduction
    - retriever_score
    - reranker_score
    """
    rows = []
    for entry in data:
        q_id = entry['id']

        # 遍历每个 retrieved doc
        if 'individual_doc_results' in entry:
            for docres in entry['individual_doc_results']:
                # 只保留 doc-level utility
                doc_u = docres['utility']
                row = {
                    'question_id': q_id,
                    'doc_entropy_reduction': doc_u['entropy_reduction'],
                    'doc_perplexity_reduction': doc_u['perplexity_reduction'],
                    'doc_seper_reduction': doc_u['seper_reduction'],
                    'doc_semantic_entropy_reduction': doc_u['semantic_entropy_reduction'],
                    'doc_retriever_score': doc_u.get('retriever_score', np.nan),
                    'doc_reranker_score': doc_u.get('reranker_score', np.nan)
                }
                rows.append(row)

    return pd.DataFrame(rows)

# -------------------------------
# 计算 Distance Correlation 矩阵
# -------------------------------
def distance_corr_matrix(df):
    """
    对 df 内所有列，两两计算 distance correlation。
    返回一个对称矩阵（DataFrame）。
    """
    cols = df.columns
    mat = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)

    for i in cols:
        for j in cols:
            mat.loc[i, j] = dcor.distance_correlation(df[i], df[j])
    return mat

# -------------------------------
# 构建三种相关性的字典
# -------------------------------
def build_all_corr(df):
    """
    给定一个数值型 DataFrame，
    返回 { 'pearson':..., 'spearman':..., 'distance':... }
    """
    return {
        'pearson': df.corr(method='pearson'),
        'spearman': df.corr(method='spearman'),
        'distance': distance_corr_matrix(df)
    }

# -------------------------------
# 绘图函数：相关性热图，强调与seper_reduction的关系
# -------------------------------
def plot_heatmap(corr_matrix, title, ax, highlight_col='doc_seper_reduction'):
    """
    corr_matrix: DataFrame
    title: 标题
    ax: matplotlib Axes
    highlight_col: 要强调的列（默认为doc_seper_reduction）
    """
    # Format labels - English
    corr_matrix = corr_matrix.copy()
    corr_matrix.index = [idx.replace('doc_', '').replace('_', ' ').title() for idx in corr_matrix.index]
    corr_matrix.columns = [col.replace('doc_', '').replace('_', ' ').title() for col in corr_matrix.columns]
    
    # 将highlight_col也格式化
    highlight_col_formatted = highlight_col.replace('doc_', '').replace('_', ' ').title()
    
    # Distance correlation range [0,1], others [-1,1]
    is_distance = ('distance' in title.lower())
    vmin = 0 if is_distance else -1
    vmax = 1
    center = None if is_distance else 0
    
    # Custom color maps
    cmap = sns.diverging_palette(230, 20, as_cmap=True) if not is_distance else "YlGnBu"
    
    # Create heatmap
    hm = sns.heatmap(
        corr_matrix,
        annot=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        linewidths=1.0,
        linecolor='black',
        annot_kws={"size": 8},
        fmt=".2f",
        ax=ax,
        cbar_kws={"shrink": .8},
        square=True
    )
    
    # 高亮显示与 seper_reduction 相关的行和列
    if highlight_col_formatted in corr_matrix.columns:
        # 找到highlight_col的索引
        col_idx = corr_matrix.columns.get_loc(highlight_col_formatted)
        row_idx = corr_matrix.index.get_loc(highlight_col_formatted)
        
        # 高亮列
        for i in range(len(corr_matrix)):
            rect = plt.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor='red', lw=2)
            ax.add_patch(rect)
        
        # 高亮行
        for i in range(len(corr_matrix.columns)):
            rect = plt.Rectangle((i, row_idx), 1, 1, fill=False, edgecolor='red', lw=2)
            ax.add_patch(rect)
    
    # Title and appearance
    ax.set_title(title, fontweight='bold', fontsize=12, pad=10)
    
    # Adjust tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    return ax

# -------------------------------
# 专门分析与 seper_reduction 的关系
# -------------------------------
def plot_seper_reduction_correlations(df, output_pdf):
    """
    专门绘制各指标与 seper_reduction 的相关性图表
    """
    # 确保只处理数值列
    df = df.select_dtypes(include=[np.number])
    
    # 如果没有 doc_seper_reduction 列，返回
    if 'doc_seper_reduction' not in df.columns:
        print("错误: 数据中没有 doc_seper_reduction 列")
        return
    
    # 计算与 seper_reduction 的相关性
    pearson_corrs = {}
    spearman_corrs = {}
    distance_corrs = {}
    
    target_col = 'doc_seper_reduction'
    
    # 计算各指标与 seper_reduction 的相关性
    for col in df.columns:
        if col != target_col:
            # Pearson correlation
            pearson_corrs[col] = df[col].corr(df[target_col], method='pearson')
            
            # Spearman correlation
            spearman_corrs[col] = df[col].corr(df[target_col], method='spearman')
            
            # Distance correlation
            distance_corrs[col] = dcor.distance_correlation(df[col], df[target_col])
    
    # 创建数据框用于绘图
    corr_data = pd.DataFrame({
        'Metric': [col.replace('doc_', '').replace('_', ' ').title() for col in pearson_corrs.keys()],
        'Pearson': list(pearson_corrs.values()),
        'Spearman': list(spearman_corrs.values()),
        'Distance': list(distance_corrs.values())
    })
    
    # 排序，按Pearson相关性绝对值降序
    corr_data = corr_data.reindex(corr_data['Pearson'].abs().sort_values(ascending=False).index)
    
    # 绘制相关性条形图
    with PdfPages(output_pdf) as pdf:
        # 添加标题页 (放在最前面)
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle("Doc-Level Utility Correlation Analysis", fontsize=20, fontweight='bold')
        plt.figtext(0.5, 0.5, "Focus on SEPER Reduction as Ground Truth", 
                   ha='center', fontsize=16, fontweight='bold')
        plt.figtext(0.5, 0.4, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", 
                   ha='center', fontsize=12)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # 只有一页，专注于与seper_reduction的关系
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle("Correlations with Seper Reduction (Ground Truth)", fontsize=16, fontweight='bold')
        
        # Pearson correlation
        ax = axes[0]
        bars = ax.barh(corr_data['Metric'], corr_data['Pearson'], color='#3498db')
        ax.set_title("Pearson Correlation", fontweight='bold')
        ax.set_xlim([-1, 1])
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            value = corr_data['Pearson'].iloc[i]
            text_color = 'black'
            ax.text(value + (0.05 if value < 0 else -0.05), 
                   i, 
                   f"{value:.2f}", 
                   va='center', 
                   ha='right' if value > 0 else 'left',
                   color=text_color,
                   fontweight='bold')
        
        # Spearman correlation
        ax = axes[1]
        bars = ax.barh(corr_data['Metric'], corr_data['Spearman'], color='#2ecc71')
        ax.set_title("Spearman Correlation", fontweight='bold')
        ax.set_xlim([-1, 1])
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            value = corr_data['Spearman'].iloc[i]
            text_color = 'black'
            ax.text(value + (0.05 if value < 0 else -0.05), 
                   i, 
                   f"{value:.2f}", 
                   va='center', 
                   ha='right' if value > 0 else 'left',
                   color=text_color,
                   fontweight='bold')
        
        # Distance correlation
        ax = axes[2]
        bars = ax.barh(corr_data['Metric'], corr_data['Distance'], color='#9b59b6')
        ax.set_title("Distance Correlation", fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            value = corr_data['Distance'].iloc[i]
            text_color = 'black'
            ax.text(value - 0.05, i, f"{value:.2f}", va='center', ha='right', color=text_color, fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(dpi=300)
        plt.close()
        
        # 添加相关性热图
        all_corrs = build_all_corr(df)
        
        # 绘制热图
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle("Doc-Level Utility Correlation Matrices", fontsize=16, fontweight='bold')
        
        # 获取三种相关性矩阵
        pearson_matrix = all_corrs.get('pearson')
        spearman_matrix = all_corrs.get('spearman')
        distance_matrix = all_corrs.get('distance')
        
        # 绘制每种相关性热图，突出显示与seper_reduction的关系
        if pearson_matrix is not None and not pearson_matrix.empty:
            plot_heatmap(pearson_matrix, "Pearson", axes[0])
        
        if spearman_matrix is not None and not spearman_matrix.empty:
            plot_heatmap(spearman_matrix, "Spearman", axes[1])
        
        if distance_matrix is not None and not distance_matrix.empty:
            plot_heatmap(distance_matrix, "Distance", axes[2])
        
        # 添加脚注
        footnotes = [
            "Pearson: Linear relationship [-1,1]",
            "Spearman: Monotonic relationship [-1,1]",
            "Distance: Non-linear dependency [0,1]"
        ]
        
        for i, note in enumerate(footnotes):
            axes[i].set_xlabel(note, fontsize=8)
        
        plt.tight_layout()
        pdf.savefig(dpi=300)
        plt.close()
        
    return corr_data

# -------------------------------
# 生成简化的文本报告，重点关注 seper_reduction
# -------------------------------
def generate_seper_report(df):
    """
    生成专注于 seper_reduction 关系的报告
    """
    # 确保只处理数值列
    df = df.select_dtypes(include=[np.number])
    
    # 如果没有 doc_seper_reduction 列，返回错误信息
    if 'doc_seper_reduction' not in df.columns:
        return "错误: 数据中没有 doc_seper_reduction 列"
    
    # 计算与 seper_reduction 的相关性
    pearson_corrs = {}
    spearman_corrs = {}
    distance_corrs = {}
    
    target_col = 'doc_seper_reduction'
    
    # 计算各指标与 seper_reduction 的相关性
    for col in df.columns:
        if col != target_col:
            # Pearson correlation
            pearson_corrs[col] = df[col].corr(df[target_col], method='pearson')
            
            # Spearman correlation
            spearman_corrs[col] = df[col].corr(df[target_col], method='spearman')
            
            # Distance correlation
            distance_corrs[col] = dcor.distance_correlation(df[col], df[target_col])
    
    # 生成报告
    report = ["# SEPER Reduction Correlation Analysis\n"]
    report.append("## Correlations with SEPER Reduction (Ground Truth)\n")
    
    # Pearson correlations
    report.append("### Pearson Correlations (Linear Relationship)")
    for col, value in sorted(pearson_corrs.items(), key=lambda x: abs(x[1]), reverse=True):
        strength = _cor_strength(value)
        formatted_col = col.replace('doc_', '').replace('_', ' ').title()
        report.append(f"- **{formatted_col}**: {value:.4f}{strength}")
    
    # Spearman correlations
    report.append("\n### Spearman Correlations (Monotonic Relationship)")
    for col, value in sorted(spearman_corrs.items(), key=lambda x: abs(x[1]), reverse=True):
        strength = _cor_strength(value)
        formatted_col = col.replace('doc_', '').replace('_', ' ').title()
        report.append(f"- **{formatted_col}**: {value:.4f}{strength}")
    
    # Distance correlations
    report.append("\n### Distance Correlations (Non-linear Dependency)")
    for col, value in sorted(distance_corrs.items(), key=lambda x: x[1], reverse=True):
        strength = _cor_distance_strength(value)
        formatted_col = col.replace('doc_', '').replace('_', ' ').title()
        report.append(f"- **{formatted_col}**: {value:.4f}{strength}")
    
    # 分析检索器得分与seper_reduction的关系
    report.append("\n## Retriever and Reranker Score Analysis\n")
    
    if 'doc_retriever_score' in pearson_corrs:
        retriever_pearson = pearson_corrs['doc_retriever_score']
        retriever_spearman = spearman_corrs['doc_retriever_score']
        retriever_distance = distance_corrs['doc_retriever_score']
        
        report.append("### Retriever Score")
        report.append(f"- Pearson: {retriever_pearson:.4f}{_cor_strength(retriever_pearson)}")
        report.append(f"- Spearman: {retriever_spearman:.4f}{_cor_strength(retriever_spearman)}")
        report.append(f"- Distance: {retriever_distance:.4f}{_cor_distance_strength(retriever_distance)}")
    
    if 'doc_reranker_score' in pearson_corrs:
        reranker_pearson = pearson_corrs['doc_reranker_score']
        reranker_spearman = spearman_corrs['doc_reranker_score']
        reranker_distance = distance_corrs['doc_reranker_score']
        
        report.append("\n### Reranker Score")
        report.append(f"- Pearson: {reranker_pearson:.4f}{_cor_strength(reranker_pearson)}")
        report.append(f"- Spearman: {reranker_spearman:.4f}{_cor_strength(reranker_spearman)}")
        report.append(f"- Distance: {reranker_distance:.4f}{_cor_distance_strength(reranker_distance)}")
    
    # 总结
    report.append("\n## Summary\n")
    
    # 找出与seper_reduction相关性最强的指标
    best_pearson = max(pearson_corrs.items(), key=lambda x: abs(x[1]))
    best_spearman = max(spearman_corrs.items(), key=lambda x: abs(x[1]))
    best_distance = max(distance_corrs.items(), key=lambda x: x[1])
    
    report.append(f"- The strongest linear correlation (Pearson) with SEPER Reduction is from **{best_pearson[0].replace('doc_', '').replace('_', ' ').title()}** ({best_pearson[1]:.4f}).")
    report.append(f"- The strongest rank correlation (Spearman) with SEPER Reduction is from **{best_spearman[0].replace('doc_', '').replace('_', ' ').title()}** ({best_spearman[1]:.4f}).")
    report.append(f"- The strongest non-linear dependency (Distance) with SEPER Reduction is from **{best_distance[0].replace('doc_', '').replace('_', ' ').title()}** ({best_distance[1]:.4f}).")
    
    # 检索器和重排器分析
    if 'doc_retriever_score' in pearson_corrs and 'doc_reranker_score' in pearson_corrs:
        retriever_pearson = abs(pearson_corrs['doc_retriever_score'])
        reranker_pearson = abs(pearson_corrs['doc_reranker_score'])
        
        if retriever_pearson > reranker_pearson:
            report.append(f"\nThe Retriever Score ({retriever_pearson:.4f}) has a stronger correlation with SEPER Reduction than the Reranker Score ({reranker_pearson:.4f}).")
        elif reranker_pearson > retriever_pearson:
            report.append(f"\nThe Reranker Score ({reranker_pearson:.4f}) has a stronger correlation with SEPER Reduction than the Retriever Score ({retriever_pearson:.4f}).")
        else:
            report.append(f"\nThe Retriever Score and Reranker Score have equal correlation with SEPER Reduction ({retriever_pearson:.4f}).")
    
    return "\n".join(report)

# -------------------------------
# 描述相关性强度的辅助函数
# -------------------------------
def _cor_strength(value):
    """返回描述相关性强度的标签"""
    abs_val = abs(value)
    if abs_val < 0.3:
        return " (weak)"
    elif abs_val < 0.5:
        return " (moderate)"
    elif abs_val < 0.7:
        return " (strong)"
    else:
        return " (very strong)"

def _cor_distance_strength(value):
    """返回描述距离相关性强度的标签"""
    if value < 0.3:
        return " (weak)"
    elif value < 0.5:
        return " (moderate)"
    elif value < 0.7:
        return " (strong)"
    else:
        return " (very strong)"

# -------------------------------
# 主函数
# -------------------------------
def main(file_path, output_pdf):
    # 设置可视化样式
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    sns.set_theme(style="white", palette="muted", font_scale=1.0)

    print("加载数据...")
    data = load_json_data(file_path)

    print("提取文档级别的utility指标...")
    doc_df = extract_doc_level_utility(data)

    print("绘制与seper_reduction相关的图表...")
    plot_seper_reduction_correlations(doc_df, output_pdf)

    # 生成文本报告
    print("生成文本报告...")
    report_text = generate_seper_report(doc_df)
    report_file = output_pdf.replace('.pdf', '_seper_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"完成!\n- 图表已保存到: {output_pdf}\n- 报告已保存到: {report_file}")

# -------------------------------
# CLI 入口
# -------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "seper_correlation_analysis.pdf"
    else:
        input_file = "/home/yijiexu/AttentionBiasForLongReasoning/CrAM/data/nq_2024_11_07_12_14_naive/results/rag_evaluation_results.json"
        output_file = "seper_correlation_analysis.pdf"

    main(input_file, output_file)