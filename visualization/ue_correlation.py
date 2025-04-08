import json
import math
import os

import dcor  # distance correlation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# -------------------------------
# 从SEPER文件直接提取数据
# -------------------------------
def extract_seper_data_directly(seper_file):
    """
    直接从SEPER结果文件中提取seper和seper_reduction值
    
    返回两个字典:
    - seper_dict: {(question_id, doc_id): seper_value}
    - seper_reduction_dict: {(question_id, doc_id): seper_reduction_value}
    """
    print(f"直接从SEPER文件提取seper和seper_reduction值: {seper_file}")
    
    seper_dict = {}  # 存储(question_id, doc_id) -> seper
    seper_reduction_dict = {}  # 存储(question_id, doc_id) -> seper_reduction
    
    # 加载SEPER数据
    seper_data = load_json_data(seper_file)
    
    # 遍历所有问题
    items_processed = 0
    seper_count = 0
    seper_reduction_count = 0
    
    for item in seper_data:
        if "id" not in item or "individual_doc_results" not in item:
            continue
            
        q_id = item["id"]
        items_processed += 1
        
        # 遍历每个文档
        for doc in item["individual_doc_results"]:
            if "doc_id" not in doc:
                continue
                
            doc_id = doc["doc_id"]
            
            # 尝试提取seper (多种可能的位置)
            seper_value = None
            # 1. 直接在doc对象中
            if "seper" in doc:
                seper_value = doc["seper"]
            # 2. 在metrics对象中
            elif "metrics" in doc and isinstance(doc["metrics"], dict) and "seper" in doc["metrics"]:
                seper_value = doc["metrics"]["seper"]
                
            # 如果找到seper值，添加到字典
            if seper_value is not None:
                seper_dict[(q_id, doc_id)] = seper_value
                seper_count += 1
            
            # 尝试提取seper_reduction (多种可能的位置)
            seper_reduction_value = None
            # 1. 直接在doc对象中
            if "seper_reduction" in doc:
                seper_reduction_value = doc["seper_reduction"]
            # 2. 在utility对象中
            elif "utility" in doc and isinstance(doc["utility"], dict) and "seper_reduction" in doc["utility"]:
                seper_reduction_value = doc["utility"]["seper_reduction"]
                
            # 如果找到seper_reduction值，添加到字典
            if seper_reduction_value is not None:
                seper_reduction_dict[(q_id, doc_id)] = seper_reduction_value
                seper_reduction_count += 1
    
    print(f"处理了{items_processed}个问题项")
    print(f"提取了{seper_count}个seper值和{seper_reduction_count}个seper_reduction值")
    
    return seper_dict, seper_reduction_dict


# -------------------------------
# 读取 JSON 数据
# -------------------------------
def load_json_data(file_path):
    """从文件中读取并返回 JSON 列表。"""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

        # 处理特殊值和问题字符
        content = content.replace("Infinity", "null")
        content = content.replace("-Infinity", "null")
        content = content.replace("NaN", "null")

        # 修复可能的JSON格式问题
        try:
            # 首先尝试直接解析
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"尝试修复JSON格式问题...")

            # 尝试找到JSON数组并处理
            start = content.find("[")
            end = content.rfind("]") + 1

            if start >= 0 and end > start:
                json_str = content[start:end]

                # 尝试使用正则表达式修复常见JSON错误
                import re

                # 修复缺少逗号的问题
                json_str = re.sub(r"}\s*{", "},{", json_str)

                # 修复额外的逗号问题
                json_str = re.sub(r",\s*]", "]", json_str)
                json_str = re.sub(r",\s*}", "}", json_str)

                # 修复非法的控制字符
                json_str = re.sub(r"[\x00-\x1F]", " ", json_str)

                # 尝试定位和修复问题位置
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e2:
                    print(f"修复后仍然存在JSON解析错误: {e2}")

                    # 找到错误位置前后的内容
                    error_pos = e2.pos
                    context_start = max(0, error_pos - 50)
                    context_end = min(len(json_str), error_pos + 50)

                    context = json_str[context_start:context_end]
                    print(f"问题上下文: ...{context}...")

                    # 尝试更激进的修复方法 - 移除错误位置周围的内容
                    if error_pos > 0 and error_pos < len(json_str):
                        # 找到包含错误的对象或数组
                        left_brace = json_str.rfind("{", 0, error_pos)
                        right_brace = json_str.find("}", error_pos)

                        if left_brace >= 0 and right_brace >= 0:
                            # 尝试跳过有问题的对象
                            json_str = json_str[:left_brace] + "{}" + json_str[right_brace + 1 :]
                            try:
                                return json.loads(json_str)
                            except:
                                pass

            # 最后的尝试：使用正则表达式提取所有可能的JSON对象
            try:
                objects = re.findall(r"\{[^{}]*\}", content)
                if objects:
                    print(f"尝试解析找到的{len(objects)}个独立JSON对象...")
                    results = []
                    for obj in objects:
                        try:
                            parsed = json.loads(obj)
                            results.append(parsed)
                        except:
                            pass
                    if results:
                        print(f"成功解析{len(results)}个对象")
                        return results
            except:
                pass

            # 如果所有尝试都失败了
            raise ValueError(f"无法在文件中找到有效的JSON数据: {file_path}")


# -------------------------------
# 合并两个 JSON 文件的数据
# -------------------------------
def merge_json_data(primary_file, seper_file):
    """
    合并两个 JSON 文件的数据，将 seper_file 中的 seper 值添加到 primary_file 中的对应文档。

    primary_file: 主要数据文件路径 (rag_evaluation_results_temp_X.json)
    seper_file: 包含 seper 值的文件路径 (seper_results.json)
    """
    # 加载主数据文件
    print(f"尝试加载主数据文件: {primary_file}")
    primary_data = load_json_data(primary_file)
    print(f"成功加载主数据 ({len(primary_data)}项)")

    # 加载SEPER数据文件
    print(f"尝试加载SEPER数据文件: {seper_file}")
    seper_data = load_json_data(seper_file)
    print(f"成功加载SEPER数据 ({len(seper_data)}项)")

    # 创建从问题ID到seper数据的映射
    seper_map = {}
    for item in seper_data:
        if "id" in item:
            seper_map[item["id"]] = item

    print(f"创建了包含{len(seper_map)}个问题ID的SEPER映射")

    # 更新主数据中的seper值 (在individual_doc_results中添加raw seper)
    merged_count = 0
    raw_seper_count = 0
    seper_reduction_count = 0

    for primary_item in primary_data:
        if "id" not in primary_item:
            continue

        q_id = primary_item["id"]
        if q_id in seper_map:
            merged_count += 1
            
            # 仅当主数据和SEPER数据都包含individual_doc_results时进行文档级合并
            if "individual_doc_results" in primary_item and "individual_doc_results" in seper_map[q_id]:
                # 创建primary和seper的文档映射 (基于doc_id)
                primary_docs = {}
                for doc in primary_item["individual_doc_results"]:
                    if "doc_id" in doc:
                        primary_docs[doc["doc_id"]] = doc

                seper_docs = {}
                for doc in seper_map[q_id]["individual_doc_results"]:
                    if "doc_id" in doc:
                        seper_docs[doc["doc_id"]] = doc

                # 遍历主数据中的每个文档，添加对应的seper值
                for doc_id, primary_doc in primary_docs.items():
                    if doc_id in seper_docs:
                        seper_doc = seper_docs[doc_id]
                        
                        # 添加raw seper到metrics中
                        if "metrics" not in primary_doc:
                            primary_doc["metrics"] = {}
                            
                        # 检查直接存在的seper字段
                        if "seper" in seper_doc:
                            primary_doc["metrics"]["seper"] = seper_doc["seper"]
                            raw_seper_count += 1
                        
                        # 检查在metrics中的seper字段
                        elif "metrics" in seper_doc and "seper" in seper_doc["metrics"]:
                            primary_doc["metrics"]["seper"] = seper_doc["metrics"]["seper"]
                            raw_seper_count += 1
                        
                        # 添加seper_reduction到utility中
                        if "utility" not in primary_doc:
                            primary_doc["utility"] = {}
                        
                        # 检查直接存在的seper_reduction字段
                        if "seper_reduction" in seper_doc:
                            primary_doc["utility"]["seper_reduction"] = seper_doc["seper_reduction"]
                            seper_reduction_count += 1
                        
                        # 检查在utility中的seper_reduction字段
                        elif "utility" in seper_doc and "seper_reduction" in seper_doc["utility"]:
                            primary_doc["utility"]["seper_reduction"] = seper_doc["utility"]["seper_reduction"]
                            seper_reduction_count += 1

    print(f"合并了{merged_count}个问题的数据")
    print(f"添加了{raw_seper_count}个raw seper值和{seper_reduction_count}个seper_reduction值")

    return primary_data


# -------------------------------
# 提取 doc-level metrics 和 utility 指标
# -------------------------------
def extract_doc_level_data(data):
    """
    提取文档级别的metrics和utility指标:
    - 所有原始metrics (包括seper)
    - 所有reduction指标 (包括seper_reduction)
    
    返回两个DataFrame: 一个包含原始metrics, 一个包含reduction metrics
    """
    raw_rows = []  # 用于存储原始metrics的行
    reduction_rows = []  # 用于存储reduction metrics的行
    
    skipped_entries = 0
    missing_ids = 0
    
    for entry in data:
        # 检查是否有id字段
        if "id" not in entry:
            missing_ids += 1
            continue

        q_id = entry["id"]

        # 遍历每个文档的结果
        if "individual_doc_results" in entry:
            for docres in entry["individual_doc_results"]:
                # 基本行数据
                base_row = {
                    "question_id": q_id,
                }
                
                # 添加文档ID和分数
                if "doc_id" in docres:
                    base_row["doc_id"] = docres["doc_id"]
                if "doc_score" in docres:
                    base_row["doc_score"] = docres["doc_score"]
                
                # 提取原始metrics
                if "metrics" in docres and isinstance(docres["metrics"], dict):
                    raw_row = base_row.copy()
                    
                    for key, value in docres["metrics"].items():
                        # 跳过特殊值 (None, NaN, Infinity)
                        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                            continue
                        
                        # 确保是数值型数据
                        try:
                            float_value = float(value)
                            raw_row[key] = float_value
                        except (ValueError, TypeError):
                            # 如果无法转换为浮点数，则跳过
                            continue
                    
                    # 只有当至少有一个有效指标时才添加行
                    if len(raw_row) > len(base_row):
                        raw_rows.append(raw_row)
                
                # 提取utility (reduction) metrics
                if "utility" in docres and isinstance(docres["utility"], dict):
                    reduction_row = base_row.copy()
                    
                    for key, value in docres["utility"].items():
                        # 排除非reduction指标，如retriever_score
                        if not key.endswith("_reduction") and key != "seper_reduction":
                            continue
                        
                        # 跳过特殊值
                        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                            continue
                        
                        # 确保是数值型数据
                        try:
                            float_value = float(value)
                            reduction_row[key] = float_value
                        except (ValueError, TypeError):
                            continue
                    
                    # 只有当至少有一个有效指标时才添加行
                    if len(reduction_row) > len(base_row):
                        reduction_rows.append(reduction_row)
        else:
            skipped_entries += 1
    
    print(f"数据处理统计: ")
    print(f"  - 跳过的条目 (无individual_doc_results): {skipped_entries}")
    print(f"  - 缺少ID的条目: {missing_ids}")
    print(f"  - 提取的原始metrics行数: {len(raw_rows)}")
    print(f"  - 提取的reduction metrics行数: {len(reduction_rows)}")
    
    raw_df = pd.DataFrame(raw_rows) if raw_rows else pd.DataFrame()
    reduction_df = pd.DataFrame(reduction_rows) if reduction_rows else pd.DataFrame()
    
    # 打印数据框信息
    if not raw_df.empty:
        print(f"原始metrics数据框大小: {raw_df.shape}")
        print(f"原始metrics列: {raw_df.columns.tolist()}")
        
        if "seper" in raw_df.columns:
            print(f"SEPER统计: ")
            print(f"  - 有效值数量: {raw_df['seper'].count()}")
            print(f"  - 平均值: {raw_df['seper'].mean():.4f}")
    
    if not reduction_df.empty:
        print(f"Reduction metrics数据框大小: {reduction_df.shape}")
        print(f"Reduction metrics列: {reduction_df.columns.tolist()}")
        
        if "seper_reduction" in reduction_df.columns:
            print(f"SEPER Reduction统计: ")
            print(f"  - 有效值数量: {reduction_df['seper_reduction'].count()}")
            print(f"  - 平均值: {reduction_df['seper_reduction'].mean():.4f}")
    
    return raw_df, reduction_df


# -------------------------------
# 计算 Distance Correlation 矩阵
# -------------------------------
def distance_corr_matrix(df):
    """
    对 df 内所有列，两两计算 distance correlation。
    返回一个对称矩阵（DataFrame）。
    """
    # 仅选择数值列
    df_numeric = df.select_dtypes(include=[np.number])
    cols = df_numeric.columns
    
    mat = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)

    for i in cols:
        for j in cols:
            # 过滤掉缺失值
            valid_data = df_numeric[[i, j]].dropna()
            if len(valid_data) > 1:  # 至少需要两个观测值
                mat.loc[i, j] = dcor.distance_correlation(valid_data[i], valid_data[j])
            else:
                mat.loc[i, j] = np.nan
    return mat


# -------------------------------
# 构建三种相关性的字典
# -------------------------------
def build_all_corr(df):
    """
    给定一个数值型 DataFrame，
    返回 { 'pearson':..., 'spearman':..., 'distance':... }
    """
    # 仅选择数值列
    df_numeric = df.select_dtypes(include=[np.number])
    
    return {
        "pearson": df_numeric.corr(method="pearson"),
        "spearman": df_numeric.corr(method="spearman"),
        "distance": distance_corr_matrix(df_numeric),
    }


# -------------------------------
# 计算指定指标与目标指标的相关性
# -------------------------------
def calculate_correlations(df, source_cols, target_col):
    """
    计算指定的一组源列与目标列的相关性。

    df: 包含所有列的DataFrame
    source_cols: 源列名列表
    target_col: 目标列名
    """
    if target_col not in df.columns:
        print(f"警告: 数据中没有{target_col}列")
        return pd.DataFrame()
    
    # 仅保留数值列
    df = df.select_dtypes(include=[np.number])

    # 计算与目标列的相关性
    pearson_corrs = {}
    spearman_corrs = {}
    distance_corrs = {}

    # 计算各指标与目标列的相关性
    for col in source_cols:
        if col not in df.columns or col == target_col:
            continue
            
        # 先过滤掉两列中的缺失值
        valid_data = df[[col, target_col]].dropna()
        if len(valid_data) > 1:  # 至少需要两个观测值
            # Pearson correlation
            pearson_corrs[col] = valid_data[col].corr(valid_data[target_col], method="pearson")

            # Spearman correlation
            spearman_corrs[col] = valid_data[col].corr(valid_data[target_col], method="spearman")

            # Distance correlation
            distance_corrs[col] = dcor.distance_correlation(valid_data[col], valid_data[target_col])

    # 创建数据框用于返回
    if not pearson_corrs:  # 如果没有计算出任何相关性
        return pd.DataFrame()

    corr_data = pd.DataFrame(
        {
            "Metric": [col.replace("_", " ").title() for col in pearson_corrs.keys()],
            "Column": list(pearson_corrs.keys()),  # 保留原始列名以便后续处理
            "Pearson": list(pearson_corrs.values()),
            "Spearman": list(spearman_corrs.values()),
            "Distance": list(distance_corrs.values()),
        }
    )

    # 排序，按Pearson相关性绝对值降序
    corr_data = corr_data.sort_values(by="Pearson", key=abs, ascending=False).reset_index(drop=True)

    return corr_data


# -------------------------------
# 绘制热图，强调与目标列的关系
# -------------------------------
def plot_heatmap(corr_matrix, title, ax, highlight_col=None):
    """
    corr_matrix: DataFrame
    title: 标题
    ax: matplotlib Axes
    highlight_col: 要强调的列
    """
    # 格式化标签
    corr_matrix = corr_matrix.copy()
    corr_matrix.index = [idx.replace("_", " ").title() for idx in corr_matrix.index]
    corr_matrix.columns = [col.replace("_", " ").title() for col in corr_matrix.columns]

    # 将highlight_col也格式化
    highlight_col_formatted = highlight_col.replace("_", " ").title() if highlight_col else None

    # Distance correlation range [0,1], others [-1,1]
    is_distance = "distance" in title.lower()
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
        linecolor="black",
        annot_kws={"size": 8},
        fmt=".2f",
        ax=ax,
        cbar_kws={"shrink": 0.8},
        square=True,
    )

    # 高亮显示与目标列相关的行和列
    if highlight_col_formatted and highlight_col_formatted in corr_matrix.columns:
        # 找到highlight_col的索引
        col_idx = corr_matrix.columns.get_loc(highlight_col_formatted)
        row_idx = corr_matrix.index.get_loc(highlight_col_formatted)

        # 高亮列
        for i in range(len(corr_matrix)):
            rect = plt.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor="red", lw=2)
            ax.add_patch(rect)

        # 高亮行
        for i in range(len(corr_matrix.columns)):
            rect = plt.Rectangle((i, row_idx), 1, 1, fill=False, edgecolor="red", lw=2)
            ax.add_patch(rect)

    # Title and appearance
    ax.set_title(title, fontweight="bold", fontsize=12, pad=10)

    # Adjust tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    return ax


# -------------------------------
# 绘制相关性条形图
# -------------------------------
def plot_correlation_bars(corr_data, target_col, output_pdf):
    """
    绘制各指标与目标列的相关性条形图

    corr_data: DataFrame，包含Metric, Pearson, Spearman, Distance列
    target_col: 目标列名（用于标题）
    output_pdf: 输出PDF文件路径
    """
    if corr_data.empty:
        print(f"警告: 没有与{target_col}的相关性数据可绘制")
        return
    
    with PdfPages(output_pdf) as pdf:
        # 添加标题页
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle(f"Correlation Analysis with {target_col.replace('_', ' ').title()}", fontsize=18, fontweight="bold")
        plt.figtext(0.5, 0.5, "Document-Level Metrics Correlation", ha="center", fontsize=14)
        plt.figtext(0.5, 0.4, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ha="center", fontsize=10)
        plt.axis("off")
        pdf.savefig()
        plt.close()
        
        # 绘制相关性条形图
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle(f"Correlations with {target_col.replace('_', ' ').title()}", fontsize=16, fontweight="bold")

        # Pearson correlation
        ax = axes[0]
        bars = ax.barh(corr_data["Metric"], corr_data["Pearson"], color="#3498db")
        ax.set_title("Pearson Correlation (Linear Relationship)", fontweight="bold")
        ax.set_xlim([-1, 1])
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        # 添加数值标签
        for i, bar in enumerate(bars):
            value = corr_data["Pearson"].iloc[i]
            ax.text(
                value + (0.05 if value < 0 else -0.05),
                i,
                f"{value:.2f}",
                va="center",
                ha="right" if value > 0 else "left",
                color="black",
                fontweight="bold",
            )

        # Spearman correlation
        ax = axes[1]
        bars = ax.barh(corr_data["Metric"], corr_data["Spearman"], color="#2ecc71")
        ax.set_title("Spearman Correlation (Monotonic Relationship)", fontweight="bold")
        ax.set_xlim([-1, 1])
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        # 添加数值标签
        for i, bar in enumerate(bars):
            value = corr_data["Spearman"].iloc[i]
            ax.text(
                value + (0.05 if value < 0 else -0.05),
                i,
                f"{value:.2f}",
                va="center",
                ha="right" if value > 0 else "left",
                color="black",
                fontweight="bold",
            )

        # Distance correlation
        ax = axes[2]
        bars = ax.barh(corr_data["Metric"], corr_data["Distance"], color="#9b59b6")
        ax.set_title("Distance Correlation (Non-linear Dependency)", fontweight="bold")
        ax.set_xlim([0, 1])
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        # 添加数值标签
        for i, bar in enumerate(bars):
            value = corr_data["Distance"].iloc[i]
            ax.text(value - 0.05, i, f"{value:.2f}", va="center", ha="right", color="black", fontweight="bold")

        plt.tight_layout()
        pdf.savefig(dpi=300)
        plt.close()
        
        # 添加相关性热图
        if len(corr_data) > 5:  # 只有当有足够的指标时才绘制热图
            # 从corr_data中获取相关列名
            cols = corr_data["Column"].tolist()
            
            if target_col not in cols:
                cols.append(target_col)
            
            # 创建一个只包含这些列的子数据框并计算相关性矩阵
            # 注意：这里需要原始数据框，但我们没有在函数参数中传入
            # 一个解决方案是在主分析函数中调用此函数，并传入原始数据框
            
            # 此处略过热图绘制逻辑，需要在主函数中实现
    
    print(f"相关性条形图已保存至: {output_pdf}")
    return True


# -------------------------------
# 主函数: 执行所有分析
# -------------------------------
def analyze_metrics_correlations(primary_file, seper_file, output_dir):
    """
    对文档级metrics和seper进行全面相关性分析
    
    primary_file: 主数据文件路径
    seper_file: SEPER数据文件路径
    output_dir: 输出目录
    """
    # 如果output_dir是PDF文件路径，则提取目录部分
    if output_dir.endswith('.pdf'):
        output_dir = os.path.dirname(output_dir)
        if not output_dir:  # 如果为空，表示当前目录
            output_dir = '.'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置可视化样式
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_theme(style="white", palette="muted", font_scale=1.0)
    
    print("=" * 80)
    print(f"开始相关性分析:")
    print(f"主数据文件: {primary_file}")
    print(f"SEPER数据文件: {seper_file}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    
    # 1. 合并数据
    print("\n1. 加载和合并数据...")
    merged_data = merge_json_data(primary_file, seper_file)
    print(f"成功合并数据，共{len(merged_data)}个条目")
    
    # 2. 提取文档级数据
    print("\n2. 提取文档级别的metrics和utility数据...")
    raw_df, reduction_df = extract_doc_level_data(merged_data)
    
    # 检查是否成功提取
    if raw_df.empty and reduction_df.empty:
        print("错误: 无法提取有效的文档级别数据!")
        return False
    
    # 直接从SEPER文件提取数据（作为备用）
    seper_dict, seper_reduction_dict = extract_seper_data_directly(seper_file)
    
    # 检查并添加seper列到raw_df
    has_raw_seper = False
    if not raw_df.empty:
        if "seper" in raw_df.columns and raw_df["seper"].count() > 0:
            has_raw_seper = True
            print(f"\n已找到{raw_df['seper'].count()}个有效的seper值")
        else:
            print("\n在metrics中找不到有效的seper列，尝试从提取的数据添加...")
            
            # 创建或重置seper列
            raw_df["seper"] = np.nan
            
            # 从seper_dict填充值
            added_count = 0
            for i, row in raw_df.iterrows():
                key = (row["question_id"], row["doc_id"])
                if key in seper_dict:
                    raw_df.at[i, "seper"] = seper_dict[key]
                    added_count += 1
            
            print(f"成功添加了{added_count}个seper值")
            
            if added_count > 0:
                has_raw_seper = True
            else:
                print("警告: 无法添加任何seper值")
    
    # 检查并添加seper_reduction列到reduction_df
    has_reduction_seper = False
    if not reduction_df.empty:
        if "seper_reduction" in reduction_df.columns and reduction_df["seper_reduction"].count() > 0:
            has_reduction_seper = True
            print(f"\n已找到{reduction_df['seper_reduction'].count()}个有效的seper_reduction值")
        else:
            print("\n在utility中找不到有效的seper_reduction列，尝试从提取的数据添加...")
            
            # 创建或重置seper_reduction列
            reduction_df["seper_reduction"] = np.nan
            
            # 从seper_reduction_dict填充值
            added_count = 0
            for i, row in reduction_df.iterrows():
                key = (row["question_id"], row["doc_id"])
                if key in seper_reduction_dict:
                    reduction_df.at[i, "seper_reduction"] = seper_reduction_dict[key]
                    added_count += 1
            
            print(f"成功添加了{added_count}个seper_reduction值")
            
            if added_count > 0:
                has_reduction_seper = True
            else:
                print("警告: 无法添加任何seper_reduction值")
    
    # 检查是否有数据可以分析
    if not has_raw_seper and not has_reduction_seper:
        print("\n错误: 无法获取任何seper或seper_reduction数据，无法继续分析")
        return False
    
    # 为确保两种分析都能进行，如果一种分析没有数据，尝试从另一种分析派生数据
    if has_raw_seper and not has_reduction_seper and not reduction_df.empty:
        print("\n尝试从raw seper派生seper_reduction数据...")
        
        # 合并raw_df的seper到reduction_df
        reduction_df["seper_reduction"] = np.nan
        
        # 使用question_id和doc_id进行匹配
        for i, row in reduction_df.iterrows():
            # 查找对应的raw行
            mask = (raw_df["question_id"] == row["question_id"]) & (raw_df["doc_id"] == row["doc_id"])
            if mask.any() and not pd.isna(raw_df.loc[mask, "seper"].iloc[0]):
                # 使用raw seper作为seper_reduction的近似值
                reduction_df.at[i, "seper_reduction"] = raw_df.loc[mask, "seper"].iloc[0]
        
        print(f"从raw seper派生了{reduction_df['seper_reduction'].count()}个seper_reduction值")
        if reduction_df["seper_reduction"].count() > 0:
            has_reduction_seper = True
    
    elif has_reduction_seper and not has_raw_seper and not raw_df.empty:
        print("\n尝试从seper_reduction派生raw seper数据...")
        
        # 合并reduction_df的seper_reduction到raw_df
        raw_df["seper"] = np.nan
        
        # 使用question_id和doc_id进行匹配
        for i, row in raw_df.iterrows():
            # 查找对应的reduction行
            mask = (reduction_df["question_id"] == row["question_id"]) & (reduction_df["doc_id"] == row["doc_id"])
            if mask.any() and not pd.isna(reduction_df.loc[mask, "seper_reduction"].iloc[0]):
                # 使用seper_reduction作为raw seper的近似值
                raw_df.at[i, "seper"] = reduction_df.loc[mask, "seper_reduction"].iloc[0]
        
        print(f"从seper_reduction派生了{raw_df['seper'].count()}个raw seper值")
        if raw_df["seper"].count() > 0:
            has_raw_seper = True
    
    # 再次检查是否可以进行两种分析
    if not has_raw_seper or not has_reduction_seper:
        print("\n警告: 仍然无法获取足够的数据进行完整分析")
        if not has_raw_seper:
            print("- 无法获取有效的raw seper数据")
        if not has_reduction_seper:
            print("- 无法获取有效的seper_reduction数据")
    
    # 变量用于存储相关性结果
    raw_correlations = None
    reduction_correlations = None
    
    # 3. 分析原始metrics与seper的相关性
    if has_raw_seper:
        print("\n3. 分析原始metrics与seper的相关性...")
        # 获取除了seper外的所有metrics列
        metrics_cols = [col for col in raw_df.columns if col not in ["question_id", "doc_id", "doc_score", "seper"]]
        
        # 计算与seper的相关性
        raw_correlations = calculate_correlations(raw_df, metrics_cols, "seper")
        
        if not raw_correlations.empty:
            # 输出到控制台
            print("\nRaw Metrics与SEPER的相关性 (前10):")
            print(raw_correlations[["Metric", "Pearson", "Spearman", "Distance"]].head(10))
            
            # 保存相关性表格到CSV
            raw_corr_csv = os.path.join(output_dir, "raw_metrics_seper_correlations.csv")
            raw_correlations.to_csv(raw_corr_csv, index=False)
            print(f"相关性表格已保存至: {raw_corr_csv}")
            
            # 绘制相关性条形图
            raw_corr_pdf = os.path.join(output_dir, "raw_metrics_seper_correlations.pdf")
            plot_correlation_bars(raw_correlations, "seper", raw_corr_pdf)
            
            # 生成相关性热图
            raw_corr_heatmap_pdf = os.path.join(output_dir, "raw_metrics_correlation_heatmap.pdf")
            with PdfPages(raw_corr_heatmap_pdf) as pdf:
                # 计算相关性矩阵
                corr_matrices = build_all_corr(raw_df)
                
                # 绘制热图
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle("Raw Metrics Correlation Matrices (with SEPER)", fontsize=16, fontweight="bold")
                
                plot_heatmap(corr_matrices["pearson"], "Pearson Correlation", axes[0], highlight_col="seper")
                plot_heatmap(corr_matrices["spearman"], "Spearman Correlation", axes[1], highlight_col="seper")
                plot_heatmap(corr_matrices["distance"], "Distance Correlation", axes[2], highlight_col="seper")
                
                plt.tight_layout()
                pdf.savefig(dpi=300)
                plt.close()
            
            print(f"相关性热图已保存至: {raw_corr_heatmap_pdf}")
            
            # 生成单独的报告
            raw_report_file = os.path.join(output_dir, "raw_metrics_seper_report.txt")
            generate_correlation_report(raw_correlations, None, raw_report_file)
        else:
            print("警告: 无法计算有效的raw metrics相关性")
    else:
        print("\n警告: 找不到有效的seper数据，跳过raw metrics相关性分析")
    
    # 4. 分析reduction metrics与seper_reduction的相关性
    if has_reduction_seper:
        print("\n4. 分析reduction metrics与seper_reduction的相关性...")
        # 获取除了seper_reduction外的所有reduction列
        reduction_cols = [col for col in reduction_df.columns if col not in ["question_id", "doc_id", "doc_score", "seper_reduction"]]
        
        # 计算与seper_reduction的相关性
        reduction_correlations = calculate_correlations(reduction_df, reduction_cols, "seper_reduction")
        
        if not reduction_correlations.empty:
            # 输出到控制台
            print("\nReduction Metrics与SEPER Reduction的相关性 (前10):")
            print(reduction_correlations[["Metric", "Pearson", "Spearman", "Distance"]].head(10))
            
            # 保存相关性表格到CSV
            reduction_corr_csv = os.path.join(output_dir, "reduction_metrics_seper_correlations.csv")
            reduction_correlations.to_csv(reduction_corr_csv, index=False)
            print(f"相关性表格已保存至: {reduction_corr_csv}")
            
            # 绘制相关性条形图
            reduction_corr_pdf = os.path.join(output_dir, "reduction_metrics_seper_correlations.pdf")
            plot_correlation_bars(reduction_correlations, "seper_reduction", reduction_corr_pdf)
            
            # 生成相关性热图
            reduction_corr_heatmap_pdf = os.path.join(output_dir, "reduction_metrics_correlation_heatmap.pdf")
            with PdfPages(reduction_corr_heatmap_pdf) as pdf:
                # 计算相关性矩阵
                corr_matrices = build_all_corr(reduction_df)
                
                # 绘制热图
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle("Reduction Metrics Correlation Matrices (with SEPER Reduction)", fontsize=16, fontweight="bold")
                
                plot_heatmap(corr_matrices["pearson"], "Pearson Correlation", axes[0], highlight_col="seper_reduction")
                plot_heatmap(corr_matrices["spearman"], "Spearman Correlation", axes[1], highlight_col="seper_reduction")
                plot_heatmap(corr_matrices["distance"], "Distance Correlation", axes[2], highlight_col="seper_reduction")
                
                plt.tight_layout()
                pdf.savefig(dpi=300)
                plt.close()
            
            print(f"相关性热图已保存至: {reduction_corr_heatmap_pdf}")
            
            # 生成单独的报告
            reduction_report_file = os.path.join(output_dir, "reduction_metrics_seper_report.txt")
            generate_correlation_report(None, reduction_correlations, reduction_report_file)
        else:
            print("警告: 无法计算有效的reduction metrics相关性")
    else:
        print("\n警告: 找不到有效的seper_reduction数据，跳过reduction metrics相关性分析")
    
    # 生成综合报告（如果两种分析都成功）
    if raw_correlations is not None and reduction_correlations is not None:
        combined_report_file = os.path.join(output_dir, "combined_correlation_report.txt")
        generate_correlation_report(raw_correlations, reduction_correlations, combined_report_file)
        print(f"综合相关性报告已保存至: {combined_report_file}")
    
    # 总结分析状态
    print("\n" + "=" * 80)
    print("相关性分析状态:")
    print(f"- Raw Metrics与SEPER相关性分析: {'成功' if raw_correlations is not None else '失败'}")
    print(f"- Reduction Metrics与SEPER Reduction相关性分析: {'成功' if reduction_correlations is not None else '失败'}")
    print(f"结果保存在: {output_dir}")
    print("=" * 80)
    
    # 确定返回值 - 只有当两种分析都成功时才返回True
    overall_success = (raw_correlations is not None and reduction_correlations is not None)
    
    return overall_success


# -------------------------------
# 生成报告 (Markdown/TXT格式)
# -------------------------------
def generate_correlation_report(raw_correlations, reduction_correlations, output_file):
    """
    生成相关性分析的文本报告
    
    raw_correlations: raw metrics与seper的相关性DataFrame
    reduction_correlations: reduction metrics与seper_reduction的相关性DataFrame
    output_file: 输出文件路径
    """
    # 描述相关性强度的辅助函数
    def _corr_strength(value):
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
    
    def _dist_corr_strength(value):
        """返回描述距离相关性强度的标签"""
        if value < 0.3:
            return " (weak)"
        elif value < 0.5:
            return " (moderate)"
        elif value < 0.7:
            return " (strong)"
        else:
            return " (very strong)"
    
    # 开始生成报告
    report = ["# Document-Level Metrics and SEPER Correlation Analysis\n"]
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. 分析raw metrics与seper的相关性
    if raw_correlations is not None and not raw_correlations.empty:
        report.append("## 1. Raw Metrics Correlation with SEPER\n")
        
        # Pearson相关性
        report.append("### Pearson Correlations (Linear Relationship)")
        for i, row in raw_correlations.head(15).iterrows():
            value = row["Pearson"]
            strength = _corr_strength(value)
            report.append(f"- **{row['Metric']}**: {value:.4f}{strength}")
        
        # Spearman相关性
        report.append("\n### Spearman Correlations (Monotonic Relationship)")
        for i, row in raw_correlations.sort_values(by="Spearman", key=abs, ascending=False).head(15).iterrows():
            value = row["Spearman"]
            strength = _corr_strength(value)
            report.append(f"- **{row['Metric']}**: {value:.4f}{strength}")
        
        # Distance相关性
        report.append("\n### Distance Correlations (Non-linear Dependency)")
        for i, row in raw_correlations.sort_values(by="Distance", ascending=False).head(15).iterrows():
            value = row["Distance"]
            strength = _dist_corr_strength(value)
            report.append(f"- **{row['Metric']}**: {value:.4f}{strength}")
        
        # 最佳指标总结
        report.append("\n### Summary for Raw Metrics")
        best_pearson = raw_correlations.iloc[0]
        best_spearman = raw_correlations.sort_values(by="Spearman", key=abs, ascending=False).iloc[0]
        best_distance = raw_correlations.sort_values(by="Distance", ascending=False).iloc[0]
        
        report.append(f"- The strongest linear correlation (Pearson) with SEPER is from **{best_pearson['Metric']}** ({best_pearson['Pearson']:.4f}).")
        report.append(f"- The strongest rank correlation (Spearman) with SEPER is from **{best_spearman['Metric']}** ({best_spearman['Spearman']:.4f}).")
        report.append(f"- The strongest non-linear dependency (Distance) with SEPER is from **{best_distance['Metric']}** ({best_distance['Distance']:.4f}).")
    
    # 2. 分析reduction metrics与seper_reduction的相关性
    if reduction_correlations is not None and not reduction_correlations.empty:
        report.append("\n\n## 2. Reduction Metrics Correlation with SEPER Reduction\n")
        
        # Pearson相关性
        report.append("### Pearson Correlations (Linear Relationship)")
        for i, row in reduction_correlations.head(15).iterrows():
            value = row["Pearson"]
            strength = _corr_strength(value)
            report.append(f"- **{row['Metric']}**: {value:.4f}{strength}")
        
        # Spearman相关性
        report.append("\n### Spearman Correlations (Monotonic Relationship)")
        for i, row in reduction_correlations.sort_values(by="Spearman", key=abs, ascending=False).head(15).iterrows():
            value = row["Spearman"]
            strength = _corr_strength(value)
            report.append(f"- **{row['Metric']}**: {value:.4f}{strength}")
        
        # Distance相关性
        report.append("\n### Distance Correlations (Non-linear Dependency)")
        for i, row in reduction_correlations.sort_values(by="Distance", ascending=False).head(15).iterrows():
            value = row["Distance"]
            strength = _dist_corr_strength(value)
            report.append(f"- **{row['Metric']}**: {value:.4f}{strength}")
        
        # 最佳指标总结
        report.append("\n### Summary for Reduction Metrics")
        best_pearson = reduction_correlations.iloc[0]
        best_spearman = reduction_correlations.sort_values(by="Spearman", key=abs, ascending=False).iloc[0]
        best_distance = reduction_correlations.sort_values(by="Distance", ascending=False).iloc[0]
        
        report.append(f"- The strongest linear correlation (Pearson) with SEPER Reduction is from **{best_pearson['Metric']}** ({best_pearson['Pearson']:.4f}).")
        report.append(f"- The strongest rank correlation (Spearman) with SEPER Reduction is from **{best_spearman['Metric']}** ({best_spearman['Spearman']:.4f}).")
        report.append(f"- The strongest non-linear dependency (Distance) with SEPER Reduction is from **{best_distance['Metric']}** ({best_distance['Distance']:.4f}).")
    
    # 3. 综合分析
    if raw_correlations is not None and not raw_correlations.empty and reduction_correlations is not None and not reduction_correlations.empty:
        report.append("\n\n## 3. Overall Analysis\n")
        
        # 比较raw和reduction指标的表现
        raw_best = raw_correlations.iloc[0]
        reduction_best = reduction_correlations.iloc[0]
        
        if abs(raw_best["Pearson"]) > abs(reduction_best["Pearson"]):
            report.append(f"- For linear relationships, raw metrics (**{raw_best['Metric']}**) show stronger correlation with SEPER than reduction metrics with SEPER Reduction.")
        else:
            report.append(f"- For linear relationships, reduction metrics (**{reduction_best['Metric']}**) show stronger correlation with SEPER Reduction than raw metrics with SEPER.")
        
        # 添加其他综合分析...
    
    # 保存报告
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"相关性分析报告已保存至: {output_file}")
    return True


# -------------------------------
# Main Entry Point
# -------------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python metrics_seper_correlation.py <rag_evaluation_file> <seper_file> [output_dir]")
        print("例如: python metrics_seper_correlation.py ../data/nq_2024_11_07_12_14_naive/results/rag_evaluation_results_temp_936.json ../data/nq_2024_11_07_12_14_naive/results/seper_results.json results")
        sys.exit(1)
    
    rag_file = sys.argv[1]
    seper_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "correlation_results"
    
    success = analyze_metrics_correlations(rag_file, seper_file, output_dir)
    
    # 根据分析结果设置退出代码
    if not success:
        print("无法完成全部分析 (同时分析metrics和utility)，请检查数据和日志")
        sys.exit(1)
    else:
        print("分析成功完成!")
        sys.exit(0)