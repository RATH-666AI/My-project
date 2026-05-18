"""
SEPER与RAGAS评估结果对比可视化分析

"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# sns.set_style("whitegrid")
# sns.set_palette("husl")


class CompleteComparisonAnalyzer:
    def __init__(self, seper_results_path: str, ragas_results_path: str):
        self.seper_results = self.load_results(seper_results_path)
        self.ragas_results = self.load_results(ragas_results_path)
        self.aligned_data = self.align_results()

    def load_results(self, path: str) -> List[Dict]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: 文件 {path} 不存在")
            return []
        except json.JSONDecodeError as e:
            print(f"警告: 文件 {path} JSON解析错误: {e}")
            return []

    def align_results(self) -> pd.DataFrame:
        if not self.seper_results or not self.ragas_results:
            print("警告: SEPER或RAGAS结果为空")
            return pd.DataFrame()

        seper_df = pd.DataFrame(self.seper_results)
        ragas_df = pd.DataFrame(self.ragas_results)

        aligned = pd.merge(
            seper_df[['question', 'delta_seper', 'seper_score', 'seper_baseline',  'confidence', 'is_correct']],
            ragas_df[['question', 'faithfulness', 'answer_relevancy', 'context_precision',  'context_recall', 'answer_similarity', 'answer_correctness']],
            on='question',
            how='inner'
        )

        print(f"成功对齐 {len(aligned)} 个问题")
        return aligned

    def _safe_polyfit(self, x, y, deg=1):
        x = np.array(x)
        y = np.array(y)

        # 步骤1: 移除NaN和无穷值
        # 创建有效数据的布尔掩码
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]

        # 步骤2: 检查数据点数量是否足够
        # 至少需要 deg+1 个数据点才能进行拟合
        if len(x_clean) < deg + 1:
            print(f"  警告: 数据点不足 ({len(x_clean)} < {deg + 1})，跳过拟合")
            return None, None

        # 步骤3: 检查数据是否有足够的方差
        # 如果x或y的方差为0，无法进行有效拟合
        if np.std(x_clean) < 1e-6 or np.std(y_clean) < 1e-6:
            print(f"  警告: 数据方差为0 (x_std={np.std(x_clean):.6f}, y_std={np.std(y_clean):.6f})，跳过拟合")
            return None, None

        try:
            # 步骤4: 进行多项式拟合
            z = np.polyfit(x_clean, y_clean, deg)
            p = np.poly1d(z)

            # 步骤5: 计算决定系数 R²
            y_pred = p(x_clean)
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return p, r2

        except np.linalg.LinAlgError as e:
            # 捕获SVD收敛错误
            print(f"  警告: 多项式拟合失败 - {e}")
            return None, None
        except Exception as e:
            print(f"  警告: 拟合过程中出现异常 - {e}")
            return None, None

    def plot_comprehensive_analysis(self, save_dir: str = "./comprehensive_analysis"):
        os.makedirs(save_dir, exist_ok=True)

        if self.aligned_data.empty:
            print("错误: 无对齐数据，无法生成图表")
            return

        print("\n开始生成综合分析图表...")
        print("=" * 50)

        # 生成6种不同类型的图表
        self._plot_correlation_matrix(os.path.join(save_dir, "1_correlation_matrix.png"))
        self._plot_scatter_comparisons(os.path.join(save_dir, "2_scatter_comparisons.png"))
        self._plot_boxplots(os.path.join(save_dir, "3_boxplots.png"))
        self._plot_radar_chart(os.path.join(save_dir, "4_radar_chart.png"))
        self._plot_parallel_coordinates(os.path.join(save_dir, "5_parallel_coordinates.png"))
        self._plot_group_comparison(os.path.join(save_dir, "6_group_comparison.png"))

        print(f"\n所有图表已保存到: {save_dir}")

    def _plot_correlation_matrix(self, save_path: str):
        # 定义要分析的数值列
        numeric_cols = [
            'delta_seper', 'seper_score', 'confidence',
            'faithfulness', 'answer_relevancy', 'context_precision',
            'context_recall', 'answer_similarity', 'answer_correctness'
        ]

        # 只保留数据中存在的列
        available_cols = [col for col in numeric_cols if col in self.aligned_data.columns]

        if len(available_cols) < 2:
            print("  警告: 可用数值列不足，跳过相关性矩阵")
            return

        # 计算相关系数矩阵
        # corr()方法会自动处理NaN值（跳过）
        corr_matrix = self.aligned_data[available_cols].corr()

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制热力图
        # annot=True: 显示数值，fmt='.3f': 保留3位小数
        # cmap='RdBu_r': 红蓝渐变色，center=0: 以0为中心
        # square=True: 方形单元格
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5, ax=ax,
                    annot_kws={'size': 9})

        ax.set_title('SEPER and RAGAS indicator correlation matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {save_path}")

    def _plot_scatter_comparisons(self, save_path: str):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # 定义要对比的指标对
        comparisons = [
            ('delta_seper', 'faithfulness', 'Δ SEPER vs Loyalty'),
            ('delta_seper', 'answer_relevancy', 'Δ SEPER vs Answer Correlation'),
            ('delta_seper', 'context_precision', 'Δ SEPER vs Context Accuracy'),
            ('confidence', 'faithfulness', 'Confidence vs Loyalty'),
            ('confidence', 'answer_relevancy', 'Confidence vs Answer Relevance'),
            ('seper_score', 'answer_correctness', 'SEPER score vs answer correctness')
        ]

        for idx, (x_col, y_col, title) in enumerate(comparisons):
            # 检查所需列是否存在
            if x_col not in self.aligned_data.columns or y_col not in self.aligned_data.columns:
                axes[idx].text(0.5, 0.5, f'缺少数据: {x_col} 或 {y_col}',
                               ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(title)
                continue

            x = self.aligned_data[x_col].values
            y = self.aligned_data[y_col].values

            # 绘制散点图
            axes[idx].scatter(x, y, alpha=0.6, s=50, c='steelblue', edgecolors='white')

            # 使用安全的拟合函数（修复关键问题）
            p, r2 = self._safe_polyfit(x, y, deg=1)

            if p is not None:
                # 生成拟合线的x坐标
                x_sorted = np.sort(x[~np.isnan(x) & ~np.isinf(x)])
                if len(x_sorted) > 0:
                    y_fit = p(x_sorted)
                    axes[idx].plot(x_sorted, y_fit, 'r--', linewidth=2, alpha=0.8,
                                   label=f'趋势线 (R²={r2:.3f})')

            # 计算相关系数（使用有效数据）
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            if valid_mask.sum() > 1:
                corr = np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
                axes[idx].text(0.05, 0.95, f'r = {corr:.3f}',
                               transform=axes[idx].transAxes, fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            axes[idx].set_xlabel(x_col, fontsize=11)
            axes[idx].set_ylabel(y_col, fontsize=11)
            axes[idx].set_title(title, fontsize=12)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend(loc='lower right', fontsize=8)

        plt.suptitle('Comparative analysis of SEPER and RAGAS indicators', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {save_path}")

    def _plot_boxplots(self, save_path: str):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ========== 左侧: SEPER指标箱线图 ==========
        seper_cols = ['delta_seper', 'seper_score', 'confidence']
        available_seper = [col for col in seper_cols if col in self.aligned_data.columns]

        if available_seper:
            data = []
            labels = []
            for col in available_seper:
                valid_vals = self.aligned_data[col].dropna().values
                if len(valid_vals) > 0:
                    data.append(valid_vals)
                    labels.append(col)

            if data:
                bp = axes[0].boxplot(data, labels=labels, patch_artist=True, showmeans=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                axes[0].set_ylabel('score', fontsize=12)
                axes[0].set_title('Distribution of seper indicators', fontsize=14)
                axes[0].grid(True, alpha=0.3, axis='y')

        # ========== 右侧: RAGAS指标箱线图 ==========
        ragas_cols = ['faithfulness', 'answer_relevancy', 'context_precision',
                      'context_recall', 'answer_similarity', 'answer_correctness']
        available_ragas = [col for col in ragas_cols if col in self.aligned_data.columns]

        if available_ragas:
            data = []
            labels = []
            for col in available_ragas:
                valid_vals = self.aligned_data[col].dropna().values
                if len(valid_vals) > 0:
                    data.append(valid_vals)
                    labels.append(col)

            if data:
                bp = axes[1].boxplot(data, labels=labels, patch_artist=True, showmeans=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightcoral')
                axes[1].set_ylabel('score', fontsize=12)
                axes[1].set_title('Distribution of RAGAS indicators', fontsize=14)
                axes[1].grid(True, alpha=0.3, axis='y')
                axes[1].tick_params(axis='x', rotation=45)

        plt.suptitle('Comparison of SEPER and RAGAS Indicator Box plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {save_path}")

    def _plot_radar_chart(self, save_path: str):
        metrics_data = {}

        # 收集SEPER指标
        if 'delta_seper' in self.aligned_data.columns:
            # ΔSEPER可能为负，映射到0-1范围
            delta_mean = np.mean(self.aligned_data['delta_seper'])
            # 将[-1, 1]范围映射到[0, 1]
            metrics_data['ΔSEPER'] = max(0, min(1, (delta_mean + 1) / 2))
            metrics_data['confidence level'] = np.mean(self.aligned_data['confidence'])

        # 收集RAGAS指标
        ragas_metrics = {
            'faithfulness': 'Loyalty',
            'answer_relevancy': 'Answer relevance',
            'context_precision': 'Context accuracy',
            'context_recall': 'Context recall rate',
            'answer_correctness': 'Correctness'
        }

        for key, name in ragas_metrics.items():
            if key in self.aligned_data.columns:
                metrics_data[name] = np.mean(self.aligned_data[key])

        if not metrics_data:
            print("  警告: 无有效指标数据，跳过雷达图")
            return

        # 准备雷达图数据
        categories = list(metrics_data.keys())
        values = list(metrics_data.values())

        # 雷达图角度计算
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合图形
        values += values[:1]

        # 创建雷达图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')

        # 设置刻度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.set_title('SEPER and RAGAS evaluation index radar chart \n(Normalized)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {save_path}")

    def _plot_parallel_coordinates(self, save_path: str):
        # 选择要展示的指标
        metrics = ['delta_seper', 'confidence', 'faithfulness', 'answer_relevancy',
                   'context_precision', 'answer_correctness']
        available_metrics = [m for m in metrics if m in self.aligned_data.columns]

        if len(available_metrics) < 2:
            print("  警告: 可用指标不足，跳过平行坐标图")
            return

        # Min-Max归一化
        plot_data = self.aligned_data[available_metrics].copy()
        for col in plot_data.columns:
            min_val = plot_data[col].min()
            max_val = plot_data[col].max()
            if max_val > min_val:
                plot_data[col] = (plot_data[col] - min_val) / (max_val - min_val)
            else:
                plot_data[col] = 0.5  

        plot_data['correct'] = self.aligned_data['is_correct'].astype(int)

        fig, ax = plt.subplots(figsize=(14, 8))

        for idx, row in plot_data.iterrows():
            color = 'green' if row['correct'] == 1 else 'red'
            alpha = 0.6 if row['correct'] == 1 else 0.3
            ax.plot(range(len(available_metrics)), row[available_metrics].values,
                    color=color, alpha=alpha, linewidth=0.8)

        # 添加均值点
        for i, col in enumerate(available_metrics):
            mean_val = plot_data[col].mean()
            ax.scatter(i, mean_val, color='blue', s=100, zorder=5, marker='D',
                       label='mean' if i == 0 else "")

        # 设置坐标轴
        ax.set_xticks(range(len(available_metrics)))
        ax.set_xticklabels(available_metrics, rotation=45, ha='right')
        ax.set_ylabel('Normalized score', fontsize=12)
        ax.set_title('Parallel coordinate chart of SEPER and RAGAS indicators \n(Green=correct answer, red=incorrect answer)', fontsize=14)
        ax.grid(True, alpha=0.3)

        # 添加图例
        legend_elements = [
            Patch(facecolor='green', alpha=0.6, label='correct answer'),
            Patch(facecolor='red', alpha=0.3, label='incorrect answer'),
            Patch(facecolor='blue', label='mean')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {save_path}")

    def _plot_group_comparison(self, save_path: str):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ========== 左侧: 按ΔSEPER符号分组 ==========
        positive_mask = self.aligned_data['delta_seper'] > 0
        negative_mask = self.aligned_data['delta_seper'] <= 0

        # 选择要对比的RAGAS指标
        ragas_metrics = ['faithfulness', 'answer_relevancy', 'context_precision',
                         'context_recall', 'answer_correctness']
        available_ragas = [m for m in ragas_metrics if m in self.aligned_data.columns]

        if available_ragas and positive_mask.any() and negative_mask.any():
            x = np.arange(len(available_ragas))
            width = 0.35

            positive_means = []
            negative_means = []
            for m in available_ragas:
                pos_vals = self.aligned_data[positive_mask][m].dropna()
                neg_vals = self.aligned_data[negative_mask][m].dropna()
                positive_means.append(pos_vals.mean() if len(pos_vals) > 0 else 0)
                negative_means.append(neg_vals.mean() if len(neg_vals) > 0 else 0)

            axes[0].bar(x - width / 2, positive_means, width, label='Positive improvement (Δ>0)',
                        color='lightgreen', alpha=0.7)
            axes[0].bar(x + width / 2, negative_means, width, label='Negative improvement (Δ≤0)',
                        color='lightcoral', alpha=0.7)

            axes[0].set_xticks(x)
            axes[0].set_xticklabels(available_ragas, rotation=45, ha='right')
            axes[0].set_ylabel('average score', fontsize=12)
            axes[0].set_title('Comparison of RAGAS indicators among different Δ SEPER groups', fontsize=14)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')

        # ========== 右侧: 正确/错误回答组的ΔSEPER对比 ==========
        correct_mask = self.aligned_data['is_correct'] == True
        incorrect_mask = self.aligned_data['is_correct'] == False

        if correct_mask.any() and incorrect_mask.any():
            correct_delta = self.aligned_data[correct_mask]['delta_seper'].dropna()
            incorrect_delta = self.aligned_data[incorrect_mask]['delta_seper'].dropna()

            bp = axes[1].boxplot([correct_delta, incorrect_delta],
                                 labels=['correct answer', 'incorrect answer'],
                                 patch_artist=True, showmeans=True)

            for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
                patch.set_facecolor(color)

            axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
            axes[1].set_ylabel('ΔSEPER', fontsize=12)
            axes[1].set_title('Comparison of Δ SEPER between correct and incorrect answers', fontsize=14)
            axes[1].grid(True, alpha=0.3, axis='y')

            stats_text = f"Correct group: n={len(correct_delta)}, μ={correct_delta.mean():.4f}\n"
            stats_text += f"Incorrect group: n={len(incorrect_delta)}, μ={incorrect_delta.mean():.4f}"
            axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Group comparison analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {save_path}")

    def print_summary(self):
        if self.aligned_data.empty:
            print("错误: 无对齐数据")
            return

        print("\n" + "=" * 80)
        print("Summary of complete comparison between SEPER and RAGAS evaluations")
        print("=" * 80)

        print(f"\n【Data alignment status】")
        print(f"  SEPER evaluation sample size: {len(self.seper_results)}")
        print(f"  RAGAS evaluation sample size: {len(self.ragas_results)}")
        print(f"  Successfully aligned sample size: {len(self.aligned_data)}")

        # SEPER指标统计
        print(f"\n【SEPER indicator statistics】")
        for col in ['delta_seper', 'seper_score', 'confidence']:
            if col in self.aligned_data.columns:
                valid_vals = self.aligned_data[col].dropna()
                if len(valid_vals) > 0:
                    print(f"  {col}: mean={valid_vals.mean():.4f}, "
                          f"standard deviation={valid_vals.std():.4f}, "
                          f"median={valid_vals.median():.4f}")

        # 正向改进率
        if 'delta_seper' in self.aligned_data.columns:
            positive_rate = (self.aligned_data['delta_seper'] > 0).mean() * 100
            print(f"\n  Positive improvement rate (ΔSEPER > 0): {positive_rate:.1f}%")

        # RAGAS指标统计
        print(f"\n【RAGAS indicator statistics】")
        for col in ['faithfulness', 'answer_relevancy', 'context_precision',
                    'context_recall', 'answer_similarity', 'answer_correctness']:
            if col in self.aligned_data.columns:
                valid_vals = self.aligned_data[col].dropna()
                if len(valid_vals) > 0:
                    print(f"  {col}: mean={valid_vals.mean():.4f}, "
                          f"standard deviation={valid_vals.std():.4f}")

        # 相关性分析
        print(f"\n【Correlation analysis (with Δ SEPER)】")
        correlations = {}
        for col in ['faithfulness', 'answer_relevancy', 'context_precision',
                    'context_recall', 'answer_correctness']:
            if col in self.aligned_data.columns:
                # 移除NaN值后计算相关系数
                valid_mask = ~(self.aligned_data['delta_seper'].isna() |
                               self.aligned_data[col].isna())
                if valid_mask.sum() > 1:
                    corr = self.aligned_data['delta_seper'][valid_mask].corr(
                        self.aligned_data[col][valid_mask])
                    correlations[col] = corr
                    print(f"  ΔSEPER vs {col}: {corr:.4f}")

        if correlations:
            best = max(correlations.items(), key=lambda x: abs(x[1]))
            print(f"\n  The indicator with the strongest correlation with Δ SEPER: {best[0]} (r={best[1]:.4f})")

    def generate_report(self, output_path: str = "comparison_report.html"):
        if self.aligned_data.empty:
            print("错误: 无对齐数据，无法生成报告")
            return

        delta_mean = np.mean(self.aligned_data['delta_seper'])
        delta_std = np.std(self.aligned_data['delta_seper'])
        positive_rate = (self.aligned_data['delta_seper'] > 0).mean() * 100

        # 计算各指标均值
        metrics_stats = {}
        for col in ['faithfulness', 'answer_relevancy', 'context_precision',
                    'context_recall', 'answer_similarity', 'answer_correctness',
                    'delta_seper', 'confidence']:
            if col in self.aligned_data.columns:
                metrics_stats[col] = np.mean(self.aligned_data[col])

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SEPER and RAGAS Evaluation Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #333; }}
        .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .summary-card {{ background-color: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .summary-card .value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .footer {{ text-align: center; margin-top: 30px; padding: 20px; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>SEPER and RAGAS Evaluation Comparison Report</h1>

    <div class="summary">
        <h2>Comparative Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Align sample size</h3>
                <div class="value">{len(self.aligned_data)}</div>
            </div>
            <div class="summary-card">
                <h3>Average Δ SEPER</h3>
                <div class="value">{delta_mean:.4f}</div>
            </div>
            <div class="summary-card">
                <h3>Δ SEPER standard deviation</h3>
                <div class="value">{delta_std:.4f}</div>
            </div>
            <div class="summary-card">
                <h3>Positive improvement rate</h3>
                <div class="value">{positive_rate:.1f}%</div>
            </div>
        </div>
    </div>

    <h2>Indicator Statistics</h2>
    <table>
        <thead>
            <tr><th>indicator</th><th>mean</th><th>Belonging framework</th></tr>
        </thead>
        <tbody>
"""

        for col, mean_val in metrics_stats.items():
            framework = "SEPER" if col in ['delta_seper', 'seper_score', 'confidence'] else "RAGAS"
            html_content += f"""
            <tr>
                <td><strong>{col}</strong></strong></td>
                <td>{mean_val:.4f}</strong></td>
                <td>{framework}</strong></td>
            </tr>
"""

        html_content += f"""
        </tbody>
    </table>

    <div class="footer">
        <p>报告生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>注: ΔSEPER > 0 表示RAG系统相对于无上下文有正面改进</p>
        <p>图表文件请查看 comprehensive_analysis 目录</p>
    </div>
</div>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML报告已保存到: {output_path}")


def main():
    print("=" * 80)
    print("Visual analysis of SEPER and RAGAS evaluation results comparison")
    print("=" * 80)

    seper_results_path = "seper_evaluation_results.json"
    ragas_results_path = "ragas_evaluation_results.json"

    if not os.path.exists(seper_results_path):
        print(f"错误: SEPER结果文件 {seper_results_path} 不存在")
        print("请先运行 rag_seper_evaluation_complete.py 生成SEPER评估结果")
        return

    if not os.path.exists(ragas_results_path):
        print(f"错误: RAGAS结果文件 {ragas_results_path} 不存在")
        print("请先运行 ragas_evaluation_simple.py 生成RAGAS评估结果")
        return

    # 创建分析器并执行分析
    print("\n正在加载和分析数据...")
    analyzer = CompleteComparisonAnalyzer(seper_results_path, ragas_results_path)

    # 打印统计摘要
    analyzer.print_summary()

    # 生成可视化图表
    analyzer.plot_comprehensive_analysis()

    # 生成HTML报告
    analyzer.generate_report()

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()



