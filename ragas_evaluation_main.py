# ragas_evaluation_simple.py
"""
使用RAGAS框架评估RAG系统 - 简化版，使用API评估
与SEPER结果进行可视化对比
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from dataclasses import dataclass, asdict
from langchain_community.embeddings import HuggingFaceEmbeddings
# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from langchain_core.callbacks import CallbackManagerForLLMRun

# Transformers for local model
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ragSystem.rag_system import RAGSystem

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from datasets import Dataset

# LangChain
from langchain_openai import ChatOpenAI

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# RAGAS评估器 - 使用API
# ============================================================================

@dataclass
class RAGASEvaluationResult:
    """RAGAS评估结果数据类"""
    question: str
    ground_truth: str
    answer: str
    contexts: List[str]
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_similarity: float
    answer_correctness: float

device = "cuda" if torch.cuda.is_available() else "cpu"
class RAGASEvaluatorSimple:
    """RAGAS框架评估器 - 使用API评估"""

    def __init__(self, rag_system: RAGSystem, api_key: str = None):
        self.rag_system = rag_system



        """
        self.eval_llm = AutoModelForCausalLM.from_pretrained(
            "D:/models/Qwen/Qwen2___5-1___5B-Instruct",
            torch_dtype=torch.float32,
            device_map=None,  # Disable auto device map
            trust_remote_code=True
        ).to(device)  # Manually move to device
        self.tokenizer = AutoTokenizer.from_pretrained("D:/models/Qwen/Qwen2___5-1___5B-Instruct", trust_remote_code=True)
        # Set pad_token for Qwen
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.encoder = SentenceTransformer('D:/models/all-MiniLM-L6-v2')

        model_kwargs = {'device': 'cpu'} 
        encode_kwargs = {'normalize_embeddings': True}  
        self.embeddings = HuggingFaceEmbeddings(
            model_name='D:/models/all-MiniLM-L6-v2',
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        """

        # 初始化评估用的LLM - 使用SiliconFlow API
        print("初始化评估LLM (SiliconFlow API)...")
        self.eval_llm = ChatOpenAI(
            base_url="https://api.siliconflow.cn/v1/",
            api_key=api_key or "sk-vhyipdsfixdpplloxbxsfoycbvkvshbomlcqsxwogrqogdfc",
            model="deepseek-ai/DeepSeek-V3",
            temperature=0
        )
        # 尝试使用本地模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="/root/autodl-tmp/models/BAAI/bge-large-zh-v1___5",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # model_kwargs = {'device': 'cpu'}  
        # encode_kwargs = {'normalize_embeddings': True}  
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name='D:/models/all-MiniLM-L6-v2',
        #     model_kwargs=model_kwargs,
        #     encode_kwargs=encode_kwargs
        # )

        # 定义评估指标
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_similarity,
            answer_correctness
        ]

        print(f"RAGAS评估器初始化完成")
        print(f"  使用指标: {[m.name for m in self.metrics]}")

    def load_hotpotqa_data(self, json_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """加载HotpotQA数据"""
        data = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    sample = json.loads(line)

                    # 提取上下文
                    context_text = ""
                    context = sample.get('context', {})
                    if 'sentences' in context:
                        for sentences in context['sentences']:
                            context_text += " ".join(sentences) + "\n"

                    data.append({
                        'id': sample.get('id', f'sample_{i}'),
                        'question': sample.get('question', ''),
                        'answer': sample.get('answer', ''),
                        'type': sample.get('type', ''),
                        'level': sample.get('level', ''),
                        'context': context_text[:3000]
                    })
                except json.JSONDecodeError:
                    continue

        print(f"加载了 {len(data)} 条HotpotQA数据")
        return data

    def get_rag_response(self, question: str, context: str) -> Dict:
        """获取RAG系统响应"""
        from langchain_core.documents import Document as LCDocument

        doc = LCDocument(page_content=context, metadata={"source": "hotpotqa"})
        self.rag_system.setup_knowledge_base(documents=[doc])

        result = self.rag_system.query(question)
        return result

    def evaluate_single_question(self, question: str, ground_truth: str,
                                 context: str) -> Optional[RAGASEvaluationResult]:
        """评估单个问题"""
        try:
            # 获取RAG响应
            rag_result = self.get_rag_response(question, context)
            answer = rag_result['result']
            contexts = rag_result.get('contexts', [])

            if not contexts:
                contexts = [context[:500]]

            # eval_data = {
            #     'question': [question],
            #     'answer': [answer],
            #     'contexts': [contexts],
            #     'ground_truth': [ground_truth]
            # }
            eval_data = {
                'user_input': [question],
                'response': [answer],
                'retrieved_contexts': [contexts],
                'reference': [ground_truth]
            }

            dataset = Dataset.from_dict(eval_data)

            # evaluation_data = []
            # for result in tqdm(self.results, desc="准备数据"):
            #     evaluation_data.append({
            #         "user_input": result.question,
            #         "response": result.generated_answer,
            #         "retrieved_contexts": result.retrieved_contexts,
            #         "reference": result.ground_truth
            #     })
            #
            # self.ragas_dataset = Dataset.from_list(evaluation_data)

            # print(f"准备完成，共 {len(evaluation_data)} 条数据")

            # 运行RAGAS评估
            print(f"  评估中...")
            result_dict = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.eval_llm,
                embeddings=self.embeddings,
            )



            # 获取结果
            # result_dict = dict(result)

            return RAGASEvaluationResult(
                question=question,
                ground_truth=ground_truth,
                answer=answer,
                contexts=contexts,
                faithfulness=result_dict['faithfulness'][0] if result_dict['faithfulness'][0] is not None else 0,
                answer_relevancy = result_dict['answer_relevancy'][0] if result_dict['answer_relevancy'][0] is not None else 0,
                context_precision=result_dict['context_precision'][0] if result_dict['context_precision'][0] is not None else 0,
                context_recall=result_dict['context_recall'][0] if result_dict['context_recall'][0] is not None else 0,
                answer_similarity=result_dict['answer_similarity'][0] if result_dict['answer_similarity'][0] is not None else 0,
                answer_correctness=result_dict['answer_correctness'][0] if result_dict['answer_correctness'][0] is not None else 0,

            )

        except Exception as e:
            print(f"评估出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_dataset(self, data_path: str, max_samples: Optional[int] = None,
                         save_path: str = "ragas_evaluation_results.json") -> List[RAGASEvaluationResult]:
        """评估数据集"""
        data = self.load_hotpotqa_data(data_path, max_samples)

        results = []

        print(f"\n开始RAGAS评估 {len(data)} 个问题...")
        print("=" * 80)

        for idx, sample in enumerate(tqdm(data, desc="RAGAS评估")):
            print(f"\n问题 {idx + 1}/{len(data)}: {sample['question'][:80]}...")

            result = self.evaluate_single_question(
                question=sample['question'],
                ground_truth=sample['answer'],
                context=sample['context']
            )

            if result:
                results.append(result)
                print(f"  答案: {result.answer[:100]}...")
                print(f"  忠实度: {result.faithfulness:.4f}")
                print(f"  答案相关性: {result.answer_relevancy:.4f}")
                print(f"  上下文精确度: {result.context_precision:.4f}")
                print(f"  上下文召回率: {result.context_recall:.4f}")
                print(f"  答案正确性: {result.answer_correctness:.4f}")

            # # 定期保存
            # if (idx + 1) % 5 == 0 and results:
            #     self.save_results(results, f"{save_path}.tmp")

        if results:
            self.save_results(results, save_path)

        return results

    def save_results(self, results: List[RAGASEvaluationResult], output_path: str):
        """保存评估结果"""
        output_data = []
        for r in results:
            output_data.append({
                'question': r.question,
                'ground_truth': r.ground_truth,
                'answer': r.answer,
                'contexts': r.contexts[:3],
                'faithfulness': r.faithfulness,
                'answer_relevancy': r.answer_relevancy,
                'context_precision': r.context_precision,
                'context_recall': r.context_recall,
                'answer_similarity': r.answer_similarity,
                'answer_correctness': r.answer_correctness
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {output_path}")

    def print_summary(self, results: List[RAGASEvaluationResult]):
        """打印评估摘要"""
        if not results:
            print("没有评估结果")
            return

        print("\n" + "=" * 80)
        print("RAGAS评估摘要")
        print("=" * 80)

        metrics = [
            ('faithfulness', '忠实度'),
            ('answer_relevancy', '答案相关性'),
            ('context_precision', '上下文精确度'),
            ('context_recall', '上下文召回率'),
            ('answer_similarity', '答案相似度'),
            ('answer_correctness', '答案正确性')
        ]

        for metric_key, metric_name in metrics:
            scores = [getattr(r, metric_key) for r in results]
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                print(f"\n{metric_name}:")
                print(f"  均值: {np.mean(valid_scores):.4f} ± {np.std(valid_scores):.4f}")
                print(f"  中位数: {np.median(valid_scores):.4f}")
                print(f"  最小值: {np.min(valid_scores):.4f}")
                print(f"  最大值: {np.max(valid_scores):.4f}")


# ============================================================================
# 对比分析器
# ============================================================================

class ComparisonAnalyzer:
    """SEPER与RAGAS评估结果对比分析器"""

    def __init__(self, seper_results_path: str, ragas_results_path: str):
        self.seper_results = self.load_results(seper_results_path)
        self.ragas_results = self.load_results(ragas_results_path)
        self.aligned_data = self.align_results()

    def load_results(self, path: str) -> List[Dict]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []

    def align_results(self) -> pd.DataFrame:
        if not self.seper_results or not self.ragas_results:
            print(f"SEPER结果数: {len(self.seper_results)}")
            print(f"RAGAS结果数: {len(self.ragas_results)}")
            return pd.DataFrame()

        seper_df = pd.DataFrame(self.seper_results)
        ragas_df = pd.DataFrame(self.ragas_results)

        # 按问题对齐
        aligned = pd.merge(
            seper_df[['question', 'delta_seper', 'seper_score', 'seper_baseline',
                      'confidence', 'is_correct']],
            ragas_df[['question', 'faithfulness', 'answer_relevancy', 'context_precision',
                      'context_recall', 'answer_similarity', 'answer_correctness']],
            on='question',
            how='inner'
        )

        # 处理NaN值 - 用0填充
        for col in ['faithfulness', 'answer_relevancy', 'context_precision',
                    'context_recall', 'answer_similarity', 'answer_correctness']:
            if col in aligned.columns:
                aligned[col] = aligned[col].fillna(0)

        print(f"对齐后数据量: {len(aligned)}")
        return aligned

    def plot_correlation_matrix(self, save_path: str = "correlation_matrix.png"):
        """绘制相关性矩阵热力图"""
        if self.aligned_data.empty:
            print("无对齐数据")
            return

        numeric_cols = ['delta_seper', 'seper_score', 'confidence',
                        'faithfulness', 'answer_relevancy', 'context_precision',
                        'context_recall', 'answer_similarity', 'answer_correctness']

        available_cols = [col for col in numeric_cols if col in self.aligned_data.columns]
        if len(available_cols) < 2:
            print("没有足够的数值列")
            return

        corr_matrix = self.aligned_data[available_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5, ax=ax,
                    annot_kws={'size': 9})

        ax.set_title('SEPER与RAGAS指标相关性矩阵', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

    def plot_scatter_comparisons(self, save_path: str = "scatter_comparisons.png"):
        """绘制散点图对比"""
        if self.aligned_data.empty:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        comparisons = [
            ('delta_seper', 'faithfulness', 'ΔSEPER vs 忠实度'),
            ('delta_seper', 'answer_relevancy', 'ΔSEPER vs 答案相关性'),
            ('delta_seper', 'context_precision', 'ΔSEPER vs 上下文精确度'),
            ('confidence', 'faithfulness', '置信度 vs 忠实度'),
            ('confidence', 'answer_relevancy', '置信度 vs 答案相关性'),
            ('seper_score', 'answer_correctness', 'SEPER分数 vs 答案正确性')
        ]

        for idx, (x_col, y_col, title) in enumerate(comparisons):
            if x_col not in self.aligned_data.columns or y_col not in self.aligned_data.columns:
                axes[idx].text(0.5, 0.5, f'缺少数据: {x_col} 或 {y_col}',
                               ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(title)
                continue

            x = self.aligned_data[x_col]
            y = self.aligned_data[y_col]

            axes[idx].scatter(x, y, alpha=0.6, s=50, c='steelblue', edgecolors='white')

            # 添加趋势线
            if len(x) > 1 and np.std(x) > 0:
                try:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_sorted = np.sort(x)
                    axes[idx].plot(x_sorted, p(x_sorted), 'r--', linewidth=2, alpha=0.8)

                    corr = np.corrcoef(x, y)[0, 1]
                    axes[idx].text(0.05, 0.95, f'r = {corr:.3f}',
                                   transform=axes[idx].transAxes, fontsize=10,
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except:
                    pass

            axes[idx].set_xlabel(x_col, fontsize=11)
            axes[idx].set_ylabel(y_col, fontsize=11)
            axes[idx].set_title(title, fontsize=12)
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle('SEPER与RAGAS指标对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

    def plot_boxplots(self, save_path: str = "boxplots.png"):
        """绘制箱线图"""
        if self.aligned_data.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # SEPER指标
        seper_cols = ['delta_seper', 'seper_score', 'confidence']
        available_seper = [col for col in seper_cols if col in self.aligned_data.columns]

        if available_seper:
            data = [self.aligned_data[col].values for col in available_seper]
            bp = axes[0].boxplot(data, labels=available_seper, patch_artist=True, showmeans=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            axes[0].set_ylabel('分数', fontsize=12)
            axes[0].set_title('SEPER指标分布', fontsize=14)
            axes[0].grid(True, alpha=0.3, axis='y')

        # RAGAS指标
        ragas_cols = ['faithfulness', 'answer_relevancy', 'context_precision',
                      'context_recall', 'answer_similarity', 'answer_correctness']
        available_ragas = [col for col in ragas_cols if col in self.aligned_data.columns]

        if available_ragas:
            data = [self.aligned_data[col].values for col in available_ragas]
            bp = axes[1].boxplot(data, labels=available_ragas, patch_artist=True, showmeans=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightcoral')
            axes[1].set_ylabel('分数', fontsize=12)
            axes[1].set_title('RAGAS指标分布', fontsize=14)
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].tick_params(axis='x', rotation=45)

        plt.suptitle('SEPER与RAGAS指标箱线图对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

    def plot_group_comparison(self, save_path: str = "group_comparison.png"):
        """绘制分组对比图"""
        if self.aligned_data.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 按ΔSEPER符号分组
        positive_mask = self.aligned_data['delta_seper'] > 0
        negative_mask = self.aligned_data['delta_seper'] <= 0

        if positive_mask.any() and negative_mask.any():
            metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'answer_correctness']
            available_metrics = [m for m in metrics if m in self.aligned_data.columns]

            if available_metrics:
                x = np.arange(len(available_metrics))
                width = 0.35

                positive_means = [self.aligned_data[positive_mask][m].mean() for m in available_metrics]
                negative_means = [self.aligned_data[negative_mask][m].mean() for m in available_metrics]

                axes[0].bar(x - width / 2, positive_means, width, label='正向改进 (Δ>0)',
                            color='lightgreen', alpha=0.7)
                axes[0].bar(x + width / 2, negative_means, width, label='负向改进 (Δ≤0)',
                            color='lightcoral', alpha=0.7)

                axes[0].set_xticks(x)
                axes[0].set_xticklabels(available_metrics, rotation=45, ha='right')
                axes[0].set_ylabel('平均分数', fontsize=12)
                axes[0].set_title('不同ΔSEPER组的RAGAS指标对比', fontsize=14)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3, axis='y')

        # 按答案正确性分组
        if 'is_correct' in self.aligned_data.columns:
            correct_mask = self.aligned_data['is_correct'] == True
            incorrect_mask = self.aligned_data['is_correct'] == False

            if correct_mask.any() and incorrect_mask.any():
                bp = axes[1].boxplot([self.aligned_data[correct_mask]['delta_seper'],
                                      self.aligned_data[incorrect_mask]['delta_seper']],
                                     labels=['正确回答', '错误回答'],
                                     patch_artist=True, showmeans=True)

                for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
                    patch.set_facecolor(color)

                axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
                axes[1].set_ylabel('ΔSEPER', fontsize=12)
                axes[1].set_title('正确与错误回答的ΔSEPER对比', fontsize=14)
                axes[1].grid(True, alpha=0.3, axis='y')

        plt.suptitle('分组对比分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

    def plot_radar_chart(self, save_path: str = "radar_chart.png"):
        """绘制雷达图"""
        if self.aligned_data.empty:
            return

        metrics_data = {}

        # SEPER指标
        if 'delta_seper' in self.aligned_data.columns:
            delta_mean = np.mean(self.aligned_data['delta_seper'])
            metrics_data['ΔSEPER'] = max(0, min(1, (delta_mean + 1) / 2))
            metrics_data['置信度'] = np.mean(self.aligned_data['confidence'])

        # RAGAS指标
        ragas_metrics = {
            'faithfulness': '忠实度',
            'answer_relevancy': '答案相关性',
            'context_precision': '上下文精确度',
            'context_recall': '上下文召回率',
            'answer_correctness': '答案正确性'
        }

        for key, name in ragas_metrics.items():
            if key in self.aligned_data.columns:
                metrics_data[name] = np.mean(self.aligned_data[key])

        if not metrics_data:
            return

        categories = list(metrics_data.keys())
        values = list(metrics_data.values())

        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        values += values[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.set_title('SEPER与RAGAS评估指标雷达图\n(归一化后)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

    def print_summary(self):
        if self.aligned_data.empty:
            print("无对比数据")
            return

        print("\n" + "=" * 80)
        print("SEPER与RAGAS评估对比摘要")
        print("=" * 80)

        print(f"\n数据对齐情况:")
        print(f"  SEPER评估样本数: {len(self.seper_results)}")
        print(f"  RAGAS评估样本数: {len(self.ragas_results)}")
        print(f"  对齐样本数: {len(self.aligned_data)}")

        print(f"\nSEPER指标统计:")
        print(f"  ΔSEPER: {np.mean(self.aligned_data['delta_seper']):.4f}")
        print(f"  置信度: {np.mean(self.aligned_data['confidence']):.4f}")
        print(f"  正向改进率: {np.mean(self.aligned_data['delta_seper'] > 0) * 100:.1f}%")

        print(f"\nRAGAS指标统计:")
        for col in ['faithfulness', 'answer_relevancy', 'context_precision',
                    'context_recall', 'answer_similarity', 'answer_correctness']:
            if col in self.aligned_data.columns:
                valid_vals = self.aligned_data[col][~np.isnan(self.aligned_data[col])]
                if len(valid_vals) > 0:
                    print(f"  {col}: {np.mean(valid_vals):.4f}")

        print(f"\n相关性分析 (与ΔSEPER):")
        for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'answer_correctness']:
            if col in self.aligned_data.columns:
                valid_mask = ~np.isnan(self.aligned_data[col])
                if valid_mask.sum() > 1:
                    corr = np.corrcoef(self.aligned_data['delta_seper'][valid_mask],
                                       self.aligned_data[col][valid_mask])[0, 1]
                    print(f"  ΔSEPER vs {col}: {corr:.4f}")

    def generate_all_plots(self, output_dir: str = "./comparison_plots"):
        """生成所有对比图表"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("生成对比分析图表")
        print("=" * 60)

        self.plot_correlation_matrix(os.path.join(output_dir, "1_correlation_matrix.png"))
        self.plot_scatter_comparisons(os.path.join(output_dir, "2_scatter_comparisons.png"))
        self.plot_boxplots(os.path.join(output_dir, "3_boxplots.png"))
        self.plot_group_comparison(os.path.join(output_dir, "4_group_comparison.png"))
        self.plot_radar_chart(os.path.join(output_dir, "5_radar_chart.png"))

        print(f"\n所有图表已保存到: {output_dir}")

    def generate_report(self, output_path: str = "comparison_report.html"):
        """生成HTML报告"""
        if self.aligned_data.empty:
            return

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SEPER与RAGAS评估对比报告</title>
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
    <h1>SEPER与RAGAS评估对比报告</h1>

    <div class="summary">
        <h2>对比摘要</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>对齐样本数</h3>
                <div class="value">{len(self.aligned_data)}</div>
            </div>
            <div class="summary-card">
                <h3>平均ΔSEPER</h3>
                <div class="value">{np.mean(self.aligned_data['delta_seper']):.4f}</div>
            </div>
            <div class="summary-card">
                <h3>平均忠实度</h3>
                <div class="value">{np.mean(self.aligned_data['faithfulness']):.4f}</div>
            </div>
            <div class="summary-card">
                <h3>平均答案相关性</h3>
                <div class="value">{np.mean(self.aligned_data['answer_relevancy']):.4f}</div>
            </div>
        </div>
    </div>

    <h2>指标统计</h2>
    <table>
        <thead>
            <tr><th>指标</th><th>均值</th><th>与ΔSEPER相关性</th></tr>
        </thead>
        <tbody>
"""

        for col in self.aligned_data.columns:
            if col not in ['question', 'answer', 'ground_truth', 'contexts']:
                valid_mask = ~np.isnan(self.aligned_data[col])
                mean_val = np.mean(self.aligned_data[col][valid_mask]) if valid_mask.any() else 0
                if col != 'delta_seper' and valid_mask.sum() > 1:
                    corr = np.corrcoef(self.aligned_data['delta_seper'][valid_mask],
                                       self.aligned_data[col][valid_mask])[0, 1]
                else:
                    corr = 1.0
                html_content += f"""
            <tr>
                <td><strong>{col}</strong></td>
                <td>{mean_val:.4f}</td>
                <td>{corr:.4f}</td>
            </tr>
"""

        html_content += """
        </tbody>
    </table>

    <div class="footer">
        <p>报告生成时间: """ + str(__import__('datetime').datetime.now()) + """</p>
        <p>注: ΔSEPER > 0 表示RAG系统相对于无上下文有正面改进</p>
    </div>
</div>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML报告已保存到: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 80)
    print("RAGAS框架评估 - SEPER vs RAGAS对比")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_model_path =  "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"
    # llm_model_path = "D:/models/Qwen/Qwen2.5-0.5B-Instruct"
    embedding_model_path = '/root/autodl-tmp/models/sentence-transformers/all-MiniLM-L6-v2'
    data_path = "./data/hotpotqa_100_samples.json"
    max_samples = 50  

    print(f"\n配置:")
    print(f"  设备: {device}")
    print(f"  评估样本数: {max_samples}")

    # seper_results_path = "seper_evaluation_results.json"
    ragas_results_path = "ragas_evaluation_results.json"

    # if not os.path.exists(seper_results_path):
    #     print(f"\n错误: SEPER结果文件 {seper_results_path} 不存在")
    #     print("请先运行 SEPER评估脚本 生成SEPER评估结果")
    #     return

    # 初始化RAG系统
    print("\n初始化RAG系统...")
    rag_system = RAGSystem(
        model_name=llm_model_path,
        embedding_model_path=embedding_model_path,
        temperature=0.1,
        max_tokens=256,
        chunk_size=500,
        chunk_overlap=100,
        top_k=3
    )

    # 运行RAGAS评估
    print("\n" + "=" * 60)
    print("运行RAGAS评估 (使用API)")
    print("=" * 60)

    evaluator = RAGASEvaluatorSimple(rag_system=rag_system)

    results = evaluator.evaluate_dataset(data_path, max_samples, ragas_results_path)
    # evaluator.print_summary(results)

if __name__ == "__main__":
    main()