import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ragSystem.rag_system import RAGSystem

from seper.huggingface_models import HuggingfaceModel
from seper.semantic_entropy import EntailmentDeberta
from seper.calculate import (
    gen_answers_batch,
    calculate_uncertainty_soft_batch,
    create_collate_fn,
    process_item_for_seper
)

@dataclass
class EvaluationResult:
    """评估结果数据类"""
    question: str
    ground_truth_answer: str
    rag_answer: str
    seper_score: float
    seper_baseline: float
    delta_seper: float
    contexts_used: List[str]
    confidence: float
    is_correct: bool = False


class RAGSEPEREvaluator:
    """RAG系统的SEPER语义困惑度评估器"""

    def __init__(self,
                 rag_system: RAGSystem,
                 generator: HuggingfaceModel,
                 entailment_model: EntailmentDeberta,
                 num_generations: int = 5,  # 减少生成次数以加快速度
                 sub_batch_size: int = 5,
                 temperature: float = 1.0,
                 max_new_tokens: int = 64,  # 减少token数以加快速度
                 max_context_words: int = 2048,
                 prompt_type: str = 'default',
                 computation_chunk_size: int = 4,  # 减小块大小
                 device: str = "cuda"):
        """
        初始化评估器
        """
        self.rag_system = rag_system
        self.generator = generator
        self.entailment_model = entailment_model
        self.num_generations = num_generations
        self.sub_batch_size = sub_batch_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_context_words = max_context_words
        self.prompt_type = prompt_type
        self.computation_chunk_size = computation_chunk_size
        self.device = device

        # 创建collate函数
        self.seper_collate_fn = create_collate_fn([
            'question', 'response_text', 'answers',
            'likelihood', 'context_label', 'log_liks_agg', 'context'
        ])

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

                    # 限制上下文长度
                    if len(context_text) > self.max_context_words * 4:  # 粗略估计
                        context_text = context_text[:self.max_context_words * 4]

                    data.append({
                        'id': sample.get('id', f'sample_{i}'),
                        'question': sample.get('question', ''),
                        'answer': sample.get('answer', ''),
                        'type': sample.get('type', ''),
                        'level': sample.get('level', ''),
                        'context': context_text
                    })
                except json.JSONDecodeError:
                    continue

        print(f"加载了 {len(data)} 条HotpotQA数据")
        return data

    def get_rag_response(self, question: str, context: str) -> Dict:
        """使用RAG系统获取回答"""
        from langchain_core.documents import Document

        # 创建文档
        doc = Document(page_content=context, metadata={"source": "hotpotqa"})

        # 设置知识库
        self.rag_system.setup_knowledge_base(documents=[doc])

        # 查询
        result = self.rag_system.query(question)

        return result

    def evaluate_single_question(self,
                                 question: str,
                                 ground_truth: str,
                                 context: str) -> Optional[EvaluationResult]:
        """
        评估单个问题的SEPER分数
        """
        try:
            # 1. 获取RAG系统的回答
            rag_result = self.get_rag_response(question, context)
            rag_answer = rag_result['result']
            contexts = rag_result.get('contexts', [])

            # 检查RAG答案是否正确（简单包含检查）
            is_correct = ground_truth.lower() in rag_answer.lower()

            # 2. 使用上下文进行生成（RAG场景）
            example_with_context = {
                'question': question,
                'context': context[:self.max_context_words * 4],  # 限制上下文长度
                'answers': [ground_truth]
            }

            result_with_context = gen_answers_batch(
                example_with_context,
                self.generator,
                self.temperature,
                self.num_generations,
                self.sub_batch_size,
                self.max_new_tokens,
                self.prompt_type,
                self.device,
                self.max_context_words
            )

            # 3. 无上下文基线
            example_baseline = {
                'question': question,
                'context': '',
                'answers': [ground_truth]
            }

            result_baseline = gen_answers_batch(
                example_baseline,
                self.generator,
                self.temperature,
                self.num_generations,
                self.sub_batch_size,
                self.max_new_tokens,
                self.prompt_type,
                self.device,
                self.max_context_words
            )

            # 4. 计算SEPER分数
            with torch.no_grad():
                r_context = process_item_for_seper(result_with_context)
                r_baseline = process_item_for_seper(result_baseline)

                seper_input = self.seper_collate_fn([r_context, r_baseline])
                seper_scores = calculate_uncertainty_soft_batch(
                    seper_input,
                    self.entailment_model,
                    self.computation_chunk_size
                )

                seper_context = seper_scores[0]
                seper_baseline = seper_scores[1]
                delta_seper = seper_context - seper_baseline

            # 5. 获取置信度
            confidence = self.rag_system.get_retrieval_confidence(question)

            return EvaluationResult(
                question=question,
                ground_truth_answer=ground_truth,
                rag_answer=rag_answer,
                seper_score=seper_context,
                seper_baseline=seper_baseline,
                delta_seper=delta_seper,
                contexts_used=contexts[:2],
                confidence=confidence,
                is_correct=is_correct
            )

        except Exception as e:
            print(f"  评估出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_dataset(self,
                         data_path: str,
                         max_samples: Optional[int] = None,
                         save_results: bool = True,
                         output_path: str = "seper_evaluation_results.json") -> List[EvaluationResult]:
        """评估整个数据集"""
        data = self.load_hotpotqa_data(data_path, max_samples)

        results = []

        print(f"\n开始评估 {len(data)} 个问题...")
        print("=" * 80)

        for idx, sample in enumerate(tqdm(data, desc="评估进度")):
            print(f"\n问题 {idx + 1}: {sample['question'][:80]}...")

            result = self.evaluate_single_question(
                question=sample['question'],
                ground_truth=sample['answer'],
                context=sample['context']
            )

            if result:
                results.append(result)

                print(f"  标准答案: {sample['answer']}")
                print(f"  RAG答案: {result.rag_answer[:100]}...")
                print(f"  ΔSEPER: {result.delta_seper:.4f} ({'改进' if result.delta_seper > 0 else '退化'})")

            if len(results) > 0 and (idx + 1) % 5 == 0:
                avg_delta = np.mean([r.delta_seper for r in results])
                correct_rate = np.mean([r.is_correct for r in results])
                print(f"\n--- 进度统计 (已完成 {len(results)} 个) ---")
                print(f"  平均ΔSEPER: {avg_delta:.4f}")
                print(f"  RAG准确率: {correct_rate * 100:.1f}%")
                print("-" * 40)

        if save_results and results:
            self.save_results(results, output_path)

        return results

    def save_results(self, results: List[EvaluationResult], output_path: str):
        """保存评估结果"""
        output_data = []
        for r in results:
            output_data.append({
                'question': r.question,
                'ground_truth_answer': r.ground_truth_answer,
                'rag_answer': r.rag_answer,
                'seper_score': r.seper_score,
                'seper_baseline': r.seper_baseline,
                'delta_seper': r.delta_seper,
                'confidence': r.confidence,
                'is_correct': r.is_correct,
                'contexts_used': r.contexts_used
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {output_path}")

    def print_summary(self, results: List[EvaluationResult]):
        """打印评估摘要"""
        if not results:
            print("没有评估结果")
            return

        seper_scores = [r.seper_score for r in results]
        seper_baselines = [r.seper_baseline for r in results]
        delta_sepers = [r.delta_seper for r in results]
        confidences = [r.confidence for r in results]
        correct_rates = [r.is_correct for r in results]

        print("\n" + "=" * 80)
        print("评估摘要")
        print("=" * 80)
        print(f"总评估问题数: {len(results)}")

        print(f"\nSEPER分数统计:")
        print(f"  有上下文平均SEPER: {np.mean(seper_scores):.4f} ± {np.std(seper_scores):.4f}")
        print(f"  无上下文平均SEPER: {np.mean(seper_baselines):.4f} ± {np.std(seper_baselines):.4f}")
        print(f"  平均ΔSEPER: {np.mean(delta_sepers):.4f} ± {np.std(delta_sepers):.4f}")

        print(f"\nRAG系统性能:")
        print(f"  RAG答案准确率: {np.mean(correct_rates) * 100:.2f}%")
        print(f"  平均检索置信度: {np.mean(confidences):.4f}")

        # 按ΔSEPER分组
        positive = [d for d in delta_sepers if d > 0]
        negative = [d for d in delta_sepers if d < 0]
        neutral = [d for d in delta_sepers if d == 0]

        print(f"\nΔSEPER分布:")
        print(f"  ΔSEPER > 0: {len(positive)} ({len(positive) / len(results) * 100:.1f}%)")
        print(f"  ΔSEPER < 0: {len(negative)} ({len(negative) / len(results) * 100:.1f}%)")
        print(f"  ΔSEPER = 0: {len(neutral)} ({len(neutral) / len(results) * 100:.1f}%)")

        # 最佳和最差案例
        if delta_sepers:
            best_idx = np.argmax(delta_sepers)
            worst_idx = np.argmin(delta_sepers)

            print(f"\n最佳改进案例 (ΔSEPER = {delta_sepers[best_idx]:.4f}):")
            print(f"  Q: {results[best_idx].question[:100]}...")
            print(f"  GT: {results[best_idx].ground_truth_answer}")
            print(f"  RAG: {results[best_idx].rag_answer[:100]}...")

            print(f"\n最差改进案例 (ΔSEPER = {delta_sepers[worst_idx]:.4f}):")
            print(f"  Q: {results[worst_idx].question[:100]}...")
            print(f"  GT: {results[worst_idx].ground_truth_answer}")
            print(f"  RAG: {results[worst_idx].rag_answer[:100]}...")


def main():
    """主函数"""
    print("=" * 80)
    print("RAG系统SEPER语义困惑度评估")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")

    llm_model_path = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"
    embedding_model_path =  '/root/autodl-tmp/models/sentence-transformers/all-MiniLM-L6-v2'
    entailment_model_path = "/root/autodl-tmp/models/microsoft/deberta-v2-xlarge-mnli"

    num_generations = 3  
    max_samples = 50  
    data_path = "./data/hotpotqa_100_samples.json"

    print(f"\n配置参数:")
    print(f"  LLM模型: {llm_model_path}")
    print(f"  嵌入模型: {embedding_model_path}")
    print(f"  蕴含模型: {entailment_model_path}")
    print(f"  每个问题生成次数: {num_generations}")
    print(f"  评估样本数: {max_samples}")

    try:
        # 1. 初始化RAG系统
        print("\n[1/4] 初始化RAG系统...")
        rag_system = RAGSystem(
            model_name=llm_model_path,
            embedding_model_path=embedding_model_path,
            temperature=0.1,
            max_tokens=256,
            chunk_size=500,  # 减小块大小
            chunk_overlap=100,
            top_k=3  # 减少检索数量
        )

        # 2. 初始化生成器
        print("\n[2/4] 初始化生成器...")
        generator = HuggingfaceModel(
            stop_sequences='default',
            max_new_tokens=64,
            device=device,
            rag_sys=rag_system,
        )
        generator.model.eval()

        # 3. 初始化蕴含检测模型
        print("\n[3/4] 初始化蕴含检测模型...")
        entailment_model = EntailmentDeberta(
            local_path=entailment_model_path,
            device=device
        )
        entailment_model.model.eval()

        # 4. 创建评估器并运行评估
        print("\n[4/4] 开始评估...")
        evaluator = RAGSEPEREvaluator(
            rag_system=rag_system,
            generator=generator,
            entailment_model=entailment_model,
            num_generations=num_generations,
            sub_batch_size=min(num_generations, 3),
            temperature=1.0,
            max_new_tokens=64,
            max_context_words=1024,
            prompt_type='default',
            computation_chunk_size=4,
            device=device
        )

        # 运行评估
        results = evaluator.evaluate_dataset(
            data_path=data_path,
            max_samples=max_samples,
            save_results=True,
            output_path="seper_evaluation_results.json"
        )

        evaluator.print_summary(results)

    except Exception as e:
        print(f"\n评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("评估完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()