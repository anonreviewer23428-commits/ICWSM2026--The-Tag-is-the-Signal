import json
import re
from typing import Dict, List, Any
from fuzzywuzzy import fuzz
# ==============================================================================
# 辅助函数 1: 将字符串规范化 (核心修正)
# ==============================================================================
def normalize_to_unique_sorted_list(value: str) -> List[str]:
    """
    将字符串按逗号/分号分割，标准化特殊字符（如连字符），
    去重，并排序后返回一个列表。
    """
    if not value or not isinstance(value, str):
        return []

    # --- 核心修正：标准化连字符 ---
    # 定义所有需要被替换为标准连字符的字符
    # U+2011: 不换行连字符, U+2013: En Dash, U+2014: Em Dash
    hyphen_variants = ['‑', '–', '—']
    normalized_value = value
    for variant in hyphen_variants:
        # 全部替换为键盘上的标准连字符 '-' (U+002D)
        normalized_value = normalized_value.replace(variant, '-')
    # --- 标准化结束 ---
    
    items = re.split('[,;]', normalized_value)
    
    unique_items_set = {item.strip() for item in items if item and item.strip()}
    
    return sorted(list(unique_items_set))

# ==============================================================================
# 辅助函数 2: 转换一个字典里的所有值为列表
# ==============================================================================
def convert_dict_values_to_lists(data: Dict[str, str]) -> Dict[str, List[str]]:
    """将一个字典内的所有值从字符串转为唯一的、排序后的列表。"""
    if not data:
        return {}
    return {field: normalize_to_unique_sorted_list(raw_value) for field, raw_value in data.items()}

# ==============================================================================
# 辅助函数 3: 根据两个“列表化”的字典计算得分
# ==============================================================================
def calculate_scores_from_lists(pred_lists: Dict[str, List[str]], gt_lists: Dict[str, List[str]]) -> Dict[str, float]:
    """根据已经转换为列表的预测和真实标签，计算得分。"""
    scores = {}
    fields = ['theme', 'claim_types', 'ctas', 'evidence']
    
    for field in fields:
        # 将列表转换为可变的集合，以便进行匹配和移除操作
        pred_items = list(pred_lists.get(field, []))
        gt_items = list(gt_lists.get(field, [])) # 使用列表以便可以迭代和按索引移除
        
        matched_gt_indices = set() # 记录已匹配的gt项的索引，避免重复匹配

        intersection_count = 0

        # 对预测集中的每个项，尝试在真实集中找到最佳匹配
        for p_item in pred_items:
            best_score = 0
            best_gt_idx = -1

            for i, g_item in enumerate(gt_items):
                # 如果这个gt项已经被匹配过了，跳过
                if i in matched_gt_indices:
                    continue
                
                # 使用 fuzz.token_sort_ratio 进行模糊匹配
                # 它可以处理单词顺序和缺失/多余单词，对于短语非常有效
                # 也可以尝试 fuzz.ratio (简单字符匹配) 或 fuzz.partial_ratio (部分匹配)
                similarity_score = fuzz.token_sort_ratio(p_item, g_item)
                
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_gt_idx = i
            
            # 如果找到了一个匹配项，且相似度达到阈值
            if best_gt_idx != -1 and best_score >= 60:
                intersection_count += 1
                matched_gt_indices.add(best_gt_idx) # 标记此gt项已匹配

        # 分母是预测集和真实集的大小中的较大者
        denominator = max(len(pred_items), len(gt_items))
        
        scores[field] = (intersection_count / denominator) if denominator > 0 else 1.0 # 如果两者都为空，则得分为1.0
        
    return scores

# ==============================================================================
# 辅助函数 4: 打印最终的总结报告
# ==============================================================================
def print_summary_report(report_data: Dict[str, Any]):
    """将最终的总结报告以美观的格式打印到控制台。"""
    print("\n" + "="*50)
    print(" " * 18 + "评估总结报告")
    print("="*50)
    print(f" 输入文件: {report_data['source_file']}")
    print(f" 总样本数: {report_data['total_samples']}")
    print(f" 成功处理样本数: {report_data['processed_samples']}")
    print("-"*50)
    print(" 各字段平均分:")
    for field, avg_score in report_data['average_scores'].items():
        print(f"  - {field:<15}: {avg_score:.4f}")
    print("-"*50)
    print(f" 总体平均分: {report_data['overall_average_score']:.4f}")
    print("="*50)

# ==============================================================================
# 主函数：处理单个输入文件，并生成总结
# ==============================================================================
def process_evaluation_file(input_path: str, output_path: str):
    """
    读取一个JSON文件，执行所有处理步骤，保存详细文件，并打印总结报告。
    """
    # ... (这部分主函数逻辑不变)
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_path}'")
        return
    except json.JSONDecodeError as e:
        print(f"错误：文件 '{input_path}' 不是一个有效的JSON文件。错误: {e}")
        return

    if not isinstance(samples, list):
        print("错误：JSON文件的顶层结构必须是一个列表 [...]。")
        return
        
    final_results_for_file = []
    all_scores_accumulator = {
        'theme': [], 'claim_types': [], 'ctas': [], 'evidence': []
    }
    
    for sample in samples:
        processed_sample = sample.copy()
        processed_sample.pop('prompt', None)
        
        pred_dict_str = sample.get('parsed_prediction', {})
        gt_dict_str = sample.get('parsed_ground_truth', {})

        pred_dict_list = convert_dict_values_to_lists(pred_dict_str)
        gt_dict_list = convert_dict_values_to_lists(gt_dict_str)
        
        scores = calculate_scores_from_lists(pred_dict_list, gt_dict_list)
        
        for field, score in scores.items():
            all_scores_accumulator[field].append(score)
        
        processed_sample['parsed_prediction'] = pred_dict_list
        processed_sample['parsed_ground_truth'] = gt_dict_list
        processed_sample['evaluation_scores'] = scores
            
        final_results_for_file.append(processed_sample)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results_for_file, f, indent=4, ensure_ascii=False)
        print(f"\n处理完成！详细结果已保存到: {output_path}")
    except IOError as e:
        print(f"错误：无法将详细结果写入文件 '{output_path}'. 错误信息: {e}")

    average_scores = {}
    for field, score_list in all_scores_accumulator.items():
        if score_list:
            average_scores[field] = sum(score_list) / len(score_list)
        else:
            average_scores[field] = 0.0

    overall_avg = sum(average_scores.values()) / len(average_scores) if average_scores else 0.0

    summary_report = {
        "source_file": input_path,
        "total_samples": len(samples),
        "processed_samples": len(final_results_for_file),
        "average_scores": average_scores,
        "overall_average_score": overall_avg
    }
    
    print_summary_report(summary_report)

# ==============================================================================
# 用法
# ==============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute per-field agreement scores for tagger outputs.")
    parser.add_argument("--input", required=True, help="Path to evaluation_results.json")
    parser.add_argument(
        "--output",
        default="final_processed_results.json",
        help="Where to write the processed JSON with per-sample scores.",
    )
    args = parser.parse_args()
    process_evaluation_file(args.input, args.output)
