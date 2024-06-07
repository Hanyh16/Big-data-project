import numpy as np
from tqdm import tqdm
import copy


def apriori(dataset, min_support):
    # 参数列表
    frequent_itemsets = []
    # 候选项集 (n,l)
    candidate_itemset = np.arange(dataset.elements_num).reshape(-1, 1)
    while True:
        # 支持度汇总(n,)
        sum_support = np.zeros(candidate_itemset.shape[0])
        # 遍历每个bucket，得到每个候选项集的支持度
        for event in tqdm(
            dataset,
            desc=f"Getting support scores, candidate len={len(frequent_itemsets)+1}",
        ):
            sum_support += np.all(np.isin(candidate_itemset, event), axis=1)
        # 提取出支持度大于阈值的候选项集作为频繁项集
        k_itemsets = candidate_itemset[sum_support >= min_support]
        if len(k_itemsets) == 0:
            break
        else:
            frequent_itemsets.append(k_itemsets)
            # 构建新的候选项集集合，其首先为set，便于剔除重复的候选项集
            candidate_itemset = set()
            # 遍历每个刚得到的频繁项集，构建新的候选项集
            for k_itemset in tqdm(
                k_itemsets,
                desc=f"Getting candidate itemsets,new candidate len={len(frequent_itemsets)+1}",
            ):
                # 找到不在该频繁项集中的元素
                new_elements = np.setdiff1d(frequent_itemsets[0], k_itemset)
                # 利用这些元素，构建新的候选项集
                for new_element in new_elements:
                    new_cadidate = sorted(np.append(k_itemset, new_element))    # 对候选项集进行内部排序，来去除重复的候选项集
                    # 这里可以利用不成功的项集进行剪枝，而不是直接导入
                    candidate_itemset.add(tuple(new_cadidate))
            # 将每个元组转换为 NumPy 数组并添加到列表中
            candidate_itemset = [np.array(tensor) for tensor in candidate_itemset]

            # 将列表转换为 NumPy 张量
            candidate_itemset = np.array(candidate_itemset)

    return frequent_itemsets

def apriori_cut(dataset, min_support):
    frequent_itemsets = []
    candidate_itemset = np.arange(dataset.elements_num).reshape(-1, 1)
    Fail_itemsets = [[[]] for i in range(dataset.elements_num)]
    while True:
        sum_support = np.zeros(candidate_itemset.shape[0])
        for event in tqdm(
            dataset,
            desc=f"Getting support scores, candidate len={len(frequent_itemsets)+1}",
        ):
            sum_support += np.all(np.isin(candidate_itemset, event), axis=1)
        # sum_support/=len(dataset)
        k_itemsets = candidate_itemset[sum_support >= min_support]
        if len(k_itemsets) == 0:
            break
        else:
            frequent_itemsets.append(k_itemsets)

            fail_candidates = candidate_itemset[sum_support < min_support]
            for i, fail_candidate in enumerate(fail_candidates):
                for element in fail_candidate:
                    Fail_itemsets[element][-1].append(fail_candidates[i])
            for element, fail_itemset in enumerate(Fail_itemsets):
                if len(fail_itemset[-1]) != 0:
                    fail_itemset[-1] = np.array(fail_itemset[-1])
                    fail_itemset.append([])

            candidate_itemset = set()
            for k_itemset in tqdm(
                k_itemsets,
                desc=f"Getting candidate itemsets,new candidate len={len(frequent_itemsets)+1}",
            ):
                new_elements = np.setdiff1d(frequent_itemsets[0], k_itemset)
                for new_element in new_elements:
                    new_cadidate = np.sort(np.append(k_itemset, new_element))
                    if tuple(new_cadidate) not in candidate_itemset and candidate_check(
                        new_cadidate, Fail_itemsets[new_element][:-1]
                    ):
                        candidate_itemset.add(tuple(new_cadidate))
            if len(candidate_itemset) == 0:
                break

            # 将每个元组转换为 NumPy 数组并添加到列表中
            candidate_itemset = [np.array(tensor) for tensor in candidate_itemset]

            # 将列表转换为 NumPy 张量
            candidate_itemset = np.array(candidate_itemset)

    return frequent_itemsets


def candidate_check(candidate, fail_itemsets):
    for length, fail_itemset in enumerate(fail_itemsets):
        check_result = np.sum(np.all(np.isin(fail_itemset, candidate), axis=1))
        if check_result != 0:
            return False
    return True