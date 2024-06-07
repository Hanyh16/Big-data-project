from dataset import Dataset
from model import apriori, apriori_cut
from time import time


def write_result(result, path_result, run_time, min_support):
    with open(path_result, "w") as fout:
        fout.write("Apriori 频繁项集结果\n\n")
        fout.write("参数:\n")
        fout.write(f"\tmin_support={min_support}\n")
        fout.write("\n")

        fout.write(f"运行时间:{run_time}\n\n")

        num_result = 0
        for i in result:
            num_result += i.shape[0]
        fout.write(f"频繁项集(总计: {num_result}个,长度: [1,{result[-1].shape[1]}]):\n")
        num_element_map = dataset.index2element_map
        for length, frequent_itemsets in enumerate(result):
            fout.write("#" * 5 + f" {length+1}(总计{frequent_itemsets.shape[0]}个):\n")
            if frequent_itemsets.shape[0] < 10000:
                for frequent_itemset in frequent_itemsets:
                    fout.write(" " * 5 + "\t")
                    fout.write(
                        ",".join(
                            num_element_map[element] for element in frequent_itemset
                        )
                    )
                    fout.write("\n")
            fout.write("\n")
    return


task = "Groceries"
# task = "test"
path_data = f"./data/{task}.json"
dataset = Dataset(path_data)
min_support_list = [5]

for min_support in min_support_list:
    path_result = f"./result/{task}_{min_support}_cut_new.txt"

    start_time = time()
    result = apriori_cut(dataset, min_support)
    end_time = time()

    write_result(result, path_result, end_time - start_time, min_support)

    print(f"End_cut_{min_support}.")
