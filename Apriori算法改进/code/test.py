from dataset import Dataset


def read_results(filename, test_id: int):
    lines = []
    count=0
    with open(filename, "r") as file:
        reading_block = False
        for line in file:
            line = line.strip()  # 去掉开头和结尾的空格符
            if reading_block:
                if line:  # 如果不是空行，则添加到结果列表中
                    lines.append(line)
                else:  # 如果是空行，则停止读取
                    break
            elif line.startswith(f"##### {test_id}"):
                reading_block = True
                # 提取括号中的数字
                count_str = line.split("计")[-1].split("个")[0]
                if count_str.isdigit():
                    count = int(count_str)
    return lines, count


task = "Groceries"
min_support = 5
# task = "test"
path_data = f"./data/{task}.json"
path_result = f"./result/{task}_{min_support}.txt"
dataset = Dataset(path_data)

test_ids = range(1,11)
my_sum=0
for test_id in test_ids:
    results, count = read_results(path_result, test_id)
    my_sum+=count
    print(count,end="|")

print(f"\nthe sum is {my_sum}")

# for i, result in enumerate(results):
#     if dataset.check_vector(result.split(","), min_support) == False:
#         print(f"Error!The set:\n\n{result}\n\ndoesn't meet the min_support.")
#         break

# print(f"The test_{test_id} for {task}_{min_support} pass.")
