import numpy as np
import json


class Dataset:
    def __init__(self, data_path="./data/test.json") -> None:
        data = self.load_data(data_path)

        unique_set = set()
        for event in data:
            unique_set.update(event)

        self.element2index_map = {}
        self.index2element_map = {}
        for index, element in enumerate(unique_set):
            self.element2index_map[element] = index
            self.index2element_map[index] = element

        self.data = []
        for event in data:
            event_index = [self.element2index_map[element] for element in event]
            self.data.append(np.array(event_index))

        self.elements_num = len(unique_set)
        self.index = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.data):
            value = self.data[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration

    def load_data(self, file_path):
        # 从JSON文件加载购买记录数据
        with open(file_path, "r") as file:
            data = json.load(file)
        return [set(record.split(",")) for record in data.values()]

    def check_vector(self, list_str: list, min_support: int):
        candidate = np.zeros(len(list_str))
        for i, element in enumerate(list_str):
            if element not in self.element2index_map:
                return False
            else:
                candidate[i] = self.element2index_map[element]
        sum_support = 0
        for event in self.data:
            sum_support += np.all(np.isin(candidate, event), axis=0)
            if sum_support >= min_support:
                return True

        return False


if __name__ == "__main__":
    a = Dataset()
    print(a.train_data)
