




import os
import warnings
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from .CodeLoaders import RolledCodes, UnrolledCodes
from .CacheHandler import PairwiseCache, LinearCache

def find_device():
    device = 'cpu'
    # if torch.cuda.is_available():
    #     device = 'cuda:0'
    #     print("CUDA device found: defaulting to CUDA device 0.\n")
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    #     print("\nApple Silicon found: defaulting to MPS.\n")
    # else:
    #     warnings.warn("No CUDA device or Apple Silicon found. Operations may be slow...\n")
    return device

class Hamming(object):
    def __init__(self, hamming_params):
        self.__comparison_batch_size = None
        self.__roll = bool(hamming_params['roll'])
        self.__verbose = bool(hamming_params['verbose'])
        self.__pairwise = bool(hamming_params['pairwise'])
        self.__reference_batch_size = int(hamming_params['reference_batch_size'])
        self.__data_paths = hamming_params['data_paths']
        self.__reference_tag = self.__data_paths[0].rsplit("__", 1)[0].split("/")[-1]
        self.__results_path = hamming_params['results_path']
        self.__device = find_device()
        self.__reference_loader = self.__set_loader(0, True)

    def calculator(self):
        print("Comparing datasets. This may take some time...\n")
        
        # 只处理第一个非reference数据集
        if len(self.__data_paths) < 2:
            print("Error: Need both reference and generated datasets")
            return
            
        dataset_tag = self.__data_paths[1].rsplit("__", 1)[0].split("/")[-1]
        print(f"\nCalculating hamming distances between {self.__reference_tag} & {dataset_tag}")
        
        # 只进行cross-comparison，不做内部比较
        self.__comparison_batch_size = int(self.__reference_batch_size * 3)
        self.__calculate_pairwise(1, dataset_tag)

    def __set_loader(self, i, reference=False):
        dataset = None
        batch_size = self.__reference_batch_size if reference else self.__comparison_batch_size
        codes = self.__load_codes(i)
        print(f"Loaded codes for dataset {i}: {codes}")
        if codes is None or len(codes.files) == 0:
            raise ValueError("iris_codes 中没有数据。")
        if reference and self.__roll:
            dataset = RolledCodes(codes)
        else:
            dataset = UnrolledCodes(codes)
        loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
        return loader

    def __load_codes(self, i):
        try:
            codes = np.load(self.__data_paths[i], allow_pickle=True)
            print(f"Codes for dataset {i}: {codes}")
            # 打印 NpzFile 对象中的键和值
            for key in codes.files:
                print(f"Key: {key}, Value shape: {codes[key].shape}")
            return codes
        except Exception as e:
            print(f"Error loading codes for dataset {i}: {e}")
            return None

    def __calculate_pairwise(self, i, tag):
        # 固定使用跨数据集比较模式
        comparison_str = 'pairwise-inter__'
        # 使用预加载的reference_loader
        reference_loader = self.__reference_loader
        # 加载generated数据集
        comparison_loader = self.__set_loader(i)
        # 设置输出路径
        target = f'{self.__results_path}/{comparison_str}{self.__reference_tag}_&_{tag}'
        os.makedirs(target, exist_ok=True)
        # 创建缓存
        Cache = PairwiseCache(target, self.__verbose, False)  # intra_comparison 固定为 False
        # 计算距离
        self.__pairwise_hamming_distance(reference_loader, comparison_loader, Cache)
        Cache.save()
        Cache.clear()

    def __calculate_linear(self, i, data_path):
        set_tag = data_path.split("__", 1)[0]
        comparison_loader = self.__set_loader(i)
        target = f'{self.__results_path}/linear__{self.__reference_tag}_&_{set_tag}'
        os.makedirs(target, exist_ok=True)
        Cache = LinearCache(target, False)
        self.__linear_hamming_distance(comparison_loader, Cache)
        Cache.save()
        Cache.clear()

    def __find_conditions(self, comparison_names):
        conditions = []
        for comparison_name in comparison_names:
            condition = comparison_name.split('___cond___')[1] if '___cond___' in comparison_name else 'NA_NA'
            conditions.append(condition)
        return tuple(conditions)

    def __pairwise_hamming_distance(self, reference_loader, comparison_loader, Cache):
        for reference_batch in tqdm(reference_loader):
            comparison_unsqueeze = 1 if self.__roll else 0
            size = reference_batch[0].shape[-1]
            reference_batch_gpu = reference_batch[0].to(self.__device).unsqueeze(1)
            for comparison_batch in comparison_loader:
                comparison_batch_gpu = comparison_batch[0].to(self.__device).unsqueeze(comparison_unsqueeze)
                result = torch.bitwise_xor(reference_batch_gpu, comparison_batch_gpu).sum(dim=-1)
                if self.__roll:
                    result = result.min(dim=-1)[0]
                for i in range(len(reference_batch[1])):
                    for j in range(len(comparison_batch[1])):
                        proto_string = sorted([reference_batch[1][i], comparison_batch[1][j]])
                        Cache.new_line(f'{proto_string[0]}|{proto_string[1]}', result[i][j] / size)

    def __linear_hamming_distance(self, comparison_loader, Cache):
        for (reference_codes, reference_names), (comparison_codes, comparison_names) in tqdm(zip(self.__reference_loader, comparison_loader)):
            reference_codes = reference_codes.to(self.__device)
            reference_names = reference_names.to(self.__device)
            comparison_codes = comparison_codes.to(self.__device)
            comparison_names = comparison_names.to(self.__device)
            comparison_codes_local = comparison_codes.unsqueeze(1).expand(-1, reference_codes.shape[1], -1) if self.__roll else comparison_codes
            xor_result = torch.bitwise_xor(reference_codes, comparison_codes_local)
            hamming_distances = xor_result.sum(dim=2)
            if self.__roll:
                hamming_distances, _ = torch.min(hamming_distances, dim=1)
            conditions = self.__find_conditions(comparison_names)
            for i in range(reference_codes.shape[0]):
                proto_string = f'{reference_names[i]}|{conditions[i]}'
                Cache.new_line(f'{proto_string}', hamming_distances[i])