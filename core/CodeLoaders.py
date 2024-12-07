import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CodeLoader(Dataset):
    def __init__(self, iris_codes):
        self.iris_codes = iris_codes
        self.names = list(iris_codes.files)
        self.names.sort()
        
        # 调试信息
        print(f"加载的 iris_codes 键: {self.names}")
        
        if len(self.names) == 0:
            raise ValueError("iris_codes 中没有数据。")
        
        self.height = iris_codes[self.names[0]].shape[2] * iris_codes[self.names[0]].shape[1] # 实部和虚部
        self.width = iris_codes[self.names[0]].shape[3]        

    def __len__(self):
        return len(self.names)

class RolledCodes(CodeLoader):
    def __getitem__(self, idx):
        # 调试信息
        print(f"处理 RolledCodes 索引: {idx}")
        
        iris_code = torch.empty((self.width, self.height * self.width), dtype=torch.bool) # 分配内存
        for i in range(self.width):
            iris_code[i] = torch.from_numpy(np.roll(self.iris_codes[self.names[idx]], shift=i, axis=3).reshape(1, self.height*self.width)) # 滚动，展平，存储

        return iris_code, self.names[idx]
    
class UnrolledCodes(CodeLoader):
    def __getitem__(self, idx):
        # 调试信息
        print(f"处理 UnrolledCodes 索引: {idx}")
        
        iris_code = torch.from_numpy(self.iris_codes[self.names[idx]].reshape(-1))

        return iris_code, self.names[idx]

def __set_loader(self, i, reference=False):
    dataset = None
    batch_size = None
    codes = np.load(self.__data_paths[i], allow_pickle=True)
    
    # 调试信息
    print(f"从 {self.__data_paths[i]} 加载的代码，键: {codes.files}")

    if reference:
        batch_size = self.__reference_batch_size
    else:
        batch_size = self.__comparison_batch_size

    if reference and self.__roll:
        dataset = RolledCodes(codes)      
    else:
        dataset = UnrolledCodes(codes)
    
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    return loader