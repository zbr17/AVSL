import numpy as np 
import torch

def topk_mask(input, k, dim=0, ismax=True, eps=1e-8):
    topk_value = torch.topk(input, k=k, dim=dim, largest=ismax)[0]
    if ismax:
        topk_value_bound = torch.min(topk_value, dim=dim, keepdim=True)[0] - eps
        topk_mask = input >= topk_value_bound
    else:
        topk_value_bound = torch.max(topk_value, dim=dim, keepdim=True)[0] + eps
        topk_mask = input <= topk_value_bound
    return topk_mask

def generate_slice(batch_size, split_num):
    if batch_size <= split_num:
        return {
            0: np.arange(batch_size)
        }
    else:
        num_last_group = int(batch_size) % int(split_num)
        interval = int(batch_size) // int(split_num)
        # generate split-dict
        slice_dict = {
            i: np.arange(i * interval, (i+1) * interval)
            for i in range(split_num)
        }
        # add the last
        if num_last_group != 0:
            index = split_num
            while(num_last_group > 0):
                if num_last_group > interval:
                    slice_dict[index] = np.arange(
                        index * interval,
                        (index + 1) * interval
                    )
                else:
                    slice_dict[index] = np.arange(
                        index * interval,
                        index * interval + num_last_group
                    )
                index += 1
                num_last_group -= interval
        
        return slice_dict

if __name__ == "__main__":
    # data = generate_slice(100, 10)
    input = torch.randn(3, 4, 5)
    mask = topk_mask(input, k=2, dim=0, ismax=True)
    print(input)
    print(mask)
    pass