# coding=utf8


import torch
import torch.distributed as dist
import numpy as np


def test(device, rank, local_rank):
    _shape = [30592, 64, 1024]
    data_bytes = np.prod(_shape) * 2
    x1 = torch.randn(_shape, dtype=torch.float16)
    x1 = x1.to(device)
    x2 = torch.ones(_shape, dtype=torch.float16)
    x2 = x2.to(device)
    iters = 1
    torch.cuda.synchronize()
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)
    t_start.record()
    for i  in range(iters):
        dist.all_reduce(x1)
    t_end.record()
    torch.cuda.synchronize()
    gen_time = t_start.elapsed_time(t_end) / 1000 # convert mill to sec
    bandwith = data_bytes / gen_time / 1e6  * iters
    print(f"the local_rank is {local_rank}; the transfer time is:{gen_time}, the bandwith is :{bandwith} MB/s")
    y = x1 * x2
    return y

if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    device_num = torch.cuda.device_count()
    rank = dist.get_rank()
    local_rank = rank % device_num
    device = torch.device('cuda', local_rank)
    test(device, rank, local_rank)
