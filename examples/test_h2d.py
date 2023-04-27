import numpy as np
import torch
import time



shape=[4096, 32, 1024]
host_mem = np.zeros(shape, dtype=np.float32)
torch_tensor = torch.Tensor(host_mem)

iters=1000
start = time.time()
for i in range(iters):
    device_mem = torch_tensor.cuda()
end = time.time() 
h2d_time = end - start
print(f"H2D time: {h2d_time*1e3} ms")


# Calculate bandwidth 
bandwidth = (np.prod(shape) * 4) / h2d_time / 1e6 * iters # MB/s

print(f"Bandwidth: {bandwidth} MB/s")
