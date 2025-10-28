import torch
import random
from batch_invariant_ops import set_batch_invariant_mode

# A = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
# B = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
# ref = torch.mm(A, B)
# for _ in range(1000):
#     assert (torch.mm(A, B) - ref).abs().max().item() == 0


# vals = [1e-10, 1e-5, 1e-2, 1]
# vals = vals + [-v for v in vals]

# results = []
# random.seed(42)
# for _ in range(10000):
#     random.shuffle(vals)
#     results.append(sum(vals))

# results = sorted(set(results))
# print(f"There are {len(results)} unique results: {results}")



torch.set_default_device('cuda')

# Just to get the logging out of the way haha
with set_batch_invariant_mode(True):
    pass

def test_batch_invariance():
    B, D = 2048, 4096
    a = torch.linspace(-100, 100, B*D).reshape(B, D)
    b = torch.linspace(-100, 100, D*D).reshape(D, D)
    
    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = torch.mm(a[:1], b)
    
    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out2 = torch.mm(a, b)[:1]
    
    # Check if results are identical
    diff = (out1 - out2).abs().max()
    print(f"Difference: {diff.item()}")
    return diff.item() == 0

# Test with standard PyTorch (likely to show differences)
print("Standard PyTorch:")
with set_batch_invariant_mode(False):
    is_deterministic = test_batch_invariance()
    print(f"Deterministic: {is_deterministic}")

# Test with batch-invariant operations
print("\nBatch-Invariant Mode:")
with set_batch_invariant_mode(True):
    is_deterministic = test_batch_invariance()
    print(f"Deterministic: {is_deterministic}")