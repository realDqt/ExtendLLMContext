import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version (编译时):", torch.version.cuda)           # e.g. "11.7"
print("CUDA driver version (运行时):", torch._C._cuda_getDriverVersion())  # e.g. 12100
print("当前 GPU 设备数:", torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    prop = torch.cuda.get_device_properties(0)
    print("GPU 名称:", prop.name)
    print("可用显存:", prop.total_memory / 1024**3, "GB")
