import torch

print('Pytorch version: ', torch.__version__)
if torch.cuda.is_available():
    print("CUDA is available")
    print('CUDA version: ', torch.version.cuda)
    print('count of devices: ', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'device {i} name: ', torch.cuda.get_device_name(i))
        print(f'device {i} capability: ', torch.cuda.get_device_capability(i))
        print(f'device {i} total memory: ', torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024, 'GB')
        print(f'device {i} memory usage: ',
              torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory * 100, '%')
        print('Is TensorCore supported: ', 'Yes' if (torch.cuda.get_device_properties(i).major >= 7) else 'Na')
        print('is BFloat16 supported: ', 'Yes' if (torch.cuda.is_bf16_supported()) else 'Na')

elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("MPS is available")

else:
    print("you are using CPU")