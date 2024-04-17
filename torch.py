import os
import torch

def main():
    os.system('python -m torch.utils.collect_env')
    print("\n")
    os.system('nvcc --version')
    print("\n")
    os.system('nvidia-smi')
    print("\n")
    print("Cuda available: ", torch.cuda.is_available())
    print("\n") 
    
    
    # select device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA/CPU device:", device)
    print("pyTorch version:", torch.__version__)
       
if __name__ == '__main__':
    main()
    
    
