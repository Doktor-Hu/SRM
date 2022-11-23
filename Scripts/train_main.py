import os

py_TF='C:/Users/zh_hu/Documents/Test/TF2/Scripts/python' # VENV TF2


# --- MSE ALL
code_preprocess='train.py SRResNet ALL_MSE ALL 3 T T 16 2 F T 1e-5 0 V1'
os.system(py_TF+' '+code_preprocess)

code_preprocess='train.py HAN ALL_MSE ALL 3 T T 16 2 F T 1e-5 0 V1'
os.system(py_TF+' '+code_preprocess)

code_preprocess='train.py Swin ALL_MSE ALL 3 T T 16 2 F T 1e-5 64 V1'
os.system(py_TF+' '+code_preprocess)


# --- MAE ALL
code_preprocess='train.py SRResNet ALL_MAE ALL 3 T T 16 2 F T 1e-5 0 V1'
os.system(py_TF+' '+code_preprocess)

code_preprocess='train.py HAN ALL_MAE ALL 3 T T 16 2 F T 1e-5 0 V1'
os.system(py_TF+' '+code_preprocess)

code_preprocess='train.py Swin ALL_MAE ALL 3 T T 16 2 F T 1e-5 64 V1'
os.system(py_TF+' '+code_preprocess)


'''
# --- MASE ALL
code_preprocess='train.py SRResNet ALL_MASE ALL 3 T T 16 2 F T 1e-5 0'
os.system(py_TF+' '+code_preprocess)

code_preprocess='train.py HAN ALL_MASE ALL 3 T T 16 2 F T 1e-5 0'
os.system(py_TF+' '+code_preprocess)

code_preprocess='train.py Swin ALL_MASE ALL 3 T T 16 2 F T 1e-5 64'
os.system(py_TF+' '+code_preprocess)

code_preprocess='train.py Swin ALL_MAE ALL 3 T T 16 2 F T 1e-5 64'
os.system(py_TF+' '+code_preprocess)

code_preprocess='train_SR.py Swin ALL_MSE ALL 3 T T 16 2 F T 1e-5 64'
os.system(py_TF+' '+code_preprocess)

code_preprocess='train_SR.py Swin LR_MAE ALL 3 T T 16 2 F T 1e-5 64'
os.system(py_TF+' '+code_preprocess)

code_preprocess='train_SR.py Swin LR_MSE ALL 3 T T 16 2 F T 1e-5 64'
os.system(py_TF+' '+code_preprocess)

code_preprocess='train_SR.py Swin VGG_MAE ALL 3 T T 16 2 F T 1e-5 64'
os.system(py_TF+' '+code_preprocess)

code_preprocess='train_SR.py Swin VGG_MSE ALL 3 T T 16 2 F T 1e-5 64'
os.system(py_TF+' '+code_preprocess)

code_preprocess='train_SR.py Swin MAE ALL 3 T T 16 2 F T 0.005 64'
os.system(py_TF+' '+code_preprocess)

code_preprocess='train_SR.py Swin MSE ALL 3 T T 16 2 F T 0.005 64'
os.system(py_TF+' '+code_preprocess)
'''
