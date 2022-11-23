# Transferability of single-image super resolution to geophysical downscaling

This is a reference implementation of geographical downscaling using super resolution, intended for use in the climate sciences domain. This code supports the paper " "Transferability of single-image super resolution to geophysical downscaling" - Zhongyang Hu, Peter Kuipers Munneke, Stef Lhermitte, Yao Sun, Brice Noël, Melchior van Wessem, Lichao Mou, and Xiao Xiang Zhu. 2022.

## Data Availability

RACMO2 27 km and 5.5 km simulations are freely available from (https://www.projects.science.uu.nl/iceclimate/models/racmo-model.php#1-1, IMAU, 2022; latest accessed on 3 October 2022) are provided by Van Wessem et al. (2018, 2016) and are available upon request from the original authors. QuikSCAT-derived surface melt estimations are provided by Trusel et al. (2013) and are available upon request from the original authors. Automatic weather station observations from AWS 14, 15, 17, and 18 are available on PANGAEA via DOI: https://doi.org/10.1594/PANGAEA.910473.

## Framework

[1] Preprocessing RACMO2 27 km and 5.5 km Simulations using Preprocessing_main.py

    -> 1. Cropping and coregistrating RACMO2 27 and 5.5 km resolution (including data NetCDF to Geo Tiff and AOI partitioning)
    -> 2. Generating Training, development, testing data sets

    
[2] Training Super Resolution Models using train_main.py

    python 
    
Training a super resolution model

positional arguments:
  SR_Model              SRResNet, HAN, Swin
  LOSS_Mode             lOSS mode: MSE, VGG, LR, ALL
  AOIs                  Area of Intersts
  Numver_of_Input_Variables
                        Input Variables
  Write_NPY             Write Numpy Array [Y/N]
  Plot_NPY              Plot the Written Numpy Array [Y/N]
  Batch_Size            Batch Size
  Training_Padding      Training Padding
  Standization          Standization [Y/N]
  Save_Format           Save mMdel as TF [Y/N]
  MD                    Early Stopping Minimal Delta
  Embedding_Dim         Embedding Dimension for Swin
  model_version         Model Version

[3] Applying super resolution models

[4] Visual Evaluation

