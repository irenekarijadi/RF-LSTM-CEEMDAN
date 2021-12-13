# Building-Energy-Prediction-using-Hybrid-RF-LSTM-based-CEEMDAN 
 [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/irenekarijadi/RF-LSTM-CEEMDAN/HEAD)
 
This repository contains all the code for the hybrid method using RF-LSTM based CEEMDAN

## Authors
*Irene Karijadi, Shuo-Yan Chou*

*corresponding author: irenekarijadi92@gmail.com (Irene Karijadi)*

Table of Contents
=================

* [Background](#Background) 
* [Framework](#Framework) 
* [Installation Requirement](#Installation-Requirement) 
* [Dataset](#Dataset) 
* [Setting](#Setting)
* [Reproducibility workflow](#Reproducibility-workflow)


## Background

> An accurate method for building energy consumption prediction is important for building energy management system. However, building energy consumption data often exhibits nonlinear and nonstationary patterns, which makes prediction more difficult. In this study, we proposed a hybrid method of Random Forest (RF), and Long Short-Term Memory (LSTM) based on Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN) to predict building energy consumption.


## Framework
This is the framework of the proposed method
![alt text](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Figures/Framework.png)

## Installation Requirement

The proposed method is coded in Python 3.7.6
In order to run the model, a number of packages need to be installed. Here are the list of the package  version that we used to run all the experiments

* EMD-signal 0.2.10
* pandas 0.25.3
* keras 2.4.3
* tensorflow>=2.0.0
* sklearn 0.22.1
* numpy 1.18.1


The complete list of packages can be found in [requirements.txt.](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/requirements.txt)

## Dataset

We used a public dataset from the [Building Data Genome Project](https://www.google.com/search?q=building+data+genome+project&oq=Building+Data+Genome+Project&aqs=chrome.0.35i39j69i59l2j69i64j69i59j69i60l3.558j0j7&sourceid=chrome&ie=UTF-8) 


Five different buildings are used in this study and you can download it here:
[Office Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20Office_Abigail.csv), [University Classroom Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20UnivClass_Abby.csv),[University Dormitory Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20UnivDorm_Prince.csv),[University Laboratory Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20UnivLab_Christy.csv), [Primary Classroom Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20PrimClass_Jaden.csv)

To visualize hourly energy consumption from five buildings, please  running the [Plot Dataset.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Plot%20Dataset.ipynb)


![alt text](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Figures/hourly%20energy%20consumption%20from%20five%20buildings.png)

## Setting
A number of parameters (e.g. LSTM learning rate, RF feature number, etc) are defined in the [Setting.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Setting.ipynb)

## Reproducibility workflow

1. In order to run the model, the packages need to be installed first using this code:
`pip install -r requirements.txt()`
2. To visualize the data, run the [Plot Dataset.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Plot%20Dataset.ipynb)
3. For experiments, the results of the proposed method and other benchmarking methods can be obtained by running the [! main.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/!%20main.ipynb)


### Results
The performance of the proposed method was compared with other prediction methods, including linear regression (LR), random forest (RF), support vector regression (SVR), Artificial Neural Network, Long Short-Term Memory (LSTM), Complete Ensemble Empirical Mode Decomposition with Adaptive Noise-Random Forest (CEEMDAN-RF), and Complete Ensemble Empirical Mode Decomposition with Adaptive Noise - Long Short-Term Memory (CEEMDAN-LSTM). 

![alt text](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Figures/univdorm%20result.PNG)
![alt text](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Figures/univlab%20result.PNG)
![alt text](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Figures/univclass%20result.PNG)
![alt text](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Figures/office%20result.PNG)
![alt text](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Figures/primclass%20result.PNG)

According to the experimental results, the proposed method has the lowest error and has the best prediction accuracy among the benchmark methods.

### Percentage Improvement
We further measure the improvement of the proposed method to other benchmarking methods.
The percentages of error improvement with other benchmark methods are computed in the file [! main.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/!%20main.ipynb)

![alt text](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Figures/PI_univdorm.PNG)
![alt text](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Figures/PI_univlab.PNG)
![alt text](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Figures/PI_univclass.PNG)
![alt text](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Figures/PI_office.PNG)
![alt text](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Figures/PI_primclass.PNG)



File Organization
------------

    ├── LICENSE
    ├── README.md          <- The README file for using this code
    ├── Dataset            <- Data used in this study can be found in Dataset directory with a description of the 
    ├── ! main.ipynb       <- This is the main file used to conduct all the experiments using settings specified in Setting.ipynb
    ├── Setting.ipynb      <-  This is a file used to set a number of parameters that are used throughout the functions in the directory.
    ├── Plot Dataset.ipynb <-  This file contains the script to visualize the data
    ├── Linear_regression.ipynb <-  This file contains all the functions for the Linear Regression that are used in the experiments
    ├── support_vector_regression.ipynb <-  This file contains all the functions for the Support Vector Regression that are used in the experiments
    ├── artificial_neural_network.ipynb <-  This file contains all the functions for the Artificial Neural Networks that are used in the experiments
    ├── random_forest.ipynb <-  This file contains all the functions for the Random Forest hat are used in the experiments
    ├── lstm.ipynb <-  This file contains all the functions for the LSTM that are used in the experiments
    ├── hybrid_ceemdan_rf.ipynb <-  This file contains all the functions for the hybrid ceemdan rf that are used in the experiments
    ├── hybrid_ceemdan_lstm.ipynb <-  This file contains all the functions for the hybrid ceemdan lstm that are used in the experiments
    ├── proposed_model_hybrid_ceemdan_rf_lstm.ipynb <-  This file contains all the functions for the proposed method that are used in the experiments

