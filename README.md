# RF-LSTM-CEEMDAN

# Building-Energy-Prediction-using-Hybrid-RF-LSTM-based-CEEMDAN 
 [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/irenekarijadi/Building-Energy-Prediction-using-Hybrid-RF-LSTM-based-CEEMDAN/HEAD)

This repository contains all the code for the hybrid method using RF-LSTM based CEEMDAN

## Authors
*Irene Karijadi, Shuo-Yan Chou*

*corresponding author: irenekarijadi92@gmail.com (Irene Karijadi)*

## Table of Contents

* [Background](#Background) 
* [Framework](#Framework) 
* [Requirement](#Requirement) 
* [Dataset](#Dataset) 
* [Setting](#Setting)
* [Results](#Results)


## Background

> An accurate method for building energy consumption prediction is important for building energy management system. However, building energy consumption data often exhibits nonlinear and nonstationary patterns, which makes prediction more difficult. In this study, we proposed a hybrid method of Random Forest (RF), and Long Short-Term Memory (LSTM) based on Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN) to predict building energy consumption.


## Framework
This is the framework of the proposed method
![alt text](https://github.com/irenekarijadi/Building-Energy-Prediction-using-Hybrid-RF-LSTM-based-CEEMDAN/blob/main/Framework.png)

## Requirement
Related packages:

* EMD-signal
* pandas
* python-math
* tensorflow>=2.0.0
* sklearn
* numpy
* python-time


The complete list of packages can be found in [requirements.txt.](https://github.com/irenekarijadi/Building-Energy-Prediction-using-Hybrid-RF-LSTM-based-CEEMDAN/blob/v1/requirements.txt)

In order to run the model, the packages need to be installed first using this code:
`!pip install -r requirements.txt()`


## Dataset

We used a public dataset from the [Building Data Genome Project](https://www.google.com/search?q=building+data+genome+project&oq=Building+Data+Genome+Project&aqs=chrome.0.35i39j69i59l2j69i64j69i59j69i60l3.558j0j7&sourceid=chrome&ie=UTF-8) 


Five different buildings are used in this study and you can download it here:
[Office Building](https://github.com/irenekarijadi/Building-Energy-Prediction-using-Hybrid-RF-LSTM-based-CEEMDAN/blob/v1/data%20of%20Office_Abigail.csv), [University Classroom Building](https://github.com/irenekarijadi/Building-Energy-Prediction-using-Hybrid-RF-LSTM-based-CEEMDAN/blob/v1/data%20of%20UnivClass_Abby.csv),[University Dormitory Building](https://github.com/irenekarijadi/Building-Energy-Prediction-using-Hybrid-RF-LSTM-based-CEEMDAN/blob/v1/data%20of%20UnivDorm_Prince.csv),[University Laboratory Building](https://github.com/irenekarijadi/Building-Energy-Prediction-using-Hybrid-RF-LSTM-based-CEEMDAN/blob/v1/2.%20Experiment%20University%20Laboratory.ipynb), [Primary Classroom Building](https://github.com/irenekarijadi/Building-Energy-Prediction-using-Hybrid-RF-LSTM-based-CEEMDAN/blob/v1/5.%20Experiment%20Primary%20Classroom%20Building.ipynb)

To visualize hourly energy consumption from five buildings, please  running the [Plot Dataset.ipynb](https://github.com/irenekarijadi/Building-Energy-Prediction-using-Hybrid-RF-LSTM-based-CEEMDAN/blob/main/Plot%20Dataset.ipynb)

![alt text](https://github.com/irenekarijadi/Building-Energy-Prediction-using-Hybrid-RF-LSTM-based-CEEMDAN/blob/main/hourly%20energy%20consumption%20from%20five%20buildings.png)


## Setting
A number of parameters (e.g. LSTM learning rate, RF feature number, etc) are defined in the setting.ipynb file

## Results

The performance of the proposed method was compared with other prediction methods, including linear regression (LR), random forest (RF), support vector regression (SVR), Artificial Neural Network, Long Short-Term Memory (LSTM), Complete Ensemble Empirical Mode Decomposition with Adaptive Noise-Random Forest (CEEMDAN-RF), and Complete Ensemble Empirical Mode Decomposition with Adaptive Noise - Long Short-Term Memory (CEEMDAN-LSTM). 


* To obtain the experimental result for University Dormitory, please run the [1. Experiment University Dormitory Building.ipynb](https://github.com/irenekarijadi/Building-Energy-Prediction-using-Hybrid-RF-LSTM-based-CEEMDAN/blob/main/1.%20Experiment%20University%20Dormitory%20Building.ipynb)


According to the results in Table 3, the proposed method has the lowest error and has the best prediction accuracy among the benchmark methods.


## A list of python files:

* Plot Dataset.ipynb : Exploratory analysis and plots of data
* Linear_regression.ipynb : Implement function for the Linear Regression model. 
* 1.Experiment University Dormitory Building.ipynb : The executable python program of proposed method and other benchmarking methods for university dormitory building dataset

