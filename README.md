# Building Energy Consumption Prediction  
## A hybrid RF-LSTM based on CEEMDAN for improving the accuracy of building energy consumption prediction  

This is the original source code used for all experiments in the paper [A hybrid RF-LSTM based on CEEMDAN for improving the accuracy of building energy consumption prediction](https://www.sciencedirect.com/science/article/pii/S0378778822000792)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/irenekarijadi/RF-LSTM-CEEMDAN/HEAD)


Access this Binder by clicking the blue badge above

If you use the code in this paper, please cite the paper as below:


*Karijadi, I., & Chou, S. Y. (2022). A hybrid RF-LSTM based on CEEMDAN for improving the accuracy of building energy consumption prediction. Energy and Buildings, 259, 111908*

## Authors

*Irene Karijadi, Shuo-Yan Chou*


*corresponding author: irenekarijadi92@gmail.com (Irene Karijadi)*

## Background
An accurate method for building energy consumption prediction is important for building energy management system. However, building energy consumption data often exhibits  nonlinear and nonstationary patterns, which makes prediction more difficult. In this study, we proposed a hybrid method of Random Forest (RF), and Long Short-Term Memory (LSTM) based on Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN) to predict building energy consumption.


## Framework
This is the framework of the proposed method      
![Alt text](Figures/Framework.png)

## Prerequisites
The proposed method is coded in Python 3.7.6 and the experiments were performed on Intel Core i3-8130U CPU, 2.20GHz, with a memory size of 4.00 GB.
The python version is specified in [runtime.txt.](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/runtime.txt)
In order to run the experiments, a number of packages need to be installed. Here are the list of the package  version that we used to run all the experiments

* EMD-signal==0.2.10
* pandas==0.25.3
* keras==2.4.3
* tensorflow>=2.0.0
* sklearn==0.22.1
* numpy==1.18.1
* matplotlib
* dataframe_image

The complete list of packages can be found in [requirements.txt.](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/requirements.txt)

In order to run the model, the packages need to be installed first using this line of code:

`pip install -r requirements.txt()`


## Description of File
Non-python file
* Dataset - This folder includes all dataset used in this study
* Figures - This folder includes all generated figures to be used in reporting
* README.md - The README file for using this code 
* License - The License file
* requirement.txt - This file contains list of packages used in this study
* runtime.txt - This file contains python version used in this study 

Python Files:
* `1.Experiments for University Dormitory Building.ipynb` - This notebook is the main file used to conduct the experiments for university dormitory building using parameter settings specified in Setting.py
* `2.Experiments for University Laboratory Building.ipynb` - This notebook is the main file used to conduct the experiments for university laboratory building using parameter settings specified in Setting.py
* `3.Experiments for University Classroom Building.ipynb` - This notebook is the main file used to conduct the experiments for university classroom building using parameter settings specified in Setting.py
* `4.Experiments for Office Building.ipynb` - This notebook is the main file used to conduct the experiments for office building using parameter settings specified in Setting.py
* `5.Experiments for Primary Classroom Building.ipynb`- This is the main file used to conduct the experiments for primary classroom building using parameter settings specified in Setting.py
* `Plot CEEMDAN result.ipynb` - This notebook contains the script to plot the Decomposition results
* `Plot Prediction results using proposed hybrid RF-LSTM based CEEMDAN method.ipynb` - This notebook contains the script to plot the prediction results from proposed method 
* `Plot Dataset.ipynb` - This notebook contains the script to visualize the data
* `myfunctions.py` - This python script includes all functions required for building proposed method and other benchmark methods that are used in the experiments
* `Setting.py` - This is python script includes a number of parameters that are used throughout the functions in the directory


## Dataset
We used a public dataset from the [Building Data Genome Project](https://www.google.com/searchq=building+data+genome+project&oq=Building+Data+Genome+Project&aqs=chrome.0.35i39j69i59l2j69i64j69i59j69i60l3.558j0j7&sourceid=chrome&ie=UTF-8) 

Five different buildings are used in this study and you can download it here:
[Office Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20Office_Abigail.csv),
[University Classroom Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20UnivClass_Abby.csv),
[University Dormitory Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20UnivDorm_Prince.csv),
[University Laboratory Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20UnivLab_Christy.csv), 
[Primary Classroom Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20PrimClass_Jaden.csv)

To visualize hourly energy consumption from five buildings, please  running the `Plot Dataset.ipynb`

![alt text](Figures/hourly%20energy%20consumption%20from%20five%20buildings.png)


## Parameter Setting
A number of parameters (e.g. LSTM learning rate, RF feature number, etc) are defined in the `Setting.py`

## Experiments
The code that generated results presented in table 3 and 4 Section 4.4 in the paper can be executed from these notebooks:

`1. Experiments for University Dormitory Building.ipynb`

`2. Experiments for University Dormitory Building.ipynb`

`3. Experiments for University Dormitory Building.ipynb`

`4. Experiments for University Dormitory Building.ipynb`

`5. Experiments for University Dormitory Building.ipynb`

notes: The number indicate the order of the files need to be run

### Workflow

1. Run the `1. Experiments for University Dormitory Building.ipynb`
   By running all the cell in this notebook, it will:
   - Train and test the proposed method and other benchmark methods on University Dormitory dataset. 
   - Generate Table which summarize the performance results of the proposed method and other benchmark methods on University Dormitory dataset. This output is used as part of Table 3 Section 4.4 in the paper
   - Generate Table which calculate the percentage of improvement of the proposed method and other benchmark methods on University Dormitory dataset. This output is used as part of Table 4 Section 4.4 in the paper 

2. Run the `2. Experiments for University Laboratory Building.ipynb`
   By running all the cell in this notebook, it will:
   - Train and test the proposed method and other benchmark methods on University Laboratory dataset. 
   - Generate Table which summarize the performance results of the proposed method and other benchmark methods on University Laboratory dataset. This output is used as part of  Table 3 Section 4.4 in the paper
   - Generate Table which calculate the percentage of improvement of the proposed method and other benchmark methods on University Laboratory dataset. This output is used as part of Table 4 Section 4.4 in the paper 

3. Run the `3. Experiments for University Classroom Building.ipynb`
   By running all the cell in this notebook, it will:
   - Train and test the proposed method and other benchmark methods on University Classroom dataset. 
   - Generate Table which summarize the performance results of the proposed method and other benchmark methods on University Classroom dataset. This output is used as part of Table 3 Section 4.4 in the paper
   - Generate Table which calculate the percentage of improvement of the proposed method and other benchmark methods on University Classroom dataset. This output is used as part of Table 4 Section 4.4 in the paper 


4. Run the `4. Experiments for Office Building.ipynb`
   By running all the cell in this notebook, it will:
   - Train and test the proposed method and other benchmark methods on Office dataset. 
   - Generate Table which summarize the performance results of the proposed method and other benchmark methods on Office dataset. This output is used as part of Table 3 Section 4.4 in the paper
   - Generate Table which calculate the percentage of improvement of the proposed method and other benchmark methods on Office dataset. This output is used as part of Table 4 Section 4.4 in the paper 

5. Run the `5. Experiments for Primary Classroom Building.ipynb`
   By running all the cell in this notebook, it will:
   - Train and test the proposed method and other benchmark methods on Primary Classroom dataset. 
   - Generate Table which summarize the performance results of the proposed method and other benchmark methods on Primary Classroom dataset. This output is used as part of Table 3 Section 4.4 in the paper
   - Generate Table which calculate the percentage of improvement of the proposed method and other benchmark methods on Primary Classroom dataset. This output is used as part of Table 4 Section 4.4 in the paper 


### Plotting of results

1. To visualize the decomposition results `Plot CEEMDAN result.ipynb` must be executed. The output generated from this notebook is used in the Figure 5 Section 4.4 in the paper
2. To obtain and visualize prediction results using proposed hybrid RF-LSTM based CEEMDAN method,`Plot Prediction results using proposed hybrid RF-LSTM based CEEMDAN method.ipynb` must be executed. The output generated from this notebook is used in the Figure 6 Section 4.4 in the paper         
 
 
