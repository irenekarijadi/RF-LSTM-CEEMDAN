<p align="center"> 
</p>
<h1 align="center"> Building Energy Consumption Prediction  </h1>
<h3 align="center"> A hybrid RF-LSTM based on CEEMDAN for improving the accuracy of building energy consumption prediction </h3>  
</br>
 
This is the original source code used for all experiments in the paper  *"A hybrid RF-LSTM based on CEEMDAN for improving the accuracy of building energy consumption prediction"* 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/irenekarijadi/RF-LSTM-CEEMDAN/HEAD)<br />
Access this Binder by clicking the blue badge above

## Authors
*Irene Karijadi, Shuo-Yan Chou*

*corresponding author: irenekarijadi92@gmail.com (Irene Karijadi)*

<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> :book: Table of Contents</h2>
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Background"> ➤ Background</a></li>
    <li><a href="#Framework"> ➤ Framework</a></li>
    <li><a href="#Prerequisites"> ➤ Prerequisites</a></li>
    <li><a href="#Description-of-files"> ➤ Description of files</a></li>
    <li><a href="#Dataset"> ➤ Dataset</a></li>
    <li><a href="#Parameter-Setting"> ➤ Parameter Setting</a></li>
    <li><a href="#Reproducibility-workflow"> ➤ Reproducibility workflow</a></li>
    <li><a href="#Result"> ➤ Result</a></li>
  </ol>
</details>

<!-- Background -->
<h2 id="Background"> :pencil: Background</h2>
<p align="justify"> 
An accurate method for building energy consumption prediction is important for building energy management system. However, building energy consumption data often exhibits  nonlinear and nonstationary patterns, which makes prediction more difficult. In this study, we proposed a hybrid method of Random Forest (RF), and Long Short-Term Memory (LSTM) based on Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN) to predict building energy consumption.
</p>

   
<!-- Framework-->
<h2 id="Framework"> :page_with_curl: Framework</h2>
This is the framework of the proposed method
<p align="center">
  <img src="Figures/Framework.png" alt="Table1" width="70%" height="70%">        
  <!--figcaption>Caption goes here</figcaption-->
</p>


<!-- Prerequisites -->
<h2 id="prerequisites"> :file_folder: Prerequisites</h2>

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

<!-- Description of File -->
<h2 id="Description of File"> :floppy_disk: Description of File</h2>

<h4>Non-python Files:</h4>
<ul>
  <li><b>Dataset</b> - This folder includes all dataset used in this study</li>
  <li><b>Figures</b> - This folder includes all generated figures to be used in reporting.</li>
  <li><b>README.md</b> - The README file for using this code </li>
  <li><b>License</b> - The License file </li>
  <li><b>requirement.txt</b> - This file contains list of packages used in this study </li>
  <li><b>runtime.txt</b> - This file contains python version used in this study </li>
</ul>

<h4>Python Files:</h4>
<ul>
  <li><b>1. Experiments for University Dormitory Building.ipynb</b> - This is the main file used to conduct the experiments for university dormitory building using parameter settings specified in Setting.ipynb</li>
  <li><b>2. Experiments for University Laboratory Building.ipynb</b> - This is the main file used to conduct the experiments for university laboratory building using parameter settings specified in Setting.ipynb</li>
  <li><b>3. Experiments for University Classroom Building.ipynb</b> - This is the main file used to conduct the experiments for university classroom building using parameter settings specified in Setting.ipynb</li>
  <li><b>4. Experiments for Office Building.ipynb</b> - This is the main file used to conduct the experiments for office building using parameter settings specified in Setting.ipynb</li>
  <li><b>5. Experiments for Primary Classroom Building.ipynb</b> - This is the main file used to conduct the experiments for primary classroom building using parameter settings specified in Setting.ipynb</li>
  <li><b>Setting.ipynb</b> - This is a file used to set a number of parameters that are used throughout the functions in the directory</li>
  <li><b>Plot CEEMDAN result.ipynb</b> - This file contains the script to plot the Decomposition results </li> 
  <li><b>Plot Prediction results using proposed hybrid RF-LSTM based CEEMDAN method.ipynb</b> - This file contains the script to plot the prediction results from proposed method </li>
  <li><b>Plot Dataset.ipynb</b> - This file contains the script to visualize the data</li>
  <li><b>Linear_regression.ipynb.ipynb</b> - This file includes all functions required for LR that are used in the experiments</li>
  <li><b>support_vector_regression.ipynb.ipynb</b> - This file includes all functions required for SVR that are used in the experiments</li>
  <li><b>artificial_neural_network.ipynb.ipynb</b> - This file includes all functions required for ANN that are used in the experiments</li>
  <li><b>random_forest.ipynb.ipynb</b> - This file includes all functions required for RF that are used in the experiments</li>
  <li><b>lstm.ipynb.ipynb</b> - This file includes all functions required for LSTM that are used in the experiments</li>
  <li><b>hybrid_ceemdan_rf.ipynb.ipynb</b> - This file includes all functions required for hybrid CEEMDAN RF that are used in the experiments</li>
  <li><b>hybrid_ceemdan_lstm.ipynb.ipynb</b> - This file includes all functions required for hybrid CEEMDAN LSTM that are used in the experiments</li>
  <li><b>hybrid_ceemdan_lstm.ipynb.ipynb</b> - This file includes all functions required for proposed method that are used in the experiments</li>
</ul>

<!-- DATASET -->
<h2 id="dataset"> :chart_with_upwards_trend: Dataset</h2>

We used a public dataset from the
[Building Data Genome Project](https://www.google.com/searchq=building+data+genome+project&oq=Building+Data+Genome+Project&aqs=chrome.0.35i39j69i59l2j69i64j69i59j69i60l3.558j0j7&sourceid=chrome&ie=UTF-8) 


Five different buildings are used in this study and you can download it here:
[Office Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20Office_Abigail.csv), [University Classroom Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20UnivClass_Abby.csv),[University Dormitory Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20UnivDorm_Prince.csv),[University Laboratory Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20UnivLab_Christy.csv), [Primary Classroom Building](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Dataset/data%20of%20PrimClass_Jaden.csv)

To visualize hourly energy consumption from five buildings, please  running the [Plot Dataset.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Plot%20Dataset.ipynb)


![alt text](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Figures/hourly%20energy%20consumption%20from%20five%20buildings.png)

<!-- Parameter Setting -->
<h2 id=Parameter Setting">:clipboard: Parameter Setting</h2>

A number of parameters (e.g. LSTM learning rate, RF feature number, etc) are defined in the [Setting.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Setting.ipynb)

<!-- Reproducibility workflow -->
<h2 id=Reproducibility workflow>:computer: Reproducibility workflow</h2>

➤In order to run the model, the packages need to be installed first using this line of code:</p>
`pip install -r requirements.txt()`</p>
➤To visualize the data, run the [Plot Dataset.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Plot%20Dataset.ipynb)</p>
➤To visualize the decomposition results [Plot CEEMDAN result.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Plot%20CEEMDAN%20result.ipynb) must be executed. </p>
➤To obtain and visualize prediction results using proposed hybrid RF-LSTM based CEEMDAN method, must be executed [Plot Prediction results using proposed hybrid RF-LSTM based CEEMDAN method.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/Plot%20Prediction%20results%20using%20proposed%20hybrid%20RF-LSTM%20based%20CEEMDAN%20method.ipynb) must be executed. </p>                       
➤To obtain the performance results of the proposed method compared with other prediction methods for University Dormitory dataset, [1. Experiments for University Dormitory Building.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/1.%20Experiments%20for%20University%20Dormitory%20Building.ipynb) must be executed. </p>  
➤To obtain the performance results of the proposed method compared with other prediction methods for University Laboratory dataset, [2. Experiments for University Laboratory Building.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/2.%20Experiments%20for%20University%20Laboratory%20Building.ipynb)must be executed. </p>  
➤To obtain the performance results of the proposed method compared with other prediction methods for University Classroom dataset, [3. Experiments for University Classroom Building.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/3.%20Experiments%20for%20University%20Classroom%20Building.ipynb) must be executed. </p>  
➤To obtain the performance results of the proposed method compared with other prediction methods result for Office dataset, [4. Experiments for Office Building.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/4.%20Experiments%20for%20Office%20Building.ipynb) must be executed. </p>  
➤To obtain the performance results of the proposed method compared with other prediction methods result for Primary Classroom dataset, [5. Experiments for Primary Classroom Building.ipynb](https://github.com/irenekarijadi/RF-LSTM-CEEMDAN/blob/main/5.%20Experiments%20for%20Primary%20Classroom%20Building.ipynb) must be executed. </p>  
                

                         
                    
<!-- Results-->
<h2 id=Result>:computer: Result</h2>
<p align="justify">                                                  
The performance of the proposed method was compared with other prediction methods, including linear regression (LR), random forest (RF), support vector regression (SVR), Artificial Neural Network, Long Short-Term Memory (LSTM), Complete Ensemble Empirical Mode Decomposition with Adaptive Noise-Random Forest (CEEMDAN-RF), and Complete Ensemble Empirical Mode Decomposition with Adaptive Noise - Long Short-Term Memory (CEEMDAN-LSTM). 
</p>
After executing these files:</p> 
➤  <b>1. Experiments for University Dormitory Building.ipynb</b></p>
➤  <b>2. Experiments for University Laboratory Building.ipynb</b></p> 
➤  <b>3. Experiments for University Classroom Building.ipynb</b></p> 
➤  <b>4. Experiments for Office Building.ipynb</b></p> 
➤  <b>5. Experiments for Primary Classroom Building.ipynb</b></p> 
</p>
We can obtain the performance results and percentage of improvement of the proposed method compared with other benchmarking methods </p>       
The performance of the proposed method compared with other benchmarking methods are shown in the following tables.</p>
<p align="center">
  <img src="Figures/univdorm_summary_table.png" alt="Table2" width="70%" height="70%">        
  <!--figcaption>Caption goes here</figcaption-->
</p>
<p align="center">
  <img src="Figures/univlab_summary_table.png" alt="Table3" width="70%" height="70%">        
  <!--figcaption>Caption goes here</figcaption-->
</p>
<p align="center">
  <img src="Figures/univclass_summary_table.png" alt="Table4" width="70%" height="70%">        
  <!--figcaption>Caption goes here</figcaption-->
</p>
<p align="center">
  <img src="Figures/office_summary_table.png" alt="Table5" width="70%" height="70%">        
  <!--figcaption>Caption goes here</figcaption-->
</p>
<p align="center">
  <img src="Figures/primclass_summary_table.png" alt="Table6" width="70%" height="70%">        
  <!--figcaption>Caption goes here</figcaption-->
</p>
</p>                                                                                     

The percentage of improvement of the proposed method compared with other benchmarking methods are shown in the following tables.
<p align="center">
  <img src="Figures/PI_univdorm_table.png" alt="Table2" width="70%" height="70%">        
  <!--figcaption>Caption goes here</figcaption-->
</p>
<p align="center">
  <img src="Figures/PI_univlab_table.png" alt="Table3" width="70%" height="70%">        
  <!--figcaption>Caption goes here</figcaption-->
</p>
<p align="center">
  <img src="Figures/PI_univclass_table.png" alt="Table4" width="70%" height="70%">        
  <!--figcaption>Caption goes here</figcaption-->
</p>
<p align="center">
  <img src="Figures/PI_office_table.png" alt="Table5" width="70%" height="70%">        
  <!--figcaption>Caption goes here</figcaption-->
</p>
<p align="center">
  <img src="Figures/PI_primclass_table.png" alt="Table6" width="70%" height="70%">        
  <!--figcaption>Caption goes here</figcaption-->
</p>
                                                                                      
According to the experimental results above, the proposed method has the lowest error and has the best prediction accuracy among the benchmark methods.


