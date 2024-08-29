# Counterfactual Explanation for Time Series Data via Learned Saliency Maps
This is the repository for our paper titled "[CELS: Counterfactual Explanation for Time Series Data via Learned Saliency Maps](https://ieeexplore.ieee.org/document/10386229)". This paper has been accepted at the [2023 IEEE International Conference on Big Data (Big Data)](https://bigdataieee.org/BigData2023/). 
 
# Approach
![main](fig2.png)

# Prerequisites and Instructions
All python packages needed are listed in [pip-requirements.txt](pip-requirements.txt) file and can be installed simply using the pip command.

# Get the results for Coffee dataset by running
python3 main.py --pname CELS_Coffee --task_id 0 --run_mode turing --jobs_per_task 10 --samples_per_task 28 --dataset Coffee --algo cf --seed_value 1 --enable_lr_decay False --background_data train --background_data_perc 100 --enable_seed True --max_itr 1000 --run_id 0 --bbm dnn --enable_tvnorm True --enable_budget True --dataset_type test --l_budget_coeff 1 --run 1 --l_tv_norm_coeff 1 --l_max_coeff 1
# The results would be saved into the bigdata_cels folder

# Data
The data used in this project comes from the [UCR](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) archive.
