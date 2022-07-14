# A Framework for Adapting Machine Learning Components

This document details how to set-up the environment for reproducing the results from the paper **A Framework for Adapting Machine Learning Components**, accepted at ACSOS 2022.

## Citing

If you use our work, please cite:

```
@inproceedings{casimiro2022acsos,
  title={A Framework for Adapting Machine Learning Components},
  author={Casimiro, Maria and Romano, Paolo and Garlan, David and Rodrigues, Luis},
  year={2022},
  booktitle={Proceedings of the 2022 IEEE International Conference on Autonomic Computing and Self-Organizing Systems (ACSOS)}
}
```

## Table of Contents
1. [Setting-up the environment for the experiments](#set-up)
	1. [Hardware requirements](#hardware)
	2. [Install VirtualBox](#install-virtualbox)
	3. [Install Vagrant](#install-vagrant)
	4. [Manage the Vagrant VM](#manage)
		1. [Set-up the Vagrant VM](#set-up-vagrant)
		2. [Power-on Vagrant VM](#power-on)
		3. [Connect to the Vagrant VM](#connect)
		4. [Copy files between local host and Vagrant VM](#copy-files)
		5. [Exit the Vagrant VM](#exit)
		6. [Power-off the Vagrant VM](#power-off)
		7. [Check state of local Vagrant repository and VMs](#check-state)
		8. [Destroy the Vagrant VM](#destroy)
2. [Contents of the Vagrant VM](#contents)
	1. [Datasets](#data)
	2. [Source code](#source)
	3. [PRISM code](#prism)
3. [Reproducing the experiments](#reproduce)
	1. [Pre-generated results files](#pre-generated-results)
	2. [Re-generating retrain files and dataset](#re-generate-retrain-files-dataset)
		1. [Re-generate retrain files](#re-generate-retrain-files)
		2. [Re-generate retrain dataset](#re-generate-retrain-dataset)
	3. [Re-generating baselines and adaptive framework results](#re-generate-results)


## Setting-up the environment for the experiments <a name="set-up"></a>

### 1. Hardware requirements <a name="hardware"></a>

To deploy the setup, you need at least 16GB RAM and we advise you to deploy the experiments on a machine with at least 16 CPUs. Re-generating the retrain files will involve retraining a lot of ML models which takes a non-negligible amount of time. We ran the experiments on an AMD EPYC 7282 CPU@2.8GHz, with 16 cores and it took approximatelly 12 hours.

The base Vagrant box which is used to instantiate the VM used to reproduce the experiments is set up to run with 32GB of RAM and with a maximum of 32 cores. 

### 2. Install [VirtualBox](https://www.virtualbox.org) <a name="install-virtualbox"></a>

To install VirtualBox access their [downloads page](https://www.virtualbox.org/wiki/Downloads) and follow the instructions. Ensure you **install version ≥ 6.1.30**.


### 3. Install [Vagrant](https://www.vagrantup.com/) <a name="install-vagrant"></a>

We provide the environment for the experiments as a Vagrant box. As such, having Vagrant installed is required for deploying the provided virtual machine. To install Vagrant you can follow the steps on the [official vagrant downlowads page](https://www.vagrantup.com/downloads), selecting the adequate version for you operating system. Vagrant requires [VirtualBox](https://www.virtualbox.org/wiki/Downloads) to be installed as well.

**Note:** VirtualBox version should be ≥ 6.1.30

**MacOS:** if you have an M1 processor, which does not yet support virtualization, this set-up will not run


### 4. Manage the Vagrant VM <a name="manage"></a>

Clone the repository and **[download the base Vagrant box](https://drive.google.com/file/d/1mycdGTPOp1VeP8kE3kXgewnKpkqUW95I/view?usp=sharing)** to the repository's directory.

#### 4.1. Setting-up the Vagrant VM <a name="set-up-vagrant"></a>

**Note:** Ensure you are in the directory where you have the files `sa4ml.box` and `Vagrantfile` before running the following commands.

**a)** Install Vagrant plugins by issuing the following commands:

```
vagrant plugin install vagrant-reload
vagrant plugin install vagrant-vbguest	
vagrant plugin install vagrant-disksize
```

**b)** Add provided Vagrant box to your local Vagrant repository with command 

```
vagrant box add sa4ml_box sa4ml.box
```

**c)** Confirm that the box was correctly added to Vagrant by running command `vagrant box list`

#### 4.2. Power-on Vagrant VM <a name="power-on"></a>
To spin up a Vagrant VM or power it on, issue command `vagrant up`

The VM is created with default user `vagrant` with password `vagrant`

**Updating VM resources:** If the default resources allocated to the VM (32GB RAM and 32 cpus) do not work on your local host, you can adjust them:

```
# edit the Vagrantfile and update fields
nano Vagrantfile
vb.memory = []
vb.cpus = []

# bring the machine up
vagrant up

# [alternative] if the machine is already running when you update the Vagrantfile
vagrant reload

# confirm VM memory
free -h

# confirm VM cpus
nproc
```


#### 4.3. Connect to the Vagrant VM <a name="connect"></a>
Connect to the Vagrant VM via ssh with command `vagrant ssh sa4ml`

#### 4.4. Copy files between local host and the Vagrant VM <a name="copy-files"></a>
To copy files to/from your computer from/to the Vagrant VM, **on the local host** (i.e., your computer), execute one of the following commands:

```
# verify VM port to establish connection with the VM
# default is port 2222
vagrant port
	
# copy from local host to Vagrant VM:
scp -P 2222 [file] vagrant@127.0.0.1:[target_directory]

# copy from Vagrant VM to local host:
scp -P 2222 vagrant@127.0.0.1:[path_to_file] [target_directory]
```

**Attention -- vagrant port may change:** Although port 2222 is the default port used by Vagrant to launch the VM, if there was a previous machine running in that port, when `vagrant up` is issued it may assign a new port to the connection. In such case, the above commands should be updated to used the new port number.

#### 4.5. Exit the Vagrant VM <a name="exit"></a>
To exit the Vagrant VM simply issue comand `exit` inside the VM
 
#### 4.6. Power-off the Vagrant VM <a name="power-off"></a>
To power-off the Vagrant VM run command `vagrant halt`

#### 4.7. Check the state of your local Vagrant repository and VMs <a name="check-state"></a>
To see which Vagrant VMs are set-up on your local Vagrant repository and to see their state (e.g., running or powered off) issue command `vagrant global-status`

#### 4.8. Destroy the Vagrant VM <a name="destroy"></a>
You can destroy the Vagrant VM with command `vagrant destroy`. Note that when you destroy the VM, the next time you bring it up the process will take longer, as it will have to boot the VM from scratch.


## Contents of the Vagrant VM <a name="contents"></a>

When you connect to the Vagrant VM with `vagrant ssh sa4ml` you should see two folders in the home directory of the VM: directory `SA4ML` and directory `prism-4.6-linux64`. You don't have to worry about directory `prism-4.6-linux64` as it contains the [PRISM](https://www.prismmodelchecker.org/) installation, which is already set-up. Directory `SA4ML` contains the code and datasets to reproduce the experiments.

### Data <a name="data"></a>
All the data files (a copy is available for [download](https://drive.google.com/file/d/1gxVUwnRk1tdW3PTCJPgYC6mmlc3EXh60/view?usp=sharing)) are stored in directory `~/SA4ML/datasets/ieee-fraud-detection/`. This directory contains the following sub-folders:

- `./original/` contains the [original kaggle datasets](https://www.kaggle.com/competitions/ieee-fraud-detection/overview). The training set files `train_identity.csv` and `train_transaction.csv` are the ones used in the paper, as they contain the actual fraud labels. Files `test_identity.csv` and `test_transaction.csv` are used for data pre-processing only
- `./new/` contains the retrain dataset that we generate and which is used by the AIP to predict the impact of an adaptation
- `./tmp/` contains the retrain files which are used to generate the retrain dataset. Sub-directory `./tmp/models/` contains all the retrained models
- `./results/files/` contains the results of the execution of the proposed self-adaptation mechanism and of all the baselines
- `./results/figures/` contains the plots to analyze the performance of each baseline
- `./pre-generated/` contains all the files we generated while evaluating our hypothesis. Specifically, this folder contains the sub-folders `./tmp/`, `./new/`, and `./results/files/`


### Source code <a name="source"></a>
The source code and scripts to reproduce the results are in directory `~/SA4ML/src/`. To reproduce the experiments of the paper, you will have to use the following files/scripts:

- **`defs.py`** contains definitions for executing the different scripts (described below). For example, you can set the time interval for consecutive retrains. To reproduce the results of the paper you do not need to change this file 
- **`generate_retrain_files.py`** execute this file to generate the retrain files. These files store the performance metrics of the fraud detection system before and after the retrain is executed. The script generates one file for each retrain (at each time interval). The retrain file and the retrained model are saved in directories `~/SA4ML/datasets/ieee-fraud-detection/tmp/` and `~/SA4ML/datasets/ieee-fraud-detection/tmp/models`, respectively. By default, the retrain time interval is set to 10 hours and the script executes 315 retrains. These span the entire duration of the test set
- **`generate_retrain_dataset.py`** this script reads the retrain files, computes the metrics described in the paper (Section 3.D) Adaptation Impact Dataset), and generates the retrain dataset. The dataset is saved in directory `~/SA4ML/datasets/ieee-fraud-detection/new/` and is used to train the adaptation impact predictors (AIP) that estimate the impact of adapting the ML model
- **`exp_settings.py`** modify this file to set the parameters for the different experiments you want to run. For example, you can test different values for the recall threshold, fpr threshold, retrain latency, retrain cost, SLA violation costs
- **`run_adaptive_framework.py`** run the baselines and AIP to compare their performances. The experiments are based on the parameters specified in `exp_settings.py`. The results are stored in directory `~/SA4ML/datasets/ieee-fraud-detection/results/files`
- **`plot.py`** reproduce the figures in the paper by reading the performance result files of the baselines and of the AIP. This script generates all the figures at once, so all result files need to be computed before executing this 


### PRISM code <a name="prism"></a>
Directory `~/SA4ML/PRISM/` contains script `test_prism_ieee_cis.sh`  which is called from the main program and that executes PRISM at each time interval to decide the adaptation strategy to execute. Each time this script is called, the resulting PRISM log files with the results of the model checking are written to this directory. Specifically, the two log files are `nop_output_log.txt` and `retrain_output_log.txt`, and the results files are `adv-nop.txt` and `adv-retrain.txt`, corresponding to each of the available tactics. Directory `~/SA4ML/PRISM/model/` contains the PRISM model (`system_model.prism`, formal model of the fraud detection system) and properties (`properties.props`) files. You do not need to worry about any of these files. You will not have to change any of them to reproduce the experiments and they will not be overwritten when you deploy the experiments.


## Reproducing the experiments <a name="reproduce"></a>

To verify the hypothesis presented in the paper, i.e., that retraining is not always worth it, only when the benefits outweigh the costs, we had to build a retrain dataset. This dataset was build based on the original kaggle dataset by running multiple retrains, at different time instants, and with different sets of data. As such, there are three main steps that generate files:

1. Generate files with the results of the execution of model retrains
2. Generate the retrain dataset by processing the retrain results obtained in the previous step
3. Run the adaptive framework and test the baselines based on the retrain dataset and on the retrain results obtained in step 1

### 1. Pre-generated results files <a name="pre-generated-results"></a>
We provide the files obtained at all steps in directory `~/SA4ML/datasets/ieee-fraud-detection/pre-generated/`. More specifically in the following directories:

- retrain files: `~/SA4ML/datasets/ieee-fraud-detection/pre-generated/tmp/`
- retrain models: `~/SA4ML/datasets/ieee-fraud-detection/pre-generated/tmp/models/`
- retrain dataset: `~/SA4ML/datasets/ieee-fraud-detection/pre-generated/new/`
- baselines and adaptive framework results: `~/SA4ML/datasets/ieee-fraud-detection/pre-generated/results/files/`

You can use the pre-generated files to do any of the following:

- generate the figures in the paper by executing 
`python3 plot.py --use-pgf`. The figures will be stored in `~/SA4ML/datasets/ieee-fraud-detection/results/figures/`
- run experiments with the baselines and AIP with commnad `python3 run_adaptive_framework.py --use-pgf`. The result files will be stored in `~/SA4ML/datasets/ieee-fraud-detection/results/files/`
- re-generate the retrain dataset by executing `python3 generate_retrain_dataset.py --use-pgf`. The retrain dataset will be stored in `~/SA4ML/datasets/ieee-fraud-detection/new/`



### 2. Re-generating retrain files and dataset <a name="re-generate-retrain-files-dataset"></a>

**Note:** The following commands assume you are in the directory where the scripts are located, i.e., `~/SA4ML/src/`

To re-generate each set of files, you should do the following:

#### 2.1. Re-generate retrain files <a name="re-generate-retrain-files"></a>

To generate the files with the results of the execution of model retrains execute the following:

```
# open a tmux session
tmux new -s [session_name]

# [alternative] attach to an existing tmux session
tmux a -t [session_name]

# move to directory with the script
cd ~/SA4ML/src/

# execute the script to generate the retrain files
python3 generate_retrain_files.py

# [optional] if you want time estimates, execute the following
time python3 generate_retrain_files.py
```

The script will take a while (12 hours to 1 day) to execute as it will execute 316 model retrains. The tmux session prevents the experiment to be killed if you lose connection to the VM. Each model retrain generates 3 files, which are saved in `~/SA4ML/datasets/ieee-fraud-detection/tmp/` and `~/SA4ML/datasets/ieee-fraud-detection/tmp/models/`: 

	- metrics: a file that starts with `metrics` and contains metrics such as the real labels, predictions, scores, confusion matrix
	- times: a file that starts with `times` and contains information about each retrain, such as how long the retrain process took, and the timestamp at which it was executed
	- model: a file with the retrained model


#### 2.2. Re-generate retrain dataset<a name="re-generate-retrain-dataset"></a>
**Note:** The retrain files are required to generate the retrain dataset. Thus, if you do not have the retrain files the pre-generated retrain files will be used.

To generate the dataset which is used by the adaptive framework to predict the impact of retraining the model execute the following:

```
# open a tmux session
tmux new -s [session_name]

# [alternative] attach to an existing tmux session
tmux a -t [session_name]

# move to directory with the script
cd ~/SA4ML/src/

# execute the script to generate the retrain dataset
python3 generate_retrain_dataset.py

# [optional] if you want time estimates, execute the following
time python3 generate_retrain_dataset.py
```

This processes the retrain files obtained previously and computes metrics such as variations in the distributions of thescores of the fraud detection model. Generating this dataset is also computationally expensive (also ≈12 hours), however both this and the previous step should be performed only once. The generated dataset is saved to `~/SA4ML/datasets/ieee-fraud-detection/new/`


### 3. Re-generating baselines and adaptive framework results<a name="re-generate-results"></a>
**Note:** The adaptive framework relies on the retrain dataset to predict the impact of adapting the fraud detection model. Thus, if you do not have generated the retrain dataset this will use the pre-generated retrain files and dataset by default.

1. To reproduce the figures in the paper, you will have to change some settings in `exp_settings.py` according to the figure you want to reproduce

	1.1. Generate the result files corresponding to the experiment of Figure 2:
	
		# open a tmux session
		tmux new -s [session_name]
		
		# [alternative] attach to an existing tmux session
		tmux a -t [session_name]
		
		# move to directory with the script
		cd ~/SA4ML/src/

		# ensure exp_settings.py has the following:
		RETRAIN_COSTS = [8] 
		RETRAIN_LATENCIES = [0] # in hours
		RECALL_T = [70] # in %
		
		# execute the following command to run the baselines and AIP
		# results are saved in ~/SA4ML/datasets/ieee-fraud-detection/results/
		python3 run_adaptive_framework.py

	
	1.2. Generate the result files corresponding to the experiment of Figure 3.a):
		
		# open a tmux session
		tmux new -s [session_name]
		
		# [alternative] attach to an existing tmux session
		tmux a -t [session_name]
		
		# move to directory with the script
		cd ~/SA4ML/src/

		# ensure exp_settings.py has the following:
		RETRAIN_COSTS = [8] 
		RETRAIN_LATENCIES = [0] # in hours
		RECALL_T = [50, 60, 70, 80, 90] # in %

		# execute the following command to run the baselines and AIP
		# results are saved in ~/SA4ML/datasets/ieee-fraud-detection/results/
		python3 run_adaptive_framework.py
	
	1.3. Generate the result files corresponding to the experiment of Figure 3.b):
		
		# open a tmux session
		tmux new -s [session_name]
		
		# [alternative] attach to an existing tmux session
		tmux a -t [session_name]
		
		# move to directory with the script
		cd ~/SA4ML/src/

		# ensure exp_settings.py has the following:
		RETRAIN_COSTS = [0, 1, 5, 8, 10, 15]
		RETRAIN_LATENCIES = [0] # in hours
		RECALL_T = [70] # in %
		
		# execute the following command to run the baselines and AIP
		# results are saved in ~/SA4ML/datasets/ieee-fraud-detection/results/
		python3 run_adaptive_framework.py
	
	1.4. Generate the result files corresponding to the experiment of Figure 3.c):
		
		# open a tmux session
		tmux new -s [session_name]
		
		# [alternative] attach to an existing tmux session
		tmux a -t [session_name]
		
		# move to directory with the script
		cd ~/SA4ML/src/
		
		# ensure exp_settings.py has the following:
		RETRAIN_COSTS = [8]
		RETRAIN_LATENCIES = [0, 1, 5] # in hours
		RECALL_T = [70] # in %
		
		# execute the following command to run the baselines and AIP
		# results are saved in ~/SA4ML/datasets/ieee-fraud-detection/results/
		python3 run_adaptive_framework.py
	
	**Note:** To get time estimates run command `time python3 run_adaptive_framework.py` 

2. Once you have deployed the experiments described above, you can reproduce the  plots of the paper by calling `python3 plot.py`


