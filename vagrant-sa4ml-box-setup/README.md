# Setting-up the base Vagrant Box

Instructions on how to create the base Vagrant box:

1. **[optional]** In directory `./vagrant-box-setup/` adapt the file `Vagrantfile` to your desired settings. For example, you can add extra python packages, or configure extra vagrant features such as networking features or shared folders.

2. Run command `vagrant up` to boot the VM.

3. [Download](http://www.prismmodelchecker.org/download.php) the Linux version of the PRISM model checker, version 4.6 

4. Copy the source code and data files to the Vagrant VM:

	```
	# verify VM port to establish connection with the VM
	vagrant port
	
	# use the local host port output by the previous command
	# and copy directory SA4ML to the Vagrant VM
	# default user: vagrant
	# password: vagrant
	scp -P [local-host-port] -r ./SA4ML/ vagrant@127.0.0.1:/home/vagrant/
	
	# copy PRISM to VM
	scp -P [local-host-port] prism-4.6-linux64.tar.gz vagrant@127.0.0.1:/home/vagrant/
	
	# download all the pre-generated data files and datasets
	https://drive.google.com/file/d/1gxVUwnRk1tdW3PTCJPgYC6mmlc3EXh60/view?usp=sharing
	
	# copy it to the machine 
	scp -P ./datasets.tar.gz -r ./SA4ML/ vagrant@127.0.0.1:/home/vagrant/SA4ML/
	```

5. Log-in to the Vagrant VM with command `vagrant ssh sa4ml`

6. Install PRISM

	```
	# extract PRISM
	tar xzvf prism-4.6-linux64.tar.gz
	
	# change to PRISM dir
	cd prism-4.6-linux64
	
	# install PRISM
	./install.sh
	```
	
7. Extract data files

	```
	cd ~/SA4ML/
	tar xzvf datasets.tar.gz
	```
	
At this point, your VM is ready and you can deploy the experiments to reproduce the paper, using the scripts in directory `~/SA4ML/src/` 

If you want to package the vagrant VM to generate the box we provide, you have to execute the command `vagrant package --base sa4ml --output sa4ml.box`, which should take approximately 25 minutes to run.
	