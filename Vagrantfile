# -*- mode: ruby -*-
# vi: set ft=ruby : 
VAGRANTFILE_API_VERSION = "2"



Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|

    # Always use Vagrant's default insecure key
    config.ssh.insert_key = false
	# Always use X11 Forwarding with this machine
    config.ssh.forward_x11 = true
	# To avoid/allow install and uninstall of VBoxGuessAdditions.
    config.vbguest.auto_update = false  
	# Disable/enable automatic box update checking.
    config.vm.box_check_update = false  


	config.vm.define "sa4ml" do |sa4ml_config|
		sa4ml_config.vm.box = "sa4ml_box"
		sa4ml_config.vm.hostname = "sa4ml"


		#VM Settings
		sa4ml_config.vm.provider "virtualbox" do |vb|
				vb.name = "sa4ml"
				vb.memory = "32767"	# 32GB
				vb.cpus = "32"
			end # of vb
		end # of sa4ml_config end

end # of config
