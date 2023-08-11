# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

# User specific environment and startup programs

PATH=$PATH:$HOME/.local/bin:$HOME/bin

export PATH

# User specific environment and startup programs for Argon
module load stack/2021.1 
module load matlab/R2021a
module load hdf5/1.10.7_intel-2021.2.0-mpi


