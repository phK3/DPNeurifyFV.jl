# taken from https://github.com/intelligent-control-lab/NeuralVerification.jl/blob/vnncomp/vnncomp_scripts/install_tool.sh
apt-get update -y
apt-get install sudo -y
sudo apt-get update -y
sudo apt-get install wget -y
#sudo wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz
sudo wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.2-linux-x86_64.tar.gz
sudo tar -xvzf julia-1.9.2-linux-x86_64.tar.gz
sudo cp -r julia-1.9.2 /opt/
sudo ln -s /opt/julia-1.9.2/bin/julia /usr/local/bin/julia
sudo apt-get install build-essential -y
sudo apt-get install git -y
sudo apt-get install python3 -y
sudo apt-get install python3-pip -y
sudo apt-get install psmisc
source ~/.bashrc

# vnnlib reader
pip3 install git+https://github.com/dlshriver/vnnlib.git@main

script_name=$0
script_path=$(dirname "$0")
#project_path=$(dirname "$script_path")
project_path=$(dirname $(dirname $(realpath $0)))

cd $project_path
echo '
using Pkg
Pkg.activate(".")
Pkg.instantiate()
#Pkg.add("https://github.com/sisl/NeuralVerification.jl#0d9be34")
#Pkg.add("https://github.com/sisl/NeuralPriorityOptimizer.jl#main")
#Pkg.add("https://github.com/phK3/VnnlibParser.jl#master")
#Pkg.add("https://github.com/phK3/PyVnnlib.jl#main")
#Pkg.add("https://github.com/phK3/VNNLib.jl#extension")
#Pkg.add("PyCall")
' | julia

script_name=$0
script_path=$(dirname "$0")
chmod +x ${script_path}/*.sh