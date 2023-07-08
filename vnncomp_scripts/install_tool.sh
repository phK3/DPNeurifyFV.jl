# modified from https://github.com/intelligent-control-lab/NeuralVerification.jl/blob/vnncomp/vnncomp_scripts/install_tool.sh
apt-get update -y
apt-get install sudo -y
sudo apt-get update -y
sudo apt-get install wget -y
sudo wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.2-linux-x86_64.tar.gz
sudo tar -xvzf julia-1.9.2-linux-x86_64.tar.gz
sudo cp -r julia-1.9.2 /opt/
sudo apt-get install build-essential -y
sudo apt-get install git -y
sudo apt-get install python3 -y
sudo apt-get install python3-pip -y
sudo apt-get install psmisc
source ~/.bashrc

script_name=$0
script_path=$(dirname "$0")
project_path=$(dirname $(dirname $(realpath $0)))

julia_path="/opt/julia-1.9.2/bin/julia"

cd $project_path
# configure PyCall to use conda environment proprietary to julia installation
# install vnnlib parser (also install numpy and onnx for legacy readers)
echo '
using Pkg
Pkg.activate(".")
Pkg.add("PyCall")
ENV["PYTHON"] = ""
Pkg.add("Conda")
using Conda
Conda.pip_interop(true)
Conda.pip("install", "git+https://github.com/dlshriver/vnnlib.git@main")
Conda.add("numpy")
Conda.add("onnx")
Pkg.build("PyCall")
' | $julia_path

# restart julia for setting to take effect and install remainder of package
echo '
using Pkg
Pkg.instantiate()
' | $julia_path

echo creating sysimage, can take some time
echo '
using Pkg
Pkg.activate(".")
include("vnncomp_scripts/create_sysimage.jl")
' | $julia_path

script_name=$0
script_path=$(dirname "$0")
chmod +x ${script_path}/*.sh
