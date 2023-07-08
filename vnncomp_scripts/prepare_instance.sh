# modified from https://github.com/intelligent-control-lab/NeuralVerification.jl/blob/vnncomp/vnncomp_scripts/prepare_instance.sh
TOOL_NAME=DPNeurifyFV
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4

echo "Preparing $TOOL_NAME for benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE' and vnnlib file '$VNNLIB_FILE'"

supported_benchmarks=( acasxu tllverifybench )

match=0
for bm in $supported_benchmarks; do
	if [ $CATEGORY = $bm ]; then
		match=1
		break
	fi
done

if [ $match = 0 ]; then
	echo $CATEGORY "not supported"
	exit 1
fi

# kill any zombie processes
killall -q julia

yes n | case $ONNX_FILE in *.gz) gzip -kd $ONNX_FILE;; esac

# script returns a 0 exit code if successful. If you want to skip a benchmark category you can return non-zero.
exit 0