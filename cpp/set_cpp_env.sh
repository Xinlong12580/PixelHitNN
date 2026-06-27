export PYTF=/afs/cern.ch/user/x/xinlong/public/NNCPE/PixelHitNN/venv/lib64/python3.9/site-packages/tensorflow
export TF_INC=/afs/cern.ch/user/x/xinlong/public/NNCPE/PixelHitNN/venv/lib64/python3.9/site-packages/tensorflow/include
export TF_LIB=/afs/cern.ch/user/x/xinlong/public/NNCPE/PixelHitNN/venv/lib64/python3.9/site-packages/tensorflow
export TF_CFLAGS='-I/afs/cern.ch/user/x/xinlong/public/NNCPE/PixelHitNN/venv/lib64/python3.9/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=1 --std=c++17 -DEIGEN_MAX_ALIGN_BYTES=64'
export TF_LFLAGS='-L/afs/cern.ch/user/x/xinlong/public/NNCPE/PixelHitNN/venv/lib64/python3.9/site-packages/tensorflow -l:libtensorflow_framework.so.2'

####Uncomment the section below for your venv

#export PYTF=$(python - <<'EOF'
#import tensorflow as tf, os
#print(os.path.dirname(tf.__file__))
#EOF
#)
#export TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#export TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#export TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
#export TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

echo PYTF: $PYTF
echo TF_INC: $TF_INC
echo TF_LIB: $TF_LIB
echo TF_CFLAGS: $TF_CFLAGS
echo TF_LFLAGS: $TF_LFLAGS

export LD_LIBRARY_PATH=$TF_LIB:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTF/../nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTF/../nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTF/../nvidia/cufft/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTF/../nvidia/curand/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTF/../nvidia/cusolver/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTF/../nvidia/cusparse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTF/../nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTF/../nvidia/cuda_cupti/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
