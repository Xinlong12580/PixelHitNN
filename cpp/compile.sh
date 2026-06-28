g++ -std=c++17 "$1".cc -o $1 \
  $TF_CFLAGS \
  -I"$TF_INC" \
  -L"$TF_LIB" \
  -Wl,--no-as-needed \
  -l:libtensorflow_cc.so.2 \
  -l:libtensorflow_framework.so.2 \
  $TF_LFLAGS \
  -Wl,-rpath,"$TF_LIB"
