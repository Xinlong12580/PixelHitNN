g++ -std=c++17 test_tf_cpp_api.cc -o test_tf_cpp_api \
  $TF_CFLAGS \
  -I"$TF_INC" \
  -L"$TF_LIB" \
  -Wl,--no-as-needed \
  -l:libtensorflow_cc.so.2 \
  -l:libtensorflow_framework.so.2 \
  $TF_LFLAGS \
  -Wl,-rpath,"$TF_LIB"
