import cffi

ffi = cffi.FFI()
ffi.cdef("""
void sosfilter(float*, int, float*, int, float*);
void sosfilter_double(double*, int, double*, int, double*);
void sosfilter_double_mimo(double*, int, int, double*, int, int, double*);
""")
ffi.set_source("pyfilterbank._sosfilt", open("src/pyfilterbank/sosfilt.c").read())
ffi.compile()