import cffi

source_code = open("src/pyfilterbank/sosfilt.c").read()

ffi = cffi.FFI()
ffi.cdef("""
void sosfilter(float*, int, float*, int, float*);
void sosfilter_double(double*, int, double*, int, double*);
void sosfilter_double_mimo(double*, int, int, double*, int, int, double*);
""")
ffi.set_source("pyfilterbank._sosfilt", source_code)

if __name__ == "__main__":
    ffi.compile(verbose=True)
