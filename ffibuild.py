import cffi

ffibuilder = cffi.FFI()
with open('./pyfilterbank/sosfilt.c', 'r') as fp:
    ffibuilder.set_source(
        "pyfilterbank._sosfilt",
        fp.read(),
        extra_compile_args=["-std=c99"]
        )

ffibuilder.cdef("""
void sosfilter(float*, int, float*, int, float*);
void sosfilter_double(double*, int, double*, int, double*);
void sosfilter_double_mimo(double*, int, int, double*, int, int, double*);
""")
ffibuilder.set_unicode(True)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)