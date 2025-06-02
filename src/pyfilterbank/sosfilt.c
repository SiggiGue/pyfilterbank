#include <stdio.h>
#include <math.h>


void sosfilter(float* signal, int nsamp, float* sos, int ksos, float* states){

    for (int k=0; k<ksos; ++k){
        float w1 = states[k*2];
        float w2 = states[k*2 + 1];
        float b0 = *sos++;
        float b1 = *sos++;
        float b2 = *sos++;
        float a0 = *sos++;
        float a1 = *sos++;
        float a2 = *sos++;

        for (int n=0; n<nsamp; ++n){
            float w0 = signal[n];
            w0 = w0 - a1*w1 - a2*w2;
            float yn = b0*w0 + b1*w1 + b2*w2;
            w2 = w1;
            w1 = w0;
            signal[n] = yn;
        }
        states[k*2] = w1;
        states[k*2 + 1] = w2;
    }
}


void sosfilter_double(double* signal, int nsamp, double* sos, int ksos, double* states){

    for (int k=0; k<ksos; ++k){
        double w1 = states[k*2];
        double w2 = states[k*2 + 1];
        double b0 = *sos++;
        double b1 = *sos++;
        double b2 = *sos++;
        double a0 = *sos++;
        double a1 = *sos++;
        double a2 = *sos++;

        for (int n=0; n<nsamp; ++n){
            double w0 = signal[n];
            w0 = w0 - a1*w1 - a2*w2;
            double yn = b0*w0 + b1*w1 + b2*w2;
            w2 = w1;
            w1 = w0;
            signal[n] = yn;
        }
        states[k*2] = w1;
        states[k*2 + 1] = w2;
    }
}


void sosfilter_double_mimo(double* signal, int nframes, int nchan, double* sos, int ksos, int kbands, double* states){

    for (int c=0; c<nchan; ++c){
        for (int b=0; b<kbands; ++b){
            for (int k=0; k<ksos; ++k){
                double w1 = states[c*ksos*kbands*2 + b*ksos*2 + k*2];
                double w2 = states[c*ksos*kbands*2 + b*ksos*2 + k*2 + 1];
                double b0 = *sos++;
                double b1 = *sos++;
                double b2 = *sos++;
                double a0 = *sos++;
                double a1 = *sos++;
                double a2 = *sos++;

                for (int n=0; n<nframes; ++n){
                    double w0 = signal[c*nframes*kbands + b*nframes + n];
                    w0 = w0 - a1*w1 - a2*w2;
                    double yn = b0*w0 + b1*w1 + b2*w2;
                    w2 = w1;
                    w1 = w0;
                    signal[c*nframes*kbands + b*nframes + n] = yn;
                }
            states[c*ksos*kbands*2 + b*ksos*2 + k*2] = w1;
            states[c*ksos*kbands*2 + b*ksos*2 + k*2 + 1] = w2;
            }
        }
    }
}
