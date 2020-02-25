#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>

void DERIV_1D(	double *wave_re ,	double *wave_im, \
				double *fft_re  ,	double *fft_im,	\
				double *deriv_re,	double *deriv_im, \
				double *omega, int Nsteps){

	int tstep;

	fftw_complex *FFTin  = fftw_alloc_complex(Nsteps);
	fftw_complex *DERIV  = fftw_alloc_complex(Nsteps);
	fftw_complex *FFTout = fftw_alloc_complex(Nsteps);

	for(tstep=0; tstep < Nsteps; tstep++){

		FFTin[tstep][0] = wave_re[tstep];
		FFTin[tstep][1] = wave_im[tstep];

		DERIV[tstep][0] = wave_re[tstep];
		DERIV[tstep][1] = wave_im[tstep];
	}

	fftw_plan DERIV_FORWARD  = fftw_plan_dft_1d(Nsteps, FFTin, DERIV , FFTW_FORWARD , FFTW_ESTIMATE);
	fftw_plan DERIV_BACKWARD = fftw_plan_dft_1d(Nsteps, DERIV, DERIV , FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_plan FFT1D_FORWARD  = fftw_plan_dft_1d(Nsteps, FFTin, FFTout, FFTW_FORWARD , FFTW_ESTIMATE);

	// Perform FORWARD FFT.
	fftw_execute(DERIV_FORWARD);
	fftw_execute(FFT1D_FORWARD);

	// Return the result.
	for(tstep=0; tstep < Nsteps; tstep++){
		fft_re[tstep] = FFTout[tstep][0];
		fft_im[tstep] = FFTout[tstep][1];
	}

	// Multiply I * omega.
	double real, imag;
	for(tstep=0; tstep < Nsteps; tstep++){
		
		real = DERIV[tstep][0];
		imag = DERIV[tstep][1];

		DERIV[tstep][0] = -omega[tstep] * imag;
		DERIV[tstep][1] = omega[tstep] * real;

	}


	// Perform BACKWARD FFT.
	fftw_execute(DERIV_BACKWARD);

	// Return the result.
	for(tstep=0; tstep < Nsteps; tstep++){
		deriv_re[tstep] = DERIV[tstep][0];
		deriv_im[tstep] = DERIV[tstep][1];
	}
		
	// Destroy plans and free the memory.
	fftw_destroy_plan(FFT1D_FORWARD);
	fftw_destroy_plan(DERIV_FORWARD);
	fftw_destroy_plan(DERIV_BACKWARD);
	fftw_free(DERIV);
	fftw_free(FFTin);
	fftw_free(FFTout);

	return;

}
