import numpy as np
from reikna import cluda as cld
from reikna.fft import FFT

api = cld.get_api('cuda')
dev1 = api.get_platforms()[0].get_devices()[0]
thr1 = api.Thread(dev1)

IEp, JEp, KEp = 64, 128, 512
totalshape= (IEp,JEp,KEp)
totalsize = IEp * JEp * KEp

dx, dy, dz, sig = 0.05, 0.05, 0.05, 1.

dtype = np.complex128
ones = np.ones((IEp,JEp,KEp), dtype=dtype)

program = thr1.compile("""
KERNEL void MUL(
    GLOBAL_MEM ${ctype} *dest,
    GLOBAL_MEM ${ctype} *a,
    GLOBAL_MEM ${ctype} *b)
{
    SIZE_T i = get_global_id(0);
    dest[i] = ${mul}(a[i],b[i]);
}

KERNEL void ADD(
	GLOBAL_MEM ${ctype} *dest,
	GLOBAL_MEM ${ctype} *a,
	GLOBAL_MEM ${ctype} *b)
{
	SIZE_T i = get_global_id(0);
	dest[i] = ${add}(a[i],b[i]);
}

KERNEL void CONJ(
	GLOBAL_MEM ${ctype} *target)
{
	SIZE_T i = get_global_id(0);
	target[i] = ${conj}(target[i]);
}
""",render_kwds=dict( ctype=cld.dtypes.ctype(dtype),
                        mul =cld.functions.mul(dtype,dtype,out_dtype=dtype),
                        add =cld.functions.add(dtype,dtype,out_dtype=dtype),
						conj=cld.functions.conj(dtype)))
ADD = program.ADD
MUL = program.MUL
CONJ= program.CONJ

fftx = FFT(ones, axes=(0,))
ffty = FFT(ones, axes=(1,))
fftz = FFT(ones, axes=(2,))

fftxc = fftx.compile(thr1, fast_math=True)
fftyc = ffty.compile(thr1, fast_math=True)
fftzc = fftz.compile(thr1, fast_math=True)


x = np.arange(IEp, dtype=dtype) * dx
y = np.arange(JEp, dtype=dtype) * dy
z = np.arange(KEp, dtype=dtype) * dz

kx = np.fft.fftfreq(IEp, dx) * 2. * np.pi
ky = np.fft.fftfreq(JEp, dx) * 2. * np.pi
kz = np.fft.fftfreq(KEp, dx) * 2. * np.pi

nax = np.newaxis
ikx = kx[:,nax,nax] * ones * 1j
iky = ky[nax,:,nax] * ones * 1j
ikz = kz[nax,nax,:] * ones * 1j

xexp1 = np.exp((-.5)*(x/sig)**2)  ## Gaussian distribution
yexp1 = np.exp((-.5)*(y/sig)**2)
zexp1 = np.exp((-.5)*(z/sig)**2)

xexp = xexp1[:,nax,nax] * ones
yexp = yexp1[nax,:,nax] * ones
zexp = zexp1[nax,nax,:] * ones

xexp_dev = thr1.to_device(xexp)
yexp_dev = thr1.to_device(yexp)
zexp_dev = thr1.to_device(zexp)

ikx_dev = thr1.to_device(ikx)
iky_dev = thr1.to_device(iky)
ikz_dev = thr1.to_device(ikz)

ikx_conj_dev = thr1.to_device(ikx)
iky_conj_dev = thr1.to_device(iky)
ikz_conj_dev = thr1.to_device(ikz)

CONJ(ikx_conj_dev, local_size=KEp, global_size=totalsize)
CONJ(iky_conj_dev, local_size=KEp, global_size=totalsize)
CONJ(ikz_conj_dev, local_size=KEp, global_size=totalsize)

conjtestx_dev = thr1.empty_like(ones)
conjtesty_dev = thr1.empty_like(ones)
conjtestz_dev = thr1.empty_like(ones)

ADD(conjtestx_dev, ikx_dev, ikx_conj_dev, local_size=512, global_size=totalsize)
ADD(conjtesty_dev, iky_dev, iky_conj_dev, local_size=512, global_size=totalsize)
ADD(conjtestz_dev, ikz_dev, ikz_conj_dev, local_size=512, global_size=totalsize)

conjtestx = conjtestx_dev.get()
conjtesty = conjtesty_dev.get()
conjtestz = conjtestz_dev.get()

print(ikx)

assert conjtestx.imag.all() == 0.
assert conjtesty.imag.all() == 0.
assert conjtestz.imag.all() == 0.

fft_xexp_dev = thr1.empty_like(ones)
fft_yexp_dev = thr1.empty_like(ones)
fft_zexp_dev = thr1.empty_like(ones)

ikx_fft_xexp_dev = thr1.empty_like(ones)
iky_fft_yexp_dev = thr1.empty_like(ones)
ikz_fft_zexp_dev = thr1.empty_like(ones)

ifft_ikx_fft_xexp_dev = thr1.empty_like(ones)
ifft_iky_fft_yexp_dev = thr1.empty_like(ones)
ifft_ikz_fft_zexp_dev = thr1.empty_like(ones)

ft = np.fft.fftn
ift= np.fft.ifftn

fftxc(fft_xexp_dev, xexp_dev)
fftyc(fft_yexp_dev, yexp_dev)
fftzc(fft_zexp_dev, zexp_dev)

MUL(ikx_fft_xexp_dev, ikx_dev, fft_xexp_dev, local_size=512, global_size = totalsize)
MUL(iky_fft_yexp_dev, iky_dev, fft_yexp_dev, local_size=512, global_size = totalsize)
MUL(ikz_fft_zexp_dev, ikz_dev, fft_zexp_dev, local_size=512, global_size = totalsize)

fftxc(ifft_ikx_fft_xexp_dev, ikx_fft_xexp_dev, inverse=True)
fftyc(ifft_iky_fft_yexp_dev, iky_fft_yexp_dev, inverse=True)
fftzc(ifft_ikz_fft_zexp_dev, ikz_fft_zexp_dev, inverse=True)

ifft_ikx_fft_xexp_gpu = ifft_ikx_fft_xexp_dev.get()
ifft_iky_fft_yexp_gpu = ifft_iky_fft_yexp_dev.get()
ifft_ikz_fft_zexp_gpu = ifft_ikz_fft_zexp_dev.get()

ifft_ikx_fft_xexp_cpu = ift( ikx * ft( xexp, axes=(0,)), axes=(0,))
ifft_iky_fft_yexp_cpu = ift( iky * ft( yexp, axes=(1,)), axes=(1,))
ifft_ikz_fft_zexp_cpu = ift( ikz * ft( zexp, axes=(2,)), axes=(2,))

assert np.allclose(ifft_ikx_fft_xexp_gpu, ifft_ikx_fft_xexp_cpu)
assert np.allclose(ifft_iky_fft_yexp_gpu, ifft_iky_fft_yexp_cpu)
assert np.allclose(ifft_ikz_fft_zexp_gpu, ifft_ikz_fft_zexp_cpu)

#import os
#myname = os.path.basename(__file__)
#os.system("/root/dropbox_uploader.sh upload /root/3D_PSTD/%s Python/MyPSTD/3D_PSTD" %myname)
