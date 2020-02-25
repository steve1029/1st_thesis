#include "lib_01_eps_func.h"

struct D1 {
	double EPS;
	double wp[2], gamma[2];
	double mindex;
};

struct D1CP1 {
	double EPS;
	double wp[2], gamma[2];
	double A[2], GAMMA[2], OMEGA[2], PHI[2];
	double mindex;
};

struct D1CP2 {
	double EPS; 
	double wp[2], gamma[2]; 
	double A[3], GAMMA[3], OMEGA[3], PHI[3];
	double mindex;
} ;

struct ActM {
	double wa, wb, GammaA, GammaB, SigA, SigB;
    double La, Lb, tau32, tau21, tau10, N0;
	double mindex;
} ;





struct D1 epsConverter(double epsinf, double mindex, double epsRE, double epsIM, double w)
{
    struct D1 tmp;
    tmp.EPS = epsinf;
    tmp.wp[1] = sqrt(w * w * (epsIM*epsIM + (epsinf-epsRE)*(epsinf-epsRE) ) / (epsinf - epsRE)) ;
    tmp.gamma[1] = w * epsIM / (epsinf - epsRE) ; 
    tmp.mindex = mindex;

    printf("epsRE : %f, epsIM : %f, converts to Drude,, epsinf : %f, wp : %e, gamma : %e\n",epsRE, epsIM, epsinf, tmp.wp[1],tmp.gamma[1]);

    return tmp;
}




struct ActM ActM_Shalaev(void)
{
    struct ActM tmp;
    tmp.wa = 0;
    tmp.wb = 0;
    tmp.GammaA = 0;
    tmp.GammaB = 0;
    tmp.SigA = 0;
    tmp.SigB = 0;
    tmp.La = 746e-9;
    tmp.Lb = 718e-9;
    tmp.tau32 = 0;
    tmp.tau21 = 0;
    tmp.tau10 = 0;
    tmp.N0 = 1.2e25; // this is in MKS, 1.2e25 m^-3 = 1.2e19 cm^-3 = 1.2e-2 nm^-3
    tmp.mindex = 3.0;
    return tmp;
}


struct ActM ActM_Soukoulis(void)
{
    struct ActM tmp;
    tmp.wa = 2.0 * 3.141592 * 1e14;
    tmp.wb = 4.0 * 3.141592 * 1e14;
    tmp.La = 2.0 * 3.141592 * 299792458.0 / tmp.wa; 
    tmp.Lb = 2.0 * 3.141592 * 299792458.0 / tmp.wb;
    tmp.GammaA = 2 * 3.141592 * 5 * 1e12;   // tmp.GammaA = 2 * 3.141592 * 20e12;
    tmp.GammaB = 2 * 3.141592 * 10 * 1e12;
    tmp.SigA = 1e-4;
    tmp.SigB = 5e-6;
    tmp.tau32 = 5e-14;
    tmp.tau21 = 5e-12;
    tmp.tau10 = 5e-14;
    tmp.N0 = 5.0e23; // this is in MKS, 1.2e25 m^-3 = 1.2e19 cm^-3 = 1.2e-2 nm^-3
    tmp.mindex = 4.0;
    return tmp;
}


struct ActM ZnO_emit380_pump(void)
{
    struct ActM tmp;
    tmp.wb = 0;
    tmp.Lb = 0;
    tmp.GammaB = 0;
    tmp.SigB = 0;

    tmp.wa = 4.95698e15;
    tmp.La = 2.0 * 3.141592 * 299792458.0 / tmp.wa; 
    tmp.GammaA = 5.01294e13;
    tmp.SigA = 1.82925e-6;
    tmp.tau32 = 1e-12;
    tmp.tau21 = 1e-10;
    tmp.tau10 = 1e-12;
    tmp.N0 = 1e24; // this is in MKS, 1.2e25 m^-3 = 1.2e19 cm^-3 = 1.2e-2 nm^-3
    tmp.mindex = 4.0;
    return tmp;
}



///////////////////////
// Jong-Ho's fitting //
///////////////////////


///////////////////// SILVER //////////////////////////

struct D1CP2 D1CP2Ag300nm900nmJOHNSON(void) // Johnson, STerr: 0.0271776, (2012.03.28)
{
	struct D1CP2 temp;
	temp.EPS     = 1.0;
	temp.wp[1]   = 1.394932972644896e16;
	temp.gamma[1]= 1.7781630288180668e13;
	temp.A[1]    = 0.284373188260343;
	temp.GAMMA[1]= 5.960612191043039e14;
	temp.OMEGA[1]= 6.222080372689097e15;
	temp.PHI[1]  = 5.106945466121798;
	temp.A[2]    = 2.0963112499088674;
	temp.GAMMA[2]= 1.2152751718223278e16;
	temp.OMEGA[2]= 1.217425709766018e16;
	temp.PHI[2]  =-0.23587497382082553;
	temp.mindex = -4.7;
	return temp;
}

struct D1CP2 D1CP2Ag300nm900nmJOHNSONloss(void) // Johnson, STerr: 0.0271776, (2012.03.28)
{
	struct D1CP2 temp;
	temp.EPS     = 1.0;
	temp.wp[1]   = 1.394932972644896e16;
	temp.gamma[1]= 10*1.7781630288180668e13;
	temp.A[1]    = 0.284373188260343;
	temp.GAMMA[1]= 5.960612191043039e14;
	temp.OMEGA[1]= 6.222080372689097e15;
	temp.PHI[1]  = 5.106945466121798;
	temp.A[2]    = 2.0963112499088674;
	temp.GAMMA[2]= 1.2152751718223278e16;
	temp.OMEGA[2]= 1.217425709766018e16;
	temp.PHI[2]  =-0.23587497382082553;
	temp.mindex = -4.7;
	return temp;
}

struct D1CP2 D1CP2Ag300nm900nmHAGEMANN(void) // Hagemann, STerr: 0.0311755, (2012.05.30) 
{
	struct D1CP2 temp;
	temp.EPS     = 1.56248;
	temp.wp[1]   = 1.4192603116074208e16;
	temp.gamma[1]= 1.2648883383214597e14;
	temp.A[1]    =-12.585227965523826;
	temp.GAMMA[1]= 5.4011525995196622e15;
	temp.OMEGA[1]= 8.460715921479399e14;
	temp.PHI[1]  =-4.339369858022698;
	temp.A[2]    = 0.42268507467006494;
	temp.GAMMA[2]= 5.838264705324864e14;
	temp.OMEGA[2]= 6.212336151322572e15;
	temp.PHI[2]  =-1.4203005565642786;

	temp.mindex = -4.7;
	return temp;
}

struct D1CP2 D1CP2Ag370nm900nmPALIK_D1CP1(void) // Winsemius, STerr: 0.0147798, (2012.05.31)  
{
	struct D1CP2 temp;
	temp.EPS     = 4.125 ;
	temp.wp[1]   = 1.3462259458308412e16;
	temp.gamma[1]= 5.191123309227644e13;
	temp.A[1]    = 3.988543261091087;
	temp.GAMMA[1]= 1.4256692514480055e15;
	temp.OMEGA[1]= 3.397350990276859e14;
	temp.PHI[1]  = 4.693855988590337;
	temp.A[2]    = 0;
	temp.GAMMA[2]= 1.0e15;
	temp.OMEGA[2]= 1.011e15;
	temp.PHI[2]  = 0;

	temp.mindex = -4.7;
	return temp;
}

struct D1CP2 D1CP2Ag900nm1700nmSOUKOULIS_Drude(void) // for reproducing 
{ // 2006, Opt. Lett., Vol. 31, page 1800, G. Dolling, ... C. M. Soukoulis
	struct D1CP2 temp;
	temp.EPS     = 1.0 ;
	temp.wp[1]   = 1.37e16;
	temp.gamma[1]= 8.5e13;

	temp.A[1]    = 0;	temp.GAMMA[1]= 1.0e15;	temp.OMEGA[1]= 1.011e15;	temp.PHI[1]  = 0;
	temp.A[2]    = 0;	temp.GAMMA[2]= 1.0e15;	temp.OMEGA[2]= 1.011e15;	temp.PHI[2]  = 0;

	temp.mindex = -4.7;
	return temp;
}

//struct D1CP2 D1CP2Ag370nm900nmPALIK(void) // STerr: 0.00694864, (2012.05.30) BLOW EPS=4.2
//struct D1CP2 D1CP2Ag300nm900nmPALIK(void) // STerr: 0.00734828, (2012.05.30) BLOW EPS=4.81
//struct D1CP2 D1CP2Ag500nm900nmPALIK(void) // STerr: 0.00519611, (2012.04.30) BLOW EPS=1.0 



////////////////////// GOLD ///////////////////////////

struct D1CP2 D1CP2Au400nm1000nmJOHNSON(void) // Johnson and Christy,, STerr : 0.0294043 (2013.1.25)
{
	struct D1CP2 temp;
	temp.EPS     = 1.0;
   	temp.wp[1]   = 1.3116170613829832e16;
	temp.gamma[1]= 1.1501700343487477e14;
	temp.A[1]    = 2.559635958760465;
	temp.GAMMA[1]= 2.0986871407450945e15;
	temp.OMEGA[1]= 4.706954941760446e15;
	temp.PHI[1]  =-0.9231394055396535;
	temp.A[2]    =-0.29813628368518946;
	temp.GAMMA[2]= 4.759121356224116e14;
	temp.OMEGA[2]= 3.9510687439963015e15;
	temp.PHI[2]  =-3.942078027009713;
	temp.mindex = -7.9;
	return temp;
}

struct D1CP2 D1CP2Au300nm900nmJOHNSON(void) // Johnson and Christy,, STerr : 0.0313929 (2013.01.31)
{
	struct D1CP2 temp;
	temp.EPS     = 2.5;
   	temp.wp[1]   = 1.3148109152556018e16;
	temp.gamma[1]= 9.505359374012766e13;
	temp.A[1]    = 1.5340437478846873;
	temp.GAMMA[1]= 2.2162289463678742e15;
	temp.OMEGA[1]= 6.315056318675713e15;
	temp.PHI[1]  = 0.034309981641033445;
	temp.A[2]    =-0.7120539753544073;
	temp.GAMMA[2]= 7.135781715940518e14;
	temp.OMEGA[2]= 3.9459276514390085e15;
	temp.PHI[2]  =-4.169089425678387;
	temp.mindex = -7.9;
	return temp;
}

struct D1CP2 D1CP2Au1000nm3000nm_Drude(void) // Palik,, STerr : 180.603  (2012.06.27) this is Drude
{
	struct D1CP2 temp;
	temp.EPS     = 1.0;
   	temp.wp[1]   = 1.2058181639891528e16;
	temp.gamma[1]= 1.1999034535903264e14;
	temp.A[1]    = 0.0;
	temp.GAMMA[1]= 1.0e16;
	temp.OMEGA[1]= 1.0e14;
	temp.PHI[1]  = 0.0;
	temp.A[2]    = 0;
	temp.GAMMA[2]= 1.0e16;
	temp.OMEGA[2]= 1.0e15;
	temp.PHI[2]  = 0;
	temp.mindex = -7.9;
	return temp;
}


////////////////////// Aluminium ///////////////////////////

struct D1CP2 D1CP2Al300nm900nm(void) // Rakic, STerr : 0.137391  (2012.3.26)
{
	struct D1CP2 temp;
	temp.EPS     = 1.0;
   	temp.wp[1]   = 2.0356022851971e16;
	temp.gamma[1]= 2.1558899154101e14;
	temp.A[1]    = 5.6588777555160;
	temp.GAMMA[1]= 3.5735790649025e14;
	temp.OMEGA[1]= 2.2590644979400e15;
	temp.PHI[1]  =-0.4850996787410;
	temp.A[2]    = 3.1116433236877;
	temp.GAMMA[2]= 1.4710570697224e15;
	temp.OMEGA[2]= 2.9387088835961e15;
	temp.PHI[2]  = 0.5083054567608;
	temp.mindex = -1.3; // atom number : 13 
	return temp;
}

struct D1CP2 D1CP2Al300nm900nm_3(void) // Rakic, STerr : 0.232058  (2012.3.26)
{
	struct D1CP2 temp;
	temp.EPS     = 1.0;
   	temp.wp[1]   = 2.366522367001322e16;
	temp.gamma[1]= 9.655523715011404e14;
	temp.A[1]    = 0;
	temp.GAMMA[1]= 3.781e14;
	temp.OMEGA[1]= 2.218e15;
	temp.PHI[1]  = 0;
	temp.A[2]    = 0;
	temp.GAMMA[2]= 6.421e15;
	temp.OMEGA[2]= 9.454e15;
	temp.PHI[2]  = 0;
	temp.mindex = -1.3; // atom number : 13 
	return temp;
}

struct D1CP2 D1CP2Al2500nm20um_Rakic(void) // Rakic, STerr : 4339.99  (2014.12.31) not proved yet
{
	struct D1CP2 temp;
	temp.EPS     = 10.0;
   	temp.wp[1]   = 1.8110413855758624e16; 
	temp.gamma[1]= 9.477738681090427e13;
	temp.A[1]    = 138.81050697190886;
	temp.GAMMA[1]= 87098.34804175566;
	temp.OMEGA[1]= 1.2913060603756078e17;
	temp.PHI[1]  = 2.136541683106582;
	temp.A[2]    = 1282.8390489461387;
	temp.GAMMA[2]= 3.16432230299051e13;
	temp.OMEGA[2]= 8.930811378051955e13;
	temp.PHI[2]  =-2.6149060674484863;
	temp.mindex = -1.3; // atom number : 13 
	return temp;
}

///////////////////// SILICON ////////////////////////// silicon atom number : 14

struct D1CP2 D1CP2Si300nm800nmPALIK(void) // Palik, STerr: 0.944317, (2013.04.28)
{
	struct D1CP2 temp;
	temp.EPS     = 2.0; 
	temp.wp[1]   = 0;
	temp.gamma[1]= 0;
	temp.A[1]    =-1.3281240719318108;
	temp.GAMMA[1]= 2.418956338146272e14;
	temp.OMEGA[1]= 5.115832370098545e15;
	temp.PHI[1]  = 2.3078590980047147;
	temp.A[2]    =-4.410075824564194;
	temp.GAMMA[2]= 7.795088221370624e14;
	temp.OMEGA[2]= 6.492623507279991e15;
	temp.PHI[2]  = 3.290102642882403;
	temp.mindex = -1.4;
	return temp;
}

struct D1CP2 D1CP2Si300nm800nmPALIK_2(void) // Palik, STerr: 0.944317, (2013.04.28)
{
	struct D1CP2 temp;
	temp.EPS     = 4.0;
	temp.wp[1]   = 1.1231275999647966e10 * 0;  // try 0
	temp.gamma[1]= 8.335330727043009e15;
	temp.A[1]    =-1.4327321393565065;
	temp.GAMMA[1]= 2.563237475701456e14;
	temp.OMEGA[1]= 5.108836820561175e15;
	temp.PHI[1]  = 2.2545987900496454;
	temp.A[2]    =-3.93520817586869;
	temp.GAMMA[2]= 6.484860512520668e14;
	temp.OMEGA[2]= 6.579562414662795e15;
	temp.PHI[2]  =-2.7762596154236685;
	temp.mindex = -1.4;
	return temp;
}

////////////////////// Nickel ///////////////////////////

struct D1CP2 D1CP2Ni300nm900nm(void) // Palik, STerr : 0.0141569  (2012.3.26)
{
	struct D1CP2 temp;
	temp.EPS     = 1.0;
   	temp.wp[1]   = 1.325874380259268e16;
	temp.gamma[1]= 1.442377560503351e15;
	temp.A[1]    = 3.214814494683990;
	temp.GAMMA[1]= 4.816375073060784e15;
	temp.OMEGA[1]= 9.690661318230162e15;
	temp.PHI[1]  = 0.428870654403958;
	temp.A[2]    = 2.663011434478377;
	temp.GAMMA[2]= 6.261403123618544e14;
	temp.OMEGA[2]= 1.977760585715344e15;
	temp.PHI[2]  =-1.277875400574552;
	temp.mindex = -2.8; // atom number : 28
	return temp;
}

struct D1CP2 D1CP2Ni300nm900nm_2(void) // Palik, STerr : 0.0139261  (2012.3.26)
{
	struct D1CP2 temp;
	temp.EPS     = 2.1;
   	temp.wp[1]   = 1.33501820171681e16;
	temp.gamma[1]= 1.49754687584861e15;
	temp.A[1]    = 4.00077466183896;
	temp.GAMMA[1]= 5.31430242142796e15;
	temp.OMEGA[1]= 1.10548790949880e16;
	temp.PHI[1]  = 0.71981478102242;
	temp.A[2]    = 2.31128985637125;
	temp.GAMMA[2]= 6.12440241489153e14;
	temp.OMEGA[2]= 2.01517491930328e15;
	temp.PHI[2]  =-1.24412187290537;
	temp.mindex = -2.8; // atom number : 28
	return temp;
}

struct D1CP2 D1CP2Ni300nm900nm_3(void) // Palik, STerr : 0.0139261  (2012.3.26)
{
	struct D1CP2 temp;
	temp.EPS     = 5.0;
   	temp.wp[1]   = 1.7419885228910362e16;
	temp.gamma[1]= 3.8420271476535495e15;
	temp.A[1]    = 0;
	temp.GAMMA[1]= 5.31e15;
	temp.OMEGA[1]= 1.10e16;
	temp.PHI[1]  = 0;
	temp.A[2]    = 0;
	temp.GAMMA[2]= 6.12e14;
	temp.OMEGA[2]= 2.01e15;
	temp.PHI[2]  = 0;
	temp.mindex = -2.8; // atom number : 28
	return temp;
}

////////////////////// Platinum ///////////////////////////

struct D1CP2 D1CP2Pt300nm1000nmPALIK_D1CP1(void) // PALIK, STerr: 0.0103642, (2013.02.03)
{
	struct D1CP2 temp;
	temp.EPS     = 1.2528;
	temp.wp[1]   = 1.468666843162885e16;
	temp.gamma[1]= 1.723657769636042e15;
	temp.A[1]    = 21.24739472151703;
	temp.GAMMA[1]= 2.123559785624624e15;
	temp.OMEGA[1]= 9.859574888333026e14;
	temp.PHI[1]  =-0.8927848047859648;

	temp.A[2]    = 0; temp.GAMMA[2]= 1.0e15; temp.OMEGA[2]= 2.0e15; temp.PHI[2]  = 0;

	temp.mindex = -7.8; // atom number : 78 
	return temp;
}

////////////////////// Chromium ///////////////////////////

struct D1CP2 D1CP2Cr400nm1000nmVIAL(void) // VIAL, in article 0.25875  (2013.02.03)
{
	struct D1CP2 temp;
	temp.EPS     = 1.1297 ;
	temp.wp[1]   = 8.8128e15; 
	temp.gamma[1]= 3.8828e14;
	temp.A[1]    = 33.086;
	temp.GAMMA[1]= 1.6329e15;
	temp.OMEGA[1]= 1.7398e15;
	temp.PHI[1]  =-0.25722;
	temp.A[2]    = 1.6592;
	temp.GAMMA[2]= 7.3567e14;
	temp.OMEGA[2]= 3.7925e15;
	temp.PHI[2]  = 0.83533;

	temp.mindex = -2.4; // atom number : 24 
	return temp;
}

////////////////////// Copper ///////////////////////////

struct D1CP2 D1CP2Cu400nm1000nmRakic_Ordal(void) // not tested yet
{
	struct D1CP2 temp;
	temp.EPS     = 1.0; 
	temp.wp[1]   = 1.3053428765243808e16;
	temp.gamma[1]= 1.7209298246055475e14;
	temp.A[1]    = -1.183257295364979;
	temp.GAMMA[1]= 8.54988679632223e14;
	temp.OMEGA[1]= 3.5522820868727255e15;
	temp.PHI[1]  =-4.4491395169549754;
	temp.A[2]    = -27.889005704684465;
	temp.GAMMA[2]= 0;
	temp.OMEGA[2]= 1.2414779093214122e17;
	temp.PHI[2]  = -1.6081249799192;

	temp.mindex = -2.4; // atom number : 24 
	return temp;
}







////////////////////////////
//// SeokJae's fitting ////
////////////////////////////
// Three asterisks (***) denote reliable fitting parameter which is varified by analytic Fresnel formula and practical nanostructure simulation

// Crystalline silicon (Si)
struct D1CP2 D1CP2SiSOPRA300nm800nm(void)
{
	struct D1CP2 temp;
	temp.EPS	= 1;
	temp.wp[1]	= 1.692278822642980e15;
	temp.gamma[1]	= 1.89381876845954e3;
	temp.A[1]    = -1.27932362167047;
	temp.GAMMA[1]= 2.18366306151624e14;
	temp.OMEGA[1]= 5.102676157839700e15;
	temp.PHI[1]  = 2.31431929891286;
	temp.A[2]    = -4.73795481342157;
	temp.GAMMA[2]= 8.48481229311499e14;
	temp.OMEGA[2]= 6.398823419340790e15;
	temp.PHI[2]  = 3.15695601979578;
	temp.mindex = -1.4;
	return temp;
}

// Al2O3
struct D1CP2 D1CP2Al2O3SOPRA200nm2000nm(void)
{
	struct D1CP2 temp;
	temp.EPS	= 1;
	temp.wp[1]	= 6.98085692248606e14;
	temp.gamma[1]	= 4.983166394818760e15;
	temp.A[1]    = -9.23112651799599;
	temp.GAMMA[1]= 1.3570087005378300e16;
	temp.OMEGA[1]= 8.063815654482060e15;
	temp.PHI[1]  = 0.373580469857281;
	temp.A[2]    = -3.9247032358421;
	temp.GAMMA[2]= 7.770734838045830e15;
	temp.OMEGA[2]= 4.023522533464460e15;
	temp.PHI[2]  = 1.9889191876293132;
	temp.mindex = -2;
	return temp;
}  

// Al
struct D1CP2 D1CP2Al300nm900nm2(void)
{
	struct D1CP2 temp;
	temp.EPS	= 1;
	temp.wp[1]	= 2.0383455518558504e16;
	temp.gamma[1]	= 2.5497979690494766e14;
	temp.A[1]    = 6.443269968772944;
	temp.GAMMA[1]= 3.7882335531953e14;
	temp.OMEGA[1]= 2.286120027634404e15;
	temp.PHI[1]  = 5.862266167897745;
	temp.A[2]    = 4.373314452470692;
	temp.GAMMA[2]= 7.633967261901861e15;
	temp.OMEGA[2]= 2.124386766566152e22;
	temp.PHI[2]  = 1.8832236656967871;
	temp.mindex = -1.3;
	return temp;
}

// Al_mono
struct D1CP2 D1CP2Al377nm(void)
{
	struct D1CP2 temp;
	temp.EPS	= 2.0;
	temp.wp[1]	= 2.4116002318184376e16;
	temp.gamma[1]	= 8.706726062148132e14;
	temp.A[1]    = 0;
	temp.GAMMA[1]= 0;
	temp.OMEGA[1]= 0; 0;
	temp.PHI[1]  = 0; 0;
	temp.A[2]    = 0; 0;
	temp.GAMMA[2]= 0; 0;
	temp.OMEGA[2]= 0; 0;
	temp.PHI[2]  = 0; 0;
	temp.mindex = -1.3;
	return temp;
}

// Al2
struct D1CP2 D1CP2Al300nm900nm_2(void) // Palic
{
	struct D1CP2 temp;
	temp.EPS	= 2.2;
	temp.wp[1]	= 2.075493476122525e16;
	temp.gamma[1]	= 2.692830652318520e14;
	temp.A[1]    = 6.986369587800884;
	temp.GAMMA[1]= 3.781756919663996e14;
	temp.OMEGA[1]= 2.218812242846060e15;
	temp.PHI[1]  = -0.690883344268221;
	temp.A[2]    = 2.971741559189552;
	temp.GAMMA[2]= 6.421007308803584e15;
	temp.OMEGA[2]= 9.454315381674146e15;
	temp.PHI[2]  = 3.167232355361811;
	temp.mindex = -1.3;
	return temp;
}

// Al
struct D1CP2 D1CP2Al200nm1000nm_Vial(void) // Vial DCP paper Appl Phys B
{
	struct D1CP2 temp;
	temp.EPS	= 1.0;
	temp.wp[1]	= 2.0598e16;
	temp.gamma[1]	= 2.2876e14;
	temp.A[1]    = 5.2306;
	temp.GAMMA[1]= 3.2867e14;
	temp.OMEGA[1]= 2.2694e15;
	temp.PHI[1]  = -0.51202;
	temp.A[2]    = 5.2704;
	temp.GAMMA[2]= 1.7731e15;
	temp.OMEGA[2]= 2.4668e15;
	temp.PHI[2]  = 0.42503;
	temp.mindex = -1.3;
	return temp;
}

// Ag_Palik at 400 nm
struct D1CP2 D1CP2Ag400nm_Palik(void) // Palik
{
	struct D1CP2 temp;
	temp.EPS	= 1.0;
	temp.wp[1]	= 1.03905e16;
	temp.gamma[1]	= 6.65967e14;
	temp.A[1]    = 0;
	temp.GAMMA[1]= 0;
	temp.OMEGA[1]= 0;
	temp.PHI[1]  = 0;
	temp.A[2]    = 0;
	temp.GAMMA[2]= 0;
	temp.OMEGA[2]= 0;
	temp.PHI[2]  = 0;
	temp.mindex = -4.7;
	return temp;
}

// a-Si (SOPRA) ***
struct D1CP2 D1CP2aSi_SOPRA(void) // Wavelength: 300 nm to 800 nm
{
	struct D1CP2 temp;
	temp.EPS	= 1.0;
	temp.wp[1]	= 1.850435611831103e13;
	temp.gamma[1]	= 9.583406339327194e15;
	temp.A[1]    = -10.870449387447888;
	temp.GAMMA[1]= 1.733426422846888e15;
	temp.OMEGA[1]= 5.324466171382987e15;
	temp.PHI[1]  = 2.944731850871322;
	temp.A[2]    = -5.18018639132158;
	temp.GAMMA[2]= 1.8167392076732152e15;
	temp.OMEGA[2]= 2.323289363616541e15;
	temp.PHI[2]  = -0.9939548517326946;
	temp.mindex = 14;
	return temp;
}

// a-Si (Pierce) ***
struct D1CP2 D1CP2aSi_Pierce(void) // Wavelength: 300 nm to 800 nm
{
	struct D1CP2 temp;
	temp.EPS	= 1;
	temp.wp[1]	= 0;
	temp.gamma[1]	= 8.061205596111416e15;
	temp.A[1]    =  -9.10222319585717;
	temp.GAMMA[1]= 1.8595339509450368e15;
	temp.OMEGA[1]= 4.94171527990468e15;
	temp.PHI[1]  = 2.8453371370897904;
	temp.A[2]    = -11.625322053622188;
	temp.GAMMA[2]= 1.575944678802349e15;
	temp.OMEGA[2]= 1.0072165244475052e15;
	temp.PHI[2]  = -2.1262302665474273;
	temp.mindex = 16;
	return temp;
}

struct D1CP2 D1CP2aSi(void) // Wavelength: 300 nm to 800 nm
{
	struct D1CP2 temp;
	temp.EPS	= 2.738587518697226;
	temp.wp[1]	= 1.6139197453237578e15;
	temp.gamma[1]	= 4.111985744321898e15;
	temp.A[1]    =  -3.1530715132508322;
	temp.GAMMA[1]= 5.02709596296994e14;
	temp.OMEGA[1]= 4.773078399299514e15;
	temp.PHI[1]  = 3.077898703926377;
	temp.A[2]    = -0.7918925733319867;
	temp.GAMMA[2]= 1.9569351317100125e15;
	temp.OMEGA[2]= 2.5409853403772105e15;
	temp.PHI[2]  = -0.14546067567560064;
	temp.mindex = 16;
	return temp;
}

// Cr (SOPRA) *** (NOTE: DCP2 model cannot make perfect fitting on Cr, but this parameter don't be blown up)
struct D1CP2 D1CP2Cr300to800_SOPRA(void) // Palik
{
	struct D1CP2 temp;
	temp.EPS	= 1;
	temp.wp[1]	= 1.4189265790760204e16;
	temp.gamma[1]	= 2.1135321666341358e15;
	temp.A[1]    = 7.149177217361324;
	temp.GAMMA[1]= 1.0809212270681449e15;
	temp.OMEGA[1]= 2.1115486212815998e15;
	temp.PHI[1]  = 10.862707457965554;
	temp.A[2]    = -0.7242878504699284;
	temp.GAMMA[2]= 5.1898621522682756e14;
	temp.OMEGA[2]= 3.37682808441949e15;
	temp.PHI[2]  = 2.397841968339894;
	temp.mindex = 24;
	return temp;
}

// MoS2
// MoS2 at room temperature
// This fitting cannot distinguish A- and B-exciton band at the region 600~700 nm
// Ref: Shen, C.-C.,et. al. Applied Physics Express, 6, 125801. (2013).
struct D1CP2 D1CP2MoS2RT_400to800(void)
{
	struct D1CP2 temp;
	temp.EPS	= 1.7877878776532141;
	temp.wp[1]	= 5.042429410702844e16;
	temp.gamma[1]	= 1.8700838553056144e17;
	temp.A[1]    = 4.305335953507419;
	temp.GAMMA[1]= 3.8927228925060125e14;
	temp.OMEGA[1]= 4.3074125920431435e15;
	temp.PHI[1]  = -6.5847621217867065;
	temp.A[2]    = 1.3712296287344166;
	temp.GAMMA[2]=  1.990435384450116e14;
	temp.OMEGA[2]= 2.811287665167908e15;
	temp.PHI[2]  = 4.951711455269794;
	temp.mindex = 4.2;
	return temp;
}

// ITO
struct D1CP2 D1CP2ITO_300to800(void)
{
	struct D1CP2 temp;
	temp.EPS	= 4.255350996093319;
	temp.wp[1]	= 3.0626827828368245e15;
	temp.gamma[1]	= 2.6936428300031965e13;
	temp.A[1]    = 0;
	temp.GAMMA[1]= 9.314219696115896e14;
	temp.OMEGA[1]= 7.23165662351179e15;
	temp.PHI[1]  = 2.2404322429544323;
	temp.A[2]    = 0;
	temp.GAMMA[2]=  1.0395814917565928e16;
	temp.OMEGA[2]= 1.9785002624362748e15;
	temp.PHI[2]  = 3.5752330243400845;
	temp.mindex = 4.9;
	return temp;
}