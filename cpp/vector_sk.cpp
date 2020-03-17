//rfi_mitigation without fft works

#include<iostream>
#include<fstream>
#include<complex>
#include<algorithm>
#include<random>
#include<vector>
#include<numeric>
#include<bits/stdc++.h>
using namespace std;

typedef std::vector<int> vInt;
typedef std::vector<double> vDouble;
typedef std::complex<double> Complex ;
typedef std::vector<std::complex<double>> vComplex ;

struct stats{
	double rmean, rsd, imean, isd;
};

class SpectralKurtosis{
    public:
        int nchannels, M, sample_size, nwindows, nclean_windows, nrfi_windows;
        string fname;
	vDouble rp0, ip0, rp1, ip1;
	vComplex p0, p1;
	vComplex P0, P1, clean_data, ref_data;
	vDouble power0, power1;
	vInt rfi_status,clean_win_index, rfi_win_index;
        SpectralKurtosis(int nch, int m, string file){
		nchannels=nch;
		M=m;
		fname= file;
		printf("SK for %d channels and %d window size..\n", nchannels,M);

	}
        void read_file();
	vComplex compute_fft(vComplex p);
	void compute_sk(vComplex p);
	void get_clean_window_indices(vInt x, int n);
	void get_rfi_window_indices(vInt x);
	stats get_stats(vComplex a);
	vComplex generate_reference_data(stats s);
	stats get_clean_data_stats(vComplex p, int n);
        vComplex mitigate_rfi(vComplex p);
        void run_sk_p0();
        void run_sk_p1();
};

void SpectralKurtosis::read_file(){
	printf("Reading file..\n");
	char *x;
	ifstream dada_file;
	int length;
	//openfile
	dada_file.open(fname);
	if (dada_file.is_open()){
		dada_file.seekg(-4096,dada_file.end);
		//finding length of the file ignoring headers of size 4096 bytes
		length = dada_file.tellg();
		
		//Reading data
		dada_file.seekg(4096);
		x = new char[length];
		dada_file.read(x,length);
	}
	dada_file.close();

	sample_size=length/4;
	nwindows=(sample_size/nchannels)/M;

	p0.resize(sample_size);
	p1.resize(sample_size);

	int offset;
	for(int i=0; i<sample_size;i++){
		offset=i*4;
	        p0[i]=Complex(x[offset],x[offset+1]);
	        p1[i]=Complex(x[offset+2],x[offset+3]);
	}
	printf("Done.\n");
}

vComplex SpectralKurtosis::compute_fft(vComplex p){
	printf("Computing FFT..\n");
	if(nchannels>1){
		vComplex P(sample_size);
		//Do FFT
	        printf("Done.\n");
		return P;
	}
	else
        	printf("Done.\n");
		return p;
}

void SpectralKurtosis::compute_sk(vComplex p){
	printf("Computing SK..\n");
	vDouble p1(sample_size), p2(sample_size);
	for(int i=0;i<sample_size;i++){
		p1[i]=pow(abs(p[i]),2);
		p2[i]=pow(p1[i],2);
	}

	//s1,s2
	vDouble s1(sample_size), s2(sample_size), sk(sample_size);
	rfi_status.resize(nwindows*nchannels);
	int r1,r2;
	//replace this with transform?
	for(int i=0;i<nwindows;i++){
                r1=i*M;
		r2=r1+M;
		s1[i]=accumulate(p1.begin()+r1,p1.begin()+r2,0);
		s2[i]=accumulate(p2.begin()+r1,p2.begin()+r2,0);
		sk[i]=((M+1)/(M-1))*(((M*s2[i])/pow(s1[i],2))-1);
		if(sk[i]>1.1 || sk[i]<0.9)
			rfi_status[i]=1;
		else
			rfi_status[i]=0;
		
	}
	float rfi_fraction = accumulate(rfi_status.begin(),rfi_status.end(),0.0)/nwindows;
	printf(">>>>>>>>>>>>RFI fraction: %f\n", rfi_fraction);
	printf("Done.\n");
}

void SpectralKurtosis::get_clean_window_indices(vInt x, int n){
	nclean_windows=count(x.begin(),x.end(),0);
	clean_win_index.resize(n);
	int iter=0;
	for(int index=0; index<n;index++){
		clean_win_index[index]=distance(x.begin(),min_element(x.begin()+iter,x.end()));
		iter=clean_win_index[index]+1;
	}//use transform function?

}

void SpectralKurtosis::get_rfi_window_indices(vInt x){
	nrfi_windows=count(x.begin(),x.end(),1);
	rfi_win_index.resize(nrfi_windows);
	int iter=0;
	//use transform function?
	for(int index=0; index<nrfi_windows;index++){
		rfi_win_index[index]=distance(x.begin(),max_element(x.begin()+iter,x.end()));
		iter=rfi_win_index[index]+1;
	}
}

stats SpectralKurtosis::get_stats(vComplex p){
	stats s;
	Complex sum;
	int len = p.size();

        sum=accumulate(p.begin(),p.end(),Complex(0,0));
	s.rmean=sum.real()/len;
	s.imean=sum.imag()/len;

	Complex mn,sd;
	
	vDouble preal(len),pimag(len),vr(len),vi(len);
        
	for(int i=0; i<p.size();i++){
		preal[i]=p[i].real();
		pimag[i]=p[i].imag();
	}//need a better logic - use transform?

	transform(preal.begin(), preal.end(), vr.begin(), bind2nd(minus<double>(),s.rmean));
	transform(pimag.begin(), pimag.end(), vi.begin(), bind2nd(minus<double>(),s.imean));
	s.rsd=sqrt((double)inner_product(vr.begin(), vr.end(), vr.begin(),0)/len);
	s.isd=sqrt((double)inner_product(vi.begin(), vi.end(), vi.begin(),0)/len);

	printf("Stats rmean= %lf, imean=%lf\n",s.rmean, s.imean);
	printf("Stats rsd= %lf, isd=%lf\n",s.rsd, s.isd);
	
	return s;
}

vComplex SpectralKurtosis::generate_reference_data(stats s){
	vComplex p(M);
	vDouble vreal(M), vimag(M);
	std::default_random_engine generator(time(0));
	std::normal_distribution<double> rdistribution(s.rmean, s.rsd);
	std::normal_distribution<double> idistribution(s.imean, s.isd);

	for(int i=0; i<M; i++){
		vreal[i]=(rdistribution(generator));
		vimag[i]=(idistribution(generator));
		p[i]=Complex(round(vreal[i]),round(vimag[i]));
	}

        stats ss;
	printf("stat ref_data.. \n");
	ss = get_stats(p);
	return p;
}

stats SpectralKurtosis::get_clean_data_stats(vComplex p, int n){
	printf("get clean data.. \n");
	clean_data.resize(n*M);
	int r1,r2,r11,ind;
	for(int i=0; i<n; i++){
		ind=clean_win_index[i];
                r1=ind*M;
		r2=r1+M;
		r11=i*M;
		std::copy(p.begin()+r1, p.begin()+r2,clean_data.begin()+r11);
	}
	stats s;
	printf("stat clean_data.. \n");
	s=get_stats(clean_data);
	return s;
}

vComplex SpectralKurtosis::mitigate_rfi(vComplex p){
	printf("mitigating RFI.. \n");
	stats clean_data_stat;
	vComplex r_data(M);

	//RFI present?
	if(*max_element(rfi_status.begin(),rfi_status.end())==1){
		//not all windows?
		if(nrfi_windows < nwindows){
			//get clean data stats
			clean_data_stat=get_clean_data_stats(p,5);

			//generate reference data
			r_data=generate_reference_data(clean_data_stat);

			//replace rfi data with reference data
			int r;
			for(int i=0; i<nrfi_windows; i++){
			//for(int i=0; i<9000; i++){
				r=rfi_win_index[i]*M;
				std::copy(r_data.begin(),r_data.end(),p.begin()+r);
			}
		}
	}
        return p;
}

void SpectralKurtosis::run_sk_p0(){
	int n=50;//first 'n' clean windows
	P0=compute_fft(p0);
	compute_sk(P0);
	get_clean_window_indices(rfi_status, n);
	get_rfi_window_indices(rfi_status);
        P0=mitigate_rfi(P0);
	compute_sk(P0);
}

void SpectralKurtosis::run_sk_p1(){
	int n=100;//first 'n' clean windows
	P1=compute_fft(p1);
	compute_sk(P1);
	get_clean_window_indices(rfi_status, n);
	get_rfi_window_indices(rfi_status);
        P1=mitigate_rfi(P1);
	compute_sk(P1);
}


int main(){
	SpectralKurtosis sk(1, 1000, "/media/scratch/BB_data/sk_test_data/after_sk/check/2019-08-24-21:04:10_0000000000000000.000000.dada");
	sk.read_file();
	printf("Pol0...\n");
	sk.run_sk_p0();
	printf("Pol1...\n");
	sk.run_sk_p1();
	return 0;
}
