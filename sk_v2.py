
#data conversion implented.
#To check - writing data in file alter some values - Done! 
#fixed with round() in compute_ifft and not in get_reference_data - 17.02.2020 

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

class SpectralKurtosis():
    def __init__(self, file_name, nchannels, window_size):
        self.nchannels = nchannels
        self.M=window_size
        self.file_name = file_name
        self.read_data()

    def read_data(self):
        f=open(self.file_name)
        f.seek(4096)
        self.f_dtype=np.dtype([('p0_real', np.int8), ('p0_imag', np.int8), 
                          ('p1_real', np.int8), ('p1_imag', np.int8)])
        self.data=np.fromfile(f,dtype=self.f_dtype)
        f.close()
        self.pol0=self.data['p0_real']+1j*self.data['p0_imag']
        self.pol1=self.data['p1_real']+1j*self.data['p1_imag']
        self.nwindows=self.data.size/(self.nchannels*self.M)
        self.pol0=self.pol0.reshape(self.nwindows,self.M,self.nchannels)
        self.pol1=self.pol1.reshape(self.nwindows,self.M,self.nchannels)


    def compute_fft(self, samples):
        print(">>>>>>>>>>>>computing FFT...")
        if (self.nchannels>1):
            self.ch_x=np.fft.fft(samples)
            #self.ch_x=np.fft.fftshift(np.fft.fft(samples))
        else:
            print("Skipping FFT...")
            self.ch_x = samples

        self.ch_x=self.ch_x.transpose(2,0,1)
        self.ch=np.abs(self.ch_x)**2 #power
        print("Done.")
     

    def compute_sk(self):
        print(">>>>>>>>>>>>>>computing SK...")
        #s1,s2,sk,mean,sd,rfi_status have shape(nchannels,nwindows)

        self.s1=self.ch.sum(axis=2)
        self.s2=(self.ch**2).sum(axis=2)
        self.sk=((self.M+1)/(self.M-1))*(((self.M*self.s2)/(self.s1**2))-1)
        
        self.rfi_status=np.where((self.sk>=0.9)&(self.sk<1.1),0,1)
        print("RFI fraction: {}".format(self.rfi_status.sum()/float(self.rfi_status.size)))
                                                     
        print("SK Done.")


    def rfi_mitigation_ifft(self, mad):
        print(">>>>>>>>>>>>>>>RFI mitigation...")
        for ch in range(self.nchannels):
            if self.rfi_status[ch].max() == 1: #if RFI present in that channel
                nRFIwindows = np.count_nonzero(self.rfi_status[ch] == 1) #how many windows
                if nRFIwindows < self.nwindows: #if not all windows
                    ref_data=self.get_reference_data(ch, 50, mad)
                    rfi_window_index = np.where(self.rfi_status[ch]==1) #RFI windows 
                    for i in range(nRFIwindows):
                        self.ch_x[ch][rfi_window_index[0][i]]=ref_data


    def get_reference_data(self, ch, nref_windows, mad):
        #For sample size/data from certain number of windows.
        clean_window_index = np.where(self.rfi_status[ch]==0)
        self.clean_data=[]
        for i in range(nref_windows):
            self.clean_data = np.append(self.clean_data,self.ch_x[ch][clean_window_index[0][i]])
        
        self.ref_rmean = self.clean_data.real.mean()
        self.ref_imean = self.clean_data.imag.mean()
        if(mad==True):
            rmed=np.median(self.clean_data.real)
            imed=np.median(self.clean_data.imag)
            self.ref_rsd = np.median(np.abs(self.clean_data.real-rmed))*1.48
            self.ref_isd = np.median(np.abs(self.clean_data.imag-imed))*1.48
        else:
            self.ref_rsd = self.clean_data.real.std()
            self.ref_isd = self.clean_data.imag.std()
        self.real = np.random.normal(size=self.M, loc=self.ref_rmean, scale=self.ref_rsd)
        self.imaginary = np.random.normal(size=self.M, loc=self.ref_imean, scale=self.ref_isd)
        self.ref_data = self.real+1j*(self.imaginary)
        print("ref_data check: type:{}, rmin: {}, rmax: {}, imin: {}, imax: {}".format(type(self.ref_data.real[0]),
            self.ref_data.real.min(),
            self.ref_data.real.max(), self.ref_data.imag.min(),
            self.ref_data.imag.max()))

        return self.ref_data


    def compute_ifft(self):
        print(">>>>>>>>>>>>computing IFFT...")
        self.ch_x = self.ch_x.transpose(1,2,0)
        if (self.nchannels>1):
            #self.ch_x=np.fft.fftshift(np.fft.fft(self.samples))
            samples=np.fft.ifft(self.ch_x)
        else:
            print("Skipping IFFT...")
            samples = self.ch_x 
        samples=samples.round()
        print("Done.")        
        return(samples)


    def write_file(self):
        print("Writing cleaned data to file.. ")
        self.new_data=np.zeros(self.data.size, dtype=self.f_dtype)

        self.pol0 = self.pol0.reshape(self.nwindows*self.M*self.nchannels)
        self.pol1 = self.pol1.reshape(self.nwindows*self.M*self.nchannels)
        self.new_data['p0_real']=self.pol0.real.astype(np.int8)
        self.new_data['p0_imag']=self.pol0.imag.astype(np.int8)
        self.new_data['p1_real']=self.pol1.real.astype(np.int8)
        self.new_data['p1_imag']=self.pol1.imag.astype(np.int8)
        f=open(self.file_name, "r+b")
        f.seek(4096)
        f.write(self.new_data)
        f.close()
        print("Done.")

    def run_sk_pol0(self):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Pol0")
        self.compute_fft(self.pol0)
        self.compute_sk()
        #self.rfi_mitigation_ifft(True)
        #self.pol0 = self.compute_ifft()
    #    self.compute_fft(self.pol0)
    #    self.compute_sk()
        print("Done.")


    def run_sk_pol1(self):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Pol1")
        self.compute_fft(self.pol1)
        self.compute_sk()
        #self.rfi_mitigation_ifft(True)
        #self.pol1 = self.compute_ifft()
        #self.compute_fft(self.pol1)
        #self.compute_sk()
        print("Done.")


    def run(self):
        self.run_sk_pol0()
        self.run_sk_pol1()
        #self.write_file()


sk = SpectralKurtosis("/media/scratch/BB_data/sk_test_data/after_sk/2019-08-24-21:04:10_0000000000000000.000000.dada",2,2000)
sk.run()


