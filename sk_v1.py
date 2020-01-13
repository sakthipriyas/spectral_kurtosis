# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:22:02 2019

@author: spriyas
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

class SpectralKurtosis():
    def __init__(self, file_name, nchannels, m):
        self.nchannels = nchannels
        self.M=m
        self.file_name = file_name
        self.read_file()

        
    def read_file(self):
        f=open(self.file_name)
        f.seek(4096)
        self.f_dtype=np.dtype([('p0_real', np.int8), ('p0_imag', np.int8), 
                          ('p1_real', np.int8), ('p1_imag', np.int8)])
        self.data=np.fromfile(f,dtype=self.f_dtype)
        f.close()

        self.pol0=self.data['p0_real']+1j*self.data['p0_imag']
        self.pol1=self.data['p1_real']+1j*self.data['p1_imag']
        self.samples=self.pol1
        self.sample_size = self.pol1.size
        self.nwindows=self.pol0.size/(self.nchannels*self.M)
        print("sample size: ", self.sample_size)
        self.pol0=self.pol0.reshape(self.nwindows,self.M,self.nchannels)
        self.pol1=self.pol1.reshape(self.nwindows,self.M,self.nchannels)

    def write_file(self):
        print("Writing cleaned data in file.. ")
        self.new_data=np.zeros(self.sample_size, dtype=self.f_dtype)

        self.pol0 = self.pol0.reshape(self.nwindows*self.M*self.nchannels)
        self.pol1 = self.pol1.reshape(self.nwindows*self.M*self.nchannels)

        self.new_data['p0_real']=np.real(self.pol0)
        self.new_data['p0_imag']=np.imag(self.pol0)
        self.new_data['p1_real']=np.real(self.pol1)
        self.new_data['p1_imag']=np.imag(self.pol1)

        f=open(self.file_name, "r+b")
        f.seek(4096)
        f.write(self.new_data)
        f.close()
        
                     
    def compute_fft(self, samples):
        print(">>>>>>>>>>>>computing FFT...")
        if (self.nchannels>1):
            #self.ch_x=np.fft.fftshift(np.fft.fft(self.samples))
            self.ch_x=np.fft.fft(samples)
        else:
            print("Skipping FFT...")
            self.ch_x = samples

        self.ch_x=self.ch_x.transpose(2,0,1)
        self.ch=np.abs(self.ch_x)**2 #power
                                       
        print("Done.")
        
                  
    def compute_sk(self):
        print(">>>>>>>>>>>>>>computing SK...")
        #s1,s2,sk,mean,sd,rfi_status => shape(nchannels,nwindows)
        self.s1=self.ch.sum(axis=2)
        self.s2=(self.ch**2).sum(axis=2)
        self.sk=((self.M+1)/(self.M-1))*(((self.M*self.s2)/(self.s1**2))-1)
        
        self.mean=self.ch.mean(axis=2)
        self.sd=self.ch.std(axis=2)
        self.rfi_status=np.where((self.sk>=0.9)&(self.sk<1.1),0,1)
        print("RFI fraction: {}".format(self.rfi_status.sum()/float(self.rfi_status.size)))
                                                     
        print("SK Done.")
        
        
    def plot_data(self):
        plt.clf()
        fig=plt.figure(1)
        plt.subplot(2,3,1)
        #plt.hist(abs(self.pol1.transpose(2,1,0).reshape(self.nwindows*self.M))**2, bins=1024)
        plt.hist(abs(self.samples)**2, bins=1024)
        
        
        plt.subplot(2,3,2)        
#        self.p=self.ch.reshape(self.nchannels,(self.nwindows*self.M)).mean(axis=1)
        self.p=self.ch.transpose(1,2,0).reshape(self.nchannels,(self.nwindows*self.M)).mean(axis=1)
        #self.p=self.ch.transpose(2,0,1).mean(axis=1)
        plt.plot(self.p)
        
        plt.subplot(2,3,3)
        plt.imshow(self.ch.transpose(1,2,0).sum(axis=1), aspect="auto", interpolation="nearest") 
                   #vmin=150000, vmax=200000)
        plt.colorbar()
        
        plt.subplot(2,3,4)
        cmap = mpl.colors.ListedColormap(['orange', 'cyan', 'blue', 'red'])
        bounds = [0.1, 0.9, 1, 1.1, 2]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(self.sk.transpose(), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest", 
                   vmin=-3, vmax=3)
        plt.xlabel("Channels")
        plt.ylabel("Window, M={}".format(self.M))
        cbar = plt.colorbar()
        cbar.set_label('spectral kurtotis', rotation=270)
        
        #plt.subplot(2,3,5)        
        #p=self.ch1.reshape(self.nchannels,(self.nwindows*self.M)).sum(axis=1)
        #plt.plot(p)
        plt.subplot(2,3,5)
        #self.p_td=(abs(self.pol0)**2).reshape(self.nchannels, self.M*self.nwindows).mean(axis=1)
        self.p_td=(abs((self.pol0).transpose(2,0,1))**2).reshape(self.nchannels, self.M*self.nwindows).mean(axis=1)
        plt.plot(self.p_td)

        plt.draw()
        plt.show()
        

                        
    def rfi_mitigation_ifft(self):
        print(">>>>>>>>>>>>>>>RFI mitigation...")
        for ch in range(self.nchannels):
            if self.rfi_status[ch].max() == 1: #if RFI present in that channel
                nRFIwindows = np.count_nonzero(self.rfi_status[ch] == 1) #how many windows
                #print("ch: {}, no. RFI windows: {}".format(ch, nRFIwindows))
                if nRFIwindows < self.nwindows: #if not all windows
                    #Get clean window statistics
                    clean_window_index = np.where(self.rfi_status[ch]==0)
                    
                    ref_window_index = clean_window_index[0][0]
                    print("ch:{}, ref window index:{}".format(ch,ref_window_index))
                    self.clean_data = self.ch_x[ch][ref_window_index]
                    self.ref_rmean = self.clean_data.real.mean()
                    self.ref_rsd = self.clean_data.real.std()
                    self.ref_imean = self.clean_data.imag.mean()
                    self.ref_isd = self.clean_data.imag.std()
                    #real = np.random.normal(size=self.M*self.nchannels, loc=self.ref_rmean, scale=self.ref_rsd)
                    #imaginary = np.random.normal(size=self.M*self.nchannels, loc=self.ref_imean, scale=self.ref_isd)
                    real = np.random.normal(size=self.M, loc=self.ref_rmean, scale=self.ref_rsd)
                    imaginary = np.random.normal(size=self.M, loc=self.ref_imean, scale=self.ref_isd)
                    self.ref_data = real.round()+1j*(imaginary.round())
                    
                    
                    
                    rfi_window_index = np.where(self.rfi_status[ch]==1) #RFI windows 
                    #print("RFI ch: {}, windows:{}, index:{}".format(ch, nRFIwindows, rfi_window_index[0]))
                    for i in range(nRFIwindows):
                        #print("channel:{} RFI_window_index:{} sk:{}".format(ch, rfi_window_index[0][i], 
                         #      self.sk[ch][rfi_window_index[0][i]]))
                        self.ch_x[ch][rfi_window_index[0][i]]=self.ref_data
        #self.ch=np.abs(self.ch_x)**2
 
                            
    def compute_ifft(self):
        print(">>>>>>>>>>>>computing IFFT...")
        self.ch_x = self.ch_x.transpose(1,2,0)
        if (self.nchannels>1):
            #self.ch_x=np.fft.fftshift(np.fft.fft(self.samples))
            samples=np.fft.ifft(self.ch_x)
        else:
            print("Skipping IFFT...")
            samples = self.ch_x      
        return(samples)
        print("Done.")                             
                       
    def run_time(self):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Pol0")
        self.compute_fft(self.pol0)
        self.compute_sk()
        #self.rfi_mitigation_ifft()
        #self.pol0 = self.compute_ifft()
        #self.compute_fft(self.pol0)
        #self.compute_sk()
        #self.rfi_mitigation_ifft()

#        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Pol1")
#        self.compute_fft(self.pol1)
#        self.compute_sk()
#        self.rfi_mitigation_ifft(self.pol1)
#        self.compute_fft(self.pol1)
#        self.compute_sk()

#        self.write_file()


#        self.plot_data()
        
        
sk = SpectralKurtosis("/media/scratch/BB_data/sk_test_data/after_sk/check/2019-08-24-21:04:10_0000000000000000.000000.dada",2,1000)
#sk = SpectralKurtosis("/media/scratch/BB_data/sk_test_data/after_sk/check/2019-08-24-21:04:10_0000000640000000.000000.dada",1,2000)
#sk = SpectralKurtosis("data1.dada",1,4000)
sk.run_time()


