# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:03:04 2019

Python library for analysing transient grating experiments conducted at FERMI's EIS-TIMER beamline. 

@author: Bjorn Wehinger bjorn.wehinger@unive.it
"""

#######################################
#
# required libraries 
#
##################

import h5py , numpy, os, glob
import numpy.ma as ma
import matplotlib.pylab as plt
import scipy, scipy.special

#######################################
#
# hdf5 file configuration
#
##################

I0_counter = '/photon_diagnostics/FEL02/I0_monitor/iom_sh_a_pc'
spectrum_counter = 'photon_diagnostics/Spectrometer/hor_spectrum'

#######################################
#
# harmonic models
#
##################

# thermal contribution + one harmonic oscillation
def model1(t, t0, S0, s, A1, f1):
    t = t+t0
    return (S0*numpy.exp(-t/s) + A1*numpy.cos(2*numpy.pi*f1*t) )**2

# thermal contribution + two harmonic oscillation
def model2(t, t0, S0, s, A1, f1, A2, f2):
    t = t+t0
    return (S0*numpy.exp(-t/s) + A1*numpy.cos(2*numpy.pi*f1*t) + A2*numpy.cos(2*numpy.pi*f2*t) )**2

# thermal contribution + one damped harmonic oscillation
def model3(t, t0, S0, s, A1, f1, d1):
    t = t+t0
    return (S0*numpy.exp(-t/s) + numpy.exp(-t/d1)*A1*numpy.cos(2*numpy.pi*f1*t))**2

# thermal contribution + two damped harmonic oscillation
def model4(t, t0, S0, s, A1, f1, d1, A2, f2, d2):
    t = t+t0
    return (S0*numpy.exp(-t/s) + numpy.exp(-t/d1)*A1*numpy.cos(2*numpy.pi*f1*t) + numpy.exp(-t/d2)*A2*numpy.cos(2*numpy.pi*f2*t))**2

# thermal contribution with two decay times + one damped harmonic oscillation
def model5(t, t0, S0, s0, c, s1, A1, f1, d1):
    t = t+t0
    return (S0*numpy.exp(-t/s0) + c*S0*numpy.exp(-t/s1)+ numpy.exp(-t/d1)*A1*numpy.cos(2*numpy.pi*f1*t))**2

# thermal contribution with two decay times + two damped harmonic oscillation
def model6(t, t0, S0, s0, c, s1, A1, f1, d1, A2, f2, d2):
    t = t+t0
    return (S0*numpy.exp(-t/s0) + c*S0*numpy.exp(-t/s1) + numpy.exp(-t/d1)*A1*numpy.cos(2*numpy.pi*f1*t) + numpy.exp(-t/d2)*A2*numpy.cos(2*numpy.pi*f2*t))**2

#######################################
#    
# anharmonic model
#
##################
    
# quartic damped harmonic oscillator
# define petential    
def V(q,a,b):
    return  0.5*(2*numpy.pi*a)**2*q**2 + 0.25*b*q**4
# 
def model_aho(t, t0, S0, s, S1, a, b, d):
    from scipy.integrate import odeint
    t = t+t0
    def dVdq(q,a,b):
        return (2*numpy.pi*a)**2*q + b*q**3
    
    def deriv(Q, t):
        """Return the derivatives dx/dt and d2x/dt2."""
        q, qdot = Q
        qdotdot = -dVdq(q,a,b) - qdot/d      
        return qdot, qdotdot
     
    Q0 = [-1, 0]
    Q = odeint(deriv, Q0, t)
    # Q = solve_ivp(deriv, t_span=[min(t),max(t)], y0=Q0, t_eval=t) # to be debugged
    q = Q[:,0]
    return (S0*numpy.exp(-t/s) + S1*q )**2

def model_aho2(t, t0, S0, s, S1, a, b, d, S2, a2, b2, d2):
    from scipy.integrate import odeint
    t = t+t0
    def dVdq(q,a,b):
        return (2*numpy.pi*a)**2*q + b*q**3
    
    def deriv(Q, t):
        """Return the derivatives dx/dt and d2x/dt2."""
        q, qdot, q2, q2dot = Q
        qdotdot = -dVdq(q,a,b) - qdot/d
        q2dotdot = -dVdq(q2,a2,b2) - qdot/d2
        return qdot, qdotdot, q2dot, q2dotdot
     
    Q0 = [-1, 0, -1, 0]
    Q = odeint(deriv, Q0, t)
    # Q = solve_ivp(deriv, t_span=[min(t),max(t)], y0=Q0, t_eval=t) # to be debugged
    q = Q[:,0]
    q2 = Q[:,2]
    return (S0*numpy.exp(-t/s) + S1*q + S2*q2)**2

def poly1(x,p1):
    return  p1*x

#######################
#
# magnetic signal

def expConvGaussNormalised(x, tau, sig):   # convolution of a exponential decay (or recovery) with time constant tau at x0 = 0 and a gaussian of sigma = sig    
    term1 = (sig**2-2*(x)*tau)/(2*tau**2)    
    term2 = (sig**2+(-x)*tau)/(scipy.sqrt(2)*sig*tau)    
    return 1/2 * scipy.exp(term1) * (scipy.special.erf(term2)-1)

def doubleDecaySingleConv(x, t0, tau1, tau2, A, B, sig):
# =============================================================================
#     exponential decay and recovery, convoluted with a gaussian of sigma = sig
#     decay time tau1 and recovery time tau2
#     amplitudes A and B
#     t0 = t0
# =============================================================================
    C = (A+B)
    return -A*expConvGaussNormalised(x-t0, tau1, sig) -B*expConvGaussNormalised(x-t0, tau2, sig) +1/2*C*scipy.special.erfc(-(x-t0)/(scipy.sqrt(2*sig**2))) + 1

def demag(x, t0, tau1, tau2, A, sig):
    return (A*expConvGaussNormalised(x-t0, tau1, sig) -A*expConvGaussNormalised(x-t0, tau2, sig))**2


#######################################
#
# data analysis    

class TransientGrating:
    def __init__(self, data_files='', bg_files=None):
        self.init_import(data_files,bg_files)
    
    def init_import(self,data_files,bg_files=None):
         self.file_list = [os.path.abspath(x) for x in [f for f in sorted(glob.glob(data_files))]]
         self.bg_file_list = None
         if bg_files is not None:
             self.bg_file_list = [os.path.abspath(x) for x in [f for f in sorted(glob.glob(bg_files))]] 
             self.BG_sum = numpy.zeros(h5py.File(self.file_list[0], 'r')['CCD/Image'].shape)
         else: print('No background images provided')
         self.CCD_sum = numpy.zeros(h5py.File(self.file_list[0], 'r')['CCD/Image'].shape)
         self.I0_uh_average = None
         self.I0_spectrum_average = None
         self.tgs_signal = numpy.zeros(len(self.file_list))
         self.tgs_signal_error = numpy.zeros(len(self.file_list))
         self.tgs_signal_normalized = numpy.zeros(len(self.file_list))
         self.tgs_signal_normalized_to_I0_uh = numpy.zeros(len(self.file_list))
         self.tgs_signal_normalized_to_I0_spectrum = numpy.zeros(len(self.file_list))
         self.tgs_freq = None
         self.tgs_fft = None
         self.I0_probe = numpy.ones(len(self.file_list))
         self.I0_pump = numpy.ones(len(self.file_list))
         self.I0_spectrum = numpy.ones(len(self.file_list))
         self.I0_uh = numpy.ones(len(self.file_list))
         self.delay = numpy.zeros(len(self.file_list))
         self.spectrum_background = None
         self.spectrum_ROI = None
  
    def input_parameters(self, t0 = None, ROI = None, ROI_bg = None, I0_uh_threshold = None, spectrum_threshold = None, calibration = None, pump_wavelength = None, probe_wavelength = None, theta = None, transmission_pumpA = None, transmission_pumpB = None, transmission_probe = None, beamsize = None):
        if t0 is None:
            print('t0 not specified')
        self.t0 = t0
        if ROI is None:
            print('Signal ROI not specified')
        self.ROI = ROI
        if ROI_bg is None:
            print('Background ROI not specified')
        self.ROI_bg = ROI_bg
        if I0_uh_threshold is None:
            print('No I0 threshold provided')
        self.I0_uh_threshold = I0_uh_threshold
        if spectrum_threshold is None:
            print('No spectrum threshold provided')
        self.spectrum_threshold = spectrum_threshold
        if calibration is None:
            print('No calibrtion specified')
        else:
            self.calibration = calibration
            self.spectrum_background = calibration.background
            self.spectrum_ROI = calibration.ROI
        if pump_wavelength is None:
            print('No pump wavelength specified')
        if theta is None:
            print('Scattering angle not specified')
        else:
            print('Grating period')
            self.L_TG = pump_wavelength/(2*numpy.sin(theta*numpy.pi/180.))
            print('L_TG = ', numpy.round(self.L_TG,2), '[nm]' )
            self.dq = 2*numpy.pi/self.L_TG
            print('Momentum transfer')
            print('dq =', numpy.round(self.dq,3), '[1/nm]' )
        self.pump_wavelength = pump_wavelength
        if probe_wavelength is None:
            print('No probe wavelength specified')
        self.probe_wavelength = probe_wavelength
        if theta is None:
            print('No scattering angle specified')
        self.theta = theta
        if transmission_pumpA is None:
            print('Transmission of pump A not specified')
        self.transmission_pumpA = transmission_pumpA
        if transmission_pumpB is None:
            print('Transmission of pump B not specified')
        self.transmission_pumpB = transmission_pumpB
        if transmission_probe is None:
            print('Transmission of probe not specified')
        self.transmission_probe = transmission_probe
        if beamsize is None:
            print('Beamsize not specified')
        self.beamsize = beamsize

         
    def sum_all_images(self):
        CCD_sum = numpy.zeros(h5py.File(self.file_list[0], 'r')['CCD/Image'].shape)
        I0_uh_average = 0
        I0_spectrum_average = 0   
        for f in self.file_list:
            # print('working on file', f)
            g = h5py.File(f, 'r')
            CCD_sum += numpy.array(g['CCD/Image'],dtype=numpy.float)
            I0_uh_average += numpy.mean(numpy.array(g[I0_counter],dtype=numpy.float))
            if self.spectrum_ROI is not None:
                spectrum = numpy.array(g[spectrum_counter],dtype=numpy.float)
                I0_spectrum_average +=  numpy.mean(numpy.sum(spectrum[:,self.spectrum_ROI[0]:self.spectrum_ROI[1]], axis=1)-self.spectrum_background) 
        if self.spectrum_ROI is not None:
            self.I0_uh_average = I0_uh_average/len(self.file_list)
            I0_uh_pC = self.I0_uh_average
            self.I0_spectrum_average = I0_spectrum_average/len(self.file_list)
            I0_probe_pC = poly1(self.I0_spectrum_average, self.calibration.popt)
            I0_probe = self.calibration.conversion(self.probe_wavelength, I0_probe_pC)
            if self.pump_wavelength == self.probe_wavelength:
                I0_pump_pC = I0_uh_pC
            else:
                I0_pump_pC = (I0_uh_pC - I0_probe_pC)
            I0_pump_pC = ma.masked_invalid(I0_pump_pC)
            I0_pump = self.calibration.conversion(self.pump_wavelength,I0_pump_pC)
            if self.transmission_probe is not None:
                pump_flux_transmitted = I0_pump*(self.transmission_pumpA+self.transmission_pumpB)
                probe_flux_transmitted = I0_probe*self.transmission_probe
                if self.beamsize is not None:
                    pump_flux_transmitted = pump_flux_transmitted/self.beamsize/10.
                    probe_flux_transmitted = probe_flux_transmitted/self.beamsize/10.
                    #print('I_0 pump average at sample = ', numpy.round(pump_flux_transmitted,2)[0], '[mJ/cm^2] (no threshold filter applied)')
                    #print('I_0 probe average at sample = ', numpy.round(probe_flux_transmitted,3)[0], '[mJ/cm^2](no threshold filter applied)')
                    
                else:
                    print(' ')
                    #print('I_0 pump average at sample = ', numpy.round(pump_flux_transmitted,2)[0], '[µJ] (no threshold filter applied)')
                    #print('I_0 probe average at sample = ', numpy.round(probe_flux_transmitted,3)[0], '[µJ](no threshold filter applied)')
            else:
                print(' ')
                #print('I_0_average at source = ', numpy.round(I0_pump,2)[0], '[µJ] (no threshold filter applied')
                #print('I0_spectrum_average at source = ',  numpy.round(I0_probe,3)[0], '[µJ] (no threshold filter applied')
        self.CCD_sum = CCD_sum/len(self.file_list)
        if self.bg_file_list is not None:
            BG_sum = numpy.zeros(h5py.File(self.bg_file_list[0], 'r')['CCD/Image'].shape)
            I0_uh_average = 0
            I0_spectrum_average = 0   
            for f in self.bg_file_list:
                g = h5py.File(f, 'r')
                BG_sum += numpy.array(g['CCD/Image'],dtype=numpy.float)
            self.BG_sum = BG_sum/len(self.bg_file_list)
        
    def plot_sum(self):
        if self.bg_file_list is None:
            plt.imshow(self.CCD_sum, cmap='gist_yarg')
        else:
            plt.imshow(self.CCD_sum-self.BG_sum, cmap='gist_yarg')
        plt.xlim(0,self.CCD_sum.shape[1])
        plt.ylim(0,self.CCD_sum.shape[0])
        plt.title('Sum of all images')
        
    def plot_BG(self):
        plt.imshow(self.BG_sum, cmap='gist_yarg')
        plt.xlim(0,self.CCD_sum.shape[1])
        plt.ylim(0,self.CCD_sum.shape[0])
        plt.title('Sum of all background images')
        
        
    def plot_signal_and_background(self, ROI = [60,100,60,120], ROI_bg =  [10,50,10,50], bad_pixels = None, signal_filter = None):
        self.ROI = ROI
        self.ROI_bg = ROI_bg
        self.bad_pixels = bad_pixels
        self.signal_filter = signal_filter
        self.signal_filter_mask = None
        if self.bg_file_list is None:
            CCD = self.CCD_sum
        else:
             CCD = self.CCD_sum-self.BG_sum
        if self.bad_pixels is not None:
            for i in range(len(self.bad_pixels)):
                CCD[self.bad_pixels[i][0]][self.bad_pixels[i][1]]=numpy.NaN
            CCD = ma.masked_invalid(CCD)
        signal_mask = numpy.zeros(CCD.shape)
        signal_mask[ROI[2]:ROI[3],ROI[0]:ROI[1]]=1
        signal = CCD*signal_mask
        if self.signal_filter is not None:
            signal = ma.masked_less(signal, self.signal_filter)
            self.signal_filter_mask = signal.mask[ROI[2]:ROI[3],ROI[0]:ROI[1]]   
        bg_mask = numpy.zeros(CCD.shape)    
        bg_mask[ROI_bg[2]:ROI_bg[3],ROI_bg[0]:ROI_bg[1]]=1
        background = CCD[ROI_bg[2]:ROI_bg[3],ROI_bg[0]:ROI_bg[1]]
        if self.signal_filter is not None:    
            plt.imshow(numpy.clip((signal.mask*1-1)*(-1)*CCD + bg_mask*CCD,numpy.min(background),numpy.max(signal)),cmap='gist_yarg')
        else:
            plt.imshow(numpy.clip(signal + bg_mask*CCD,numpy.min(background),numpy.max(signal)),cmap='gist_yarg')
        #plt.imshow(numpy.clip(bg_mask*CCD,numpy.min(background),numpy.max(signal)),cmap='gist_yarg')
        plt.xlim(min([ROI[0],ROI_bg[0]])-1,max([ROI[1],ROI_bg[1]]))
        plt.ylim(min([ROI[2],ROI_bg[2]])-1,max([ROI[3],ROI_bg[3]]))     
        
    def line_plots(self):
        if self.bg_file_list is None:
            CCD = self.CCD_sum
        else:
            CCD = self.CCD_sum-self.BG_sum
        signal = CCD[self.ROI[2]:self.ROI[3],self.ROI[0]:self.ROI[1]]
        plt.figure()
        for i in range(signal.shape[0]):
            plt.plot(signal[i,:])
        plt.figure()
        for i in range(signal.shape[1]):
            plt.plot(signal[:,i])

    def tgs(self):
        file_list = self.file_list
        ROI_bg    =  self.ROI_bg
        bad_pixels = self.bad_pixels
        signal_filter = self.signal_filter
        signal_filter_mask = self.signal_filter_mask
        points_before_t0 = 0 
        
        for i in range(len(file_list)):
        #for i in range(20):
            g   = h5py.File(file_list[i], 'r')
            #print('working on file',file_list[i])
            if self.bg_file_list is None:
                CCD = numpy.array(g['CCD/Image'],dtype=numpy.float)
            else:
                CCD = numpy.array(g['CCD/Image'],dtype=numpy.float)-self.BG_sum
            background = CCD[ROI_bg[2]:ROI_bg[3],ROI_bg[0]:ROI_bg[1]]
            background = numpy.sum(background)/background.shape[0]/background.shape[1]
            CCD = CCD-background
            if bad_pixels is not None:
                for j in range(len(bad_pixels)):
                    CCD[self.bad_pixels[j][0]][self.bad_pixels[j][1]]=numpy.NaN
                CCD = ma.masked_invalid(CCD)
            signal = CCD[self.ROI[2]:self.ROI[3],self.ROI[0]:self.ROI[1]]
            if signal_filter is not None:
                signal = ma.masked_array(signal, mask=signal_filter_mask)
            iom_uh_a = numpy.array(g[I0_counter],dtype=float)
            number_of_shots = iom_uh_a.shape[0]
            iom_uh_a = ma.masked_invalid(iom_uh_a)
            uh_mask = ma.masked_less(iom_uh_a, self.I0_uh_threshold*self.I0_uh_average).mask+numpy.zeros(number_of_shots,dtype=bool)
            self.I0_uh[i] = ma.masked_array(iom_uh_a, mask=uh_mask).sum()/iom_uh_a.shape[0]
            summed_signal = numpy.sum(signal)
            if self.calibration.photon_convert is not None:
                converted_signal = summed_signal*self.calibration.photon_convert
                self.tgs_signal[i] = converted_signal/number_of_shots
                self.tgs_signal_error[i] = numpy.sqrt(converted_signal)/number_of_shots
            else:
                self.tgs_signal[i] = summed_signal/number_of_shots   
            if self.pump_wavelength == self.probe_wavelength:
                self.I0_probe[i] = self.I0_uh[i]
                self.I0_pump[i] = self.I0_uh[i]
            elif self.spectrum_ROI is not None:
                spectrum = numpy.array(g[spectrum_counter],dtype=float)
                spectrum = ma.masked_invalid(spectrum)
                spectrum_int = numpy.sum(spectrum[:,self.spectrum_ROI[0]:self.spectrum_ROI[1]], axis=1)-self.spectrum_background
                if self.spectrum_threshold is None:
                    self.I0_spectrum[i] = spectrum_int.sum()/number_of_shots                    
                else:
                    spectrum_mask = ma.masked_outside(spectrum_int, self.spectrum_threshold[0]*self.I0_spectrum_average, self.spectrum_threshold[1]*self.I0_spectrum_average).mask+numpy.zeros(spectrum.shape[0],dtype=bool)
                    self.I0_spectrum[i] = ma.masked_array(spectrum_int,mask=uh_mask+spectrum_mask).sum()/number_of_shots
                self.I0_probe[i] = poly1(self.I0_spectrum[i], self.calibration.popt)
            self.delay[i] = numpy.array(g['padres/DelayLine/FEL_delay'],dtype=float)-self.t0
            if self.delay[i] < 0:
                points_before_t0 +=1
        print('Background correction before t0:', numpy.round(numpy.mean(self.tgs_signal[0:7]),2), 'averaged over', points_before_t0, 'points')
        self.I0_uh = ma.masked_invalid(self.I0_uh)
        self.tgs_signal = self.tgs_signal-numpy.mean(self.tgs_signal[0:points_before_t0-1])
        self.tgs_signal_error = self.tgs_signal_error+max(numpy.abs(numpy.abs(self.tgs_signal[0:points_before_t0])-numpy.abs(self.tgs_signal_error[0:points_before_t0])))
        self.tgs_signal_normalized_to_I0_uh = self.tgs_signal/self.I0_uh**3*numpy.mean(self.I0_uh)**3
        self.tgs_signal_normalized_to_I0_uh_e = self.tgs_signal_error/self.I0_uh**3*numpy.mean(self.I0_uh)**3
        if self.pump_wavelength == self.probe_wavelength:
            self.I0_pump = ma.masked_invalid(self.I0_pump)
            self.pump_flux = self.calibration.conversion(self.pump_wavelength, numpy.mean(self.I0_pump))  
            self.I0_probe = ma.masked_invalid(self.I0_probe)
        elif self.spectrum_ROI is not None:
            #print('Average total flux at source: ',numpy.round(numpy.mean(self.I0_uh),2), 'µJ (not corrected for setup dependent beamline transmission)' )
            self.I0_pump = (self.I0_uh - self.I0_probe)
            self.I0_pump = ma.masked_invalid(self.I0_pump)
            self.I0_probe = ma.masked_invalid(self.I0_probe)
            self.tgs_signal_normalized = self.tgs_signal/self.I0_pump**2*numpy.mean(self.I0_pump)**2/self.I0_probe*numpy.mean(self.I0_probe)
            self.tgs_signal_normalized_e = self.tgs_signal_error/self.I0_pump**2*numpy.mean(self.I0_pump)**2/self.I0_probe*numpy.mean(self.I0_probe)
            
            self.tgs_signal_normalized_to_I0_spectrum = self.tgs_signal/self.I0_spectrum**3*numpy.mean(self.I0_spectrum)**3
            self.tgs_signal_normalized_to_I0_spectrum_e = self.tgs_signal_error/self.I0_spectrum**3*numpy.mean(self.I0_spectrum)**3           
            self.pump_flux = self.calibration.conversion(self.pump_wavelength, numpy.mean(self.I0_pump))
        print('Average pump flux at source: ', numpy.round(self.pump_flux, 2), 'µJ')
        if self.transmission_pumpA is not None:
            self.pump_flux_transmitted = self.pump_flux*(self.transmission_pumpA+self.transmission_pumpB)
            if self.beamsize is not None:   
                self.pump_flux_per_surface = self.pump_flux_transmitted/self.beamsize/10.
                print('Average pump flux at sample: ', numpy.round(self.pump_flux_per_surface, 2), 'mJ/cm^2')
            else:
                print('Average pump flux at sample: ', numpy.round(self.pump_flux_transmitted, 2), 'µJ')
        self.probe_flux = self.calibration.conversion(self.probe_wavelength, numpy.mean(self.I0_probe))
        print('Average probe flux at source: ', numpy.round(self.probe_flux, 2), 'µJ')
        if self.transmission_probe is not None:
            self.probe_flux_transmitted = self.probe_flux*self.transmission_probe
            if self.beamsize is not None:
                self.probe_flux_per_surface = self.probe_flux_transmitted/self.beamsize/10.
                print('Average probe flux at sample: ', numpy.round(self.probe_flux_per_surface, 2), 'mJ/cm^2')
            else:
                print('Average probe flux at sample: ', numpy.round(self.probe_flux_transmitted, 2), 'µJ')
            
    def tgs_plot(self, normalisation = 1 ):
        if normalisation == 0:      
            if self.calibration.photon_convert is not None:
                plt.errorbar(self.delay,self.tgs_signal,yerr=self.tgs_signal_error,fmt='.-b')
            else:
                plt.plt(self.delay,self.tgs_signal,'.-b')
            plt.xlabel('Delay [ps]')
            plt.ylabel('Intensity [arb. units]')
            plt.title('No normalization')
        elif normalisation == 1:
            if self.spectrum_ROI is not None:
                if self.calibration.photon_convert is not None:
                    plt.errorbar(self.delay,self.tgs_signal_normalized, yerr=self.tgs_signal_normalized_e)
                    plt.ylabel('Intensity [Photon counts/shot]')
                else:
                    plt.plot(self.delay,self.tgs_signal_normalized,'.-b')
                    plt.ylabel('Intensity [arb. units]')
                plt.xlabel('Delay [ps]')
                plt.title(r'Normalization to $(I_{0,pump})^2 * I_{0,probe}$')
            else:
                print('calibration of spectrometer required')
        elif normalisation == 2:
            if self.calibration.photon_convert is not None:
                plt.errorbar(self.delay,self.tgs_signal_normalized_to_I0_uh, yerr= self.tgs_signal_normalized_to_I0_uh_e)
                plt.ylabel('Intensity [Photon counts/shot]')
            else:
                plt.plot(self.delay,self.tgs_signal_normalized_to_I0_uh,'.-b')
                plt.ylabel('Intensity [arb. units]')
            plt.xlabel('Delay [ps]')
            plt.title(r'Normalization to $(I_{0,uh})^3$')
        elif normalisation == 3:
            if self.spectrum_ROI is not None:
                if self.calibration.photon_convert is not None:
                    plt.errorbar(self.delay,self.tgs_signal_normalized_to_I0_spectrum, yerr=self.tgs_signal_normalized_to_I0_spectrum_e)
                    plt.ylabel('Intensity [Photon counts/shot]')
                else:
                    plt.plot(self.delay,self.tgs_signal_normalized_to_I0_spectrum,'.-b')
                    plt.ylabel('Intensity [arb. units]')
                plt.xlabel('Delay [ps]')
                plt.title(r'Normalization to $(I_{0,spectrum})^3$')
            else:
                print('calibration of spectrometer required')
            
    def fft(self, normalisation = 1, units = 'THz', delay_cutoff=0): 
        from scipy.interpolate import interp1d
        if normalisation == 0:      
            s = self.tgs_signal
        elif normalisation == 1:
            if self.spectrum_ROI is not None:
                s = self.tgs_signal_normalized
            else:
                print('calibration of spectrometer required')
        elif normalisation == 2:
            s = self.tgs_signal_normalized_to_I0_uh
        elif normalisation == 3:
            if self.spectrum_ROI is not None:
                s = self.tgs_signal_normalized_to_I0_sepectrum
            else:
                print('calibration of spectrometer required')    
        signal = s[numpy.count_nonzero(self.delay<delay_cutoff):]
        delay = self.delay
        delay = delay[numpy.count_nonzero(delay<delay_cutoff):]
        #print(delay)
        T = delay[2] - delay[1]
        delay_interp = numpy.linspace(delay[0],delay[-1], int(numpy.round(delay[-1]/T,0)))
        interp = interp1d(delay, signal)
        s = interp(delay_interp)
        N = s.size
        tgs_freq = numpy.linspace(0, 1.0 / T, N)
        self.tgs_freq = tgs_freq[:N // 2]
        tgs_fft = numpy.fft.fft(s)
        self.tgs_fft = numpy.abs(tgs_fft)[:N // 2] * 1 / N
                
        plt.ylabel('Amplitude')
        if units == 'THz':
            plt.xlabel('Frequency [THz]')
            #plt.plot(self.tgs_freq[:N // 2], numpy.abs(self.tgs_fft)[:N // 2] * 1 / N, '.-')  # 1 / N is a normalization factor
            plt.plot(self.tgs_freq, self.tgs_fft, '.-')
        elif units == 'meV':
            THz2meV = 4.13567
            plt.xlabel('Energy [meV]')
            plt.plot(self.tgs_freq*THz2meV, self.tgs_fft, '.-')
        plt.title('Amplitude of TGS signal in frequency space')
    
    def save_tgs(self, filename = 'tgs_data.dat', normalisation = 1):
        if normalisation == 0: 
            data = numpy.vstack([self.delay,self.tgs_signal,self.tgs_signal_error]).T
            numpy.savetxt(filename, data, delimiter=',') 
            print('data saved to',filename)
        elif normalisation == 1: 
            data = numpy.vstack([self.delay,self.tgs_signal_normalized,self.tgs_signal_normalized_e]).T
            numpy.savetxt(filename, data, delimiter=',') 
            print('data saved to',filename)  
        elif normalisation == 2: 
            data = numpy.vstack([self.delay,self.tgs_signal_normalized_to_I0_uh,self.tgs_signal_normalized_to_I0_uh_e]).T
            numpy.savetxt(filename, data, delimiter=',') 
            print('data saved to',filename)  

    def tgs_fit(self, model = 2, angle = 27.6 , wavelength = 41.6, p0 = [10., 0.2, 2., 0.24, 2., 0.49],  bounds =([0, 0, -numpy.inf, 0., -numpy.inf, 0.], [numpy.inf, 100., numpy.inf, 20., numpy.inf, 20. ]), max_nfev=5000, normalisation = 1, units='THz'):
        from scipy.optimize import curve_fit
        delay = self.delay
        delay_fit = delay[numpy.count_nonzero(delay<0):]
        delay_theo = numpy.linspace(0.2, max(delay), 20000)
        self.delay_theo = delay_theo
        if normalisation == 0: 
            intens = self.tgs_signal
            intens_error = self.tgs_signal_error
        elif normalisation == 1: 
            intens = self.tgs_signal_normalized
            intens_error = self.tgs_signal_normalized_e
        elif normalisation == 2: 
            intens = self.tgs_signal_normalized_to_I0_uh  
            intens_error = self.tgs_signal_normalized_to_I0_uh_e
        intens_fit = intens[numpy.count_nonzero(delay<0):]
        if model == 1:
            # Fit
            popt,pcov = curve_fit(model1, delay_fit, intens_fit, p0=p0, bounds=bounds)
            perr = numpy.sqrt(numpy.diag(pcov))
            print('Fitting model: I = [S0*exp(-(t+t0)/s) + A1*cos(2*pi*f1*(t+t0))]^2')
            print('Fit result : [t0, S0, s, A1, f1] =', popt )
            print('Error :', perr)
            intens_theo = model1(delay_theo, *popt)
        elif model == 2:
            # Fit
            popt,pcov = curve_fit(model2, delay_fit, intens_fit, p0=p0, bounds=bounds)
            perr = numpy.sqrt(numpy.diag(pcov))
            print('Fitting model: I = [S0*exp(-(t+t0)/s) + A1*cos(2*pi*f1*(t+t0)) + A2*cos(2*pi*f2*(t+t0))]^2')
            print('Fit result : [t0, S0, s, A1, f1, A2, f2] =', popt )
            print('Error :', perr)
            intens_theo = model2(delay_theo, *popt)
        elif model == 3:
            # Fit
            popt,pcov = curve_fit(model3, delay_fit, intens_fit, p0=p0, bounds=bounds)
            perr = numpy.sqrt(numpy.diag(pcov))
            print('Fitting model: I = [S0*exp(-(t+t0)/s) + A1*exp(-(t+t0)/d1)*cos(2*pi*f1*(t+t0))]^2')
            print('Fit result : [t0, S0, s, A1, f1, d1] =', popt )
            print('Error :', perr)
            intens_theo = model3(delay_theo, *popt)
        elif model == 4:
            # Fit
            popt,pcov = curve_fit(model4, delay_fit, intens_fit, p0=p0, bounds=bounds)
            perr = numpy.sqrt(numpy.diag(pcov))
            print('Fitting model: I = [S0*exp(-(t+t0)/s) + A1*exp(-(t+t0)/d1)*cos(2*pi*f1*(t+t0)) + A2*exp(-(t+t0)/d2)*cos(2*pi*f2*(t+t0))]^2')
            print('Fit result : [t0, S0, s, A1, f1, d1, A2, f2, d2] =', popt )
            print('Error :', perr)
            intens_theo = model4(delay_theo, *popt)    
        elif model == 5:
            # Fit
            popt,pcov = curve_fit(model5, delay_fit, intens_fit, p0=p0, bounds=bounds)
            perr = numpy.sqrt(numpy.diag(pcov))
            print('Fitting model: I = [S0*exp(-(t+t0)/s) + A1*exp(-(t+t0)/d1)*cos(2*pi*f1*(t+t0))]^2')
            print('Fit result : [t0, S0, s0, S1, s1, A1, f1, d1] =', popt)
            print('Error :', perr)
            intens_theo = model5(delay_theo, *popt)   
        elif model == 6:
            # Fit
            popt,pcov = curve_fit(model6, delay_fit, intens_fit, p0=p0, bounds=bounds)
            perr = numpy.sqrt(numpy.diag(pcov))
            print('Fitting model: I = [S0*exp(-(t+t0)/s) + A1*exp(-(t+t0)/d1)*cos(2*pi*f1*(t+t0)) + A2*exp(-(t+t0)/d1)*cos(2*pi*f1*(t+t0))]^2')
            print('Fit result : [t0, S0, s0, S1, s1, A1, f1, d1, A2, f2, d2] =', popt )
            print('Error :', perr)
            intens_theo = model6(delay_theo, *popt) 
        elif model =='aho':
            popt,pcov = curve_fit(model_aho, delay_fit, intens_fit, p0=p0, bounds=bounds)
            perr = numpy.sqrt(numpy.diag(pcov))
            print('Fitting model: I = [S0*(exp((t+t0)/s + numerically solved damped quartic oscillator (AHO) )]^2')
            print('AHO potential: V = 0.5(2*pi*a)**2*q**2 + 0.25*b*q**4') 
            print('AHO differential equation:')
            print('qdotdot = -dVdq(q,a,b) - qdot/d')
            print('Fit result : [t0, S0, s, a, b, d] =', popt )
            print('Error :', perr)
            intens_theo = model_aho(delay_theo, *popt)  
            plt.figure()
            qgrid = numpy.linspace(-1.,1,200)
            plt.plot(qgrid,V(qgrid,popt[3],popt[4]),label='fitted anharmonic potential')
            plt.plot(qgrid,V(qgrid,popt[3],0),label='harmonic potential')
            plt.legend()
            plt.xlabel('Phonon displacement [arb. units]')
            plt.ylabel('Potential [arb. units]')
        elif model =='aho2':
            popt,pcov = curve_fit(model_aho2, delay_fit, intens_fit, p0=p0, bounds=bounds)
            perr = numpy.sqrt(numpy.diag(pcov))
            print('Fitting model: I = [S0*(exp((t+0)/s + 2 numerically solved damped quartic oscillators (AHO) )]^2')
            print('AHO potential: V = 0.5(2*pi*a)**2*q**2 + 0.25*b*q**4') 
            print('AHO differential equation:')
            print('qdotdot = -dVdq(q,a,b) - qdot/d')
            print('Fit result : [t0, S0, s, a, b, d] =', popt )
            print('Error :', perr)
            intens_theo = model_aho2(delay_theo, *popt)  
            plt.figure()
            qgrid = numpy.linspace(-1.,1,200)
            plt.plot(qgrid,V(qgrid,popt[3],popt[4]),label='fitted anharmonic potential')
            plt.plot(qgrid,V(qgrid,popt[3],0),label='harmonic potential')
            plt.legend()
            plt.xlabel('Phonon displacement [arb. units]')
            plt.ylabel('Potential [arb. units]')
        else:
            popt,pcov = curve_fit(model, delay_fit, intens_fit, p0=p0, bounds=bounds, max_nfev=10000)
            perr = numpy.sqrt(numpy.diag(pcov))
            self.popt = popt
            self.perr = perr
            print('Fit result :', popt )
            print('Error :', perr)
            intens_theo = model(delay_theo, *popt)  
        # Plot
        plt.figure()
        if self.calibration.photon_convert is not None:
            plt.errorbar(delay+popt[0],intens,yerr=intens_error,fmt='.b', label= 'Data')
            plt.ylabel('Intensity [Photons per shot]')
        else:
            plt.plot(delay,intens,'.b', label= 'Data')
            plt.ylabel('Intensity [arb. units]')
        plt.plot(delay_theo+popt[0], intens_theo, color='orange' , label= 'Fit')
        plt.legend()
        plt.xlabel('Delay [ps]')
        
           
        #calculate FFT
        if self.tgs_fft is not None:
            plt.figure()
            s = intens_theo
            T = delay_theo[1] - delay_theo[0]  # sampling interval 
            N = s.size
            tgs_freq = numpy.linspace(0, 1.0 / T, N)
            tgs_freq = tgs_freq[:N // 2]
            tgs_fft = numpy.fft.fft(s)
            tgs_fft = numpy.abs(tgs_fft)[:N // 2] * 1 / N
                
            plt.ylabel('Amplitude')
            if units == 'THz':
                plt.xlabel('Frequency [THz]')
                if self.tgs_fft is None:
                    print('FFT on data not performed, execute "fft"')
                else:
                    plt.plot(self.tgs_freq, self.tgs_fft, '.', label='Data')  
                    plt.plot(tgs_freq, tgs_fft, '-', label='Fit')
            elif units == 'meV':
                THz2meV = 4.13567
                plt.xlabel('Energy [meV]')
                if self.tgs_fft is None:
                    print('FFT on data not performed, execute "fft"')
                else:
                    plt.plot(self.tgs_freq*THz2meV, self.tgs_fft, '.', label='Data')
                    plt.plot(tgs_freq*THz2meV, tgs_fft, '-', label='Fit')
            plt.xlim(min(self.tgs_freq),max(self.tgs_freq))
            plt.ylim(0,max(self.tgs_fft))
            plt.title('Amplitude of TGS signal in frequency space')
            plt.legend()
        return popt, perr

#ef model2(t, S0, a, A1, f1, A2, f2):
#    return (S0*numpy.exp(-a*t) + A1*numpy.cos(2*numpy.pi*f1*t) + A2*numpy.cos(2*numpy.pi*f2*t) )**2



class Calibration:
    def __init__(self, calibration_files, probe_wavelength=None):
        self.init_import(calibration_files)
        self.popt = None
        self.background = None
        self.ROI = None
        self.probe_wavelength=probe_wavelength
        if probe_wavelength is not None:
            self.photon_convert = 1/(0.75*(1239/self.probe_wavelength/3.5)*0.437)
        else:
            self.photon_convert = None
        
#        Lambda FEL = 20 nm
#        Energia  FEL = 1239/20 = 61.95 eV
#        Nelettroni per fotone assorbito = 61.95/3.5 = 17.7 el
#        QE@60eV    = 75%
#        Nelettroni per fotone assorbito reali = 0.75*( Nelettroni per fotone assorbito) = 13.3 el/Ph
#        1 ADU/Ph 
#        PixelsWell   (linear) = 1x105 el
#        Output amplifier     = 2x105 el
#        1 ADU(conteggio)   = 2*216/( PixelsWell+ Output amplifier) = 65536 ADU/1.5 x105 el = 0.437 ADU/el   
#        1 ADU/Ph = 0.437 ADU/el *13.3 el/Ph = 5.81 ADU/Ph
     

    def init_import(self,calibration_files,conversion_file='calibrazione.txt'):
        from scipy import interpolate 
        x = numpy.loadtxt(conversion_file, max_rows=1)
        y = numpy.array(numpy.loadtxt(conversion_file, skiprows=1,usecols=0))/(2**20/1800)
        z = numpy.loadtxt(conversion_file,skiprows=1,usecols=range(1,len(x)+1))
        self.conversion = interpolate.interp2d(x, y, z, kind='linear')
        file_list = [os.path.abspath(x) for x in [f for f in sorted(glob.glob(calibration_files))]]
        I0_uh = numpy.array([])
        spectrum = numpy.array([])
        for f in file_list:
            g = h5py.File(f, 'r')
            I0_uh = numpy.append(I0_uh, numpy.array(g[I0_counter],dtype=numpy.float))
            spectrum = numpy.append(spectrum, numpy.array(g[spectrum_counter],dtype=numpy.float))
        self.I0_uh = I0_uh
        self.spectrum = numpy.reshape(spectrum, (len(self.I0_uh),-1))

    def plot_spectrum(self, index = 0, background = 2650, ROI = [557,760], ylim=(0, 1e4)):
        self.background = background
        self.ROI = ROI
        plt.plot(self.spectrum[index])
        plt.plot(self.spectrum[index]*0+background, '-r')
        plt.xlim([ROI[0]-40,ROI[1]+40])
        plt.vlines(ROI,ylim[0], ylim[1], 'r')
        plt.ylim(ylim)
                
    def calibrate(self, background = 2650, ROI = [557,760]):
        self.I0_spectrum = numpy.sum(self.spectrum[:,ROI[0]:ROI[1]], axis=1)-background
        from scipy.optimize import curve_fit
        self.popt,self.pcov = curve_fit(poly1,self.I0_spectrum, self.I0_uh)
        print('Fit result : y =', self.popt, '* x')
        return self.popt
        
    def plot_calibration(self):
        I0_spectrum_fit = numpy.linspace(numpy.min(self.I0_spectrum),numpy.max(self.I0_spectrum),200)
        I0_uh_fit = poly1(I0_spectrum_fit, self.popt)
        plt.plot(self.I0_spectrum,self.I0_uh, '.')
        plt.plot(I0_spectrum_fit,I0_uh_fit, '-r')
        plt.xlabel('I0_specturm')
        plt.ylabel('I0_uh_probe')   
        
    def plot_I0_spectrum(self):
        plt.plot(self.I0_spectrum)
        plt.xlabel('Bunch number')
        plt.ylabel('I0_spectrum')

    def plot_I0_uh(self):
        plt.plot(self.I0_uh)
        plt.xlabel('Bunch number')
        plt.ylabel('I0_uh_probe')

def save_calibration(calibration = None,filename='calibration.p'):
    import pickle
    calibration_file = open(filename, 'wb') 
    pickle.dump(calibration, calibration_file)
    
def load_calibration(filename='calibration.p'):
    import pickle
    calibration_file = open(filename, 'rb') 
    calibration = pickle.load(calibration_file)
    return calibration

