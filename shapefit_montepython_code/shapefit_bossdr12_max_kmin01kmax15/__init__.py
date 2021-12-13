import os
import numpy as np
import warnings
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood
import scipy.constants as conts

class shapefit_bossdr12_max_kmin01kmax15(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # needed arguments in order to get sigma_8(z) up to z=1 with correct precision
        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'P_k_max_h/Mpc': 1.})
        self.need_cosmo_arguments(data, {'z_max_pk': 1.})

        # are there conflicting experiments?
        if 'bao_boss_aniso' in data.experiments:
            raise io_mp.LikelihoodError(
                'conflicting bao_boss_aniso measurments')

        # define arrays for values of z and data points
        self.z = np.array([], 'float64')
        self.aperp = np.array([], 'float64')
        self.apara = np.array([], 'float64')
        self.mslope = np.array([], 'float64')
        self.fAmp = np.array([], 'float64')
        self.pkshape_fid = [np.array([], 'float64'),np.array([], 'float64')]
        # read redshifts and data points
        with open(os.path.join(self.data_directory, self.data_file), 'r') as filein:
            for i, line in enumerate(filein):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    # load redshifts and alpha_parallel
                    if this_line[1] == 'apara':
                        self.z = np.append(self.z, float(this_line[0]))
                        self.apara = np.append(self.apara, float(this_line[2]))
                    # load alpha perpendicular
                    elif this_line[1] == 'aperp':
                        self.aperp = np.append(self.aperp, float(this_line[2]))
                    # load shape tilt
                    elif this_line[1] == 'mslope':
                        self.mslope = np.append(self.mslope, float(this_line[2]))
                    # load f * sigma8
                    elif this_line[1] == 'fAmp':
                        self.fAmp = np.append(self.fAmp, float(this_line[2]))

        # read covariance matrix
        self.cov_data = np.loadtxt(os.path.join(self.data_directory, self.cov_file))

        # number of bins
        self.num_bins = np.shape(self.z)[0]

        # number of data points
        self.num_points = np.shape(self.cov_data)[0]

        # fiducial EH98 power spectrum 
        self.kvec, self.pkshape_fid[0], self.pkshape_fid[1] = np.loadtxt(os.path.join(self.data_directory, self.pkshape_fid_file), unpack=True)

        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):

        # define array for  values of D_M_diff = D_M^th - D_M^obs and H_diff = H^th - H^obs,
        # ordered by redshift bin (z=[0.38, 0.51, 0.61]) as following:
        # data_array = [DM_diff(z=0.38), H_diff(z=0.38), DM_diff(z=0.51), .., .., ..]
        data_array = np.array([], 'float64')

        # for each point, compute comoving angular diameter distance D_M = (1 + z) * D_A,
        # sound horizon at baryon drag rs_d, theoretical prediction
        for i in range(self.num_bins):
            DM_at_z = cosmo.angular_distance(self.z[i]) * (1. + self.z[i])
            H_at_z = cosmo.Hubble(self.z[i]) * conts.c / 1000.0
            rd = cosmo.rs_drag() * self.rs_rescale
            theo_fAmp = cosmo.scale_independent_growth_factor_f(self.z[i])*np.sqrt(cosmo.pk_lin(self.kmpiv*self.h_fid*self.rd_fid_in_Mpc/rd,self.z[i])*(self.h_fid*self.rd_fid_in_Mpc/rd)**3.)/self.Amp[i]
            theo_aperp = DM_at_z / self.DM_fid_in_Mpc[i] / rd * self.rd_fid_in_Mpc
            theo_apara = self.H_fid[i] / H_at_z / rd * self.rd_fid_in_Mpc
            EHpk = self.EH98(cosmo,self.kvec*self.h_fid*self.rd_fid_in_Mpc/cosmo.h()/rd,self.z[i],1.0)
            Pkshape_ratio_prime = self.slope_at_x(np.log(self.kvec),np.log(EHpk/self.pkshape_fid[i]))
            theo_mslope = np.interp(self.kmpiv,self.kvec,Pkshape_ratio_prime)
            
            # calculate difference between the sampled point and observations
            apara_diff = theo_apara - self.apara[i]
            aperp_diff = theo_aperp - self.aperp[i]
            mslope_diff = theo_mslope - self.mslope[i]
            fAmp_diff = theo_fAmp - self.fAmp[i]

            # save to data array
            data_array = np.append(data_array, apara_diff)
            data_array = np.append(data_array, aperp_diff)
            data_array = np.append(data_array, mslope_diff)
            data_array = np.append(data_array, fAmp_diff)

        # compute chi squared
        inv_cov_data = np.linalg.inv(self.cov_data)
        chi2 = np.dot(np.dot(data_array,inv_cov_data),data_array)

        # return ln(L)
        loglkl = - 0.5 * chi2

        return loglkl
    
    def EH98(self, cosmo, kvector, redshift, scaling_factor):
        cdict = cosmo.get_current_derived_parameters(['z_d'])
        h = cosmo.h()
        H_at_z = cosmo.Hubble(redshift) * conts.c /1000. /(100.*h)
        Omm = cosmo.Omega_m()
        Omb = cosmo.Omega_b()
        Omc = cosmo.omegach2()/h**2.
        Omm_at_z = Omm*(1.+redshift)**3./H_at_z**2.
        OmLambda_at_z = 1.-Omm_at_z
        ns = cosmo.n_s()
        rs = cosmo.rs_drag()*h/scaling_factor
        Omnu = Omm-Omb-Omc
        fnu = Omnu/Omm
        fb = Omb/Omm
        fnub = (Omb+Omnu)/Omm
        fc = Omc/Omm
        fcb = (Omc+Omb)/Omm
        pc = 1./4.*(5-np.sqrt(1+24*fc))
        pcb = 1./4.*(5-np.sqrt(1+24*fcb))
        Neff = cosmo.Neff()
        Omg = cosmo.Omega_g()
        Omr = Omg * (1. + Neff * (7./8.)*(4./11.)**(4./3.))
        aeq = Omr/(Omb+Omc)/(1-fnu)
        zeq = 1./aeq -1.
        Heq = cosmo.Hubble(zeq)/h
        keq = aeq*Heq*scaling_factor   
        zd = cdict['z_d']
        yd = (1.+zeq)/(1.+zd)
        growth = cosmo.scale_independent_growth_factor(redshift)
        if (fnu==0):
            Nnu = 0.
        else:
            Nnu = 1.
        #alpha_gamma = 1 - 0.328*np.log(431*Omm*h**2)*Omb/Omm + 0.38*np.log(22.3*Omm*h**2)*(Omb/Omm)**2
        alpha_nu = fc/fcb * (5 - 2*(pc+pcb))/(5-4*pcb) * (1-0.553*fnub + 0.126*fnub**3.) / (1 - 0.193*np.sqrt(fnu)*Nnu**0.2 + 0.169*fnu) \
                    *(1.+yd)**(pcb-pc) * (1+(pc-pcb)/2*(1+1./(3-4*pc)/(7-4*pcb))*(1.+yd)**(-1.))
        #eff_shape = (alpha_gamma + (1.-alpha_gamma)/(1+(0.43*kvector*rs)**4.))
        eff_shape = (np.sqrt(alpha_nu) + (1.-np.sqrt(alpha_nu))/(1+(0.43*kvector*rs)**4.))
        q0 = kvector/(keq/7.46e-2)/eff_shape
        betac = (1.-0.949*fnub)**(-1.)
        L0 = np.log(np.exp(1.)+1.84*np.sqrt(alpha_nu)*betac*q0)
        C0 = 14.4 + 325./(1+60.5*q0**1.08)
        T0 = L0/(L0+C0*q0**2.)
        if (fnu==0):
            yfs=0.
            qnu=3.92*q0
        else:
            yfs = 17.2*fnu*(1.+0.488*fnu**(-7./6.))*(Nnu*q0/fnu)**2.
            qnu = 3.92*q0*np.sqrt(Nnu/fnu)
        D1 = (1.+zeq)/(1.+redshift)*Omm_at_z/2./(Omm_at_z**(4./7.) - OmLambda_at_z + (1.+Omm_at_z/2.)*(1.+OmLambda_at_z/70.))
        Dcbnu = (fcb**(0.7/pcb)+(D1/(1.+yfs))**0.7)**(pcb/0.7) * D1**(1.-pcb)
        Bk = 1. + 1.24*fnu**(0.64)*Nnu**(0.3+0.6*fnu)/(qnu**(-1.6)+qnu**0.8)

        Tcbnu = T0*Dcbnu/D1*Bk
        deltah = 1.94e-5 * Omm**(-0.785-0.05*np.log(Omm))*np.exp(-0.95*(ns-1)-0.169*(ns-1)**2.)
        Pk = 2*np.pi**2. * deltah**2. * (kvector)**(ns) * Tcbnu**2. * growth**2. /cosmo.Hubble(0)**(3.+ns)
        return Pk

    def slope_at_x(self,xvector,yvector):
        diff = np.diff(yvector)/np.diff(xvector)
        diff = np.append(diff,diff[-1])
        return diff
