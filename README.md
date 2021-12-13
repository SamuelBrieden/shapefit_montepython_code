# shapefit_montepython_code
Provides a likelihood folder ("shapefit_bossdr12_max_kmin01kmax15") and a data folder ("DR12_max_kmin01kmax15") which can be linked to MontePython to obtain cosmological constraints from BOSS DR12 data.  

Inside your montepython_xxx folder, copy the likelihod folder into "montepython/likelihoods/" and the data folder into "data/". Then you can run cosmological models on BOSS DR12 ShapeFit results in the same way as running a conventional BAO or RSD cosmological likelihood fit.

The ShapeFit results for z1=0.38 and z3=0.61 were obtained using the wavevector range of 0.01<k<0.15 and the 'max' case (free local bias parameters bs2 and b3nl) from the ShapeFit paper 2106.07641. This likelihood can be used to reproduce the results presented in 2106.11931.  
