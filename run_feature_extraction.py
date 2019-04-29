# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:49:24 2019

@author: mbobb
"""
from helper_funcs import *



ft1 = channel_line_length
ft2 = get_energy
ft3 = get_variance
ft4 = get_power_spec

patient = 5
datafolderpath = '/Volumes/TOSHIBA EXT/'

dt = extract_features_test(datafolderpath, patient, ft1,ft2,ft3,ft4)
d1, d2, d3, d4 = extract_features_train(datafolderpath, patient, ft1,ft2,ft3,ft4)
    


scipy.io.savemat('patient_'+str(patient)+'_ict_train', d1)
scipy.io.savemat('patient_'+str(patient)+'_ict_val', d2)
scipy.io.savemat('patient_'+str(patient)+'_nonict_train', d3)
scipy.io.savemat('patient_'+str(patient)+'_nonict_val', d4)
scipy.io.savemat('patient_'+str(patient)+'_test', dt)