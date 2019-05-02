# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:49:24 2019

@author: mbobb
"""
from helper_funcs import *



ft1 = get_entropy
ft2 = get_IF_mean_var
#ft4 = get_power_spec

patient = 7
datafolderpath = '/Volumes/TOSHIBA EXT/'

dt = extract_features_test(datafolderpath, patient, ft1,ft2)
d1, d2, d3, d4 = extract_features_train(datafolderpath, patient, ft1,ft2)
    


scipy.io.savemat('patient_'+str(patient)+'_ict_train3', d1)
scipy.io.savemat('patient_'+str(patient)+'_ict_val3', d2)
scipy.io.savemat('patient_'+str(patient)+'_nonict_train3', d3)
scipy.io.savemat('patient_'+str(patient)+'_nonict_val3', d4)
scipy.io.savemat('patient_'+str(patient)+'_test3', dt)