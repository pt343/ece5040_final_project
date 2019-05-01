# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:49:24 2019

@author: mbobb
"""
from helper_funcs import *



ft1 = get_skew
ft2 = get_mean
ft3 = get_kurtosis
#ft4 = get_power_spec

patient = 1
datafolderpath = '/Volumes/TOSHIBA EXT/'

dt = extract_features_test(datafolderpath, patient, ft1,ft2,ft3)
d1, d2, d3, d4 = extract_features_train(datafolderpath, patient, ft1,ft2,ft3)
    


scipy.io.savemat('patient_'+str(patient)+'_ict_train2', d1)
scipy.io.savemat('patient_'+str(patient)+'_ict_val2', d2)
scipy.io.savemat('patient_'+str(patient)+'_nonict_train2', d3)
scipy.io.savemat('patient_'+str(patient)+'_nonict_val2', d4)
scipy.io.savemat('patient_'+str(patient)+'_test2', dt)