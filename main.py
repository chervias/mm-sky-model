#!/usr/local/Python-2.7.6/bin/python
import functions
import numpy as np
import sys
import re

#---------------------------
# Physical constants
#---------------------------
h = 4.135667516e-15     # planck constant in eV*s
k = 8.6173324e-5        # boltzmann constant in eV/K

#-----------------------------
# Spectral dependence laws
#-----------------------------

def dust_law(nu,beta_dust,T_dust,f_dust):
    # this is the spectral law of the dust component
    # nu must be in GHz
    x = h/k/T_dust
    val = f_dust * (nu)**(beta_dust+1.0) / (np.exp(x*nu*1e9)-1)
    return val
def syn_law(nu,beta_syn,delta_syn,ref_freq_syn,nu_pivot):
    # this is the spectral law of the synchroton component
    # nu must be in GHz
    return (nu/ref_freq_syn)**(-beta_syn + delta_syn*np.log(nu/nu_pivot))
def ame_law(nu,m60,nu_max):
	value = - (m60*np.log10(nu_max)/(np.log10(nu_max/60.0)) + 2.0 )*np.log10(nu) + (m60/(2.0*np.log10(nu_max/60.0)))*(np.log10(nu))**2
	return 10**value

# READ THE PARAMETERSM FROM THE INPUT FILE
file_in 	= open(str(sys.argv[1]))
param		= functions.read_parameters(file_in)

# Check that everything is consistent
if len(eval(param['list_freq'])) != len(eval(param['list_beams'])):
	print 'Error: The number of bands is not consistent with the number of beams'
	exit() 
if float(param['fwhm_small'])/60.0 > np.min(eval(param['list_beams']))/60.0:
	print 'Error: You are trying to simulate CMB and/or high-multipole features with a lower resolution than the best resolution from your experiment'
	exit()
pixel_size	= functions.nside2resol(float(param['Final_nside']))
if (2*pixel_size)/60.0 > float(param['fwhm_small'])/60.0:
	print '---------------------------------------------------------------------------'
	print 'WARNING: Your choice of Nside is not enough, you should go to higher Nside'
	print 'You do not have at least 2 pixels per beam for your highest resolution map'
	print '---------------------------------------------------------------------------'

# First, we need to know if you want to create fake small scale structure
if functions.to_boolean(param['artificial_high_ell']):
	# if the answer is yes, do you want to include ame
	if functions.to_boolean(param['include_ame'])==False:
		[templ_syn,templ_dust] = functions.create_high_ell(	param['dust_pol_templ'],
															param['syn_pol_templ'],
															eval(param['ell_range_ee_syn']),
															eval(param['ell_range_ee_dust']),
															eval(param['ell_range_bb_syn']),
															eval(param['ell_range_bb_dust']),
															float(param['scale_syn']),
															float(param['scale_dust']),
															int(param['seed_syn']),
															int(param['seed_dust']),
															float(param['fwhm_small'])/60.0,
															int(param['Final_nside']),
															functions.to_boolean(param['save_templates']),
															param['label'],
															functions.to_boolean(param['reuse_templates']),
															param['syn_templ_high_ell_name'],
															param['dust_templ_high_ell_name'],
															functions.to_boolean(param['plot_high_ell']),
															float(param['tilt_syn']),
															float(param['tilt_dust']),
															param['data_folder']
															)
		templ_ame	= 0 # Dummy variable
		print 'Artificial high-multipole features created..'
	else:
		[templ_syn,templ_dust,templ_ame] = functions.create_high_ell_ame(	param['dust_pol_templ'],
																			param['syn_pol_templ'],
																			param['ame_pol_templ'],
																			eval(param['ell_range_ee_syn']),
																			eval(param['ell_range_ee_dust']),
																			eval(param['ell_range_bb_syn']),
																			eval(param['ell_range_bb_dust']),
																			eval(param['ell_range_ee_ame']),
																			eval(param['ell_range_bb_ame']),
																			eval(param['ell_range_cross']),
																			float(param['scale_syn']),
																			float(param['scale_dust']),
																			float(param['scale_ame']),
																			int(param['seed_syn']),
																			int(param['seed_dust']),
																			float(param['fwhm_small'])/60.0,
																			int(param['Final_nside']),
																			functions.to_boolean(param['save_templates']),
																			param['label'],
																			functions.to_boolean(param['reuse_templates']),
																			param['syn_templ_high_ell_name'],
																			param['dust_templ_high_ell_name'],
																			param['ame_templ_high_ell_name'],
																			functions.to_boolean(param['plot_high_ell']),
																			float(param['tilt_syn']),
																			float(param['tilt_dust']),
																			float(param['tilt_ame']),
																			param['data_folder']
																			)
		templ_ame	= 0 # dummy variable
		print 'Artificial high-multipole features created.'
else:
	if functions.to_boolean(param['include_ame'])==False:
		[templ_syn,templ_dust] = functions.just_udgrade(	param['dust_pol_templ'],
															param['syn_pol_templ'],
															int(param['Final_nside']),
															param['data_folder']
															)
		templ_ame	= 0 # Dummy variable
	else:
		[templ_syn,templ_dust,templ_ame] = functions.just_udgrade_ame(	param['dust_pol_templ'],
																		param['syn_pol_templ'],
																		param['ame_pol_templ'],
																		int(param['Final_nside']),
																		param['data_folder']
																		)
	print 'Foreground templates ud_graded.'
# Second, we create the components if they are asked for

if functions.to_boolean(param['include_cmb']):
	cmb_list = functions.create_cmb(	param['cl_th'],
										eval(param['list_freq']),
										int(param['seed_cmb']),
										int(param['Final_nside']),
										functions.to_boolean(param['pixwin']),
										float(param['fwhm_small'])/60.0,
										functions.to_boolean(param['bands_weights']),
										param['data_folder']
									)
	print 'CMB map created.'
else:
	cmb_list 	= 0 # dummy variable

if functions.to_boolean(param['include_syn']):
	if functions.to_boolean(param['beta_syn_map'])==False:
		syn_list	= functions.create_foregrounds_syn(	templ_syn,
														syn_law,
														eval(param['list_freq']),
														eval(param['list_beams']),
														functions.to_boolean(param['bands_weights']),
														float(param['beta_syn']),
														float(param['ref_freq_syn']),
														float(param['nu_pivot']),
														int(param['Final_nside']),
														float(param['delta_syn']),
														param['data_folder']
														)
	else:
		syn_list	= functions.create_foregrounds_syn(	templ_syn,
														syn_law,
														eval(param['list_freq']),
														eval(param['list_beams']),
														functions.to_boolean(param['bands_weights']),
														param['beta_syn_name'],
														float(param['ref_freq_syn']),
														float(param['nu_pivot']),
														int(param['Final_nside']),
														float(param['delta_syn']),
														param['data_folder']
														)
	print 'Synchrotron map scaled in frequency.'
else:
	syn_list 	= 0 # dummy variable

if functions.to_boolean(param['include_ame']):
	if functions.to_boolean(param['nu_max_map'])==False:
		ame_list	= functions.create_foregrounds_ame(	templ_ame,
												ame_law,
												eval(param['list_freq']),
												eval(param['list_beams']),
												functions.to_boolean(param['bands_weights']),
												float(param['m60']),
												float(param['nu_max']),
												float(param['ref_freq_ame']),
												int(param['Final_nside']),
												param['data_folder']
											)
	else:
		ame_list	= functions.create_foregrounds_ame(	templ_ame,
												ame_law,
												eval(param['list_freq']),
												eval(param['list_beams']),
												functions.to_boolean(param['bands_weights']),
												float(param['m60']),
												param['nu_max_name'],
												float(param['ref_freq_ame']),
												int(param['Final_nside']),
												param['data_folder']
											)
	print 'AME map scaled in frequency.'
else:
	ame_list 	= 0 # dummy variable

if functions.to_boolean(param['include_dust']):
	if functions.to_boolean(param['beta_dust_map'])==False:
		# Extract 3 sub-dictionaries with the values of the N greybodies.
		dict_T_dust		= {}
		dict_beta_dust	= {}
		dict_E_dust		= {}
		# Now we check every variable in the param dictionary and keep the relevant variables
		for key in param:
			if re.match('beta_dust(?!_name)', key, flags=0):
				dict_beta_dust[key] 	= param[key]
			elif re.match('T_dust', key, flags=0):
				dict_T_dust[key]		= param[key]
			elif re.match('E_dust', key, flags=0):
				dict_E_dust[key]		= param[key]
		# Check that the number of variable is consistent with N_greybodies
		if len(dict_E_dust) != int(param['N_greybodies']) or len(dict_T_dust) != int(param['N_greybodies']):
			print 'ERROR: the number of grey bodies parameters is not consistent with N_greybodies'
			exit() 
		dust_list	= functions.create_foregrounds_dust(	templ_dust,
															dust_law,
															eval(param['list_freq']),
															eval(param['list_beams']),
															functions.to_boolean(param['bands_weights']),
															dict_beta_dust, # we supply all of the N_greybodies beta_dust in a single dictionary
															dict_T_dust, # we supply all of the N_greybodies T_dust in one dictionary
															dict_E_dust,
															float(param['ref_freq_dust']),
															int(param['Final_nside']),
															int(param['N_greybodies']),
															param['data_folder']
														)
	else:
		dict_T_dust				= {}
		dict_beta_dust_names	= {}
		dict_E_dust				= {}
		# Now we check every variable in the param dictionary and keep the relevant variables
		for key in param:
			if re.match('beta_dust_name', key, flags=0):
				dict_beta_dust_names[key] 	= param[key]
			elif re.match('T_dust', key, flags=0):
				dict_T_dust[key]		= param[key]
			elif re.match('E_dust', key, flags=0):
				dict_E_dust[key]		= param[key]		
		# Check that the number of variable is consistent with N_greybodies
		if len(dict_E_dust) != int(param['N_greybodies']) or len(dict_T_dust) != int(param['N_greybodies']):
			print 'ERROR: the number of grey bodies parameters is not consistent with N_greybodies'
			exit() 
		dust_list	= functions.create_foregrounds_dust(	templ_dust,
															dust_law,
															eval(param['list_freq']),
															eval(param['list_beams']),
															functions.to_boolean(param['bands_weights']),
															dict_beta_dust_names, # we supply all of the N_greybodies beta_dust in a single dictionary
															dict_T_dust, # we supply all of the N_greybodies T_dust in one dictionary
															dict_E_dust,
															float(param['ref_freq_dust']),
															int(param['Final_nside']),
															int(param['N_greybodies']),
															param['data_folder']
														)
	print 'Thermal dust map scaled in frequency.'
else:
	dust_list 	= 0 # dummy variable
	 
if functions.to_boolean(param['include_noise']):
	if functions.to_boolean(param['noise_maps'])==False:
		noise_list	= functions.create_noises(	eval(param['list_freq']),
												eval(param['list_noises']),
												int(param['Final_nside']),
												param['data_folder']
											)
	else:
		noise_list	= functions.create_noises(	eval(param['list_freq']),
												eval(param['noise_maps_names']),
												int(param['Final_nside']),
												param['data_folder']
											)
	print 'Noise realisations created.'
else:
	noise_list 	= 0 # dummy variable

# Third, we join the desired components and smooth them to the experiment resolution
full_model_smoothed_list = functions.join_and_smooth(	cmb_list,
														noise_list,
														syn_list,
														dust_list,
														ame_list,
														functions.to_boolean(param['include_cmb']),
														functions.to_boolean(param['include_noise']),
														functions.to_boolean(param['include_syn']),
														functions.to_boolean(param['include_dust']),
														functions.to_boolean(param['include_ame']),
														eval(param['list_freq']),
														eval(param['list_beams']),
														int(param['Final_nside']),
														float(param['fwhm_small'])/60.0
													)

# Fourth, choose what to write
functions.write_function(	functions.to_boolean(param['write_syn']),
							functions.to_boolean(param['write_dust']),
							functions.to_boolean(param['write_noise']),
							functions.to_boolean(param['write_cmb']),
							functions.to_boolean(param['write_ame']),
							full_model_smoothed_list,
							syn_list,
							dust_list,
							noise_list,
							cmb_list,
							ame_list,
							eval(param['list_freq']),
							param['label'],
							eval(param['list_beams']),
							param['output_folder'],
							float(param['fwhm_small'])/60.0
							)
