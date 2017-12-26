import numpy as np
import healpy as hp
import matplotlib.pyplot as pl
import time
from numpy.linalg import cholesky,LinAlgError
from scipy.stats import pearsonr,spearmanr

default_files	= [	'Dust_template_commander_Planck_2015.fits',
					'Dust_template_smoothed_1d.fits',
					'Syn_template_smoothed_1d.fits',
					'Syn_template_commander_Planck_2015.fits',
					'AME_template_smoothed_1d.fits',
					'beta_dust_from_planck_fwhm3d.fits',
					'synchrotron_giardino_02_high_freq.fits',
					'camb_72267493_lensedtotcls.fits'
					]

def read_parameters(file):
	newDict	= {'dust_pol_templ':'Dust_template_commander_Planck_2015.fits',
				'syn_pol_templ':'Syn_template_smoothed_1d.fits',
				'ame_pol_templ':'AME_template_smoothed_1d.fits',
				'ref_freq_dust':'353.0',
				'ref_freq_syn':'30.0',
				'ref_freq_ame':'23.0',
				'include_dust':'True',
				'include_syn':'True',
				'include_ame':'False',
				'N_greybodies':'1',
				'beta_dust_1':'1.53',
				'T_dust_1':'21.0',
				'E_dust_1':'1.0',
				'beta_syn':'3.11',
				'nu_pivot':'23.0',
				'delta_syn':'0.0',
				'm60':'4.0',
				'nu_max':'19',
				'beta_dust_map':'False',
				'beta_syn_map':'False',
				'nu_max_map':'False',
				'beta_dust_name_1':'beta_dust_from_planck_fwhm3d.fits',
				'beta_syn_name':'synchrotron_giardino_02_high_freq.fits',
				'nu_max_name':'',
				'artificial_high_ell':'False',
				'ell_range_ee_syn':'(10,120)',
				'ell_range_ee_dust':'(60,600)',
				'ell_range_ee_ame':'(10,120)',
				'ell_range_bb_syn':'(4,40)',
				'ell_range_bb_dust':'(60,600)',
				'ell_range_bb_ame':'(10,120)',
				'ell_range_cross':'(20,120)',
				'scale_syn':'14.0',
				'scale_dust':'10.0',
				'scale_ame':'22.0',
				'seed_syn':'-1',
				'seed_dust':'-1',
				'tilt_syn':'0.0',
				'tilt_dust':'0.0',
				'tilt_ame':'0.0',
				'save_templates':'False',
				'reuse_templates':'False',
				'syn_templ_high_ell_name':'',
				'dust_templ_high_ell_name':'',
				'ame_templ_high_ell_name':'',
				'plot_high_ell':'False',
				'Final_nside':'1024',
				'list_freq':'(30,353)',
				'bands_weights':'False',
				'fwhm_small':'2.0',
				'list_beams':'(1.0,1.0)',
				'include_noise':'True',
				'list_noises':'(1.0,1.0)',
				'noise_maps':'False',
				'noise_maps_names':'('','')',
				'include_cmb':'True',
				'cl_th':'camb_72267493_lensedtotcls.fits',
				'seed_cmb':'-1',
				'pixwin':'True',
				'output_folder':'.',
				'label':'',
				'write_cmb':'False',
				'write_syn':'False',
				'write_dust':'False',
				'write_noise':'False',
				'write_ame':'False',
				'data_folder':'../data_inputs'
				}
	for line in file:
		listedline = line.strip().split('=') # split around the = sign
		listedline[0] = listedline[0].replace('\t','').replace(' ','')
		listedline[1] = listedline[1].replace('\t','').replace(' ','')
		if len(listedline) > 1: # we have the = sign in there
			newDict[listedline[0]] = listedline[1]
	return newDict

def to_boolean(s):
	if s == 'True':
		return True
	elif s == 'False':
		return False
	else:
		exit()

def nside2resol(nside):
	return hp.nside2resol(nside,arcmin=True)

def read_map(file_name):
	return hp.read_map(file_name,nest=True,field=(0,1,2))

def thermo2antenna(nu):
	# nu in GHz
	h = 4.135667516e-15     # planck constant in eV*s
	k = 8.6173324e-5        # boltzmann constant in eV/K
	T = 2.72548
	x = h*nu*1E9/k/T
	return ((np.exp(x) - 1.0)**2/(x**2*np.exp(x)))**(-1)

def plot_high_ell_4(map_all_l_dust,map_all_l_syn,new_cls_dust,new_cls_syn,cls_dust,cls_syn):
	fig 			= pl.figure()
	ax1 			= fig.add_subplot(2,2,1)
	ax2 			= fig.add_subplot(2,2,2)
	ax3 			= fig.add_subplot(2,2,3)
	ax4 			= fig.add_subplot(2,2,4)

	ell					= np.array(range(len(new_cls_dust[0])))
		
	map_all_l_dust_ring	= hp.reorder(map_all_l_dust,n2r=True)
	n_side_map			= hp.get_nside(map_all_l_dust_ring)
	cl_all_l_dust 		= hp.anafast(map_all_l_dust_ring,pol=True,lmax=2*n_side_map-1)
	
	ax1.plot(new_cls_dust[0],color='red',lw=0.2)
	ax1.plot(cls_dust[1],color='green',lw=0.2)
	ax1.plot(cl_all_l_dust[1],color='blue',ls='dashed',lw=0.2)
	ax1.set_xscale('log');ax1.set_yscale('log')
	ax1.set_title('Dust')
		
	ax3.plot(new_cls_dust[1],color='red',lw=0.2)
	ax3.plot(cls_dust[2],color='green',lw=0.2)
	ax3.plot(cl_all_l_dust[2],color='blue',ls='dashed',lw=0.2)
	ax3.set_xscale('log');ax3.set_yscale('log')
	
	map_all_l_syn_ring	= hp.reorder(map_all_l_syn,n2r=True)
	n_side_map			= hp.get_nside(map_all_l_syn_ring)
	cl_all_l_syn 		= hp.anafast(map_all_l_syn_ring,pol=True,lmax=2*n_side_map-1)
	
	ax2.plot(new_cls_syn[0],color='red',lw=0.2)
	ax2.plot(cls_syn[1],color='green',lw=0.2)
	ax2.plot(cl_all_l_syn[1],color='blue',ls='dashed',lw=0.2)
	ax2.set_xscale('log');ax2.set_yscale('log')
	ax2.set_title('Synchrotron')
	
	ax4.plot(new_cls_syn[1],color='red',lw=0.2)
	ax4.plot(cls_syn[2],color='green',lw=0.2)
	ax4.plot(cl_all_l_syn[2],color='blue',ls='dashed',lw=0.2)
	ax4.set_xscale('log');ax4.set_yscale('log')

	pl.tight_layout()
	pl.savefig('high_ell_features_diagnostics.pdf',format='pdf')

def plot_high_ell_6(map_all_l_dust,map_all_l_syn,map_all_l_ame,new_cls_dust,new_cls_syn,new_cls_ame,cls_dust,cls_syn,cls_ame):

	fig 			= pl.figure()
	ax1 			= fig.add_subplot(2,3,1)
	ax2 			= fig.add_subplot(2,3,2)
	ax3 			= fig.add_subplot(2,3,3)
	ax4 			= fig.add_subplot(2,3,4)
	ax5 			= fig.add_subplot(2,3,5)
	ax6 			= fig.add_subplot(2,3,6)

	ell					= np.array(range(len(new_cls_dust[0])))
		
	map_all_l_dust_ring	= hp.reorder(map_all_l_dust,n2r=True)
	n_side_map			= hp.get_nside(map_all_l_dust_ring)
	cl_all_l_dust 		= hp.anafast(map_all_l_dust_ring,pol=True,lmax=2*n_side_map-1)
	
	ax1.plot(new_cls_dust[0],color='red',lw=0.2)
	ax1.plot(cls_dust[1],color='green',lw=0.2)
	ax1.plot(cl_all_l_dust[1],color='blue',ls='dashed',lw=0.2)
	ax1.set_xscale('log');ax1.set_yscale('log')
	ax1.set_title('Dust')
		
	ax4.plot(new_cls_dust[1],color='red',lw=0.2)
	ax4.plot(cls_dust[2],color='green',lw=0.2)
	ax4.plot(cl_all_l_dust[2],color='blue',ls='dashed',lw=0.2)
	ax4.set_xscale('log');ax4.set_yscale('log')
	
	map_all_l_syn_ring	= hp.reorder(map_all_l_syn,n2r=True)
	n_side_map			= hp.get_nside(map_all_l_syn_ring)
	cl_all_l_syn 		= hp.anafast(map_all_l_syn_ring,pol=True,lmax=2*n_side_map-1)
	
	ax2.plot(new_cls_syn[0],color='red',lw=0.2)
	ax2.plot(cls_syn[1],color='green',lw=0.2)
	ax2.plot(cl_all_l_syn[1],color='blue',ls='dashed',lw=0.2)
	ax2.set_xscale('log');ax2.set_yscale('log')
	ax2.set_title('Synchrotron')
	
	ax5.plot(new_cls_syn[1],color='red',lw=0.2)
	ax5.plot(cls_syn[2],color='green',lw=0.2)
	ax5.plot(cl_all_l_syn[2],color='blue',ls='dashed',lw=0.2)
	ax5.set_xscale('log');ax5.set_yscale('log')
	
	map_all_l_ame_ring	= hp.reorder(map_all_l_ame,n2r=True)
	n_side_map			= hp.get_nside(map_all_l_ame_ring)
	cl_all_l_ame 		= hp.anafast(map_all_l_ame_ring,pol=True,lmax=2*n_side_map-1)
	
	ax3.plot(new_cls_ame[0],color='red',lw=0.2)
	ax3.plot(cls_ame[1],color='green',lw=0.2)
	ax3.plot(cl_all_l_ame[1],color='blue',ls='dashed',lw=0.2)
	ax3.set_xscale('log');ax3.set_yscale('log')
	ax3.set_title('AME')
	
	ax6.plot(new_cls_ame[1],color='red',lw=0.2)
	ax6.plot(cls_ame[2],color='green',lw=0.2)
	ax6.plot(cl_all_l_ame[2],color='blue',ls='dashed',lw=0.2)
	ax6.set_xscale('log');ax6.set_yscale('log')
	
	pl.tight_layout()
	pl.savefig('high_ell_features_diagnostics.pdf',format='pdf')


def cross_correlate_ame_dust(cl_dust_set,cl_ame_set,cl_cross_set,seed,Final_nside,fwhm_small):
	# This function will receive 3 sets of cl's and will output two sets of maps that are correlated
	# First of all, set the seed
	if seed!=-1:
		np.random.seed(seed)
	else:
		random_seed = int(time.time())
		np.random.seed(random_seed)	
	lmax			= len(cl_dust_set[0]) - 1
	print 'the lmax is',lmax
	nfields			= 4 # The fields are E_dust,B_dust,E_ame,B_ame. WE do not include T because is zero in dust
	alms			= np.zeros((nfields,lmax+1,lmax+1),dtype=complex)
	for ell in range(2,lmax+1):
		print 'MULTIPOLE ',ell
		C_ell_matrix 		= np.zeros((nfields,nfields))
		# Fill in the diagonal
		C_ell_matrix[0,0]	= cl_dust_set[1][ell] 	# dust EE
		C_ell_matrix[1,1]	= cl_dust_set[2][ell] 	# dust BB
		C_ell_matrix[2,2]	= cl_ame_set[1][ell] 	# ame EE
		C_ell_matrix[3,3]	= cl_ame_set[2][ell] 	# ame BB
		# Fill in the off-diagonal
		C_ell_matrix[0,1]	= C_ell_matrix[1,0] = 0.0 #cl_dust_set[4][ell] 	# dust EB
		C_ell_matrix[2,3]	= C_ell_matrix[3,2] = 0.0 #cl_ame_set[4][ell] 	# ame EB
		C_ell_matrix[0,2]	= C_ell_matrix[2,0] = cl_cross_set[1][ell]	# cross EE
		C_ell_matrix[1,3]	= C_ell_matrix[3,1] = cl_cross_set[2][ell]	# cross BB
		C_ell_matrix[0,3]	= C_ell_matrix[3,0] = C_ell_matrix[1,2] = C_ell_matrix[2,1] = 0.0# cl_cross_set[4][ell]	# cross EB
		# This will do the cholesky dec
		try:
			L_ell_matrix		= cholesky(C_ell_matrix)
#			print ell,L_ell_matrix
#			exit()
		except LinAlgError:
			continue
		# Now, create two arrays with gaussian random numbers
		randoms				= np.random.standard_normal(2*nfields*(ell+1))
		gauss_1				= np.zeros((nfields,ell+1))
		gauss_2				= np.zeros((nfields,ell+1))
		counter				= 0
		for i in range(nfields):
			for m in range(ell+1):
				gauss_1[i,m]	= randoms[counter] 
				gauss_2[i,m]	= randoms[counter+1]
				counter			= counter + 2
		# for m = 0
		for i in range(nfields):
			for j in range(nfields):
				alms[i,ell,0] = alms[i,ell,0] + complex(gauss_1[j,0],0.0) * L_ell_matrix[i,j]
		# for m > 0
		for m in range(1,ell+1):
			for i in range(nfields):
				for j in range(nfields):
					alms[i,ell,m] = alms[i,ell,m] + complex(gauss_1[j,m],gauss_2[j,m]) * L_ell_matrix[i,j]
			alms[:,ell,m] = np.sqrt(0.5) * alms[:,ell,m]	
	# Transform the alms to the correct format
	alms_TEB_dust	= []
	alms_TEB_ame	= []
	# The size of the array must be mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1
	size 		= lmax * (2 * lmax + 1 - lmax)/2 + lmax + 1 # In this we assume lmax = mmax	
	# fill in the first array to zeros, corresponding to alm^T
	alms_TEB_dust.append(np.zeros(size,dtype=complex))
	alms_TEB_ame.append(np.zeros(size,dtype=complex))
	# fill in the second array to zeros, corresponding to alm^E
	alms_TEB_dust.append(np.zeros(size,dtype=complex))
	alms_TEB_ame.append(np.zeros(size,dtype=complex))
	for ell in range(0,lmax+1):
		for m in range(0,ell+1):
			alms_TEB_dust[1][m*(2*lmax+1-m)/2+ell]	= alms[0,ell,m]
			alms_TEB_ame[1][m*(2*lmax+1-m)/2+ell]	= alms[2,ell,m]
	# fill in the third array to zeros, corresponding to alm^B
	alms_TEB_dust.append(np.zeros(size,dtype=complex))
	alms_TEB_ame.append(np.zeros(size,dtype=complex))
	for ell in range(0,lmax+1):
		for m in range(0,ell+1):
			alms_TEB_dust[2][m*(2*lmax+1-m)/2+ell]	= alms[1,ell,m]
			alms_TEB_ame[2][m*(2*lmax+1-m)/2+ell]	= alms[3,ell,m]
	# Finally create the TQU maps
	maps_dust 	= hp.alm2map(alms_TEB_dust,nside=Final_nside,fwhm=np.radians(fwhm_small),pol=True,verbose=False)
	maps_ame	= hp.alm2map(alms_TEB_ame,nside=Final_nside,fwhm=np.radians(fwhm_small),pol=True,verbose=False)
	return [maps_dust,maps_ame]

def just_udgrade(dust_pol_templ,syn_pol_templ,Final_nside,data_folder):
	# This method is to just ud_grade the templates to the right nside, if you do not want to create fake small structure
	syn_map 		= hp.read_map(data_folder+'/'+syn_pol_templ,field=(0,1,2),nest=True,verbose=False)
	dust_map		= hp.read_map(data_folder+'/'+dust_pol_templ,field=(0,1,2),nest=True,verbose=False)
	N_side_syn		= hp.get_nside(syn_map[0])
	N_side_dust		= hp.get_nside(dust_map[0])
	tqu_syn			= syn_map
	tqu_dust		= dust_map
	tqu_syn_final	= hp.ud_grade(tqu_syn,Final_nside,order_in='nest',order_out='nest')
	tqu_dust_final	= hp.ud_grade(tqu_dust,Final_nside,order_in='nest',order_out='nest')
	return [tqu_syn_final,tqu_dust_final]

def just_udgrade_ame(dust_pol_templ,syn_pol_templ,ame_pol_templ,Final_nside,data_folder):
	# This method is to just ud_grade the templates to the right nside, if you do not want to create fake small structure
	syn_map 		= hp.read_map(data_folder+'/'+syn_pol_templ,field=(0,1,2),nest=True,verbose=False)
	dust_map		= hp.read_map(data_folder+'/'+dust_pol_templ,field=(0,1,2),nest=True,verbose=False)
	ame_map			= hp.read_map(data_folder+'/'+ame_pol_templ,field=(0,1,2),nest=True,verbose=False)
	N_side_syn		= hp.get_nside(syn_map[0])
	N_side_dust		= hp.get_nside(dust_map[0])
	tqu_syn			= syn_map
	tqu_dust		= dust_map
	tqu_ame			= ame_map
	tqu_syn_final	= hp.ud_grade(tqu_syn,Final_nside,order_in='nest',order_out='nest')
	tqu_dust_final	= hp.ud_grade(tqu_dust,Final_nside,order_in='nest',order_out='nest')
	tqu_ame_final	= hp.ud_grade(tqu_ame,Final_nside,order_in='nest',order_out='nest')
	return [tqu_syn_final,tqu_dust_final,tqu_ame_final]

def create_high_ell(dust_pol_templ,syn_pol_templ,
							ell_range_ee_syn,ell_range_ee_dust,
								ell_range_bb_syn,ell_range_bb_dust,
										scale_syn,scale_dust,
											seed_syn,seed_dust,
												fwhm_small,Final_nside,save_templates,label,reuse_templates,
													syn_templ_high_ell_name,dust_templ_high_ell_name,plot_high_ell,
														tilt_syn,tilt_dust,data_folder):
	
	# This method is for creating high-ell features for Dust and Syn, separately
	if reuse_templates:
		# Remember that in this case the user must provide full path
		map_all_l_syn 	= hp.read_map(syn_templ_high_ell_name,nest=True,field=(0,1,2),verbose=False)
		map_all_l_dust	= hp.read_map(dust_templ_high_ell_name,nest=True,field=(0,1,2),verbose=False)
		return [map_all_l_syn,map_all_l_dust]
	if syn_pol_templ in default_files:
		tqu_syn 		= hp.read_map(data_folder+'/'+syn_pol_templ,field=(0,1,2),nest=True,verbose=False)
	else:
		tqu_syn 		= hp.read_map(syn_pol_templ,field=(0,1,2),nest=True,verbose=False)
	if dust_pol_templ in default_files:
		tqu_dust		= hp.read_map(data_folder+'/'+dust_pol_templ,field=(0,1,2),nest=True,verbose=False)
	else:
		tqu_dust		= hp.read_map(dust_pol_templ,field=(0,1,2),nest=True,verbose=False)
	
	N_side_dust		= hp.get_nside(tqu_dust[0])
	N_side_syn 		= hp.get_nside(tqu_syn[0])
	tqu_syn_ring 	= hp.reorder(tqu_syn,n2r=True);
	tqu_dust_ring	= hp.reorder(tqu_dust,n2r=True);
	
	#These are auto power spectra
	cls_syn			= hp.anafast(tqu_syn_ring,pol=True,lmax=2*N_side_syn-1)
	cls_dust		= hp.anafast(tqu_dust_ring,pol=True,lmax=2*N_side_dust-1)
	
	ell_array			= np.array(range(2*Final_nside))
	
	# fit the power law on EE
	p_syn	 		= np.polyfit(np.log10(ell_array[ell_range_ee_syn[0]:ell_range_ee_syn[1]]),
									np.log10(cls_syn[1][ell_range_ee_syn[0]:ell_range_ee_syn[1]]),1)
	p_dust	 		= np.polyfit(np.log10(ell_array[ell_range_ee_dust[0]:ell_range_ee_dust[1]]),
									np.log10(cls_dust[1][ell_range_ee_dust[0]:ell_range_ee_dust[1]]),1)
	p_syn[0]	= p_syn[0]-tilt_syn
	p_dust[0]	= p_dust[0]-tilt_dust
	
	print 'The power law fit on the EE Syn Cl ',p_syn
	print 'The power law fit on the EE Dust Cl ',p_dust
	
	new_cls_syn_ee 	= np.zeros(ell_array.shape)
	new_cls_dust_ee = np.zeros(ell_array.shape)
	for ell in ell_array:
		if (2*N_side_syn-1)>=ell>=ell_range_ee_syn[1]:
			new_cls_syn_ee[ell] 	= 10.0**(p_syn[0]*np.log10(ell) + p_syn[1] ) - cls_syn[1][ell]
		elif ell>(2*N_side_syn-1):
			new_cls_syn_ee[ell] 	= 10.0**(p_syn[0]*np.log10(ell) + p_syn[1] )
		
		if (2*N_side_dust-1)>=ell>=ell_range_ee_dust[1]:
			new_cls_dust_ee[ell] 	= 10.0**(p_dust[0]*np.log10(ell) + p_dust[1] )-cls_dust[1][ell]
		elif ell>(2*N_side_dust-1):
			new_cls_dust_ee[ell] 	= 10.0**(p_dust[0]*np.log10(ell) + p_dust[1] )
	
	# fit the power law on BB
	p_syn	 		= np.polyfit(np.log10(ell_array[ell_range_bb_syn[0]:ell_range_bb_syn[1]]),
									np.log10(cls_syn[2][ell_range_bb_syn[0]:ell_range_bb_syn[1]]),1)
	p_dust	 		= np.polyfit(np.log10(ell_array[ell_range_bb_dust[0]:ell_range_bb_dust[1]]),
									np.log10(cls_dust[2][ell_range_bb_dust[0]:ell_range_bb_dust[1]]),1)
	p_syn[0]	= p_syn[0]-tilt_syn
	p_dust[0]	= p_dust[0]-tilt_dust
	
	print 'The power law fit on the bb Syn Cl ',p_syn
	print 'The power law fit on the bb Dust Cl ',p_dust
	
	new_cls_syn_bb 	= np.zeros(ell_array.shape)
	new_cls_dust_bb = np.zeros(ell_array.shape)
	for ell in ell_array:
		if (2*N_side_syn-1)>=ell>=ell_range_bb_syn[1]:
			new_cls_syn_bb[ell] 	= 10.0**(p_syn[0]*np.log10(ell) + p_syn[1] ) - cls_syn[1][ell]
		elif ell>(2*N_side_syn-1):
			new_cls_syn_bb[ell] 	= 10.0**(p_syn[0]*np.log10(ell) + p_syn[1] )
		
		if (2*N_side_dust-1)>=ell>=ell_range_bb_dust[1]:
			new_cls_dust_bb[ell] 	= 10.0**(p_dust[0]*np.log10(ell) + p_dust[1] )-cls_dust[1][ell]
		elif ell>(2*N_side_dust-1):
			new_cls_dust_bb[ell] 	= 10.0**(p_dust[0]*np.log10(ell) + p_dust[1] )
	
	# Create the high-l map for dust
	if seed_dust!=-1:
		np.random.seed(seed_dust)
	else:
		random_seed = int(time.time())
		np.random.seed(random_seed)
	map_high_l_dust 	= hp.synfast([np.zeros(2*Final_nside),new_cls_dust_ee,new_cls_dust_bb,np.zeros(2*Final_nside)],nside=Final_nside,new=True,pol=True,fwhm=np.radians(fwhm_small),lmax=(2*Final_nside-1),verbose=False)
	map_high_l_dust 	= hp.reorder(map_high_l_dust,r2n=True)
	
	# Create the high-l map for syn
	if seed_syn!=-1:
		np.random.seed(seed_syn)
	else:
		random_seed = int(time.time())
		np.random.seed(random_seed)
	map_high_l_syn 	= hp.synfast([np.zeros(2*Final_nside-1),new_cls_syn_ee,new_cls_syn_bb,np.zeros(2*Final_nside-1)],nside=Final_nside,new=True,pol=True,fwhm=np.radians(fwhm_small),lmax=(3*Final_nside-1),verbose=False)
	map_high_l_syn 	= hp.reorder(map_high_l_syn,r2n=True)
	
	# Modulate using the Galactic plane
	
	tqu_syn_final	= hp.ud_grade(tqu_syn,Final_nside,order_in='nest',order_out='nest')
	gplane_q_syn 	= tqu_syn_final[1]/np.mean(tqu_syn_final[1])
	gplane_u_syn 	= tqu_syn_final[2]/np.mean(tqu_syn_final[2])
	map_all_l_syn 	= [tqu_syn_final[0] + map_high_l_syn[0]/scale_syn,
							tqu_syn_final[1]+gplane_q_syn*map_high_l_syn[1]/scale_syn,
								tqu_syn_final[2]+gplane_u_syn*map_high_l_syn[2]/scale_syn]
	
	tqu_dust_final	= hp.ud_grade(tqu_dust,Final_nside,order_in='nest',order_out='nest')
	gplane_q_dust 	= tqu_dust_final[1]/np.mean(tqu_dust_final[1])
	gplane_u_dust 	= tqu_dust_final[2]/np.mean(tqu_dust_final[2])
	map_all_l_dust 	= [tqu_dust_final[0] + map_high_l_dust[0]/scale_dust,
						tqu_dust_final[1]+gplane_q_dust*map_high_l_dust[1]/scale_dust,
							tqu_dust_final[2]+gplane_u_dust*map_high_l_dust[2]/scale_dust]

	if plot_high_ell:
		new_cls_dust	= [new_cls_dust_ee,new_cls_dust_bb]
		new_cls_syn		= [new_cls_syn_ee,new_cls_syn_bb]
		plot_high_ell_4(map_all_l_dust,map_all_l_syn,new_cls_dust,new_cls_syn,cls_dust,cls_syn)
	if save_templates:
		hp.write_map('%s_template_syn_with_high_ell.fits'%(label),map_all_l_syn,nest=True)
		hp.write_map('%s_template_dust_with_high_ell.fits'%(label),map_all_l_dust,nest=True)
	return [map_all_l_syn,map_all_l_dust]

def create_high_ell_ame(dust_pol_templ,syn_pol_templ,ame_pol_templ,
							ell_range_ee_syn,ell_range_ee_dust,
								ell_range_bb_syn,ell_range_bb_dust,
									ell_range_ee_ame,ell_range_bb_ame,ell_range_cross,
										scale_syn,scale_dust,scale_ame,
											seed_syn,seed_dust,
												fwhm_small,Final_nside,save_templates,label,reuse_templates,
													syn_templ_high_ell_name,dust_templ_high_ell_name,ame_templ_high_ell_name,plot_high_ell,
														tilt_syn,tilt_dust,tilt_ame,data_folder):
	
	# This method is for creating high-ell features when you want to include AME, which will be correlated to the dust
	if reuse_templates:
		# These templates must be in the output folder
		map_all_l_syn 	= hp.read_map(output_folder+'/'+syn_templ_high_ell_name,nest=True,field=(0,1,2),verbose=False)
		map_all_l_dust	= hp.read_map(output_folder+'/'+dust_templ_high_ell_name,nest=True,field=(0,1,2),verbose=False)
		map_all_l_ame	= hp.read_map(output_folder+'/'+ame_templ_high_ell_name,nest=True,field=(0,1,2),verbose=False)
		return [map_all_l_syn,map_all_l_dust,map_all_l_ame]
	
	if syn_pol_templ in default_files:
		tqu_syn 		= hp.read_map(data_folder+'/'+syn_pol_templ,field=(0,1,2),nest=True,verbose=False)
	else:
		tqu_syn 		= hp.read_map(syn_pol_templ,field=(0,1,2),nest=True,verbose=False)
	if dust_pol_templ in default_files:
		tqu_dust		= hp.read_map(data_folder+'/'+dust_pol_templ,field=(0,1,2),nest=True,verbose=False)
	else:
		tqu_dust		= hp.read_map(dust_pol_templ,field=(0,1,2),nest=True,verbose=False)
	if ame_pol_templ in default_files:
		tqu_ame			= hp.read_map(data_folder+'/'+ame_pol_templ,field=(0,1,2),nest=True,verbose=False)
	else:
		tqu_ame			= hp.read_map(ame_pol_templ,field=(0,1,2),nest=True,verbose=False)
	
	N_side_dust		= hp.get_nside(tqu_dust[0])
	N_side_ame		= hp.get_nside(tqu_ame[0])
	N_side_syn 		= hp.get_nside(tqu_syn[0]);
	
	tqu_syn_ring 	= hp.reorder(tqu_syn,n2r=True);
	tqu_dust_ring	= hp.reorder(tqu_dust,n2r=True);
	tqu_ame_ring	= hp.reorder(tqu_ame,n2r=True)
	
	# Make sure the dust and ame maps have the same Nside, the user should provide both templates at the same resolution.
	# We undersample at the lowest N_side
	if N_side_dust < N_side_ame:
		tqu_ame_ring 	= hp.ud_grade(tqu_ame_ring,N_side_dust)
		N_side_ame		= N_side_dust
	elif N_side_dust > N_side_ame:
		tqu_dust_ring 	= hp.ud_grade(tqu_dust_ring,N_side_ame)
		N_side_dust		= N_side_ame
	
	#These are auto power spectra
	cls_syn			= hp.anafast(tqu_syn_ring,pol=True,lmax=2*N_side_syn-1)
	cls_dust		= hp.anafast(tqu_dust_ring,pol=True,lmax=2*N_side_dust-1)
	cls_ame			= hp.anafast(tqu_ame_ring,pol=True,lmax=2*N_side_ame-1)
	#This is the cross-spectra
	cls_cross		= hp.anafast(map1=tqu_dust_ring,map2=tqu_ame_ring,pol=True,lmax=2*N_side_dust-1)
	
	ell_array			= np.array(range(2*Final_nside))
	
	# fit the power law on EE
	p_syn	 		= np.polyfit(np.log10(ell_array[ell_range_ee_syn[0]:ell_range_ee_syn[1]]),
									np.log10(cls_syn[1][ell_range_ee_syn[0]:ell_range_ee_syn[1]]),1)
	p_dust	 		= np.polyfit(np.log10(ell_array[ell_range_ee_dust[0]:ell_range_ee_dust[1]]),
									np.log10(cls_dust[1][ell_range_ee_dust[0]:ell_range_ee_dust[1]]),1)
	p_ame			= np.polyfit(np.log10(ell_array[ell_range_ee_ame[0]:ell_range_ee_ame[1]]),
									np.log10(cls_ame[1][ell_range_ee_ame[0]:ell_range_ee_ame[1]]),1)
	p_syn[0]	= p_syn[0]-tilt_syn
	p_dust[0]	= p_dust[0]-tilt_dust
	p_ame[0]	= p_ame[0] - tilt_ame
	
	print 'The power law fit on the EE Syn Cl ',p_syn
	print 'The power law fit on the EE Dust Cl ',p_dust
	print 'The power law fit on the EE AME Cl ',p_ame
	
	new_cls_syn_ee 	= np.zeros(ell_array.shape)
	new_cls_dust_ee = np.zeros(ell_array.shape)
	new_cls_ame_ee	= np.zeros(ell_array.shape)
	for ell in ell_array:
		if (2*N_side_syn-1)>=ell>=ell_range_ee_syn[1]:
			new_cls_syn_ee[ell] 	= 10.0**(p_syn[0]*np.log10(ell) + p_syn[1] ) - cls_syn[1][ell]
		elif ell>(2*N_side_syn-1):
			new_cls_syn_ee[ell] 	= 10.0**(p_syn[0]*np.log10(ell) + p_syn[1] )
		
		if (2*N_side_dust-1)>=ell>=ell_range_ee_dust[1]:
			new_cls_dust_ee[ell] 	= 10.0**(p_dust[0]*np.log10(ell) + p_dust[1] )-cls_dust[1][ell]
		elif ell>(2*N_side_dust-1):
			new_cls_dust_ee[ell] 	= 10.0**(p_dust[0]*np.log10(ell) + p_dust[1] )
			
		if (2*N_side_ame-1)>=ell>=ell_range_ee_ame[1]:
			new_cls_ame_ee[ell] 	= 10.0**(p_ame[0]*np.log10(ell) + p_ame[1] )-cls_ame[1][ell]
		elif ell>(2*N_side_ame-1):
			new_cls_ame_ee[ell] 	= 10.0**(p_ame[0]*np.log10(ell) + p_ame[1] )
	
	# fit the power law on BB
	p_syn	 		= np.polyfit(np.log10(ell_array[ell_range_bb_syn[0]:ell_range_bb_syn[1]]),
									np.log10(cls_syn[2][ell_range_bb_syn[0]:ell_range_bb_syn[1]]),1)
	p_dust	 		= np.polyfit(np.log10(ell_array[ell_range_bb_dust[0]:ell_range_bb_dust[1]]),
									np.log10(cls_dust[2][ell_range_bb_dust[0]:ell_range_bb_dust[1]]),1)
	p_ame			= np.polyfit(np.log10(ell_array[ell_range_bb_ame[0]:ell_range_bb_ame[1]]),
									np.log10(cls_ame[2][ell_range_bb_ame[0]:ell_range_bb_ame[1]]),1)
	p_syn[0]	= p_syn[0]-tilt_syn
	p_dust[0]	= p_dust[0]-tilt_dust
	p_ame[0]	= p_ame[0] - tilt_ame
	
	print 'The power law fit on the bb Syn Cl ',p_syn
	print 'The power law fit on the bb Dust Cl ',p_dust
	print 'The power law fit on the bb AME Cl ',p_ame
	
	new_cls_syn_bb 	= np.zeros(ell_array.shape)
	new_cls_dust_bb = np.zeros(ell_array.shape)
	new_cls_ame_bb	= np.zeros(ell_array.shape)
	for ell in ell_array:
		if (2*N_side_syn-1)>=ell>=ell_range_bb_syn[1]:
			new_cls_syn_bb[ell] 	= 10.0**(p_syn[0]*np.log10(ell) + p_syn[1] ) - cls_syn[1][ell]
		elif ell>(2*N_side_syn-1):
			new_cls_syn_bb[ell] 	= 10.0**(p_syn[0]*np.log10(ell) + p_syn[1] )
		
		if (2*N_side_dust-1)>=ell>=ell_range_bb_dust[1]:
			new_cls_dust_bb[ell] 	= 10.0**(p_dust[0]*np.log10(ell) + p_dust[1] )-cls_dust[1][ell]
		elif ell>(2*N_side_dust-1):
			new_cls_dust_bb[ell] 	= 10.0**(p_dust[0]*np.log10(ell) + p_dust[1] )
			
		if (2*N_side_ame-1)>=ell>=ell_range_bb_ame[1]:
			new_cls_ame_bb[ell] 	= 10.0**(p_ame[0]*np.log10(ell) + p_ame[1] )-cls_ame[1][ell]
		elif ell>(2*N_side_ame-1):
			new_cls_ame_bb[ell] 	= 10.0**(p_ame[0]*np.log10(ell) + p_ame[1] )
	
	# fit the power law on cross spectra EE and BB	
	p_cross_ee	 	= np.polyfit(np.log10(ell_array[ell_range_cross[0]:ell_range_cross[1]]),np.log10(cls_cross[1][ell_range_cross[0]:ell_range_cross[1]]),1)
	p_cross_bb	 	= np.polyfit(np.log10(ell_array[ell_range_cross[0]:ell_range_cross[1]]),np.log10(cls_cross[2][ell_range_cross[0]:ell_range_cross[1]]),1)
	
	print 'The power law fit on the ee Cross Cl ',p_cross_ee
	print 'The power law fit on the bb Cross Cl ',p_cross_bb
	
	new_cls_cross_ee 	= np.zeros(ell_array.shape)
	new_cls_cross_bb 	= np.zeros(ell_array.shape)
	N_side_common		= np.max([N_side_dust,N_side_ame])
	
	for ell in ell_array:
		if (2*N_side_common-1)>=ell>=ell_range_cross[1]:
			new_cls_cross_ee[ell] 	= 10.0**(p_cross_ee[0]*np.log10(ell) + p_cross_ee[1] ) - cls_cross[1][ell]
		elif ell>(2*N_side_common-1):
			new_cls_cross_ee[ell] 	= 10.0**(p_cross_ee[0]*np.log10(ell) + p_cross_ee[1] )
			
		if (2*N_side_common-1)>=ell>=ell_range_cross[1]:
			new_cls_cross_bb[ell] 	= 10.0**(p_cross_bb[0]*np.log10(ell) + p_cross_bb[1] ) - cls_cross[2][ell]
		elif ell>(2*N_side_common-1):
			new_cls_cross_bb[ell] 	= 10.0**(p_cross_bb[0]*np.log10(ell) + p_cross_bb[1] )	
	
	# Run the correlated generation of maps
	cl_dust_set		= [np.zeros(len(ell_array)),new_cls_dust_ee,new_cls_dust_bb,np.zeros(len(ell_array))]
	cl_ame_set		= [np.zeros(len(ell_array)),new_cls_ame_ee,new_cls_ame_bb,np.zeros(len(ell_array))]
	cl_cross_set	= [np.zeros(len(ell_array)),new_cls_cross_ee,new_cls_cross_bb,np.zeros(len(ell_array))]
	
	[map_high_l_dust,map_high_l_ame] 	= cross_correlate_ame_dust(cl_dust_set,cl_ame_set,cl_cross_set,seed_dust,Final_nside,fwhm_small)
	map_high_l_dust						= hp.reorder(map_high_l_dust,r2n=True)
	map_high_l_ame						= hp.reorder(map_high_l_ame,r2n=True)

	# Create the high-l map for syn
	if seed_syn!=-1:
		np.random.seed(seed_syn)
	else:
		random_seed = int(time.time())
		np.random.seed(random_seed)
	map_high_l_syn 	= hp.synfast([np.zeros(2*Final_nside),new_cls_syn_ee,new_cls_syn_bb,np.zeros(2*Final_nside)],nside=Final_nside,new=True,pol=True,fwhm=np.radians(fwhm_small),lmax=(2*Final_nside-1),verbose=False)
	map_high_l_syn 	= hp.reorder(map_high_l_syn,r2n=True)
	
	tqu_syn_final	= hp.ud_grade(tqu_syn,Final_nside,order_in='nest',order_out='nest')
	gplane_q_syn 	= tqu_syn_final[1]/np.mean(tqu_syn_final[1])
	gplane_u_syn 	= tqu_syn_final[2]/np.mean(tqu_syn_final[2])
	map_all_l_syn 	= [tqu_syn_final[0] + map_high_l_syn[0]/scale_syn,
							tqu_syn_final[1]+gplane_q_syn*map_high_l_syn[1]/scale_syn,
								tqu_syn_final[2]+gplane_u_syn*map_high_l_syn[2]/scale_syn]
	
	tqu_dust_final	= hp.ud_grade(tqu_dust,Final_nside,order_in='nest',order_out='nest')
	gplane_q_dust 	= tqu_dust_final[1]/np.mean(tqu_dust_final[1])
	gplane_u_dust 	= tqu_dust_final[2]/np.mean(tqu_dust_final[2])
	map_all_l_dust 	= [tqu_dust_final[0] + map_high_l_dust[0]/scale_dust,
						tqu_dust_final[1]+gplane_q_dust*map_high_l_dust[1]/scale_dust,
							tqu_dust_final[2]+gplane_u_dust*map_high_l_dust[2]/scale_dust]
	
	tqu_ame_final	= hp.ud_grade(tqu_ame,Final_nside,order_in='nest',order_out='nest')
	gplane_q_ame 	= tqu_ame_final[1]/np.mean(tqu_ame_final[1])
	gplane_u_ame 	= tqu_ame_final[2]/np.mean(tqu_ame_final[2])
	map_all_l_ame 	= [tqu_ame_final[0] + map_high_l_ame[0]/scale_ame,
						tqu_ame_final[1]+gplane_q_ame*map_high_l_ame[1]/scale_ame,
							tqu_ame_final[2]+gplane_u_ame*map_high_l_ame[2]/scale_ame]

	if plot_high_ell:
		new_cls_dust	= [new_cls_dust_ee,new_cls_dust_bb]
		new_cls_syn		= [new_cls_syn_ee,new_cls_syn_bb]
		new_cls_ame		= [new_cls_ame_ee,new_cls_ame_bb]
		plot_high_ell_6(map_all_l_dust,map_all_l_syn,map_all_l_ame,new_cls_dust,new_cls_syn,new_cls_ame,cls_dust,cls_syn,cls_ame)
	if save_templates:
		hp.write_map('%s_template_syn_with_high_ell.fits'%(label),map_all_l_syn,nest=True)
		hp.write_map('%s_template_dust_with_high_ell.fits'%(label),map_all_l_dust,nest=True)
		hp.write_map('%s_template_ame_with_high_ell.fits'%(label),map_all_l_ame,nest=True)
	return [map_all_l_syn,map_all_l_dust,map_all_l_ame]

def create_foregrounds_dust(templ_dust,dust_law,list_freq,list_beams,bands_weights,dict_beta_dust,dict_T_dust,dict_E_dust,ref_freq_dust,Final_nside,N_greybodies,data_folder):
	list=[]
	if 'beta_dust_1' in dict_beta_dust:
		# this means that the spectral index is constant across the sky
		for freq in list_freq:
			if bands_weights:
				band 	= np.loadtxt('band_%i.dat'%int(freq))
				NN 		= len(band[:,0])
				sum_up_dust = 0.0 ; sum_w = 0.0
				for j in range(NN):
					f_band = band[j,0]
					w_band = band[j,1]
					for g in range(N_greybodies):
						beta_dust		= float(dict_beta_dust['beta_dust_%i'%(g+1)])
						T_dust			= float(dict_T_dust['T_dust_%i'%(g+1)])
						E_dust			= float(dict_E_dust['E_dust_%i'%(g+1)])	 
						sum_up_dust     += w_band * dust_law(f_band,beta_dust,T_dust,E_dust) / dust_law(ref_freq_dust,beta_dust,T_dust,E_dust)
						sum_w           += w_band
				map_dust_pol_nu     = [templ_dust[0]*sum_up_dust/sum_w,	
										templ_dust[1]*sum_up_dust/sum_w,	
											templ_dust[2]*sum_up_dust/sum_w]
			else:
				factor = 0.0
				for g in range(N_greybodies):
					beta_dust		= float(dict_beta_dust['beta_dust_%i'%(g+1)])
					T_dust			= float(dict_T_dust['T_dust_%i'%(g+1)])
					E_dust			= float(dict_E_dust['E_dust_%i'%(g+1)])
					factor 			= factor + dust_law(freq,beta_dust,T_dust,E_dust)/dust_law(ref_freq_dust,beta_dust,T_dust,E_dust)
				map_dust_pol_nu     = [ templ_dust[0]*factor,templ_dust[1]*factor,templ_dust[2]*factor ]
			list.append(map_dust_pol_nu)
	else:
		# This means that beta_dust is a string with the name of the fits file that contains a map of beta_dust
		# First, we need to move the upgraded maps to a list .
		list_beta_dust_maps	=	[]
		N_pixels		= 12*Final_nside**2
		for g in range(N_greybodies):
			if dict_beta_dust['beta_dust_name_%i'%(g+1)] in default_files:
				beta_dust_map_before	= hp.read_map(data_folder+'/'+dict_beta_dust['beta_dust_name_%i'%(g+1)],nest=True,field=0,verbose=False)
			else:
				beta_dust_map_before	= hp.read_map(dict_beta_dust['beta_dust_name_%i'%(g+1)],nest=True,field=0,verbose=False)
			beta_dust_map			= hp.ud_grade(beta_dust_map_before,Final_nside,order_in='nested',order_out='nested')
			list_beta_dust_maps.append(beta_dust_map)
		# Then, we have to go pixel by pixel 
		for freq in list_freq:
			if bands_weights:
				band 	= np.loadtxt('band_%i.dat'%int(freq))
				NN 		= len(band[:,0])
				# in this case you must go pixel by pixel
				map_dust_pol_nu = [np.zeros(N_pixels),np.zeros(N_pixels),np.zeros(N_pixels)]
				for n in range(N_pixels):
					sum_up_dust = 0.0 ; sum_w = 0.0
					for j in range(NN):
						f_band = band[j,0]
						w_band = band[j,1]
						for g in range(N_greybodies):	
							beta_dust				= list_beta_dust_maps[g][n]
							T_dust					= float(dict_T_dust['T_dust_%i'%(g+1)])
							E_dust					= float(dict_E_dust['E_dust_%i'%(g+1)])
							sum_up_dust     		+= w_band * dust_law(f_band,beta_dust,T_dust,E_dust)/dust_law(ref_freq_dust,beta_dust,T_dust,E_dust)
							sum_w           		+= w_band
					for m in range(3):
						map_dust_pol_nu[m][n]	= templ_dust[m][n]*sum_up_dust/sum_w
			else:
				factor	= np.zeros(N_pixels)
				for g in range(N_greybodies):
					beta_dust_map	= list_beta_dust_maps[g]
					T_dust			= float(dict_T_dust['T_dust_%i'%(g+1)])
					E_dust			= float(dict_E_dust['E_dust_%i'%(g+1)])
					factor 			= factor + dust_law(freq,beta_dust_map,T_dust,E_dust)/dust_law(ref_freq_dust,beta_dust_map,T_dust,E_dust)
				map_dust_pol_nu     = [ templ_dust[0]*factor , templ_dust[1]*factor , templ_dust[2]*factor ]
			list.append(map_dust_pol_nu)
	return list

def create_foregrounds_syn(templ_syn,syn_law,list_freq,list_beams,bands_weights,beta_syn,ref_freq_syn,nu_pivot,Final_nside,delta_syn,data_folder):
	list=[]
	if isinstance(beta_syn,(int,long,float,complex)):
		# this means that the spectral index is constant across the sky
		for freq in list_freq:
			if bands_weights:
				band 	= np.loadtxt('band_%i.dat'%int(freq))
				NN 		= len(band[:,0])
				sum_up_syn=0.0 ; sum_w = 0.0
				for j in range(NN):
					f_band = band[j,0]
					w_band = band[j,1]
					sum_up_syn     += w_band * syn_law(f_band,beta_syn,delta_syn,ref_freq_syn,nu_pivot)
					sum_w          += w_band
				map_syn_pol_nu     = [templ_syn[0]*sum_up_syn/sum_w,	templ_syn[1]*sum_up_syn/sum_w,	templ_syn[2]*sum_up_syn/sum_w]
			else:
				map_syn_pol_nu     = [templ_syn[0]*syn_law(freq,beta_syn,delta_syn,ref_freq_syn,nu_pivot),	
										templ_syn[1]*syn_law(freq,beta_syn,delta_syn,ref_freq_syn,nu_pivot),		
											templ_syn[2]*syn_law(freq,beta_syn,delta_syn,ref_freq_syn,nu_pivot)
										]
			list.append(map_syn_pol_nu)
	elif isinstance(beta_syn,(str)):
		# This means that beta_syn is a string with the name of the fits file that contains a map of beta_syn
		beta_syn_map_before		= hp.read_map(data_folder+'/'+beta_syn,nest=True,field=0,verbose=False)
		beta_syn_map			= hp.ud_grade(beta_syn_map_before,Final_nside,order_in='nested',order_out='nested')
		for freq in list_freq:
			if bands_weights:
				band 	= np.loadtxt('band_%i.dat'%int(freq))
				NN 		= len(band[:,0])
				# in this case you must go pixel by pixel
				N_pixels		= 12*Final_nside**2
				map_syn_pol_nu 	= [np.zeros(N_pixels),np.zeros(N_pixels),np.zeros(N_pixels)]
				for n in range(N_pixels):
					sum_up_syn = 0.0 ; sum_w = 0.0
					for j in range(NN):
						f_band = band[j,0]
						w_band = band[j,1]
						sum_up_syn     += w_band * syn_law(f_band,beta_syn_map[n],delta_syn,ref_freq_syn,nu_pivot)
						sum_w          += w_band
					for m in range(3):
						map_syn_pol_nu[m][n]	= templ_syn[m][n]*sum_up_syn/sum_w
			else:
				map_syn_pol_nu     = [templ_syn[0]*syn_law(freq,beta_syn_map,delta_syn,ref_freq_syn,nu_pivot),	
										templ_syn[1]*syn_law(freq,beta_syn_map,delta_syn,ref_freq_syn,nu_pivot),		
											templ_syn[2]*syn_law(freq,beta_syn_map,delta_syn,ref_freq_syn,nu_pivot)
									]
			list.append(map_syn_pol_nu)
	return list

def create_foregrounds_ame(templ_ame,ame_law,list_freq,list_beams,bands_weights,m60,nu_max,ref_freq_ame,Final_nside,data_folder):
	list=[]
	if isinstance(nu_max,(int,long,float,complex)):
		# This means we were given a constant nu_max.
		for freq in list_freq:
			if bands_weights:
				band 		= np.loadtxt('band_%i.dat'%int(freq))
				NN 			= len(band[:,0])
				sum_up_ame	=0.0 ; sum_w = 0.0
				for j in range(NN):
					f_band = band[j,0]
					w_band = band[j,1]
					sum_up_ame		+= w_band * ame_law(f_band,m60,nu_max)/ame_law(ref_freq_ame,m60,nu_max)
					sum_w          	+= w_band
				map_ame_pol_nu  	= [templ_ame[0]*sum_up_ame/sum_w,	templ_ame[1]*sum_up_ame/sum_w,	templ_ame[2]*sum_up_ame/sum_w]
			else:
				factor			= ame_law(freq,m60,nu_max)/ame_law(ref_freq_ame,m60,nu_max)
				map_ame_pol_nu	= [templ_ame[0]*factor,	templ_ame[1]*factor,	templ_ame[2]*factor]
			list.append(map_ame_pol_nu)
	elif isinstance(nu_max,(str)):
		if nu_max in default_files:
			nu_max_map_before	= hp.read_map(data_folder+'/'+nu_max,nest=True,field=0,verbose=False)
		else:
			nu_max_map_before	= hp.read_map(nu_max,nest=True,field=0,verbose=False)
		nu_max_map			= hp.ud_grade(nu_max_map_before,Final_nside,order_in='nested',order_out='nested')
		for freq in list_freq:
			if bands_weights:
				band 	= np.loadtxt('band_%i.dat'%int(freq))
				NN 		= len(band[:,0])
				# in this case you must go pixel by pixel
				N_pixels		= 12*Final_nside**2
				map_syn_ame_nu 	= [np.zeros(N_pixels),np.zeros(N_pixels),np.zeros(N_pixels)]
				for n in range(N_pixels):
					sum_up_ame = 0.0 ; sum_w = 0.0
					for j in range(NN):
						f_band = band[j,0]
						w_band = band[j,1]
						sum_up_ame 	+=  w_band * ame_law(f_band,m60,nu_max_map[n])/ame_law(ref_freq_ame,m60,nu_max_map[n])
						sum_w 		+=  w_band
					for m in range(3):
						map_ame_pol_nu[m][n]	= templ_syn[m][n]*sum_up_ame/sum_w
			else:
				factor			= ame_law(freq,m60,nu_max_map)/ame_law(ref_freq_ame,m60,nu_max_map)
				map_ame_pol_nu	= [templ_ame[0]*factor,	templ_ame[1]*factor,	templ_ame[2]*factor]
			list.append(map_ame_pol_nu)
	return list

def create_cmb(cl_th_name,list_freq,seed_cmb,Final_nside,pixwin,fwhm_small,bands_weights,data_folder):
	if seed_cmb!=-1:
		np.random.seed(seed_cmb)
	else:
		random_seed = int(time.time())
		np.random.seed(random_seed)
	if cl_th_name in default_files:
		cl_th 			= hp.read_cl(data_folder+'/'+cl_th_name)
	else:
		cl_th 			= hp.read_cl(cl_th_name)
	cmb_map_ring 	= hp.synfast(cl_th,nside=Final_nside,pol=True,new=True,pixwin=pixwin,fwhm=np.radians(fwhm_small),verbose=False)
	cmb_map_nest	= hp.reorder(cmb_map_ring,r2n=True)
	list = []
	for freq in list_freq:
		if bands_weights:
			band 	= np.loadtxt('band_%i.dat'%int(freq))
			NN 		= len(band[:,0])
			sum_up = 0.0 ; sum_w = 0.0
			for j in range(NN):
				f_band = band[j,0]
				w_band = band[j,1]
				sum_up	     	+= w_band * thermo2antenna(f_band)
				sum_w           += w_band
			cmb_nest_nu			= [cmb_map_nest[0]*sum_up/sum_w,		cmb_map_nest[1]*sum_up/sum_w,			cmb_map_nest[2]*sum_up/sum_w]
		else:
			cmb_nest_nu			= [cmb_map_nest[0]*thermo2antenna(freq),	cmb_map_nest[1]*thermo2antenna(freq),	cmb_map_nest[2]*thermo2antenna(freq)]
		list.append(cmb_nest_nu)
	return list

def create_noises(list_freq,noises,Final_nside,data_folder):
	# The variable "noises" will be either a list of constant noises or a list of filenames with the covariance maps for each band
	list = []
	for j,freq in enumerate(list_freq):
		if isinstance(noises[0],(str)):
			# This means that we were given a list of maps of variances
			noise_map 			= hp.read_map(noises[j],nest=True,field=0)
			N_side_noise_map 	= hp.get_nside(noise_map)
			N_pix_final			= 12*Final_nside**2
			random_seed			= int(time.time())
			np.random.seed(random_seed)
			noise_map_final		= [np.zeros(N_pix_final),	np.random.normal(loc=0.0,scale=1.0,size=N_pix_final),	np.random.normal(loc=0.0,scale=1.0,size=N_pix_final)]
			noise_map_grade		= hp.ud_grade(noise_map,Final_nside)
			for m in range(3):
				noise_map_final[m] = noise_map_final[m]*noise_map_grade[m]*(Final_nside/N_side_noise_map)
			list.append(noise_map_final)
		elif isinstance(noises[0], (int, long, float, complex)):
			# This means we were given a list with constant value errors, which must be in (T unit antenna) deg
			N_pix_final			= 12*Final_nside**2
			random_seed			= int(time.time())
			np.random.seed(random_seed)
			noise_map_final		= [np.zeros(N_pix_final),	np.random.normal(loc=0.0,scale=1.0,size=N_pix_final),	np.random.normal(loc=0.0,scale=1.0,size=N_pix_final)]
			pixel_size_final	= hp.nside2resol(Final_nside,arcmin=False)
			for m in range(3):
				noise_map_final[m] = noise_map_final[m]*noises[j]*(1.0/np.degrees(pixel_size_final))
			list.append(noise_map_final)
	return list

def join_and_smooth(cmb_list,noise_list,syn_list,dust_list,ame_list,include_cmb,include_noise,include_syn,include_dust,include_ame,list_freq,list_beams,Final_nside,fwhm_small):
	print 'Join components and smooth'
	full_model_list = []
	# Create an empty numpy array to contain the sum of everything
	N_pix_final		= 12*Final_nside**2
	for j,freq in enumerate(list_freq):
		full_model		= [np.zeros(N_pix_final),	np.zeros(N_pix_final),	np.zeros(N_pix_final)]
		for m in range(3):
			if include_cmb:
				full_model[m] += cmb_list[j][m]
			if include_dust:
				full_model[m] += dust_list[j][m]
			if include_syn:
				full_model[m] += syn_list[j][m]
			if include_ame:
				full_model[m] += ame_list[j][m]
		full_model_list.append(full_model)
	# Now we have an added full model, we smooth to the resolution of each band
	full_model_smoothed_list = []
	for j,freq in enumerate(list_freq):
		print 'Smoothing freq ',freq,' GHz'
		if list_beams[j]/60.0 > fwhm_small:
			fwhm 					= np.sqrt((list_beams[j]/60.0)**2 - fwhm_small**2)
		else:
			fwhm 					= 0.0
		full_model_ring				= hp.reorder(full_model_list[j],n2r=True)
		full_model_smoothed_ring 	= hp.smoothing(full_model_ring,fwhm=np.radians(fwhm),pol=True,verbose=False)
		full_model_smoothed_nest	= hp.reorder(full_model_smoothed_ring,r2n=True)
		full_model_smoothed_list.append(full_model_smoothed_nest)
	# Now that we smoothed, we add the noise on top
	if include_noise:
		for j,freq in enumerate(list_freq):
			for m in range(3):
				full_model_smoothed_list[j][m] += noise_list[j][m]
	return full_model_smoothed_list

def write_function(write_syn,write_dust,write_noise,write_cmb,write_ame,full_model_smoothed_list,syn_list,dust_list,noise_list,cmb_list,ame_list,list_freq,label,list_beams,output_folder,fwhm_small):
	for j,freq in enumerate(list_freq):
		print 'Writing freq ',freq,' GHz'
		hp.write_map('%s/%s_%iGHz.fits'%(output_folder,label,int(freq)),full_model_smoothed_list[j],nest=True)
		if list_beams[j]/60.0 > fwhm_small:
			fwhm 	= np.sqrt((list_beams[j]/60.0)**2 - fwhm_small**2)
		else:
			fwhm 	= 0.0
		# Write each of the component is they are asked for
		if write_noise:
			hp.write_map('%s/%s_Noise_%iGHz.fits'%(output_folder,label,int(freq)),noise_list[j],nest=True)
		if write_dust:
			print 'Smoothing Dust map at freq ',freq,' GHz'
			dust_ring			= hp.reorder(dust_list[j],n2r=True)
			dust_ring_smooth	= hp.smoothing(dust_ring,fwhm=np.radians(fwhm),pol=True,verbose=False)
			dust_nest_smooth	= hp.reorder(dust_ring_smooth,r2n=True)
			hp.write_map('%s/%s_Dustmodel_%iGHz.fits'%(output_folder,label,int(freq)),dust_nest_smooth,nest=True)
		if write_syn:
			print 'Smoothing Syn map at ',freq,' GHz'
			syn_ring			= hp.reorder(syn_list[j],n2r=True)
			syn_ring_smooth		= hp.smoothing(syn_ring,fwhm=np.radians(fwhm),pol=True,verbose=False)
			syn_nest_smooth		= hp.reorder(syn_ring_smooth,r2n=True)
			hp.write_map('%s/%s_Synchrotronmodel_%iGHz.fits'%(output_folder,label,int(freq)),syn_nest_smooth,nest=True)
		if write_cmb:
			print 'Smoothing CMB map at ',freq,' GHz'
			cmb_ring			= hp.reorder(cmb_list[j],n2r=True)
			cmb_ring_smooth		= hp.smoothing(cmb_ring,fwhm=np.radians(fwhm),pol=True,verbose=False)
			cmb_nest_smooth		= hp.reorder(cmb_ring_smooth,r2n=True)
			hp.write_map('%s/%s_CMB_%iGHz.fits'%(output_folder,label,int(freq)),cmb_nest_smooth,nest=True)
		if write_ame:
			print 'Smoothing AME amp at ',freq,' GHz'
			ame_ring			= hp.reorder(ame_list[j],n2r=True)
			ame_ring_smooth		= hp.smoothing(ame_ring,fwhm=np.radians(fwhm),pol=True,verbose=False)
			ame_nest_smooth		= hp.reorder(ame_ring_smooth,r2n=True)
			hp.write_map('%s/%s_AME_%iGHz.fits'%(output_folder,label,int(freq)),ame_nest_smooth,nest=True)
	return 0
