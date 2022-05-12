import numpy as np
import scipy.stats as st
import pandas as pd
import math
import os
import pyfits as fits
import gc

def ReadSdssDr12FitsFile(inFilename):
	"""
	Reads a Sloan Digital Sky Survey FITS file.

	IN:
	inFilename: the filename of the FITS file

	OUT:
	objectID: the ID of the astronomical object
	redShift: the redshift of the astronomical object
	dataframe: an Pandas dataframe containing the wavelength and spectral flux density series.
	"""

	# See: http://docs.astropy.org/en/stable/io/fits/

	# Open the FITS file read-only and store a list of Header Data Units
	HduList = fits.open(inFilename)

	# Print the HDU info
#	HduList.info()

	# Get the first header and print the keys
	priHeader = HduList[1].header #HduList['PRIMARY'].header
#	print(repr(priHeader))

	# Get spectrum data is containted in the first extension
	spectrumData = HduList[1].data #HduList['COADD'].data
	spectrumColumns = HduList[1].columns.names #HduList['COADD'].columns.names
#	print(spectrumColumns)
	dataframe = pd.DataFrame(spectrumData, columns=spectrumColumns)

	# Get the number of records
	numRecords = len(dataframe.index)

	# Get range of wavelengths contained in the second extension
	spectrumProperties = HduList[2].data #HduList['SPECOBJ'].data
#	print(spectrumProperties.columns.names)
	survey        = spectrumProperties.field(0)[0]
	minWavelength = spectrumProperties.field('WAVEMIN') # in Angstroms
	maxWavelength = spectrumProperties.field('WAVEMAX') # in Angstroms
	covWavelength = spectrumProperties.field('wCoverage') # Coverage in wavelength, in units of log10 wavelength; unsure what this is
	redShift      = spectrumProperties.field('z') # Final redshift

	objectID = ['-1']
	if(survey.upper() == 'SDSS'):
		objectID = spectrumProperties.field('BESTOBJID') # Object ID
	elif(survey.upper() == 'BOSS'):
		objectID = spectrumProperties.field('OBJID') # Object ID
	elif(survey.upper() == 'SEQUELS'):
		objectID = spectrumProperties.field('OBJID') # Object ID
	else:
		print("ERROR in ReadSdssDr12FitsFile: Unknow survey type: " + survey + "...")

	# Add the wavelengths to the dataframe
	wavelengthArray = np.logspace(start=np.log10(minWavelength), stop=np.log10(maxWavelength), num = numRecords, endpoint=True)
	dataframe['wavelength'] = wavelengthArray

	# Close the FITS file
	HduList.close()

	return [objectID, redShift, dataframe]

def ConvertSdssDr12FitsToCsv(inFilename, outFilename):
	# Load data from the FITS file
	objId, z, data = ReadSdssDr12FitsFile(inFilename)
	data.to_csv(outFilename)
	print("Writing " + outFilename)

def ConvertAllSdssDr12FitsToCsv(foldername):
	for inFilename in os.listdir(foldername):
		if inFilename.endswith('.fits'):
			inFilename = foldername + inFilename
			outFilename = inFilename[0:(len(inFilename)-len('.fits'))] + '.csv'
			ConvertSdssDr12FitsToCsv(inFilename, outFilename)

def ToEmittedFrame(z, measuredFrame):
	"""
	Converts the wavelengthArray to its emitted spectrum

	IN:
	z: redshift
	measuredFrame: the measured wavelengths

	OUT:
	emittedFrame: the wavelengths in the emitted frame
	"""

	emittedFrame = measuredFrame / (1+z)
	return emittedFrame

def ConsiderAndMask(fluxArray, andMaskArray):
	"""
	Zeros all fluxes which have an and_mask > 0

	IN:
	fluxArray: array of fluxes
	andMaskArray: array of and_masks

	OUT:
	fluxArray: array of modified fluxes
	"""

	if (len(fluxArray) != len(andMaskArray)):
		print("Error in ConsiderAndMask: arrays not compatible.")
	else:
		fluxArray = fluxArray * (andMaskArray==0)

	return fluxArray

def ReBinData(spectrumList, binSize, startWavelength, stopWavelength, method = "AM"):
	"""
	Re-bins data.

	IN:
	spectrumList: list with spectra.
	binSize: new size of bins
	startWavelength: new start wavelength
	endWavelength: new end wavelenght
	method: method for combining spectra

	OUT:
	compositeDf: dataframe containing the composite spectrum
	"""

	newWavelengthArray = np.arange(start=startWavelength, stop=stopWavelength, step = binSize)
	arrayLength = len(newWavelengthArray)

	newFluxArray = np.zeros(shape=arrayLength)
	newUncFluxArray = np.zeros(shape=arrayLength)
	newNoiseFluxArray =  np.zeros(shape=arrayLength)
	newNumDataArray =  np.zeros(shape=arrayLength)

	# Define indices on the destination arrays
	oldProgress = 0.
	for index in range(arrayLength):
		# Print progress
		newProgress = 100*float(index)/float(arrayLength)
		if (newProgress - oldProgress > 1):
			print('   Rebinning progress: {0:.0F}%...'.format(newProgress))
			oldProgress = newProgress

		# Calculate bin boundaries
		lowerWavelength = newWavelengthArray[index] # - binSize/2.
		upperWavelength = newWavelengthArray[index] + binSize # + binSize/2.

		# Fill temporary arrays
		tempFluxArray = np.array([])
		tempIvarArray = np.array([])
		for spectrumDf in spectrumList:
			condition1 = (spectrumDf['wavelength'] >= lowerWavelength)
			condition2 = (spectrumDf['wavelength'] <  upperWavelength)
			tempFluxArray = np.append(tempFluxArray, spectrumDf['flux'][condition1 & condition2])
			tempIvarArray = np.append(tempIvarArray, spectrumDf['ivar'][condition1 & condition2])

		# Calculate mean flux, uncertainty and noise
		mean_f_lambda  = 0.
		unc_f_lambda   = 0.
		noise_f_lambda = 0.
		n_data = len(tempFluxArray)

		if(n_data == 0):
			print("ERROR in ReBinData: No data in bin {0:.0F} - {1:.0F}".format(lowerWavelength, upperWavelength))

		elif(n_data == 1):
			print("WARNING in ReBinData: One data point only in bin {0:.0F} - {1:.0F}".format(lowerWavelength, upperWavelength))

			mean_f_lambda  = tempFluxArray[0]
			noise_f_lambda = 1./tempIvarArray[0]
			unc_f_lambda   = 0.

		else:
			# Calculate the uncertainty of the mean value
			unc_f_lambda = np.std(a=tempFluxArray, ddof=1)/math.sqrt(float(n_data))

			if(method.upper() == 'GM'):
				# Set all values <0 to 1
				tempFluxArray[tempFluxArray<=0] = 1.

				# Calculate the geometric mean flux
				mean_f_lambda = st.gmean(tempFluxArray)

				# Calculate the the noise of the mean value
				noise_f_lambda = mean_f_lambda/float(n_data) * math.sqrt(np.sum((1./tempIvarArray)/mean_f_lambda))

			else:
				# Calculate the arithmetric mean flux
				mean_f_lambda = np.mean(tempFluxArray)

				# Calculate the noise of the mean value
				noise_f_lambda = 1./float(n_data) * math.sqrt(np.sum(1./tempIvarArray))

		# Store calculated values
		newFluxArray[index] = mean_f_lambda
		newUncFluxArray[index] = unc_f_lambda
		newNoiseFluxArray[index] = noise_f_lambda
		newNumDataArray[index] = n_data

	# Return the composite spectrum in a dataframe
	compositeDf = pd.DataFrame({'wavelength':newWavelengthArray, 'mean_f_lambda':newFluxArray, 'noise_f_lambda':newNoiseFluxArray, 'unc_f_lambda':newUncFluxArray, 'n_data':newNumDataArray})
	return compositeDf

def NormaliseSpectra(redshiftList, objectIdList, spectrumList):
	# Create a dataframe to be sorted
	spectraDf = pd.DataFrame({'z':redshiftList, 'objID':objectIdList, 'spectrum':spectrumList})

	# Sort spectra by redshift
	spectraDf.sort_values(by='z', inplace=True)
	spectraDf.index = range(len(spectraDf.index)) # Reset the index (was also sorted)

	# Find the normalisation factors
	normalisationList = []
	oldProgress = 0.
	for index in range(len(spectraDf.index)-1):
		# Print progress
		newProgress = 100*float(index)/float(len(spectraDf.index)-1)
		if (newProgress - oldProgress > 1):
			print('   Normalising progress: {0:.0F}%...'.format(newProgress))
			oldProgress = newProgress

#		# Display the redshift
#		z = spectraDf['z'][index]
#		print("   Processing redshift: z = {0:.3G}".format(z))

		# Get current and next rebinned spectra
		spectrumDf1 = spectraDf['spectrum'][index]
		spectrumDf2 = spectraDf['spectrum'][index+1]

		# Store fluxes in lists
		fluxList1 = []
		fluxList2 = []

		for flux1, flux2 in zip(spectrumDf1['flux'], spectrumDf2['flux']):
			if((flux1 != 0) and (flux2 != 0)):
				fluxList1.append(flux1)
				fluxList2.append(flux2)

		if((len(fluxList1) == 0) or (len(fluxList2) == 0)):
			print("ERROR in NormaliseSpectra: No overlapping region found.")
			print("   SpecObjectID1 = {0}; SpecObjectID2 = {1}".format(objectIdList[pointer1], objectIdList[pointer2]))

			normalisationFactor = 1.
		else:
			# Calculate the normalisation factor for spectrum 1
			normalisationFactor = np.mean(fluxList2)/np.mean(fluxList1)

		# Multiply all normalisation factors before current with the current normalisation factor
		normalisationList = [x*normalisationFactor for x in normalisationList]

		# Add the current normalisation factor
		normalisationList.append(normalisationFactor)

	# Add the normalisation factor for the last spectrum, which is 1, since all previous spectra are normalised to the last spectrum
	normalisationList.append(1.)

	# Normalise the spectra
	for index in range(len(normalisationList)):
		normalisationFactor = normalisationList[index]

		spectraDf['spectrum'][index]['flux'] = spectraDf['spectrum'][index]['flux']*normalisationFactor
		spectraDf['spectrum'][index]['ivar'] = spectraDf['spectrum'][index]['ivar']*normalisationFactor

	return spectraDf

def CompositeSpectrum(filenameList = [], wavelengthRange = (), binSize = 4., method = "AM"):
	"""
	Creates a composite spectrum from multiple spectra.

	IN:

	OUT:
	"""

	# Open each fits file and retrieve and store redshift
	objectIdList = []
	redshiftList = []
	spectrumList = []

	minWavelengthList = []
	maxWavelengthList = []

	#
	# Load data
	#
	print("Reading {0} FITS files...".format(len(filenameList)))
	for filename in filenameList:
		print("Processing: " + filename +"...")

		# Open file
		HduList = fits.open(filename)

		# Get spectrum data is containted in the first extension
		spectrumData = HduList[1].data.copy() #HduList['COADD'].data
		spectrumColumns = HduList[1].columns.names.copy() #HduList['COADD'].columns.names
		spectrumDf = pd.DataFrame(spectrumData, columns=spectrumColumns)

		# Get the number of records
		numRecords = len(spectrumDf.index)

		# Get range of wavelengths contained in the second extension
		spectrumProperties = HduList[2].data.copy() #HduList['SPECOBJ'].data

		try:
			minWavelength = float(spectrumProperties.field('WAVEMIN')) # in Angstrom
			maxWavelength = float(spectrumProperties.field('WAVEMAX')) # in Angstrom
			covWavelength = float(spectrumProperties.field('wCoverage')) # Coverage in wavelength, in units of log10 wavelength; unsure what this is
			redShift      = float(spectrumProperties.field('z')) # Final redshift
			objectId      = str(spectrumProperties.field('SPECOBJID')) # Object ID
		except KeyError:
			print("ERROR in CompositeSpectrum: Could not retrieve spectrum property...")
			print(HduList[2].columns.names)
			raise KeyError

		# Explicitly delete references to FITS data to free up file handles and prevent 'too many open files'
		# See: http://docs.astropy.org/en/stable/io/fits/appendix/faq.html#i-m-opening-many-fits-files-in-a-loop-and-getting-oserror-too-many-open-files
		del HduList[1].data
		del HduList[2].data

		# Close fits file
		HduList.close()

		# Enforce garbage collection
		gc.collect()

		# Create a wavelength array and add it to the dataframe
		wavelengthArray = np.logspace(start=np.log10(minWavelength), stop=np.log10(maxWavelength), num = numRecords, endpoint=True)
		spectrumDf['wavelength'] = wavelengthArray

		# Transform min / max wavelengths to emitted frame
		minWavelength = minWavelength/(1+redShift)
		maxWavelength = maxWavelength/(1+redShift)

		# Store all in lists
		minWavelengthList.append(minWavelength)
		maxWavelengthList.append(maxWavelength)

		objectIdList.append(objectId)
		redshiftList.append(redShift)
		spectrumList.append(spectrumDf)

	# Determine minimum and maximum wavelengths to use for rebinning
	if (len(wavelengthRange)<2):
		# Create a proper start and end wavelength based on the info in the FITS files
		minWavelength = min(minWavelengthList)
		maxWavelength = max(maxWavelengthList)

#		minWavelength = int(minWavelength/500)*500.0
#		maxWavelength = (int(maxWavelength/500)+1)*500.0
	else:
		# Use given range
		minWavelength = min(wavelenghRange)
		maxWavelength = max(wavelenghRange)

	print("Wavelength range: [{0:.3G}, {1:.3G}]".format(minWavelength, maxWavelength))

	#
	# Correct for redshift
	#
	print("Correcting for redshift...")
	for z, spectrumDf in zip(redshiftList, spectrumList):
		# Correct for redshift
		spectrumDf['wavelength'] = ToEmittedFrame(z, spectrumDf['wavelength'])

	#
	# Remove bad data
	#
	print("Removing bad data...")
	for spectrumDf in spectrumList:
		# Zero all suspicious data
		spectrumDf['flux'] = ConsiderAndMask(spectrumDf['flux'], spectrumDf['and_mask'])

	#
	# Normalise the spectra
	#
	print("Calculating normalisation factors...")
	normalisedDf = NormaliseSpectra(redshiftList, objectIdList, spectrumList)

	# Free memory
	del spectrumList

	#
	# Combine the individual spectra to a single composite spectrum
	#
	print("Rebinning data and creating composite...")
	compositeDf = ReBinData(normalisedDf['spectrum'], binSize, minWavelength, maxWavelength, method)

	# Return the composite spectrum
	return compositeDf


def CalculateRequiredRedshifts(observedWavelengthRange = (), emittedWavelengthRange = ()):
	"""
	Calculates a number of redshift for spectra to use in a composite spectrum.

	IN:
	observedWavelengthRange: the tuple with the minimum and maximum wavelength values that the instrument can measure (in observed frame)
	emittedWavelengthRange: the tuple with desired wavelength range in the emitted frame.

	OUT:
	redshiftList: A list containing all advised redshifts to use
	"""

	# emitted frame = observed frame / (1+z)

	redshiftList = []

	# First calculate at the minimum wavelength
	leftWavelength = min(emittedWavelengthRange)
	z = min(observedWavelengthRange) / leftWavelength - 1

	# Calculate the right boundary of the first spectrum
	rightWavelength = max(observedWavelengthRange) / (1+z)

	# Add redshift to lsit
	reshiftList.append(z)

	# Repeat while rightWavelength < maximum emittedWavelengthRange
	while rightWavelength < max(emittedWavelengthRange):
		# Calculate the left boundary of the spectrum
		leftWavelength = min(emittedWavelengthRange)
		z = observedWavelengthRange[1] / leftWavelength - 1

		# Calculate the right boundary of the spectrum
		rightWavelength = observedWavelengthRange[2] / (1+z)

		# Add redshift to lsit
		reshiftList.append(z)


	return redshiftList

def CreateSpecCombineParameterFile(binSize = 2., normalisationList = [], wavelengthRange = (), foldername = ".", parameterFilename = 'input.txt', csvFilename = 'output.csv'):
	"""
	Create a parameter file for the programme 'SpecCombine.exe'

	IN:
	binSize: the size of the bins in Angstrom. Default = 2A
	normalisationList: list of normalisation factors
	wavelengthRange: tuple of the minimum and maximum wavelength for the composite spectrum.
	foldername: name of the working folder. Default is current folder.
	parameterFilename: the name of the input file for SpecCombine. This file will be generated. Default 'input.txt'
	output.csv: the name of the CSV file that SpecCombine must generate. Default 'output.csv'

	OUT:
	-
	"""

	# Store the names of fits-files
	filenameList = []
	for filename in os.listdir(foldername):
		if filename.endswith(".fits"):
			filenameList.append(filename)

	# If the length of the normalisation list < length filename list, then make the normalisation factor 1. for the missing items
	if len(normalisationList) < len(filenameList):
		for index in range(len(normalisationList), len(filenameList)):
			normalisationList.append(1.)

	# Open each fits file and retrieve and store redshift
	minWavelengthList = []
	maxWavelengthList = []
	redshiftList = []

	for filename in filenameList:
		fullname = foldername +filename
		print("Processing: " + fullname +"...")
		HduList = fits.open(fullname)

		# Get range of wavelengths
		spectrumProperties = HduList[2].data # HduList['SPECOBJ'].data
		minWavelength = float(spectrumProperties.field('WAVEMIN')) # in Angstroms
		maxWavelength = float(spectrumProperties.field('WAVEMAX')) # in Angstroms
		redshift      = float(spectrumProperties.field('z')) # Final redshift

		# Transform wavelengths to emitted frame
		minWavelength = minWavelength/(1+redshift)
		maxWavelength = maxWavelength/(1+redshift)

		# Store all in lists
		minWavelengthList.append(minWavelength)
		maxWavelengthList.append(maxWavelength)
		redshiftList.append(redshift)

		# Close fits file
		HduList.close()

	if (len(wavelengthRange)<2):
		# Create a proper start and end wavelength based on the info in the FITS files
		minWavelength = min(minWavelengthList)
		maxWavelength = max(maxWavelengthList)

		minWavelength = int(minWavelength/500)*500.0
		maxWavelength = (int(maxWavelength/500)+1)*500.0
	else:
		# Use given range
		minWavelength = min(wavelenghRange)
		maxWavelength = max(wavelenghRange)

	print("Wavelength range: [{0:.3G}, {1:.3G}]".format(minWavelength, maxWavelength))

	# Store to SpecCombine input file
	filename = foldername + parameterFilename
	with open(filename, 'w') as f:
		# Write the number of files to use
		f.write(str(len(filenameList))+'\n')


		# Write the minimum wavelength, maximum wavelength, bin size
		f.write(str(minWavelength) + ',' + str(maxWavelength) + ',' + str(binSize) + '\n')

		# Write the SpecCombine output filename
		f.write(csvFilename + '\n')

		# Write for each spectrum the redshift, normalisation factor, fits filename
		for z, normalisationFactor, fitsFile in zip(redshiftList, normalisationList, filenameList):
			f.write(str(z) +',' + str(normalisationFactor) + ',' + fitsFile + '\n')

	print('Parameter file written to: ' + filename + '...')

def CreatePowerLaw(spectrumDf, breakpointList = []):
	"""
	Creates a continuum for spectrumDf.
	The continuum is calculated using a linear regression model.

	IN:
	spectrumDf: the dataframe containing the spectrum.
	breakPointList: the wavelength where one power-law transitions into another.

	OUT:
	y:              the numpy array containing the spectrum, including a column 'continuum'
	slopeList:      the list containing the slopes of the line elements in the log-log plane
	interceptList:  the list containing the intercepting points with the y-axis of the line elements in the loglog plane
	breakPointList: the list containing the final breakpoints between the line segments
	"""

	# Calculate the linear pieces
	slopeList = []
	interceptList = []
	stdErrList = []

	condition1 = (spectrumDf['mean_f_lambda'] > 0.)
	for index in range(len(breakpointList)):
		if(index==0):
			# The first piece...
			breakpoint = breakpointList[index]
			condition2 = (spectrumDf['wavelength'] <= breakpoint)

			indexList = spectrumDf[condition1 & condition2].index.tolist()
			x = np.log(spectrumDf['wavelength'].iloc[indexList])
			y = np.log(spectrumDf['mean_f_lambda'].iloc[indexList])
		else:
			# ... the middle piece...
			breakpoint1 = breakpointList[index-1]
			breakpoint2 = breakpointList[index]
			condition2 = (spectrumDf['wavelength'] >  breakpoint1)
			condition3 = (spectrumDf['wavelength'] <= breakpoint2)

			indexList = spectrumDf[condition1 & condition2 & condition3].index.tolist()
			x = np.log(spectrumDf['wavelength'].iloc[indexList])
			y = np.log(spectrumDf['mean_f_lambda'].iloc[indexList])

		slope, intercept, r_value, p_value, std_err = st.linregress(x, y)

		print("slope = {0:.3G}; intercept = {1:.3G}; std_err = {2:.3G}".format(slope, intercept, std_err))

		slopeList.append(slope)
		interceptList.append(intercept)
		stdErrList.append(std_err)

	# ...and the last piece
	breakpoint = breakpointList[-1]
	condition2 = (spectrumDf['wavelength']>breakpoint)

	indexList = spectrumDf[condition1 & condition2].index.tolist()
	x = np.log(spectrumDf['wavelength'].iloc[indexList])
	y = np.log(spectrumDf['mean_f_lambda'].iloc[indexList])

	slope, intercept, r_value, p_value, std_err = st.linregress(x, y)

	print("slope = {0:.3G}; intercept = {1:.3G}; std_err = {2:.3G}".format(slope, intercept, std_err))

	slopeList.append(slope)
	interceptList.append(intercept)
	stdErrList.append(std_err)

	# Create a new breakpoint list based on interceptions between line elements
	breakpointList = []
	for index in range(1, len(slopeList)):
		slope1 = slopeList[index-1]
		intercept1 = interceptList[index-1]

		slope2 = slopeList[index]
		intercept2 = interceptList[index]

		breakpoint = np.exp((intercept2-intercept1)/(slope1-slope2))
		breakpointList.append(breakpoint)

		print("new breakpoint at {0:.3G} A".format(breakpoint))

	# Create the continuum arrays
	x = np.zeros(len(spectrumDf.index))
	y = np.zeros(len(spectrumDf.index))

	# The first piece..
	index = 0

	breakpoint = breakpointList[index]
	slope = slopeList[index]
	intercept = interceptList[index]

	condition1 = (spectrumDf['mean_f_lambda'] > 0.)
	condition2 = (spectrumDf['wavelength'] <= breakpoint)

	indexList = spectrumDf[condition1 & condition2].index.tolist()

	x[indexList] = spectrumDf['wavelength'].iloc[indexList]
	y[indexList] = np.exp(intercept)*x[indexList]**slope

	# .. the middle pieces
	for index in range(1, len(breakpointList)):
		slope = slopeList[index]
		intercept = interceptList[index]

		breakpoint1 = breakpointList[index-1]
		breakpoint2 = breakpointList[index]

		condition2 = (spectrumDf['wavelength'] > breakpoint1)
		condition3 = (spectrumDf['wavelength'] <= breakpoint2)

		indexList = spectrumDf[condition1 & condition2 & condition3].index.tolist()

		x[indexList] = spectrumDf['wavelength'].iloc[indexList]
		y[indexList] = np.exp(intercept)*x[indexList]**slope

	# ...and the last piece
	index = -1

	slope = slopeList[index]
	intercept = interceptList[index]

	breakpoint = breakpointList[index]

	condition2 = (spectrumDf['wavelength']>breakpoint)

	indexList = spectrumDf[condition1 & condition2].index.tolist()

	x[indexList] = spectrumDf['wavelength'].iloc[indexList]
	y[indexList] = np.exp(intercept)*x[indexList]**slope

	return y, slopeList, interceptList, breakPointList

def CreateContinuum(spectrumDf, breakpointList = [], lineNameList = []):
	"""
	Approximates the continuum of a quasar spectrum using power laws.
	Will make a first estimate, then removes known emission lines, and makes the final estimate.

	IN:
	spectrumDf: the dataframe containing the spectrum.
	breakPointList: the wavelength  where one power-law transitions into another.

	OUT:
	y:              the numpy array containing the spectrum, including a column 'continuum'
	"""

	tempDf = spectrumDf.copy()

	# First estimate
	tempDf['continuum'], _, _, _ = CreatePowerLaw(tempDf, breakpointList)

	# Remove known spectrum lines
	LineDf = FindSpectrumLines(tempDf)
	for _, row in LineDf.iterrows():
		wavelengthLeft = row['WavelengthLeft']
		wavelengthRight = row['WavelengthRight']

		indexSpecLeft = tempDf[tempDf['wavelength'] == wavelengthLeft].index.tolist()[0]
		indexSpecRight = tempDf[tempDf['wavelength'] == wavelengthRight].index.tolist()[0]

		tempDf['mean_f_lambda'].iloc[indexSpecLeft:indexSpecRight] = tempDf['continuum'].iloc[indexSpecLeft:indexSpecRight]

	# Final estimate
	y, _, _, _ = CreatePowerLaw(tempDf, breakpointList)

	return y, tempDf

def FindSpectrumLines(spectrumDf):
	"""
	Finds spectrum lines in a given spectrum.

	IN:
	spectrumDf: the dataframe containing the spectrum.
	lineNameList: a list with laboratory determined wavelengths of spectrum lines.

	OUT:
	LineDf: a dataframe with all lines and the associated left and right wavelengths.
	"""

	# Load the table with known emission lines
	emissionLines = pd.read_csv('./hkSpectrumLines.csv', skiprows=5)

	# Create temporary storage
	LineNameList = []
	LineWavelengthCentre = []
	LineWavelengthLeft = []
	LineWavelengthRight = []

	# Loop over all spectrum lines
	for _, row in emissionLines.iterrows():
		lineName = row['Line']
		wavelengthLab = row['Wavelength / Å']

		# Search for the same wavelength in the actual spectrum
		indexSpec = 0 # Spectrum index
		while (spectrumDf['wavelength'].iloc[indexSpec] < wavelengthLab):
			indexSpec = indexSpec + 1

		# IndexSpec now points to the position in SpectrumDf that is associated with the laboratory spectrum line.
		# The actual spectrum line might however be shifted to the left or to the right.
		# Let's find the emission peak by checking neighbouring fluxes.

		# First check to the left
		while (spectrumDf['mean_f_lambda'].iloc[indexSpec-1] > spectrumDf['mean_f_lambda'].iloc[indexSpec]):
			indexSpec = indexSpec - 1

		# Now check to the right
		while (spectrumDf['mean_f_lambda'].iloc[indexSpec+1] > spectrumDf['mean_f_lambda'].iloc[indexSpec]):
			indexSpec = indexSpec + 1

		# indexSpec now points to the emission peak in the actual spectrum
		# Store the emissionline name and the actual wavelength in a list for later use
		LineNameList.append(lineName)
		LineWavelengthCentre.append(spectrumDf['wavelength'].iloc[indexSpec])

#		print("Emission line {0} found at {1:4.2f} Å".format(lineName, spectrumDf['wavelength'].iloc[indexSpec]))

		#
		# Find the left crossing point with the continuum
		#
		indexSpecLeft = indexSpec
		while (spectrumDf['mean_f_lambda'].iloc[indexSpecLeft] > spectrumDf['continuum'].iloc[indexSpecLeft]):
			indexSpecLeft = indexSpecLeft - 1

		wavelengthLeft = spectrumDf['wavelength'].iloc[indexSpecLeft]
		LineWavelengthLeft.append(wavelengthLeft)
#		print("1: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecLeft]))

		#
		# Find the right crossing point with the continuum
		#
		indexSpecRight = indexSpec
		while (spectrumDf['mean_f_lambda'].iloc[indexSpecRight] > spectrumDf['continuum'].iloc[indexSpecRight]):
			indexSpecRight = indexSpecRight + 1

		wavelengthRight = spectrumDf['wavelength'].iloc[indexSpecRight]
		LineWavelengthRight.append(wavelengthRight)
#		print("2: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecRight]))

#		print("Wavelength left: {0:4.2f} Å; wavelength right: {1:4.2f} Å".format(wavelengthLeft, wavelengthRight))

	# Now create the final dataframe
	LineDf = pd.DataFrame()
	LineDf['Line'] = LineNameList
	LineDf['WavelengthLeft'] = LineWavelengthLeft
	LineDf['WavelengthRight'] = LineWavelengthRight

	return LineDf

def CalculateFwhm1(spectrumDf, lineNameList = []):
	"""
	Determines the Full-Width at Half-Maximum of one or more spectrum lines.
	Achieves this by searching for troughs on both sides of an emission line.

	IN:
	spectrumDf: the dataframe defining the spectrum; should have columns 'wavelength', 'mean_f_lambda' and 'theoretical'
	lineNameList: a list with laboratory determined wavelengths of spectrum lines.

	OUT:
	FwhmDf: a dataframe with all lines and the associated FWHM values.
	"""

	# Load the table with known emission lines
	emissionLines = pd.read_csv('./hkSpectrumLines.csv', skiprows=5)

	# Create temporary storage
	FwhmLineNameList = []
	FwhmLineWavelengthList = []
	FwhmLineWidth = []

	# Loop over all spectrum lines
	for lineName in lineNameList:

		# We can have multiple lines with the same name, but at different wavelengths
		indexListLab = emissionLines['Line'][emissionLines['Line'] == lineName].index.tolist()

		# So loop over all found wavelengths for this spectrum line
		for indexLab in indexListLab:
			# Retrieve the laboratory wavelength; this is a good starting point to find an emission peak
			wavelengthLab = emissionLines['Wavelength / Å'].iloc[indexLab]

			# Search for the same wavelength in the actual spectrum
			indexSpec = 0 # Spectrum index
			while (spectrumDf['wavelength'].iloc[indexSpec] < wavelengthLab):
				indexSpec = indexSpec + 1

			# IndexSpec now points to the position in SpectrumDf that is associated with the laboratory spectrum line.
			# The actual spectrum line might however be shifted to the left or to the right.
			# Let's find the emission peak by checking neighbouring fluxes.

			# First check to the left
			while (spectrumDf['mean_f_lambda'].iloc[indexSpec-1] > spectrumDf['mean_f_lambda'].iloc[indexSpec]):
				indexSpec = indexSpec - 1

			# Now check to the right
			while (spectrumDf['mean_f_lambda'].iloc[indexSpec+1] > spectrumDf['mean_f_lambda'].iloc[indexSpec]):
				indexSpec = indexSpec + 1

			# indexSpec now points to the emission peak in the actual spectrum
			# Store the emissionline name and the actual wavelength in a list for later use
			FwhmLineNameList.append(lineName)
			FwhmLineWavelengthList.append(spectrumDf['wavelength'].iloc[indexSpec])

#			print("Emission line {0} found at {1:4.2f} Å".format(lineName, spectrumDf['wavelength'].iloc[indexSpec]))

			# Determine the half-maximum
			halfMaximum = spectrumDf['continuum'].iloc[indexSpec] + 0.5*(spectrumDf['mean_f_lambda'].iloc[indexSpec]-spectrumDf['continuum'].iloc[indexSpec])
#			print("Half-maximum = {0:.2f}".format(halfMaximum))

			#
			# Find a minimum flux value to the left or the crossing point with the half-maximum, whichever comes first
			#
			indexSpecLeft = indexSpec
			while (spectrumDf['mean_f_lambda'].iloc[indexSpecLeft-1] < spectrumDf['mean_f_lambda'].iloc[indexSpecLeft])\
			and (spectrumDf['mean_f_lambda'].iloc[indexSpecLeft] > halfMaximum):
				indexSpecLeft = indexSpecLeft - 1

#			print("1: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecLeft]))

			# If the flux is smaller than the half-maximum, then we have found the left wavelength, otherwise extrapolate
			if(spectrumDf['mean_f_lambda'].iloc[indexSpecLeft] > halfMaximum):
				A = (spectrumDf['mean_f_lambda'].iloc[indexSpec] - spectrumDf['mean_f_lambda'].iloc[indexSpecLeft]) / (spectrumDf['wavelength'].iloc[indexSpec] - spectrumDf['wavelength'].iloc[indexSpecLeft])
				B = spectrumDf['mean_f_lambda'].iloc[indexSpec] - A*spectrumDf['wavelength'].iloc[indexSpec]

				indexSpecLeft = indexSpec
				while (A*spectrumDf['wavelength'].iloc[indexSpecLeft]+B > halfMaximum):
					indexSpecLeft = indexSpecLeft - 1

			wavelengthLeft = spectrumDf['wavelength'].iloc[indexSpecLeft]
#			print("2: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecLeft]))

			#
			# Find a minimum flux value to the right or the crossing point with the half-maximum
			#
			indexSpecRight = indexSpec
			while (spectrumDf['mean_f_lambda'].iloc[indexSpecRight+1] < spectrumDf['mean_f_lambda'].iloc[indexSpecRight])\
			and (spectrumDf['mean_f_lambda'].iloc[indexSpecRight] > halfMaximum):
				indexSpecRight = indexSpecRight + 1

#			print("3: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecRight]))

			# If the flux is smaller than the half-maximum, then we have found the right wavelength, otherwise extrapolate
			if(spectrumDf['mean_f_lambda'].iloc[indexSpecRight] > halfMaximum):
				A = (spectrumDf['mean_f_lambda'].iloc[indexSpecRight] - spectrumDf['mean_f_lambda'].iloc[indexSpec]) / (spectrumDf['wavelength'].iloc[indexSpecRight] - spectrumDf['wavelength'].iloc[indexSpec])
				B = spectrumDf['mean_f_lambda'].iloc[indexSpec] - A*spectrumDf['wavelength'].iloc[indexSpec]

				indexSpecRight = indexSpec
				while (A*spectrumDf['wavelength'].iloc[indexSpecRight]+B > halfMaximum):
					indexSpecRight = indexSpecRight + 1

			wavelengthRight = spectrumDf['wavelength'].iloc[indexSpecRight]
#			print("4: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecRight]))

#			print("Wavelength left: {0:4.2f} Å; wavelength right: {1:4.2f} Å".format(wavelengthLeft, wavelengthRight))

			# Now calculate the full width and add to the list
			fwhm = wavelengthRight - wavelengthLeft
			FwhmLineWidth.append(fwhm)

#			print("Full Width at Half Maximum: {0:4.2f} Å".format((wavelengthRight - wavelengthLeft)/2.))

	# Now create the final dataframe
	FwhmDf = pd.DataFrame()
	FwhmDf['Line'] = FwhmLineNameList
	FwhmDf['Wavelength'] = FwhmLineWavelengthList
	FwhmDf['Width'] = FwhmLineWidth

	return FwhmDf

def CalculateFwhm2(spectrumDf, lineNameList = []):
	"""
	Determines the Full-Width at Half-Maximum of one or more spectrum lines.
	Achieves this by searching for crossing points with the continuum.

	IN:
	spectrumDf: the dataframe defining the spectrum; should have columns 'wavelength', 'mean_f_lambda' and 'theoretical'
	lineNameList: a list with laboratory determined wavelengths of spectrum lines.

	OUT:
	FwhmDf: a dataframe with all lines and the associated FWHM values.
	"""

	# Load the table with known emission lines
	emissionLines = pd.read_csv('./hkSpectrumLines.csv', skiprows=5)

	# Create temporary storage
	FwhmLineNameList = []
	FwhmLineWavelengthList = []
	FwhmLineWidth = []

	# Loop over all spectrum lines
	for lineName in lineNameList:

		# We can have multiple lines with the same name, but at different wavelengths
		indexListLab = emissionLines['Line'][emissionLines['Line'] == lineName].index.tolist()

		# So loop over all found wavelengths for this spectrum line
		for indexLab in indexListLab:
			# Retrieve the laboratory wavelength; this is a good starting point to find an emission peak
			wavelengthLab = emissionLines['Wavelength / Å'].iloc[indexLab]

			# Search for the same wavelength in the actual spectrum
			indexSpec = 0 # Spectrum index
			while (spectrumDf['wavelength'].iloc[indexSpec] < wavelengthLab):
				indexSpec = indexSpec + 1

			# IndexSpec now points to the position in SpectrumDf that is associated with the laboratory spectrum line.
			# The actual spectrum line might however be shifted to the left or to the right.
			# Let's find the emission peak by checking neighbouring fluxes.

			# First check to the left
			while (spectrumDf['mean_f_lambda'].iloc[indexSpec-1] > spectrumDf['mean_f_lambda'].iloc[indexSpec]):
				indexSpec = indexSpec - 1

			# Now check to the right
			while (spectrumDf['mean_f_lambda'].iloc[indexSpec+1] > spectrumDf['mean_f_lambda'].iloc[indexSpec]):
				indexSpec = indexSpec + 1

			# indexSpec now points to the emission peak in the actual spectrum
			# Store the emissionline name and the actual wavelength in a list for later use
			FwhmLineNameList.append(lineName)
			FwhmLineWavelengthList.append(spectrumDf['wavelength'].iloc[indexSpec])

#			print("Emission line {0} found at {1:4.2f} Å".format(lineName, spectrumDf['wavelength'].iloc[indexSpec]))

			# Determine the half-maximum
			halfMaximum = spectrumDf['continuum'].iloc[indexSpec] + 0.5*(spectrumDf['mean_f_lambda'].iloc[indexSpec]-spectrumDf['continuum'].iloc[indexSpec])
#			print("Half-maximum = {0:.2f}".format(halfMaximum))

			#
			# Find the left crossing point with the half-maximum
			#
			indexSpecLeft = indexSpec
			while (spectrumDf['mean_f_lambda'].iloc[indexSpecLeft] > halfMaximum):
				indexSpecLeft = indexSpecLeft - 1

			wavelengthLeft = spectrumDf['wavelength'].iloc[indexSpecLeft]
#			print("1: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecLeft]))

			#
			# Find the right crossing point with the half-maximum
			#
			indexSpecRight = indexSpec
			while (spectrumDf['mean_f_lambda'].iloc[indexSpecRight] > halfMaximum):
				indexSpecRight = indexSpecRight + 1

			wavelengthRight = spectrumDf['wavelength'].iloc[indexSpecRight]
#			print("2: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecRight]))

#			print("Wavelength left: {0:4.2f} Å; wavelength right: {1:4.2f} Å".format(wavelengthLeft, wavelengthRight))

			# Now calculate the full width and add to the list
			fwhm = wavelengthRight - wavelengthLeft
			FwhmLineWidth.append(fwhm)

#			print("Full Width at Half Maximum: {0:4.2f} Å".format((wavelengthRight - wavelengthLeft)/2.))

	# Now create the final dataframe
	FwhmDf = pd.DataFrame()
	FwhmDf['Line'] = FwhmLineNameList
	FwhmDf['Wavelength'] = FwhmLineWavelengthList
	FwhmDf['Width'] = FwhmLineWidth

	return FwhmDf

def CalculateFwhm3(spectrumDf, lineNameList = []):
	"""
	Determines the Full-Width at Half-Maximum of one or more spectrum lines.
	Achieves this by approximating an emission line by a Bell curve.

	IN:
	spectrumDf: the dataframe defining the spectrum; should have columns 'wavelength', 'mean_f_lambda' and 'theoretical'
	lineNameList: a list with laboratory determined wavelengths of spectrum lines.

	OUT:
	FwhmDf: a dataframe with all lines and the associated FWHM values.
	"""

	# Load the table with known emission lines
	emissionLines = pd.read_csv('./hkSpectrumLines.csv', skiprows=5)

	# Create temporary storage
	FwhmLineNameList = []
	FwhmLineWavelengthList = []
	FwhmLineWidth = []

	# Loop over all spectrum lines
	for lineName in lineNameList:

		# We can have multiple lines with the same name, but at different wavelengths
		indexListLab = emissionLines['Line'][emissionLines['Line'] == lineName].index.tolist()

		# So loop over all found wavelengths for this spectrum line
		for indexLab in indexListLab:
			# Retrieve the laboratory wavelength; this is a good starting point to find an emission peak
			wavelengthLab = emissionLines['Wavelength / Å'].iloc[indexLab]

			# Search for the same wavelength in the actual spectrum
			indexSpec = 0 # Spectrum index
			while (spectrumDf['wavelength'].iloc[indexSpec] < wavelengthLab):
				indexSpec = indexSpec + 1

			# IndexSpec now points to the position in SpectrumDf that is associated with the laboratory spectrum line.
			# The actual spectrum line might however be shifted to the left or to the right.
			# Let's find the emission peak by checking neighbouring fluxes.

			# First check to the left
			while (spectrumDf['mean_f_lambda'].iloc[indexSpec-1] > spectrumDf['mean_f_lambda'].iloc[indexSpec]):
				indexSpec = indexSpec - 1

			# Now check to the right
			while (spectrumDf['mean_f_lambda'].iloc[indexSpec+1] > spectrumDf['mean_f_lambda'].iloc[indexSpec]):
				indexSpec = indexSpec + 1

			# indexSpec now points to the emission peak in the actual spectrum
			# Store the emissionline name and the actual wavelength in a list for later use
			FwhmLineNameList.append(lineName)
			FwhmLineWavelengthList.append(spectrumDf['wavelength'].iloc[indexSpec])

#			print("Emission line {0} found at {1:4.2f} Å".format(lineName, spectrumDf['wavelength'].iloc[indexSpec]))

			#
			# Find the left crossing point with the continuum
			#
			indexSpecLeft = indexSpec
			while (spectrumDf['mean_f_lambda'].iloc[indexSpecLeft] > spectrumDf['continuum'].iloc[indexSpecLeft]):
				indexSpecLeft = indexSpecLeft - 1

			wavelengthLeft = spectrumDf['wavelength'].iloc[indexSpecLeft]
#			print("1: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecLeft]))

			#
			# Find the right crossing point with the continuum
			#
			indexSpecRight = indexSpec
			while (spectrumDf['mean_f_lambda'].iloc[indexSpecRight] > spectrumDf['continuum'].iloc[indexSpecLeft]):
				indexSpecRight = indexSpecRight + 1

			wavelengthRight = spectrumDf['wavelength'].iloc[indexSpecRight]
#			print("2: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecRight]))

#			print("Wavelength left: {0:4.2f} Å; wavelength right: {1:4.2f} Å".format(wavelengthLeft, wavelengthRight))

			# Now calculate the full width and add to the list
			fwhm = BellCurve(spectrumDf[indexSpecLeft:indexSpecRight])
			FwhmLineWidth.append(fwhm)

#			print("Full Width at Half Maximum: {0:4.2f} Å".format((wavelengthRight - wavelengthLeft)/2.))

	# Now create the final dataframe
	FwhmDf = pd.DataFrame()
	FwhmDf['Line'] = FwhmLineNameList
	FwhmDf['Wavelength'] = FwhmLineWavelengthList
	FwhmDf['Width'] = FwhmLineWidth

	return FwhmDf

def BellCurve(spectrumDf):
	"""
	Creates a Bell curve. This is based on a normal distribution, but used to approximate an emission line.
	Therefore the actual inetgral will not be 1, as in a normal distribution, but scaled such as to fit the peak and width of the eission line.

	IN:

	OUT:
	"""

	width = max(spectrumDf['wavelength']) - min(spectrumDf['wavelength'])
	height = max(spectrumDf['mean_f_lambda']) - min(spectrumDf['mean_f_lambda'])
	offsetX = min(spectrumDf['wavelength']) + width/2.
	offsetY = min(spectrumDf['mean_f_lambda'])

	sigma = width/6. # 99%
	mu = offsetX

#	spectrumDf['bell curve'] = height*np.exp(-1.*(spectrumDf['wavelength'] - mu)**2)/(2*sigma**2)

	xl = -np.sqrt(-1.*np.log(0.5)*2*sigma**2) + mu
	xr = +np.sqrt(-1.*np.log(0.5)*2*sigma**2) + mu

	fwhm = xr - xl
#	print("fwhm = {0:.2f}".format(fwhm))

	return fwhm
