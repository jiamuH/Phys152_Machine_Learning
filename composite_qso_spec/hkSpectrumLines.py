import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tic

def PlotSpectrumLines(ax, lineNameList = [], yText = None):
	# Load the table with known emission lines
	emissionLines = pd.read_csv('./hkSpectrumLines.csv', skiprows=5)

	# Get ymin, ymax
	yLim = ax.get_ylim()
	yDelta = yLim[1]-yLim[0]

	if yLim == None:
		y = yLim[1]
	else:
		y = yText

	if(len(lineNameList)==0):
		for lineName, wavelength in zip(emissionLines['Line'], emissionLines['Wavelength / Å']):
			yMod = -yMod

			ax.axvline(wavelength, color = 'r', linestyle = '-', linewidth = 1)
			ax.annotate(lineName, xy=(wavelength, y+yMod), arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.3'), xytext=(+25, -15), xycoords = 'data', textcoords='offset points', fontsize='x-large')

	else:
		for lineName in lineNameList:
			# We can have multiple lines with the same name, but at different wavelengths
			indexList = emissionLines['Line'][emissionLines['Line'] == lineName].index.tolist()

			for index in indexList:
				y -= 0.025*yDelta

				wavelength = emissionLines['Wavelength / Å'][index]
				ax.axvline(wavelength, color = 'r', linestyle = '-', linewidth = 1)

				ax.annotate(lineName, xy=(wavelength, y), arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.3'), xytext=(+25, -15), xycoords = 'data', textcoords='offset points', fontsize='x-large')
