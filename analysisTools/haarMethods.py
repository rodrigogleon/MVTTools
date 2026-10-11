"""
Haar wavelet methods. Vectorised implementation of various Haar wavelet algorithms found in 
https://github.com/sumanbala2210-USRA/GBM_MVT_paper. Original code by Suman Bala, modified
by Derek Pinkett
"""
# import statements
import numpy as np
from math import exp as MATHexp, log as MATHlog
from matplotlib import pyplot as plt, use as MPLUse

from scipy.optimize import minimize_scalar
from warnings import filterwarnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import cpu_count
from loadConfig import getInitialBinSize
from tqdm import tqdm

# standard logging/configuration setup
from loadConfig import config
import logging
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    from loggerSetup import initialiseLogging
    initialiseLogging()



filterwarnings('ignore', r'divide by zero encountered')
filterwarnings('ignore', r'invalid value encountered')




def haarDenoise(
        data: np.ndarray,
        error: np.ndarray | None = [],
        thresholdFactor: float = 1.0,
        estimateNoise: bool = False,
        soft: bool = False,
        floatType: str = 'float64'
        ) -> np.ndarray:
    """
    Haar denoiser using a vectorised implementation of the Haar wavelet denoising algorithm. This function takes in a 1D array of data and applies Haar wavelet denoising to reduce noise while preserving important features in the signal. Assumes Gaussian noise.

    Args:
        data (np.ndarray): 1D array of input data to be denoised.
        error (np.ndarray, optional): 1D array of error values. Defaults to [].
        thresholdFactor (float, optional): Threshold factor for denoising. Defaults to 1.0.
        estimateNoise (bool, optional): Whether to estimate noise. Defaults to False.
        soft (bool, optional): Whether to use soft thresholding. Defaults to False.
        floatType (str, optional): Data type for computations. Defaults to 'float64'.

    Returns:
        np.ndarray: Denoised data array.
    """
    data = data.astype(floatType)
    lengthOfData: int = len(data)
    useError: bool = True

    if error is None or error.size == 0:
        useError = False
    else:
        # compute the variance of the error
        varianceError: np.ndarray = error.astype(floatType) ** 2

    # determine the maximum decomposition level using the bit length of the data length minus one
    maxDecompositionLevel: int = (lengthOfData - 1).bit_length()
    # bitshifting to find the next power of two greater than or equal to the length of the data
    paddedSignalLength: int = 1 if (lengthOfData <= 0) else 1 << maxDecompositionLevel


    leftPaddingLength: int = 0
    rightPaddingLength: int = 0
    paddedSignalData: np.ndarray = data.copy()
    if useError:
        paddedNoiseVariance: np.ndarray = varianceError.copy()

    # extend to get 2^n length using symmetric reflection padding
    if paddedSignalLength > lengthOfData:
        # Calculate how much total padding is needed
        totalPaddingNeeded = paddedSignalLength - lengthOfData
        
        # Split the padding between left and right sides
        leftPaddingLength = totalPaddingNeeded // 2
        rightPaddingLength = totalPaddingNeeded - leftPaddingLength
        
        # np.pad the signal, 'reflect' mode mirrors the edge data accurately
        paddedSignalData = np.pad(
            data, 
            (leftPaddingLength, rightPaddingLength), 
            mode='reflect'
        )
        
        if useError:
            paddedNoiseVariance = np.pad(
                varianceError, 
                (leftPaddingLength, rightPaddingLength), 
                mode='reflect'
            )

    # initialize the fall-back noise estimate to 1.0
    estimatedNoiseSTD: float = 1.

    if (estimateNoise):
        # calculate the first difference of the signal to estimate noise
        signalFirstDifference: np.ndarray = np.abs(np.diff(paddedSignalData))

        # estimate the noise standard deviation using the median absolute deviation (MAD) method
        if (useError):
            squaredVarianceSum: np.ndarray = (
                paddedNoiseVariance[1:] ** 2 + paddedNoiseVariance[: -1] ** 2
                )
            propagatedNoiseSTD: np.ndarray = np.sqrt(0.5 * squaredVarianceSum)

            weightedDifferences: np.ndarray = signalFirstDifference / propagatedNoiseSTD
            estimatedNoiseSTD = 1.05 * np.median(weightedDifferences)
        else:
            estimatedNoiseSTD = 1.05 * np.median(signalFirstDifference)

    # compute and remove the baseline offset of the signal to center it around zero
    signalMeanOffset: float = paddedSignalData.mean()
    paddedSignalData  -=  signalMeanOffset

    # initialize arrays for the approximation coefficients and the reconstructed signal
    levelApproximationCoefficients: np.ndarray = np.empty(
        paddedSignalLength,
        dtype = floatType)
    reconstructedSignal: np.ndarray = np.full(
        paddedSignalLength,
        signalMeanOffset,
        dtype = floatType)
    
    # compute the cumulative sum of the padded signal data for the Haar transform
    paddedSignalData.cumsum(out = paddedSignalData)

    if (useError):
        # calculate the variance cumulative sum
        paddedNoiseVariance.cumsum(out = paddedNoiseVariance)
        levelVarianceCoefficients: np.ndarray = np.empty(
            paddedSignalLength,
            dtype = floatType)

    # calculate the squared noise variance threshold for denoising
    squaredNoiseVarianceThreshold: float = (
        (2.0 * MATHlog(2.0)) 
        * ((thresholdFactor * estimatedNoiseSTD)
        * (thresholdFactor * estimatedNoiseSTD))
        )
    for decompositionLevelIndex in range(maxDecompositionLevel):
        # calculate scale using bitshifting
        waveletScaleBlockWidth: int = 1 << (maxDecompositionLevel - decompositionLevelIndex - 1)
        twoScale: int = 2 * waveletScaleBlockWidth

        # precalculate the inverse squared scale
        inverseSquaredScale: float = 1.0 / (twoScale * twoScale)

        # Forward tranform
        levelApproximationCoefficients[:-twoScale] = (
            2 * paddedSignalData[waveletScaleBlockWidth:-waveletScaleBlockWidth]
            - paddedSignalData[:-twoScale]
            - paddedSignalData[twoScale:]
        )
        levelApproximationCoefficients[-twoScale:-waveletScaleBlockWidth] = (
            2 * paddedSignalData[-waveletScaleBlockWidth:]
            - paddedSignalData[-twoScale:-waveletScaleBlockWidth]
            - paddedSignalData[:waveletScaleBlockWidth]
            - paddedSignalData[-1]
        )
        levelApproximationCoefficients[-waveletScaleBlockWidth:] = (
            2 * paddedSignalData[:waveletScaleBlockWidth]
            - paddedSignalData[-waveletScaleBlockWidth:]
            - paddedSignalData[waveletScaleBlockWidth:twoScale]
            + paddedSignalData[-1]
        )

        # level-dependent variance calculation
        if (useError):
            levelVarianceCoefficients[:-twoScale] = (
                paddedNoiseVariance[twoScale:] - paddedNoiseVariance[:-twoScale]
            )
            levelVarianceCoefficients[-twoScale:] = (
                paddedNoiseVariance[:twoScale] + paddedNoiseVariance[-1] - paddedNoiseVariance[-twoScale:]
            )
        else:
            # avoid allocating a whole array if the variance is uniform scalar
            levelVarianceCoefficients: int = twoScale

        # denoising thresholds
        thresholdLimit: float = (
            squaredNoiseVarianceThreshold * decompositionLevelIndex * levelVarianceCoefficients
        )
        noiseMask: np.ndarray = (
            (levelApproximationCoefficients * levelApproximationCoefficients) < thresholdLimit
        )

        # hard thresholding: set coefficients below the threshold to zero
        levelApproximationCoefficients[noiseMask] = 0

        if (soft):
            signalMask: np.ndarray = ~noiseMask
            survivingCoefficients: np.ndarray = levelApproximationCoefficients[signalMask]
            # prevent division by zero by ensuring non-zero coefficients
            squaredCoefficients: np.ndarray = (
                survivingCoefficients * survivingCoefficients
                + 1e-12
            )
            if (useError):
                varienceTerm: np.ndarray = (
                    thresholdLimit[signalMask] / squaredCoefficients
                )
            else:
                varienceTerm: np.ndarray = (
                    thresholdLimit / squaredCoefficients
                )
            # apply soft thresholding by scaling the surviving coefficients
            levelApproximationCoefficients[signalMask] *= (
                np.sqrt(1.0 - varienceTerm)
            )
        
        # set all noise coefficients to zero
        levelApproximationCoefficients[noiseMask] = 0.0


        # reconstruction via inverse transformation
        levelApproximationCoefficients.cumsum(out = levelApproximationCoefficients)

        # pre-calculate internal difference blocks to keep array operations vectorized
        diffMain: np.ndarray = (
            2 * levelApproximationCoefficients[waveletScaleBlockWidth:-waveletScaleBlockWidth]
            - levelApproximationCoefficients[:-twoScale]
            - levelApproximationCoefficients[twoScale:]
        )
        diffLeft: np.ndarray = (
            2 * levelApproximationCoefficients[-waveletScaleBlockWidth:]
            - levelApproximationCoefficients[-twoScale:-waveletScaleBlockWidth]
            - levelApproximationCoefficients[:waveletScaleBlockWidth]
            - levelApproximationCoefficients[-1]
        )
        diffRight: np.ndarray = (
            2 * levelApproximationCoefficients[:waveletScaleBlockWidth]
            - levelApproximationCoefficients[-waveletScaleBlockWidth:]
            - levelApproximationCoefficients[waveletScaleBlockWidth:twoScale]
            + levelApproximationCoefficients[-1]
        )

        # accumulate the results into the reconstructed signal
        reconstructedSignal[twoScale:] += (
            diffMain[::-1] * inverseSquaredScale
        )
        reconstructedSignal[waveletScaleBlockWidth:twoScale] += (
            diffLeft[::-1] * inverseSquaredScale
        )
        reconstructedSignal[:waveletScaleBlockWidth] += (
            diffRight[::-1] * inverseSquaredScale
        )

    # return the reconstructed signal, removing any padding that was added
    return reconstructedSignal[leftPaddingLength:leftPaddingLength+lengthOfData]


def haarNDWT(
        poissonTimeSeriesData: np.ndarray,
        measurementError: np.ndarray,
        statisticalWeights: np.ndarray,
        binSizeInSeconds: float,
        timeSeriesOversampleFactor: float = 32.0,
        totalSignalRepetitions: int = 1,
        binFactor: float = 4.0,
        floatType: str = 'float64',
        intType: str = 'int32'
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    Args:
        poissonTimeSeriesData (np.ndarray): The input (Poisson) time series data.
        measurementError (np.ndarray): The measurement error data.
        statisticalWeights (np.ndarray): The statistical weights data.
        binSizeInSeconds (float): The bin size in seconds of the regularly spaced time series.
        timeSeriesOversampleFactor (float, optional): The over-sampling factor of the time series. Defaults to 32.0.
        totalSignalRepetitions (int, optional): The number of times to repeat the time series. Defaults to 1.
        binFactor (float, optional): The bin factor defining the sampling of the output data. Defaults to 4.0.
        floatType (str, optional): Data type for computations. Defaults to 'float64'.
        intType (str, optional): Data type for integer computations. Defaults to 'int32'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following:
            - Scaled lower time bounds (np.ndarray)
            - Scaled upper time bounds (np.ndarray)
            - Differential signal power (np.ndarray)
            - Differential baseline noise power (np.ndarray)
            - Differential power uncertainty (np.ndarray)
    """
    # integral signal, weights and noise variance, each with a leading zero
    integralSignalData: np.ndarray = np.concatenate((
        [0.0],
        np.cumsum(poissonTimeSeriesData.astype(floatType))
    ))
    integralWeights: np.ndarray = np.concatenate((
        [0.0],
        np.cumsum(statisticalWeights.astype(floatType))
    ))
    integralNoiseVariance: np.ndarray = np.concatenate((
        [0.0],
        np.cumsum(measurementError.astype(floatType) ** 2)
    ))

    # repeat the time series end on end, matching the original haar_nondec (each pass appends the
    # current integral minus its first and last points, offset by its final value)
    for _ in range(totalSignalRepetitions - 1):
        integralSignalData = np.concatenate((
            integralSignalData,
            integralSignalData[1:-1] + integralSignalData[-1]
        ))
        integralWeights = np.concatenate((
            integralWeights,
            integralWeights[1:-1] + integralWeights[-1]
        ))
        integralNoiseVariance = np.concatenate((
            integralNoiseVariance,
            integralNoiseVariance[1:-1] + integralNoiseVariance[-1]
        ))

    # length of the (repeated) time series
    lengthOfData: int = len(integralSignalData) - 1

    # find the maximum wavelet scale index, equivalent to ceil(log2(lengthOfData))
    maximumWaveletScaleIndex: int = (lengthOfData - 1).bit_length()

    # boundary guard checks
    if binFactor <= 0:
        binFactor = 1.0
    if timeSeriesOversampleFactor < binFactor:
        timeSeriesOversampleFactor = binFactor

    def generateScales(samplesPerOctave: float) -> np.ndarray:
        """Dyadic scales merged with even-rounded scales sampled samplesPerOctave times per octave."""
        dyadicScales: np.ndarray = 1 << np.arange(
            maximumWaveletScaleIndex,
            dtype = intType
        )
        intermediateScales: np.ndarray = 2.0 * np.round(
            (2.0 ** (np.arange(maximumWaveletScaleIndex * samplesPerOctave, dtype = floatType) / samplesPerOctave)) / 2.0
        )
        combinedScales: np.ndarray = np.unique(
            np.hstack((dyadicScales, intermediateScales)).astype(intType)
        )
        # keep scales whose full wavelet fits inside the time series
        return combinedScales[(combinedScales > 0) & (2 * combinedScales <= lengthOfData)]

    # finely oversampled scales at which the power spectrum is evaluated
    waveletScaleBlockWidths: np.ndarray = generateScales(timeSeriesOversampleFactor)

    # coarser output scales defining the bands the power spectrum is averaged into
    if binFactor < timeSeriesOversampleFactor:
        outputScaleBoundaries: np.ndarray = generateScales(binFactor)
    else:
        outputScaleBoundaries: np.ndarray = waveletScaleBlockWidths.copy()

    # number of scales to be used in the analysis
    totalWaveletScales: int = len(waveletScaleBlockWidths)

    # preallocate arrays for the power spectrum, baseline noise power spectrum, and power spectrum variance
    signalPowerSpectrum: np.ndarray = np.zeros(
        totalWaveletScales,
        dtype = floatType
    )
    baselineNoisePowerSpectrum: np.ndarray = np.zeros(
        totalWaveletScales,
        dtype = floatType
    )
    powerSpectrumVariance: np.ndarray = np.zeros(
        totalWaveletScales,
        dtype = floatType
    )

    # non-decimated forward Haar wavelet transform
    for scaleIndex in range(totalWaveletScales):
        # extract the block width for the current scale
        waveletScaleBlockWidth: int = int(waveletScaleBlockWidths[scaleIndex])
        twoScale: int = waveletScaleBlockWidth << 1
        squaredScaleBlockWidth: float = float(waveletScaleBlockWidth) * float(waveletScaleBlockWidth)

        # calculate the boundary correction for the active window
        validWindowCount: int = int(lengthOfData - twoScale + 1)
        boundaryCorrectionFactor: float = lengthOfData / validWindowCount

        # non-decimated calculation and squaring of the Haar wavelet coefficients
        waveletCoefficients: np.ndarray = (
            integralSignalData[twoScale:lengthOfData + 1]
            - 2 * integralSignalData[waveletScaleBlockWidth:lengthOfData - waveletScaleBlockWidth + 1]
            + integralSignalData[:validWindowCount]
        )
        squaredWaveletCoefficients: np.ndarray = waveletCoefficients * waveletCoefficients

        # variance and weight window slicing
        waveletVarianceWindows: np.ndarray = (
            integralNoiseVariance[twoScale:lengthOfData + 1]
            - integralNoiseVariance[:validWindowCount]
        )
        normalisedLocalWeights: np.ndarray = (
            (integralWeights[twoScale:lengthOfData + 1]
            - integralWeights[:validWindowCount])
            / float(twoScale)
        )
        meanLocalWeights: float = normalisedLocalWeights.mean()

        # spectrum accumulation
        inverseScaleWeightDenominator: float = (
            1.0 / (squaredScaleBlockWidth * meanLocalWeights)
        )
        signalPowerSpectrum[scaleIndex] = (
            (squaredWaveletCoefficients * normalisedLocalWeights).sum()
            * boundaryCorrectionFactor * inverseScaleWeightDenominator
        )

        varianceWindowProducts: np.ndarray = (
            waveletVarianceWindows * normalisedLocalWeights
        )
        currentNoiseSpectrum: float = (
            varianceWindowProducts.sum() * boundaryCorrectionFactor * inverseScaleWeightDenominator
        )
        baselineNoisePowerSpectrum[scaleIndex] = currentNoiseSpectrum

        # variance of the power spectrum (signal = 0, error only)
        inverseSquaredMeanWeight: float = 1.0 / (meanLocalWeights * meanLocalWeights)
        varianceTerm: float = (
            ((varianceWindowProducts * varianceWindowProducts).sum() / squaredScaleBlockWidth * inverseSquaredMeanWeight)
            + (0.5 * currentNoiseSpectrum)
        )
        powerSpectrumVariance[scaleIndex] = (
            (varianceTerm * (boundaryCorrectionFactor * boundaryCorrectionFactor))
            / float(waveletScaleBlockWidth)
        )

    # adjacent output band boundaries. These are deliberately views of outputScaleBoundaries, as in
    # the original haar_nondec: updating one band's limits also updates its neighbour's
    lowerBoundScales: np.ndarray = outputScaleBoundaries[:-1]
    upperBoundScales: np.ndarray = outputScaleBoundaries[1:]

    # number of adjacent scale bands
    adjacentScalePairCount: int = len(lowerBoundScales)

    # allocate arrays for the differential power spectrum calculations
    differentialSignalPower: np.ndarray = np.zeros(adjacentScalePairCount, dtype = floatType)
    differentialBaselineNoisePower: np.ndarray = np.zeros(adjacentScalePairCount, dtype = floatType)
    differentialPowerUncertainty: np.ndarray = np.zeros(adjacentScalePairCount, dtype = floatType)

    # scalar constant multiplier
    varianceReplicationMultiplier: float = float(totalSignalRepetitions + 1)

    # average the finely sampled power spectrum within each output band
    for scalePairIndex in range(adjacentScalePairCount):
        matchingScaleMask: np.ndarray = (
            (waveletScaleBlockWidths >= lowerBoundScales[scalePairIndex])
            & (waveletScaleBlockWidths < upperBoundScales[scalePairIndex])
        )
        matchingScaleCount: int = np.count_nonzero(matchingScaleMask)

        if (matchingScaleCount > 0):
            matchingScales: np.ndarray = waveletScaleBlockWidths[matchingScaleMask]
            inverseCount: float = 1.0 / float(matchingScaleCount)

            # tighten the band limits to the scales actually inside it
            lowerBoundScales[scalePairIndex] = matchingScales.min()
            upperBoundScales[scalePairIndex] = matchingScales.max()

            differentialSignalPower[scalePairIndex] = (
                signalPowerSpectrum[matchingScaleMask].sum() * inverseCount
            )
            differentialBaselineNoisePower[scalePairIndex] = (
                baselineNoisePowerSpectrum[matchingScaleMask].sum() * inverseCount
            )
            differentialPowerUncertainty[scalePairIndex] = np.sqrt(
                powerSpectrumVariance[matchingScaleMask].sum() * varianceReplicationMultiplier * inverseCount
            )

    # return the results scaled by the bin size in seconds
    scaledLowerTimeBounds: np.ndarray = lowerBoundScales.astype(floatType) * binSizeInSeconds
    scaledUpperTimeBounds: np.ndarray = upperBoundScales.astype(floatType) * binSizeInSeconds

    return (
        scaledLowerTimeBounds,
        scaledUpperTimeBounds,
        differentialSignalPower,
        differentialBaselineNoisePower,
        differentialPowerUncertainty
    )


def evaluateBreakPointResiduals(
    logBreakPointLocation: float,
    fitLogTimescales: np.ndarray,
    modeledLogSignalPower: np.ndarray,
    fitPowerVariance: np.ndarray,
    weightedPowerSum: float,
    totalInverseVarianceWeight: float,
    initialFitParameters: np.ndarray,
) -> float:
    """
    Evaluates the residuals for a given break point location in a piecewise linear fit.

    Args:
        logBreakPointLocation (float): The logarithm of the break point location in the timescale.
        fitLogTimescales (np.ndarray): The logarithm of the timescales for the fit.
        modeledLogSignalPower (np.ndarray): The logarithm of the signal power for the fit.
        fitPowerVariance (np.ndarray): The variance of the signal power for the fit.
        weightedPowerSum (float): The weighted sum of the signal power.
        totalInverseVarianceWeight (float): The total inverse variance weight.
        initialFitParameters (np.ndarray): The initial parameters for the fit.

    Returns:
        float: The evaluated residuals for the given break point location.
    """
    # Used to be in haarPowerMod, but moved here to allow garbage collection to consistently trigger
    # fallback guard against undefined NaN inputs
    if np.isnan(logBreakPointLocation):
        logBreakPointLocation = float(fitLogTimescales.min())

    # divide timescales across the current break point boundary
    lowerScaleMask: np.ndarray = fitLogTimescales < logBreakPointLocation
    upperScaleMask: np.ndarray = ~lowerScaleMask

    # re-cache active masked slices
    timescalesUpperView: np.ndarray = fitLogTimescales[upperScaleMask]
    varianceUpperView: np.ndarray = fitPowerVariance[upperScaleMask]
    signalUpperView: np.ndarray = modeledLogSignalPower[upperScaleMask]

    # calculate relative distance coordinates from the break location
    relativeDistanceUpper: np.ndarray = (
        timescalesUpperView - logBreakPointLocation
    )
    inverseVarianceUpper: np.ndarray = 1.0 / varianceUpperView

    # compute structural covariance moments
    crossProductMoment: float = float(
        (relativeDistanceUpper * inverseVarianceUpper).sum()
    )
    varianceWeightedMoment: float = float(
        (
            (relativeDistanceUpper * relativeDistanceUpper)
            * inverseVarianceUpper
        ).sum()
    )

    covarianceDeterminant: float = (
        totalInverseVarianceWeight * varianceWeightedMoment
    ) - (crossProductMoment * crossProductMoment)
    weightedSignalMoment: float = float(
        ((signalUpperView * relativeDistanceUpper) * inverseVarianceUpper).sum()
    )

    # optimize model parameter estimates
    if covarianceDeterminant > 0.0:
        initialFitParameters[0] = (
            (varianceWeightedMoment * weightedPowerSum)
            - (crossProductMoment * weightedSignalMoment)
        ) / covarianceDeterminant
        initialFitParameters[1] = (
            (totalInverseVarianceWeight * weightedSignalMoment)
            - (crossProductMoment * weightedPowerSum)
        ) / covarianceDeterminant

    # evaluate residuals and structural curvature uncertainty
    if any(lowerScaleMask):
        signalLowerView: np.ndarray = modeledLogSignalPower[lowerScaleMask]
        varianceLowerView: np.ndarray = fitPowerVariance[lowerScaleMask]

        lowerResiduals: np.ndarray = signalLowerView - initialFitParameters[0]
        lowerRegionChiSquared: float = float(
            ((lowerResiduals * lowerResiduals) / varianceLowerView).sum()
        )

        inverseVarianceUpperSum: float = float(inverseVarianceUpper.sum())
        curvatureMetric1: float = (
            -initialFitParameters[1] * inverseVarianceUpperSum
        )
        curvatureMetric0: float = -initialFitParameters[1] * curvatureMetric1

        signalResidualsUpper: np.ndarray = (
            signalUpperView - initialFitParameters[0]
        )
        curvatureMetric2: float = float(
            (signalResidualsUpper * inverseVarianceUpper).sum()
        ) - (2.0 * initialFitParameters[1] * crossProductMoment)

        numeratorTermA: float = curvatureMetric1 * (
            (curvatureMetric2 * crossProductMoment)
            - (curvatureMetric1 * varianceWeightedMoment)
        )
        numeratorTermB: float = curvatureMetric2 * (
            (curvatureMetric1 * crossProductMoment)
            - (curvatureMetric2 * totalInverseVarianceWeight)
        )

        initialFitParameters[2] = 1.0 / np.sqrt(
            curvatureMetric0
            + ((numeratorTermA + numeratorTermB) / covarianceDeterminant)
        )
    else:
        lowerRegionChiSquared = 0.0

    # compute final model residual sum across the entire distribution
    modelPredictionsUpper: np.ndarray = (
        initialFitParameters[1] * relativeDistanceUpper
    ) + initialFitParameters[0]
    upperResiduals: np.ndarray = signalUpperView - modelPredictionsUpper
    upperRegionChiSquared: float = float(
        ((upperResiduals * upperResiduals) * inverseVarianceUpper).sum()
    )

    return lowerRegionChiSquared + upperRegionChiSquared


haarPowerModSettings = config.mvtAnalysisConfig.haarPowerModelSettings

def haarPowerMod(
        emissionRateSignal: np.ndarray,
        emissionRateUncertainty: np.ndarray,
        minimumBinSizeSeconds: float = 1.0e-4,
        maximumBinSizeSeconds: float = float(haarPowerModSettings.maximumBinSizeSeconds),
        maxBackgroundTimescale: float = float(haarPowerModSettings.maxBackgroundTimescale),
        totalSignalRepetitions: int = int(haarPowerModSettings.totalSignalRepetitions),
        shouldGeneratePlots: bool = bool(haarPowerModSettings.shouldGeneratePlots),
        outputBinningRatio: int = int(haarPowerModSettings.outputBinningRatio),
        shouldVerifyZeroBaseline: bool = bool(haarPowerModSettings.shouldVerifyZeroBaseline),
        scalingAdjustmentFactor: float = float(haarPowerModSettings.scalingAdjustmentFactor),
        signalToNoiseRatioThreshold: float = float(config.mvtAnalysisConfig.significanceSigmaThreshold),
        shouldApplyStatisticalWeight: bool = bool(haarPowerModSettings.shouldApplyStatisticalWeight),
        outputExportFilename: str = str(haarPowerModSettings.outputExportFilename),
        floatType = 'float64',
        intType = 'int32'
        ) -> tuple[float, float, float, float, float, float, float]:
    """
    Modulated Haar wavelet power transform pipeline for time-series rates.
    set scalingAdjustmentFactor to negative to have the code estimate it
    largest timescale used will be maxBackgroundTimescale

    All defaults other than minimumBinSizeSeconds, floatType and intType are read from mvtAnalysisConfig in config.yaml (haarPowerModelSettings, and significanceSigmaThreshold for signalToNoiseRatioThreshold). Explicitly passed arguments override the config values.

    Args:
        emissionRateSignal (np.ndarray): The input emission rate signal.
        emissionRateUncertainty (np.ndarray): The uncertainty associated with the emission rate signal.
        minimumBinSizeSeconds (float, optional): The bin size of the input signal in seconds. Defaults to 1.0e-4.
        maximumBinSizeSeconds (float, optional): The maximum bin size in seconds.
        maxBackgroundTimescale (float, optional): The maximum timescale for background noise.
        totalSignalRepetitions (int, optional): The number of times to repeat the signal.
        shouldGeneratePlots (bool, optional): Whether to generate plots.
        outputBinningRatio (int, optional): The binning ratio for output.
        shouldVerifyZeroBaseline (bool, optional): Whether to verify zero baseline.
        scalingAdjustmentFactor (float, optional): The scaling adjustment factor.
        signalToNoiseRatioThreshold (float, optional): The significance (in sigma) of power over background noise required for a timescale to count as signal.
        shouldApplyStatisticalWeight (bool, optional): Whether to apply statistical weights.
        outputExportFilename (str, optional): The filename for exporting output.
        floatType (str, optional): Data type for computations. Defaults to 'float64'.
        intType (str, optional): Data type for integer computations. Defaults to 'int32'.

    Returns:
        Tuple[float, float, float, float, float, float, float]: A tuple containing the following:
            - peakSignalTimescale (float): The timescale corresponding to the peak signal-to-noise ratio.
            - peakSignalToNoiseRatio (float): The peak signal-to-noise ratio.
            - spectralIndexSlope (float): The slope of the spectral index.
            - minimumVariabilityTimescale (float): The minimum variability timescale.
            - minimumVariabilityTimescaleUncertainty (float): The uncertainty associated with the minimum variability timescale.
            - fittedPowerLawSlope (float): The slope of the fitted power law.
            - peakSignalToNoiseUncertainty (float): The uncertainty associated with the peak signal-to-noise ratio.
    """
    timeDeltaLabelString: str = r'$\Delta t$'

    if (shouldApplyStatisticalWeight):
        # first pass: extract primary denoised signal
        statisticalWeightVector: np.ndarray = haarDenoise(
            emissionRateSignal,
            emissionRateUncertainty,
            floatType = floatType
            )
        # second pass: further isolate the low-frequency baseline flux
        statisticalWeightVector = haarDenoise(
            statisticalWeightVector,
            emissionRateUncertainty,
            floatType = floatType
        )
        # clip the statistical weight vector
        statisticalWeightVector.clip(
            0.0,
            out = statisticalWeightVector
        )
    else:
        statisticalWeightVector: np.ndarray = np.empty(
            len(emissionRateSignal),
            dtype=floatType
            )
        statisticalWeightVector.fill(1.0) # faster to fill empty than to use np.ones()

    # non-decimated Haar wavelet transform
    (
        scaledLowerTimeBounds,
        scaledUpperTimeBounds,
        signalPowerSpectrum,
        baselineNoisePowerSpectrum,
        differentialPowerUncertainty
    ) = haarNDWT(
        emissionRateSignal,
        emissionRateUncertainty,
        statisticalWeightVector,
        binSizeInSeconds = minimumBinSizeSeconds,
        totalSignalRepetitions = totalSignalRepetitions,
        binFactor = outputBinningRatio,
        timeSeriesOversampleFactor = outputBinningRatio * 8.0,
        floatType = floatType,
        intType = intType
    )

    # filter values that are less than zero
    validNoiseSpectrumMask: np.ndarray = (
        baselineNoisePowerSpectrum > 0.0
    )

    scaledLowerTimeBounds = scaledLowerTimeBounds[validNoiseSpectrumMask]
    scaledUpperTimeBounds = scaledUpperTimeBounds[validNoiseSpectrumMask]
    signalPowerSpectrum = signalPowerSpectrum[validNoiseSpectrumMask]
    baselineNoisePowerSpectrum = baselineNoisePowerSpectrum[validNoiseSpectrumMask]
    differentialPowerUncertainty = differentialPowerUncertainty[validNoiseSpectrumMask]

    # calculate the characteristic timescales
    characteristicTimescales: np.ndarray = (
        0.5 * (scaledLowerTimeBounds + scaledUpperTimeBounds)
    )

    # find the peak signal-to-noise timescale
    peakScaleIndex: int = int(
        (signalPowerSpectrum / baselineNoisePowerSpectrum).argmax()
    )
    peakSignalTimescale: float = float(characteristicTimescales[peakScaleIndex])

    # pre-allocate tracking and statistical variables
    peakSignalToNoiseRatio: float = 0.0
    spectralIndexSlope: float = 0.0
    minimumVariabilityTimescale: float = 0.0
    minimumVariabilityTimescaleUncertainty: float = 0.0
    fittedPowerLawSlope: float = 0.0
    peakSignalToNoiseUncertainty: float = 0.0
    variabilityTimescaleSTD: float = 0.0

    measurementClassificationType: str = 'limit'

    # filter for short background noise timescales
    backgroundNoiseMask: np.ndarray = characteristicTimescales < maxBackgroundTimescale

    # count matching elements
    matchingBackgroundCount: int = np.count_nonzero(backgroundNoiseMask)

    # determine and apply the background scaling factor
    if (matchingBackgroundCount < 2 or scalingAdjustmentFactor > 0):
        scalingAdjustmentFactor = np.abs(scalingAdjustmentFactor)
        
        # apply scaling factor
        baselineNoisePowerSpectrum *= scalingAdjustmentFactor
        differentialPowerUncertainty *= scalingAdjustmentFactor
    else:
        # estimate scaling factor via median ratio of noise-dominant bands
        estimatedBackgroundScalingFactor: float = np.median(
            signalPowerSpectrum[backgroundNoiseMask]
            / baselineNoisePowerSpectrum[backgroundNoiseMask]
        )

        # apply estimated scaling factor
        baselineNoisePowerSpectrum *= estimatedBackgroundScalingFactor
        differentialPowerUncertainty *= estimatedBackgroundScalingFactor

#        commmented out for now, as later code calls this hundreds of times.
#        logger.debug(f" {outputExportFilename} a factor: {estimatedBackgroundScalingFactor}")

    # subtract background noise from the signal power spectrum
    signalPowerSpectrum -= baselineNoisePowerSpectrum

    # define signal-to-noise ratio threshold and set everything on shorter timescales to be background noise
    # flag initial noisy regions below the SNR threshold
    backgroundNoiseMask = signalPowerSpectrum < signalToNoiseRatioThreshold * differentialPowerUncertainty
    validSignalMask = ~backgroundNoiseMask

    # limit the mask to timescales below the peak signal window
    backgroundNoiseMask &= characteristicTimescales < peakSignalTimescale

    # locate the last noise index before the signal
    noiseIndices: np.ndarray = np.flatnonzero(backgroundNoiseMask)
    lastNoiseIndex: int = int(
        noiseIndices[-1] if noiseIndices.size > 0 else 0
    )

    # enforce background constraints on all preceding shorter timescales
    backgroundNoiseMask[:lastNoiseIndex] = True
    validSignalMask[:lastNoiseIndex] = False

    # look backward to preserve valid boundary signales exceeding 1σ
    fallBackStepCounter: int = 0
    while (lastNoiseIndex > 0
        and signalPowerSpectrum[lastNoiseIndex] > differentialPowerUncertainty[lastNoiseIndex]
        and fallBackStepCounter < outputBinningRatio):
        backgroundNoiseMask[lastNoiseIndex] = False
        validSignalMask[lastNoiseIndex] = True
        lastNoiseIndex -= 1
        fallBackStepCounter += 1



    # check if diagnostic baseline plots are requested
    if shouldVerifyZeroBaseline:
        figure1 = plt.figure(1)
        plt.clf()
        
        # plot total raw power spectrum and noise-subtracted signal
        plt.plot(
            characteristicTimescales,
            signalPowerSpectrum + baselineNoisePowerSpectrum,
            'o',
            label='Total Raw Power'
            )
        plt.plot(
            characteristicTimescales,
            signalPowerSpectrum,
            'o',
            label='Noise-Subtracted Signal'
            )
        
        # highlight points matching the background noise floor
        if any(backgroundNoiseMask):
            plt.plot(
                characteristicTimescales[backgroundNoiseMask],
                signalPowerSpectrum[backgroundNoiseMask],
                'kv',
                label='Background Noise Mask'
                )
            
        # plot baseline reference lines
        plt.plot(
            characteristicTimescales,
            baselineNoisePowerSpectrum,
            'k-',
            label='Baseline Noise'
            )
        plt.plot(
            characteristicTimescales,
            differentialPowerUncertainty,
            'k-.',
            label='Differential Power Uncertainty'
            )
        
        # force logarithmic scaling across both axes
        plt.loglog()
        plt.legend()
        
        figure2 = plt.figure(2)
        plt.clf()

    # verify a baseline number of signal elements exist
    if np.count_nonzero(validSignalMask) < 2:
        logger.debug(f"{outputExportFilename} Not enough significant data!")
    else:
        # filter out NaN or Inf values from the active data set
        cleanSignalMask: np.ndarray = validSignalMask & np.isfinite(signalPowerSpectrum)

        # guard against empty subsets after cleaning
        if np.count_nonzero(cleanSignalMask) < 2:
            logger.debug(f"{outputExportFilename} Not enough significant data after removing NaNs!")
            # exit the function gracefully if no valid data is left
            return (
                peakSignalToNoiseRatio,
                spectralIndexSlope,
                minimumVariabilityTimescale,
                minimumVariabilityTimescaleUncertainty,
                fittedPowerLawSlope,
                peakSignalToNoiseUncertainty,
                variabilityTimescaleSTD
            )
        
        # update the valid signal mask with the cleaned mask
        validSignalMask = cleanSignalMask

        # extract valid signal bounds < 1e7
        clampedNormalisationMask: np.ndarray = (
            validSignalMask & (signalPowerSpectrum < 1e7)
        )
        clampedNormalisationSignal: np.ndarray = signalPowerSpectrum[clampedNormalisationMask]

        # calculate the peak scaling factor with a fallback
        if clampedNormalisationSignal.size > 0:
            peakSpectrumNormalisationFactor: float = clampedNormalisationSignal.max()
        else:
            # fallback to mean if no valid points are below the threshold
            peakSpectrumNormalisationFactor: float = float(signalPowerSpectrum[validSignalMask].mean())
        
        # normalize the spectra and uncertainties
        signalPowerSpectrum /= peakSpectrumNormalisationFactor
        baselineNoisePowerSpectrum /= peakSpectrumNormalisationFactor
        differentialPowerUncertainty /= peakSpectrumNormalisationFactor

        # extract peak timescale offset
        peakSignalToNoiseRatio = float(characteristicTimescales[lastNoiseIndex + 1].max())

        # isolate the net signal power excess above the noise floor
        netSignalPowerExcess: np.ndarray = (
            signalPowerSpectrum[validSignalMask] - baselineNoisePowerSpectrum[validSignalMask]
        )

        # extract indices where power excess is positive
        positiveExcessIndices: np.ndarray = np.flatnonzero(netSignalPowerExcess > 0.0)
        isSpectralSlopeFixedToBaseline: bool = False

        validTimescalesView: np.ndarray = characteristicTimescales[validSignalMask]

        if positiveExcessIndices.size > 2:
            firstPositiveIndex: int = positiveExcessIndices[0]
            precedingNegativeIndex: int = firstPositiveIndex - 1

            if precedingNegativeIndex >= 0:
                # extract starting timescale baseline
                spectralIndexSlope: float = validTimescalesView[precedingNegativeIndex]

                # perform linear interpolation to isolate the zero crossing intercept
                y0: float = netSignalPowerExcess[precedingNegativeIndex]
                y1: float = netSignalPowerExcess[firstPositiveIndex]

                if y1 != y0:
                    timeDelta: float = (
                        validTimescalesView[firstPositiveIndex] - validTimescalesView[precedingNegativeIndex]
                    )
                    spectralIndexSlope -= y0 * (timeDelta / (y1 - y0))
            else:
                isSpectralSlopeFixedToBaseline = True
        else:
            # fallback path if there are insufficient positive excess indices
            spectralIndexSlope = validTimescalesView.max()

        # find departures from a linear fit, will be starting point for tmin calculation
        # pre-cache signal and uncertainty slices
        validSignalView: np.ndarray = signalPowerSpectrum[validSignalMask]
        validUncertaintyView: np.ndarray = differentialPowerUncertainty[validSignalMask]

        # map characteristic timescales and signal power into logarithmic space
        logCharacteristicTimescales: np.ndarray = np.log(validTimescalesView)
        normalisedLogSignalPower: np.ndarray = (
            np.log(validSignalView) - 2.0 * logCharacteristicTimescales
            )

        # compute fractional uncertainties and flattened inverse squared weights
        fractionalPowerUncertainty: np.ndarray = (
            validUncertaintyView / validSignalView
        )
        inverseSquaredUncertainty: np.ndarray = (
            1.0 / (fractionalPowerUncertainty * fractionalPowerUncertainty)
        )

        # calculate cumulative running variance
        cumulativeRunningVariance: np.ndarray = (
            inverseSquaredUncertainty.cumsum()
        )
        np.reciprocal(
            cumulativeRunningVariance,
            out = cumulativeRunningVariance
        )

        # compute cumulative running weighted mean
        weightedSignalPowerTerms: np.ndarray = (
            normalisedLogSignalPower * inverseSquaredUncertainty
        )
        cumulativeRunningWeightedMean: np.ndarray = (
            weightedSignalPowerTerms.cumsum() * cumulativeRunningVariance
        )


        # departure condition: change in slope by 0.1, but at least 0.5σ, relative to a time a factor 2 shorter
        # compute the departure threshold
        combinedVarianceTerms: np.ndarray = (
            (fractionalPowerUncertainty[1:] * fractionalPowerUncertainty[1:])
            + (float(outputBinningRatio) * cumulativeRunningVariance[:-1])
        )
        departureThreshold: np.ndarray = (
            0.5 * np.sqrt(combinedVarianceTerms)
        )

        # enforce minimum change floor of 0.1 * MATHlog(2) to avoid overly sensitive detections
        minimumFloorLimit: float = 0.1 * MATHlog(2.0)
        departureThreshold.clip(
            minimumFloorLimit,
            out = departureThreshold
        )

        # identify indices where the signal breaks below the running threshold
        departureCondition: np.ndarray = (
            (cumulativeRunningWeightedMean[:-1] - normalisedLogSignalPower[1:])
            > departureThreshold
        )
        departureIndices: np.ndarray = np.flatnonzero(departureCondition)

        # determing the primary fallback point index
        firstPositiveIndex: int = len(cumulativeRunningWeightedMean) - 2
        if departureIndices.size > 0:
            firstPositiveIndex = int(departureIndices[0])
        
        precedingNegativeIndex: int = firstPositiveIndex - 1
        if precedingNegativeIndex < 0:
            precedingNegativeIndex = 0
        
        # extract the combined baseline reference level
        meanReferenceBaseline: float = (
            0.5 * (cumulativeRunningWeightedMean[precedingNegativeIndex] 
            + cumulativeRunningWeightedMean[firstPositiveIndex])
        )

        # linear region is firstPositiveIndex-1 -> firstPositiveIndex
        # flat region is firstPositiveIndex+1 -> firstPositiveIndex+outputBinningRatio
        # intersection should be bracketed by logCharacteristicTimescales[firstPositiveIndex-1], logCharacteristicTimescales[firstPositiveIndex+1]
        # define the fitting window upper coundary index
        fittingWindowEndIndex: int = (
            firstPositiveIndex + outputBinningRatio + 1
            )
        
        # extract linear contiguous subsets for regression fitting
        fitLogTimescales: np.ndarray = logCharacteristicTimescales[:fittingWindowEndIndex]
        modeledLogSignalPower: np.ndarray = normalisedLogSignalPower[:fittingWindowEndIndex]

        # extract the uncertainty subset and compute the variance track
        fitPowerUncertainty: np.ndarray = fractionalPowerUncertainty[:fittingWindowEndIndex]
        fitPowerVariance: np.ndarray = fitPowerUncertainty * fitPowerUncertainty
        
        # optomise uncertainty weights
        fitInverseVarianceWeights: np.ndarray = inverseSquaredUncertainty[:fittingWindowEndIndex]

        # compute baseline statistical weights for regression
        weightedPowerSum: float = (
            modeledLogSignalPower * fitInverseVarianceWeights
        ).sum()
        totalInverseVarianceWeight: float = float(fitInverseVarianceWeights.sum())

        # initialise the regression parameters: [baseline, slope, location of break point]
        initialFitParameters: np.ndarray = np.array(
            [meanReferenceBaseline, 0.5, 0.0],
            dtype = floatType
        )

        # optimisation arguments for the residual evaluation function
        optimisationArguments: tuple = (
            fitLogTimescales,
            modeledLogSignalPower,
            fitPowerVariance,
            weightedPowerSum,
            totalInverseVarianceWeight,
            initialFitParameters
        )

        # scalar optimisation pass over the bounded timescale region
        optimisationResult = minimize_scalar(
            evaluateBreakPointResiduals,
            bounds = (
                float(fitLogTimescales.min()),
                float(fitLogTimescales.max() - 1.0e-5)),
                method = 'bounded',
                args = optimisationArguments
        )

        # extract the optimal break point and associated residuals
        minimisedResidualSum: float = float(optimisationResult['fun'])
        optimalLogBreakPoint: float = float(optimisationResult['x'])

        # structural uncertainty extraction and fallback guard against NaN values
        logBreakPointUncertainty: float = float(initialFitParameters[2])
        if np.isnan(logBreakPointUncertainty):
            logBreakPointUncertainty = abs(optimalLogBreakPoint)

        logBreakPointUncertainty = abs(logBreakPointUncertainty)

        # map optomisation parameters back to physical space
        meanReferenceBaseline: float = float(initialFitParameters[0])
        fittedPowerLawSlope: float = 1.0 + 0.5 * float(initialFitParameters[1])
        
        # restore logarithmic break point to physical timescale space
        peakSignalToNoiseUncertainty: float = MATHexp(
            (0.5 * meanReferenceBaseline) + float(fitLogTimescales[0])
        )
        variabilityTimescaleSTD: float = MATHexp(
            (0.5 * meanReferenceBaseline) + optimalLogBreakPoint
        )

        # evaluate classification limits from time bands
        if (
            optimalLogBreakPoint >= fitLogTimescales[1]
            and optimalLogBreakPoint - logBreakPointUncertainty > fitLogTimescales[0]
        ):
            measurementClassificationType = 'measurement'
        else:
            # iteratively refine the uncertainty
            for refinementItterationIndex in range(3):
                # perturb the break point and evaluate residuals
                perturbedResidualSum: float = evaluateBreakPointResiduals(
                    optimalLogBreakPoint + MATHlog(1.0 + logBreakPointUncertainty),
                    *optimisationArguments
                )

                # tune the error bound width based on the local slope
                residualDelta: float = (
                    1.0e-5 + np.abs(perturbedResidualSum - minimisedResidualSum)
                )
                clampedErrorStep: float = min(
                    logBreakPointUncertainty / residualDelta,
                    1.5 * logBreakPointUncertainty
                )
                logBreakPointUncertainty = max(
                    0.5 * logBreakPointUncertainty,
                    clampedErrorStep
                )

            # scale the STD boundary
            varianceScalingFactor: float = (
                signalToNoiseRatioThreshold * logBreakPointUncertainty
                * np.sqrt(float(outputBinningRatio))
            )
            variabilityTimescaleSTD *= (
                MATHexp(fittedPowerLawSlope * MATHlog(1.0 + varianceScalingFactor))
            )

        # map the log-space break points back to physical timescale space
        minimumVariabilityTimescale: float = MATHexp(optimalLogBreakPoint)
        minimumVariabilityTimescaleUncertainty: float = (
            minimumVariabilityTimescale * abs(logBreakPointUncertainty) * np.sqrt(float(outputBinningRatio))
        )

        # fallback calculation if the spectral slope is fixed to the baseline
        if isSpectralSlopeFixedToBaseline:
            # model synthetic log signal power for all timescales
            modeledLogSignalPower: np.ndarray = (
                2.0 * np.log(characteristicTimescales) + meanReferenceBaseline
                - np.log(baselineNoisePowerSpectrum)
            )

            # extract indices where the modeled log signal power is positive
            positiveModelIndices: np.ndarray = np.flatnonzero(modeledLogSignalPower > 0.0)

            if positiveModelIndices.size > 2:
                # identify the bounding indices around the zero crossing
                firstPositiveModelIndex: int = int(positiveModelIndices[0])
                precedingNegativeModelIndex: int = firstPositiveModelIndex - 1

                # extract starting timescale baseline
                spectralIndexSlope: float = characteristicTimescales[precedingNegativeModelIndex]

                # perform linear interpolation to isolate the intercept
                y0: float = float(modeledLogSignalPower[precedingNegativeModelIndex])
                y1: float = float(modeledLogSignalPower[firstPositiveModelIndex])

                if y1 != y0:
                    timeDelta: float = float(
                        characteristicTimescales[firstPositiveModelIndex]
                        - characteristicTimescales[precedingNegativeModelIndex]
                    )
                    spectralIndexSlope -= y0 * (timeDelta / (y1 - y0))

        if (shouldGeneratePlots):
            outputPlotFilename=outputExportFilename+'haar_mod.png'

            # square-root transformation
            safeSignalPower: np.ndarray = np.clip(
                signalPowerSpectrum[validSignalMask],
                0.0,
                None
            )
            transformedSignalPower: np.ndarray = np.sqrt(safeSignalPower)
            transformedUncertainty: np.ndarray = np.divide(
                differentialPowerUncertainty[validSignalMask],
                2.0 * transformedSignalPower,
                out = np.zeros_like(transformedSignalPower),
                where = transformedSignalPower > 0.0
            )
            transformedBaselineNoise: np.ndarray = np.sqrt(
                np.clip(
                    baselineNoisePowerSpectrum[validSignalMask],
                    0.0,
                    None
                )
            )

            # plot labels
            plt.xlabel(
                f"{timeDeltaLabelString} [s]",
                fontsize = 14
            )
            plt.ylabel(
                r'Flux Variation $\sigma_{X,\Delta t}$',
                fontsize = 14
            )

            # plot breakthrough scalar profile
            plt.plot(
                characteristicTimescales[validSignalMask][firstPositiveIndex + 1],
                transformedSignalPower[firstPositiveIndex + 1],
                'mo',
                ms = 10,
                mew = 0,
                label = 'Breakthrough Point'
            )

            # generate piecewise structured fit
            lowerModelPlotDomain: np.ndarray = np.array(
                [minimumBinSizeSeconds / 2.0, minimumVariabilityTimescale],
                dtype = floatType
            )
            upperModelPlotDomain: np.ndarray = np.array(
                [minimumVariabilityTimescale, maximumBinSizeSeconds * 2.0],
                dtype = floatType
            )

            # plot lower model domain
            plt.plot(
                lowerModelPlotDomain,
                lowerModelPlotDomain * MATHexp(meanReferenceBaseline / 2.0),
                'r-',
                alpha = 0.5,
                label = 'Fitted Power-Law Model'
            )

            # plot upper logarithmic power-law decay string path
            logScaleDelta: float = (
                0.5 * meanReferenceBaseline - (fittedPowerLawSlope - 1.0)
                * MATHlog(minimumVariabilityTimescale)
            )
            upperModelPrediction: np.ndarray = np.exp(
                logScaleDelta + fittedPowerLawSlope * np.log(upperModelPlotDomain)
            )
            plt.plot(
                upperModelPlotDomain,
                upperModelPrediction,
                'r-',
                alpha = 0.5
            )

            # plot error bars
            horizontalTimeErrors: np.ndarray = (
                0.5 * (scaledUpperTimeBounds - scaledLowerTimeBounds)[validSignalMask]
            )

            plt.errorbar(
                characteristicTimescales[validSignalMask],
                transformedSignalPower,
                yerr = transformedUncertainty,
                xerr = horizontalTimeErrors,
                fmt = 'bo',
                capsize = 0,
                linestyle = 'None',
                markersize = 3,
                label = 'Signal Power Spectrum'
            )

            # background grid
            peakSignalIndex: int = int(signalPowerSpectrum[validSignalMask].argmax())
            xAnchor: float = characteristicTimescales[validSignalMask][peakSignalIndex]
            yAnchor: float = transformedSignalPower[peakSignalIndex]

            gridTimescaleBounds: np.ndarray = np.array(
                [minimumBinSizeSeconds / 2.0, maximumBinSizeSeconds * 2.0],
                dtype = floatType
            )
            baseGridSlope: np.ndarray = (
                yAnchor * gridTimescaleBounds / xAnchor
            )

            for gridLineIndex in range(-20, 20):
                plt.plot(
                    gridTimescaleBounds,
                    baseGridSlope * 2.0 ** gridLineIndex,
                    'k:',
                    alpha = 0.5
                )
            
            # find the limits for the plot axes
            plt.xlim(
                (characteristicTimescales[validSignalMask].min() / 4.0,
                characteristicTimescales[validSignalMask].max() * 1.5)
            )
            plt.ylim(
                (np.nanmin(transformedSignalPower) / 2.0,
                np.nanmax(transformedSignalPower) * 1.5)
            )

            import matplotlib.ticker as ticker


            plt.xscale('log')
            plt.yscale('log')

            plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))


            # labels
            signalToNoiseLabel: str = rf"$\Delta t_{{\rm snr}}=$ {peakSignalToNoiseRatio:.4f}"
            spectralSlopeLabel: str = rf"$t_{{\beta}}=$ {spectralIndexSlope:.4f}"

            if measurementClassificationType == 'limit':
                upperLimitThreshold: float = (
                    minimumVariabilityTimescale
                    + signalToNoiseRatioThreshold * minimumVariabilityTimescaleUncertainty
                )
                finalTimescaleLabel: str = rf"$\Delta t_{{\rm min}}< {upperLimitThreshold:.4f} \pm {minimumVariabilityTimescaleUncertainty:.4f}$"
            else:
                finalTimescaleLabel: str = rf"$\Delta t_{{\rm min}}= {minimumVariabilityTimescale:.4f} \pm {minimumVariabilityTimescaleUncertainty:.4f}$"
            
            plt.title(finalTimescaleLabel)

            # overlay baseline error ceiling arrows if present
#            if any(backgroundNoiseMask):
#                clampedBackgroundNoise: np.ndarray = signalPowerSpectrum[backgroundNoiseMask].clip(0.0)
#                backgroundUpperBounds: np.ndarray = np.sqrt(
#                    clampedBackgroundNoise
#                    + signalToNoiseRatioThreshold * differentialPowerUncertainty[backgroundNoiseMask]
#                )
#                plt.plot(
#                    characteristicTimescales[backgroundNoiseMask],
#                    backgroundUpperBounds,
#                    'bv',
#                    label = 'Background Noise Upper Bound',
#                )
            plt.legend(
                loc = 'lower right',
                fontsize = 10
            )

            plt.tight_layout()

            # save plot and close figure
            plt.savefig(
                outputPlotFilename,
                dpi = 300
            )
            plt.close()

        # convert output to a strict upper bound if no breakthrough is confirmed
        if measurementClassificationType == 'limit':
            minimumVariabilityTimescale += (
                signalToNoiseRatioThreshold * minimumVariabilityTimescaleUncertainty
            )
            minimumVariabilityTimescaleUncertainty = 0.0

#    # print the final results if requested commented out for now as later functions call this hundreds of times
#    logger.debug(
#        f"{outputExportFilename} "
#        f"T_snr={peakSignalToNoiseRatio:f} "
#        f"T_beta={spectralIndexSlope:f} "
#        f"T_min={minimumVariabilityTimescale:f} +/- {minimumVariabilityTimescaleUncertainty:f}"
#    )

    # return the computed parameters as a tuple
    return (
        peakSignalToNoiseRatio,
        spectralIndexSlope,
        minimumVariabilityTimescale,
        minimumVariabilityTimescaleUncertainty,
        fittedPowerLawSlope,
        peakSignalToNoiseUncertainty,
        variabilityTimescaleSTD
    )


def _runSinglerealisation(args):
    """
    Worker function for a single Monte Carlo realisation.

    Returns both the MVT and its uncertainty, because haarPowerMod uses them to signal the outcome:
        * failed realisation (not enough significant signal): MVT = 0
        * upper limit (no confirmed linear rise before the break): uncertainty = 0
        * valid measurement: MVT > 0 and uncertainty > 0
    """
    binCounts, binErrors, resampleErrors, binSizeSeconds = args
    sampledCounts = np.random.poisson(lam=binCounts)
    iterationErrors = np.sqrt(sampledCounts) if resampleErrors else binErrors
    (_, _, minimumVariabilityTimescale, minimumVariabilityTimescaleUncertainty, _, _, _) = haarPowerMod(
        sampledCounts, iterationErrors, minimumBinSizeSeconds=binSizeSeconds
    )
    return minimumVariabilityTimescale, minimumVariabilityTimescaleUncertainty


def estimateMvtDistribution(
        binCounts,
        binErrors,
        haarPowerMod,
        binSizeSeconds,
        nRealisations=300,
        resampleErrors=False,
        nWorkers=None) -> dict:
    """
    Run the Monte Carlo realisations and return the full MVT distribution, following Bala et al. (2026),
    Section 2.3: only valid measurements (MVT > 0 with a non-zero uncertainty) are kept; failed
    realisations and upper limits are counted separately and excluded from the summary statistics.

    Parameters are the same as estimateMvtUncertainty.

    Returns
    -------
    dict with keys:
        mvtValues, mvtUncertainties : np.ndarray
            Raw MVT (seconds) and uncertainty from every realisation.
        validMask, limitMask, failedMask : np.ndarray (bool)
            Which realisations were valid measurements, upper limits, or failures.
        nRealisations, nValid, nLimit, nFailed : int
        successRate : float
            nValid / nRealisations (reported as "Success" in Bala et al. Table 3).
        median, p16, p84 : float
            Median and 16th/84th percentiles of the valid MVTs (seconds). NaN if nValid == 0.
        mvt84 : float
            The 84th percentile, used by Bala et al. as the upper-limit value.
        mean, std : float
            Mean and standard deviation of the valid MVTs (seconds). NaN if too few values.
        meanAll, stdAll : float
            Mean and standard deviation of ALL realisations, including failures and limits
            (the original behaviour, kept for comparison).
    """
    binCounts = np.asarray(binCounts)
    binErrors = np.asarray(binErrors)
    mvtValues = np.empty(nRealisations)
    mvtUncertainties = np.empty(nRealisations)

    args = [(binCounts, binErrors, resampleErrors, binSizeSeconds)] * nRealisations

    with ProcessPoolExecutor(max_workers=nWorkers) as executor:
        futures = {executor.submit(_runSinglerealisation, arg): i for i, arg in enumerate(args)}
        progressBar = tqdm(as_completed(futures), total=nRealisations, desc="Monte Carlo MVT sampling", unit="realisation")
        for future in progressBar:
            i = futures[future]
            mvtValues[i], mvtUncertainties[i] = future.result()
            progressBar.set_postfix(mvt=f"{mvtValues[i]:.4f}")

    # classify every realisation
    failedMask = ~np.isfinite(mvtValues) | (mvtValues <= 0.0)
    limitMask = ~failedMask & ~(mvtUncertainties > 0.0)
    validMask = ~failedMask & ~limitMask
    validValues = mvtValues[validMask]

    nValid = int(validMask.sum())
    nLimit = int(limitMask.sum())
    nFailed = int(failedMask.sum())

    if nValid > 0:
        p16, median, p84 = (float(v) for v in np.percentile(validValues, [16, 50, 84]))
        mean = float(np.mean(validValues))
    else:
        p16 = median = p84 = mean = float("nan")
    std = float(np.std(validValues, ddof=1)) if nValid > 1 else float("nan")

    logger.info(
        f"Monte Carlo MVT: {nValid} valid, {nLimit} upper limits, {nFailed} failed "
        f"out of {nRealisations} (success rate {nValid / nRealisations:.1%})"
    )

    return {
        "mvtValues": mvtValues,
        "mvtUncertainties": mvtUncertainties,
        "validMask": validMask,
        "limitMask": limitMask,
        "failedMask": failedMask,
        "nRealisations": int(nRealisations),
        "nValid": nValid,
        "nLimit": nLimit,
        "nFailed": nFailed,
        "successRate": nValid / nRealisations,
        "median": median,
        "p16": p16,
        "p84": p84,
        "mvt84": p84,
        "mean": mean,
        "std": std,
        "meanAll": float(np.mean(mvtValues)),
        "stdAll": float(np.std(mvtValues, ddof=1)) if nRealisations > 1 else float("nan"),
    }


def estimateMvtUncertainty(
        binCounts,
        binErrors,
        haarPowerMod,
        binSizeSeconds,
        nRealisations=300,
        resampleErrors=False,
        nWorkers=None,
        statistic="median"):
    """
    Estimate the MVT and its uncertainty using Monte Carlo sampling,
    with realisations computed in parallel. Wrapper around estimateMvtDistribution that keeps the
    original three-value return so existing callers keep working.

    Parameters
    ----------
    binCounts : array-like
        Observed counts in each bin (treated as Poisson means).
    binErrors : array-like
        Observational error for each bin. Either held fixed across all realisations (geometric instrument errors) or recomputed as sqrt(sampledCounts) each iteration (Poisson-derived errors).
    haarPowerMod : callable
        The GB14 method function that calculates the MVT for a given set of bin counts. Expected return order: (peakSignalToNoiseRatio, spectralIndexSlope, minimumVariabilityTimescale, minimumVariabilityTimescaleUncertainty, fittedPowerLawSlope, peakSignalToNoiseUncertainty, variabilityTimescaleSTD)
    binSizeSeconds : float
        Width of each bin in binCounts, in seconds. Used to convert timescales from bins to seconds.
    nRealisations : int, optional
        Number of Monte Carlo realisations (default: 300).
    resampleErrors : bool, optional
        If False (default), binErrors are held fixed across all realisations (geometric instrument i.e. Swift). If True, errors are recomputed as sqrt(sampledCounts) each iteration (Poisson-derived instrument).
    nWorkers : int or None, optional
        Number of parallel worker processes. Defaults to None, which lets ProcessPoolExecutor use the number of available CPU cores.
    statistic : str, optional
        "median" (default, Bala et al. 2026): median of the valid MVTs, error = half the 16th-84th percentile range.
        "mean": mean and standard deviation of the valid MVTs.
        "meanAll": original behaviour, mean and standard deviation of all realisations including failures and limits.

    Returns
    -------
    mvtCentral : float
        Central MVT value (seconds). 0.0 if no realisation gave a valid measurement.
    mvtError : float
        Uncertainty on the MVT (seconds). 0.0 if no realisation gave a valid measurement.
    mvtValues : np.ndarray
        The valid MVT values (seconds). All values for statistic="meanAll".
    """
    result = estimateMvtDistribution(
        binCounts,
        binErrors,
        haarPowerMod,
        binSizeSeconds,
        nRealisations=nRealisations,
        resampleErrors=resampleErrors,
        nWorkers=nWorkers
    )

    if statistic == "meanAll":
        return result["meanAll"], result["stdAll"], result["mvtValues"]

    validValues = result["mvtValues"][result["validMask"]]
    if result["nValid"] == 0:
        logger.warning("No valid MVT measurements in any Monte Carlo realisation; returning 0.")
        return 0.0, 0.0, validValues

    match statistic:
        case "median":
            mvtCentral = result["median"]
            mvtError = 0.5 * (result["p84"] - result["p16"])
        case "mean":
            mvtCentral = result["mean"]
            mvtError = result["std"] if np.isfinite(result["std"]) else 0.0
        case _:
            raise ValueError(f"Unknown statistic: {statistic}. Use 'median', 'mean' or 'meanAll'.")

    return mvtCentral, mvtError, validValues


def processSingleWindowWorker(
    windowStartBin: int,
    windowEndBin: int,
    windowCountsView: np.ndarray,
    windowErrorsView: np.ndarray,
    absoluteStartTimeSeconds: float,
    binSizeInSeconds: float,
    windowDurationBins: int
) -> dict:
    """Independent worker task that runs MVT denoising on a single window."""
    
    # calculate absolute physical timestamps
    windowCenterTimeSeconds: float = (
        absoluteStartTimeSeconds + (windowStartBin + windowDurationBins / 2.0) * binSizeInSeconds
        )
    windowStartTimeSeconds: float = (
        absoluteStartTimeSeconds + windowStartBin * binSizeInSeconds
    )
    windowEndTimeSeconds: float = (
        absoluteStartTimeSeconds + windowEndBin * binSizeInSeconds
    )
    
    try:
        # get the MVT result for the current window
        result: tuple = haarPowerMod( # TODO this should be changed to the MCMC version of the MVT calculation
            windowCountsView,
            windowErrorsView,
            minimumBinSizeSeconds = binSizeInSeconds
            )
        
        # Extract minimumVariabilityTimescale (index 2) and minimumVariabilityTimescaleUncertainty (index 3)
        minimumVariabilityTimescaleSeconds: float = float(result[2])
        minimumVariabilityTimescaleUncertaintySeconds: float = float(result[3])

        if (
            not np.isfinite(minimumVariabilityTimescaleSeconds)
            or minimumVariabilityTimescaleSeconds <= 0.0
            or minimumVariabilityTimescaleSeconds > windowDurationBins * binSizeInSeconds
        ):
            minimumVariabilityTimescaleMs = np.nan
        else:
            minimumVariabilityTimescaleMs = round(minimumVariabilityTimescaleSeconds * 1000.0, 3)

        if (
            not np.isfinite(minimumVariabilityTimescaleUncertaintySeconds)
            or minimumVariabilityTimescaleUncertaintySeconds < 0.0
        ):
            minimumVariabilityTimescaleUncertaintyMs = np.nan
        else:
            minimumVariabilityTimescaleUncertaintyMs = round(minimumVariabilityTimescaleUncertaintySeconds * 1000.0, 3)
    except Exception as calculationException:
        minimumVariabilityTimescaleMs: float = np.nan
        minimumVariabilityTimescaleUncertaintyMs: float = np.nan
        logger.warning(f"Error during MVT calculation for window at {windowCenterTimeSeconds}s: {calculationException}")
        
    # return an isolated dictionary
    return {
        'centerTimeSeconds': windowCenterTimeSeconds,
        'startTimeSeconds': windowStartTimeSeconds,
        'endTimeSeconds': windowEndTimeSeconds,
        'mvtMs': minimumVariabilityTimescaleMs,
        'mvtErrMs': minimumVariabilityTimescaleUncertaintyMs,
    }


def timeResolvedMVT(
    counts: np.ndarray, 
    errors: np.ndarray, 
    source: str,
    absoluteStartTimeSeconds: float = 0.0,
    timeWindowSizeSeconds: float = None,
) -> list:
    """
    Calculates the Minimum Variability Timescale (MVT) using a sliding time window,
    correctly handling an absolute start time.

    Args:
        counts (np.ndarray): The full binned light curve data.
        errors (np.ndarray): The errors for the counts.
        source (str): The source of the data.
        absoluteStartTimeSeconds (float): The absolute start time of the counts array. Defaults to 0.0.
        timeWindowSizeSeconds (float, optional): The sliding window size in seconds. Defaults to the timeWindowSize value in config.yaml.

    Note:
        haarPowerMod settings are taken from config.yaml via haarPowerMod's defaults.

    Returns:
        list: A list of dictionaries containing the results for each time window.
    """
    # get configuration settings from the config file
    binSizeInSeconds: float = getInitialBinSize(source)
    if timeWindowSizeSeconds is None:
        timeWindowSizeSeconds = float(config.mvtAnalysisConfig.timeResolvedMVTSettings.timeWindowSize)
    windowDurationSeconds: float = timeWindowSizeSeconds
    stepDurationSeconds: float = float(config.mvtAnalysisConfig.timeResolvedMVTSettings.stepDurationSeconds)

    # convert physical time into integer bin metrics
    windowDurationBins = int(
        round(windowDurationSeconds / binSizeInSeconds)
    )
    stepDurationBins = int(
        round(stepDurationSeconds / binSizeInSeconds)
    )

    # guard against window size exceeding the light curve length
    if windowDurationBins > len(counts):
        logger.warning("Window size is larger than the light curve. No analysis performed.")
        return []
    
    # initialise a list to hold the results for each window
    windowedAnalysisResults = []

    # calculate the window slice arguments
    windowTasksList: list = []
    windowStartBin: int = 0

    while windowStartBin + windowDurationBins <= len(counts):
        windowEndBin: int = windowStartBin + windowDurationBins
        
        # create views for the current window
        windowCountsView: np.ndarray = counts[windowStartBin:windowEndBin]
        windowErrorsView: np.ndarray = errors[windowStartBin:windowEndBin]
        
        # append the task to the list for processing
        windowTasksList.append((
            windowStartBin,
            windowEndBin,
            windowCountsView,
            windowErrorsView,
            absoluteStartTimeSeconds,
            binSizeInSeconds,
            windowDurationBins
        ))
        
        # move to the next window
        windowStartBin += stepDurationBins

    # spawn a pool of workers to process the windows in parallel
    cpuCoreCount: int = cpu_count() or 4
    windowedAnalysisResults = []

    with ProcessPoolExecutor(
        max_workers=cpuCoreCount
        ) as processPool:
        # submit all window tasks to the pool boys
        futuresTracker = [
            processPool.submit(
                processSingleWindowWorker,
                *taskArgs
            ) for taskArgs in windowTasksList
        ]

        # collect results as they complete
        for completedFuture in as_completed(futuresTracker):
            windowedAnalysisResults.append(completedFuture.result())

    # asynchronous processing can return windows out of order, so sort by center time
    windowedAnalysisResults.sort(key=lambda windowData: windowData['centerTimeSeconds'])

    return windowedAnalysisResults


def csvReader(
        filepath: str
    ) -> tuple:
    """
    Reads a CSV file containing photon counts and errors, returning them as numpy arrays.
    """
    filePath: str = filepath + "/photonCounts.csv"
    data: np.ndarray = np.genfromtxt(filePath, delimiter=',', skip_header=1)
    counts: np.ndarray = data[:, 0]
    errors: np.ndarray = data[:, 1]
    return counts, errors


if __name__ == "__main__":
    # get the photon counts and errors from the CSV file
    photonCounts, photonErrors = csvReader("/home/derekpinkett/coding/MVTTools/data/processed/GRB080319B")

    # find the haar power model results for the photon counts and errors
    results = haarPowerMod(
        photonCounts,
        photonErrors,
        shouldGeneratePlots=True,
        outputExportFilename="/home/derekpinkett/coding/MVTTools/plots/GRB080319B_haar_power_mod"
    )
