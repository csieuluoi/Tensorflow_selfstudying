import tensorflow.keras as keras
import librosa 
import numpy as np 


MODEL_PATH = 'model.h5'
# SAMPLES_TO_CONSIDER = 22050
class _Keyword_Spotting_Service:
	model = None
	_mappings =  [
		"down",
		"off",
		"on",
		"no",
		"yes",
		"stop",
		"up",
		"right",
		"left",
		"go"
	]
	_instance = None

	def predict(self, file_path):
		"""
		:param file_path (str): Path to audio file to predict
		:return predicted_keyword (str): Keyword predicted by the model
		"""

		# extract MFCC
		MFCCs = self.preprocess(file_path)
		if MFCCs is not None:
			MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
			# get the predicted label

			predictions = self.model.predict(MFCCs)
			preidcted_index = np.argmax(predictions)
			predicted_keyword = self._mappings[preidcted_index]


		else: 
			predicted_keyword = None

		return predicted_keyword


	def preprocess(self, file_path, SAMPLES_TO_CONSIDER = 22050, num_mfcc = 13, n_fft = 2048, hop_length = 512):
		"""Extract MFCCs from audio file.
		:param file_path (str): Path of audio file
		:param num_mfcc (int): # of coefficients to extract
		:param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
		:param hop_length (int): Sliding window for STFT. Measured in # of samples
		:return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
		"""

		# load audio file
		signal, sample_rate = librosa.load(file_path)

		if len(signal) >= SAMPLES_TO_CONSIDER:
			# ensure consistency of the length of the signal
			signal = signal[:SAMPLES_TO_CONSIDER]

			# extract MFCCs
			MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, 
				hop_length=hop_length)
			MFCCs = MFCCs.T
		else:
			MFCCs = None
		return MFCCs

def Keyword_Spotting_Service():
	"""Factory function for Keyword_Spotting_Service class.
	:return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
	"""

	# ensure that we only have 1 instance of KSS
	if _Keyword_Spotting_Service._instance is None:
		_Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
		_Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)

	return _Keyword_Spotting_Service._instance


if __name__ == "__main__":

	# create 2 instances of the keyword spotting service
	kss = Keyword_Spotting_Service()
	kss1 = Keyword_Spotting_Service()

	# check that different instances of the keyword spotting service point back to the object (sigleton)

	assert kss is kss1

	# make a prediction
	keyword = kss.predict("test/down.wav")
	print(keyword)