import librosa
import os
import json
from tqdm import tqdm

def preprocess_dataset(dataset_path, json_path, num_mfcc = 13, n_fft = 2048, hop_length = 512):
	"""Extracts MFCCs from music dataset and saves them into a json file.

	:param dataset_path (str): Path to dataset
	:param json_path (str): Path to json file used to save MFCCs
	:param num_mfcc ( int): Number of coefficients to extract
	:param n_nfft (int): Interval we consider to apply FFT. Measures in # of samples
	:param hop_length (int): Sliding window for FFT. Measures in # of samples
	:return:
	"""

	data = {
		'mapping': [],
		'labels': [],
		'MFCCs': [],
		'files': []
	}

	# loop through all sub-dirs
	total_samples = 0
	valid_samples = 0
	for i, (dirpath, dirname, filenames) in tqdm(enumerate(os.walk(dataset_path))):

		# ensure we're at sub-folder level
		if dirpath is not dataset_path:
			# save label (i.e., sub-folder name) in the mapping
			label = dirpath.partition('speech_commands_subset')[-1][1:]

			data['mapping'].append(label)
			print("\nProcessing: '{}'".format(label))
			print("number of files for each class: ", len(filenames))
			# process all audio files
			for f in filenames:
				total_samples+=1
				file_path = os.path.join(dirpath, f)

				# load audio file and slice it to ensure length consistency among different files
				signal, sample_rate = librosa.load(file_path)
				# print(signal.shape)
				# print(type(signal[0]))

				# drop audio files with less than pre-decided number of samples
				if len(signal) >= SAMPLES_TO_CONSIDER:
					valid_samples+=1
					# ensure consistency of the length of the signal
					signal = signal[:SAMPLES_TO_CONSIDER]

					# extract MFCCs
					MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc = num_mfcc, n_fft = n_fft, 
						hop_length = hop_length) 
					# print(MFCCs.shape)
					# print(type(MFCCs[0,0]))

					# store data for analysed track
					data['MFCCs'].append(MFCCs.T.tolist())
					data['labels'].append(i-1)
					# data['files'].append(file_path)

					# print("{}: {}".format(file_path, i-1))

					# if valid_samples == 20:
					# 	valid_samples =0
					# 	break
					# break
	print("\ntotal samples: ", total_samples)
	print("\nvalid_samples: ", valid_samples)
	with open(json_path, 'w') as fp:
		json.dump(data, fp, indent = 4)
	
	# return data

if __name__ == '__main__':
	DATASET_PATH = 'D:/python/Data/Audio/speech_commands_subset'
	JSON_PATH = 'data.json'
	SAMPLES_TO_CONSIDER = 22050

	preprocess_dataset(DATASET_PATH, JSON_PATH, SAMPLES_TO_CONSIDER)
