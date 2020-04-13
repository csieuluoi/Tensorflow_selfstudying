import requests
import os

URL = "http://127.0.0.1:5000/predict"

TEST_AUDIO_PATH = "D:/python/Audio-Deeplearning-From-Design-To-Deployment/test/"



if __name__ == '__main__':
	

	for filename in os.listdir(TEST_AUDIO_PATH):
		test_audio_file_path = os.path.join(TEST_AUDIO_PATH, filename)
		# open file

		audio_file = open(test_audio_file_path, 'rb')
		# pakages stuff to send and perform POST request

		values = {'file': (test_audio_file_path, audio_file, "audio/wav")}
		response = requests.post(URL, files = values)
		data = response.json()

		if data['keyword'] is not None:

			print("\nPredicted keyword for the file named {} is : {}".format(filename, data['keyword']))
		else: 
			print("the input audio file is too short")