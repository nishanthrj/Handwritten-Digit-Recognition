import os
import string
import random
import requests
from pathlib import Path
import numpy as np
from decouple import config
from PIL import Image, ImageOps


API_KEY = config('API_KEY')

def random_name_generator(n):
	return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))


def use_model(img):
	token_response = requests.post('https://iam.cloud.ibm.com/identity/token', 
								data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})

	mltoken = token_response.json()["access_token"]

	header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

	payload_scoring = {"input_data": [{"fields": [], "values": [img.tolist()]}]}

	response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/ae43e79c-1fbc-450a-b0b4-9a54c451033b/predictions', 
									json=payload_scoring,  headers=header)
 
 
	return (response_scoring.json()['predictions'][0]['values'][0][1],
         	list(map(lambda x: round(x*100, 2), response_scoring.json()['predictions'][0]['values'][0][2])))


def recognize(image):
	img = Image.open(image).convert("L")
	img_name = random_name_generator(10) + '.jpg'
	if not os.path.exists("./static/data/"): 
		os.mkdir(os.path.join('./static/', 'data'))
	img.save(Path(f"./static/data/{img_name}"))
	img = ImageOps.grayscale(img)
	img = ImageOps.invert(img)
	img = img.resize((28, 28))
	img2arr = np.array(img)
	img2arr = img2arr / 255.0
	img2arr = img2arr.reshape(28, 28, 1)
	results  = use_model(img2arr)
	best = results[0]
	others = results[1]
	values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	pred = list(zip(values, others))
	best = pred.pop(best)

	return best, pred, img_name
