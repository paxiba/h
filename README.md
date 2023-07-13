from google.colab import drive
drive.mount('/content/drive')
!pip install googletrans==3.1.0a0
import os
import subprocess
from googletrans import Translator
import tensorflow as tf
from PIL import Image
from urllib import request
from io import BytesIO
import matplotlib.pyplot as plt
import json
from keras.preprocessing.text import tokenizer_from_json
import numpy as np
import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import sequence
## ACTUALIZAR LA RUTA A LA CARPETA DE DRIVE
path_drive = '/content/drive/MyDrive/Flickr8K'
os.chdir(path_drive)
print('Directorio actual es:', subprocess.check_output('pwd').decode("utf-8"))
if not(os.path.exists('image-captioning')):
    os.system('git clone https://github.com/kahotsang/image-captioning.git ')

os.chdir('image-captioning')
nfiles = ["build_model/build.py"]
for nn in nfiles:
    fin = open(nn, "rt")
    data = fin.read()
    data = data.replace('keras.preprocessing.image.load_img','keras.utils.load_img')
    data = data.replace('keras.preprocessing.image.img_to_array','keras.utils.img_to_array')
    data = data.replace('keras.preprocessing.sequence.pad_sequences','keras.utils.pad_sequences')
    fin.close()
    fin = open(nn, "wt")
    fin.write(data)
    fin.close()
    # Cargar tokenizer
with open('tokenizer.json', 'r') as f:
    tokenizer_json = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_json)

# Cargar modelo
model = keras.models.load_model('./sample_model.h5')

# Obtener el tamaño del vocabulario
vocab_size = tokenizer.num_words

# Longitud máxima de la secuencia de subtítulos
max_length = 37
# Función para extraer características de la imagen
def feature_extractions_from_url(url):
    res = request.urlopen(url).read()
    image = Image.open(BytesIO(res)).resize((299, 299))
    arr = np.array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    features = modelIM.predict(arr)
    return features, image
!pip install tensorflow
import tensorflow as tf

# Cargar modelo InceptionV3
modelIM = tf.keras.applications.InceptionV3()
modelIM = keras.models.Model(inputs=modelIM.input, outputs=modelIM.layers[-2].output)
# URL de la imagen que deseas probar
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzrt_qNFCb91Pf1MWSvHXqt5qo7Qp-K7Emkg&usqp=CAU"

def sample_caption(model, tokenizer, max_length, vocab_size, features):
    start_word = tokenizer.word_index['<start>']
    end_word = tokenizer.word_index['<end>']

    input_seq = np.zeros((1, max_length))
    output_seq = []

    for i in range(max_length):
        prediction = model.predict([features, input_seq])
        prediction = np.argmax(prediction)
        output_seq.append(prediction)
        input_seq[0, i] = prediction

        if prediction == end_word:
            break

    caption = tokenizer.sequences_to_texts([output_seq])[0]
    caption = caption.replace(' <end>', '')

    return caption
import os

tokenizer_file = 'tokenizer.json'
if os.path.exists(tokenizer_file):
    print("El archivo tokenizer.json existe en el directorio actual.")
else:
    print("El archivo tokenizer.json no se encuentra en el directorio actual.")
import json

tokenizer_file = 'tokenizer.json'
with open(tokenizer_file, 'r') as f:
    tokenizer_json = json.load(f)

print(tokenizer_json)
# Generar las características de la imagen
feat, img = feature_extractions_from_url(url)

# Generar el subtítulo de la imagen
caption = sample_caption(model, tokenizer, max_length, vocab_size, feat)

# Imprimir la imagen y el subtítulo
plt.imshow(img)
plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
