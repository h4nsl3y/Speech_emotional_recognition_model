import flask
import numpy as np
import librosa
import tensorflow as tf
import base64
import os
from codecs import encode
import soundfile as sf
from glob import glob
from flask import Flask, request, jsonify, json

from pydub import AudioSegment

app = Flask(__name__)

global model
speech_emotional_recognition_model = tf.keras.models.load_model(
    "C:\\Users\\hansley\\Desktop\\API\\model\\model_multi_input_1.h5")
music_emotional_recognition_model = tf.keras.models.load_model(
    "C:\\Users\\hansley\\Desktop\\API\\model\\music_model.h5")


# model = tf.keras.models.load_model('multi_merged_model_1.h5')
# model = tf.keras.models.load_model('old_model\\Model\\CNN_LSTM1.h5')

# process
# Purpose - extract features and return list of 4 arrays
# Input - path of audio
# Output - list of array 
def process(inputs):
    X = []
    res_1 = np.array([])
    res_2 = np.array([])
    res_3 = np.array([])
    res_4 = np.array([])
    result = ""
    try:
        data, sample_rate = librosa.load(inputs)
        # ZCR
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        res_1 = np.hstack((res_1, zcr))  # stacking horizontally
        # Root Mean Square Value
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        res_1 = np.hstack((res_1, rms))  # stacking horizontally
        # Chroma_stft
        stft = np.abs(librosa.stft(data))
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        res_2 = np.hstack((res_2, chroma_stft))  # stacking horizontally
        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        res_3 = np.hstack((res_3, mfcc))  # stacking horizontally
        # MelSpectogram
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        res_4 = np.hstack((res_4, mel))  # stacking horizontally

    except:
        x1 = [0.1] * 2
        x2 = [0.1] * 12
        x3 = [0.1] * 20
        x4 = [0.1] * 128

        res_1 = np.hstack((res_1, x1))
        res_2 = np.hstack((res_2, x2))
        res_3 = np.hstack((res_3, x3))
        res_4 = np.hstack((res_4, x4))

    res_1 = np.expand_dims(res_1, axis=0)
    res_2 = np.expand_dims(res_2, axis=0)
    res_3 = np.expand_dims(res_3, axis=0)
    res_4 = np.expand_dims(res_4, axis=0)

    result = res_1
    X.append(result)
    result = res_2
    X.append(result)
    result = res_3
    X.append(result)
    result = res_4
    X.append(result)

    return X


def split_audio(file_loc, format):
    temp = 'C:\\Users\\hansley\\Desktop\\API\\audio\\temp'
    y, sr = librosa.load(file_loc, sr=None)
    y = librosa.to_mono(y)
    segment_length = sr * 2
    num_sections = int(np.ceil(len(y) / segment_length))
    split = []
    for i in range(num_sections):
        t = y[i * segment_length: (i + 1) * segment_length]
        split.append(t)

    for i in range(num_sections):
        recording_name = os.path.basename(file_loc[:-4])
        out_file = f"{recording_name}_{str(i)}." + format
        sf.write(os.path.join(temp, out_file), split[i], sr)


# evaluate( speech )
# Purpose - Use neural network model to evaluate features extracted
# Input - path of audio
# Output  - result of evaluation
def evaluateAudio(input):
    data = process(input)
    data = speech_emotional_recognition_model.predict(data)
    return (data)

# evaluate( music )
# Purpose - Use neural network model to evaluate features extracted
# Input - path of audio
# Output  - result of evaluation
def evaluateMusic(input):
    data = process(input)
    data = music_emotional_recognition_model.predict(data)
    return (data)

def mp3_to_wav(file):
    file_1 = str(file)
    out = (file_1.split("\\"))
    out_file = out[-1]
    out_file = "audio\\temp\\wav\\" + (out_file) + ".wav"
    audio = AudioSegment.from_mp3(file)
    audio.export(out_file, format="wav")

def m4a_to_wav(file):
    file_1 = str(file)
    out = (file_1.split("\\"))
    out_file = out[-1]
    out_file = "audio\\temp\\wav\\" + (out_file) + ".wav"
    audio = AudioSegment.from_m4a(file)
    audio.export(out_file, format="wav")

@app.route("/recieve", methods=["POST", "GET"])
def recieved():
    # file = flask.request.files['audio']
    global result
    if flask.request.method == "POST":

        content = request.get_json(silent=True)
        # print(content["type"])
        type = content["type"]
        # print(content["format"])
        format = content["format"]
        ans = base64.b64decode(bytes(content["audio"], 'utf-8'))

        with open("audio\\audioToSave." + format, "wb") as fh:
            fh.write(ans)

        if type == "recording":
            anger, calm, disgust, fear, happiness, sadness = 0, 0, 0, 0, 0, 0

            # file processing and evaluation file
            data = evaluateAudio("audio\\audioToSave." + format)

            data_values = 0
            for values in data[0]:
                data_values = data_values + values

            anger = round(((data[0][0] / data_values) * 100), 2)
            calm = round(((data[0][1] / data_values) * 100), 2)
            disgust = round(((data[0][2] / data_values) * 100), 2)
            fear = round(((data[0][3] / data_values) * 100), 2)
            happiness = round(((data[0][4] / data_values) * 100), 2)
            sadness = round(((data[0][5] / data_values) * 100), 2)

            result = {
                "anger": str(anger),
                "calm": str(calm),
                "disgust": str(disgust),
                "fear": str(fear),
                "happiness": str(happiness),
                "sadness": str(sadness),
            }

        if type == "file":
            fierce, frightful, happy, peaceful, sad = 0, 0, 0, 0, 0
            data_values = 0
            fierce, frightful, happy, peaceful, sad = 0, 0, 0, 0, 0
            category_lst = ["fierce", "frightful", "happy", "peaceful", "sad"]
            value_count = 0
            print(format)

            split_audio("audio\\audioToSave." + format, format)
            file_dir = glob("audio\\temp\\*.*")  # + format

            if(format == "mp3"):
                for file in file_dir:
                    mp3_to_wav(file)
                    os.remove(file)

            if(format == "m4a"):

                for file in file_dir:
                    m4a_to_wav(file)
                    os.remove(file)

            file_dir = glob("C:\\Users\\hansley\\Desktop\\API\\audio\\temp\\wav\\*.wav")
            for file in file_dir:
                # print("preprocess output : ")
                # print(data

                data = music_emotional_recognition_model.predict(process(file))
                # data = data.numpy()
                print(data)

                data_values = (data[0][0] + data[0][1] + data[0][2] + data[0][3] + data[0][4])

                # print(data[0][0], data[0][1], data[0][2], data[0][3], data[0][4])

                value_count = value_count + 1
                print(data[0][0])

                fierce = fierce + (round(((data[0][0] / data_values) * 100), 2))
                frightful = frightful + (round(((data[0][1] / data_values) * 100), 2))
                happy = happy + (round(((data[0][2] / data_values) * 100), 2))
                peaceful = peaceful + (round(((data[0][3] / data_values) * 100), 2))
                sad = sad + (round(((data[0][4] / data_values) * 100), 2))
                os.remove(file)

            average_count = [
                (fierce / value_count),
                (frightful / value_count),
                (happy / value_count),
                (peaceful / value_count),
                (sad / value_count)
            ]

            print(average_count)
            print(category_lst[average_count.index(max(average_count))])
            result = {
                "category": category_lst[average_count.index(max(average_count))],
            }
            print(result)

        return result


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(debug=True,
            host='0.0.0.0',
            port=9000,
            threaded=True)
