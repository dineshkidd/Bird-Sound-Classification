import streamlit as st
import numpy as np
import soundfile
import matplotlib.pyplot as plt
import librosa
from librosa.display import specshow
from keras.models import load_model
from keras.preprocessing import image

# converting to spectrogram
# path="song.wav"
st.markdown(
    """
    <style>
    .reportview-container {
        margin-top:-50px;
    }
    h3{
        color:blue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """<h1 style="text-align:center;padding-botton:5000px;">Bird Sound Classification</h1>""",
    unsafe_allow_html=True,
)

st.set_option('deprecation.showfileUploaderEncoding', False)
file_buffer = st.file_uploader("Choose a WAV File...", type="wav")


def run(path):
    # audio_file = open(path, 'r')
    # audio_bytes = audio_file.read()
    # st.audio(audio_bytes)
    if path==None:
        st.error('Please select an wav file....')
        return 
    else:
        with st.spinner('Detecting...'):
            audio_signal, sampling_rate = soundfile.read(path)
            window_length = int(0.025 * sampling_rate)
            hop_length = int(0.01 * sampling_rate)
            spectrogram = np.abs(
                librosa.stft(audio_signal, hop_length=hop_length, win_length=window_length)
            )
            specshow(
                librosa.amplitude_to_db(spectrogram, ref=np.max),
                sr=sampling_rate,
                hop_length=hop_length,
            )
            # plt.savefig(path.replace("wav", "png"), bbox_inches="tight", pad_inches=0)
            plt.savefig("spec.png", bbox_inches="tight", pad_inches=0)
            plt.close()

            # dimensions of our images
            img_width, img_height = 64, 64

            # load the model we saved
            model = load_model("model1.h5")
            model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

            # predicting images
            # img = image.load_img(path.replace("wav","png"), target_size=(img_width, img_height))
            img = image.load_img("spec.png", target_size=(img_width, img_height))

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = model.predict_classes(images, batch_size=10)
            a = [
                "Black Redstart",
                "Chestnut-collared Swift",
                "Eurasian Nuthatch",
                "European Greenfinch",
                "Greater Rhea",
                "Indian Spot-billed Duck",
                "Southern Cassowary",
                "Surf Scoter",
                "Swallow-tailed Nightjar",
                "White-tipped Sicklebill",
            ]
            st.subheader(a[classes[0]])
            st.image("spec.png",width=350)
    


if st.button("Detect Bird"):
    run(file_buffer)
