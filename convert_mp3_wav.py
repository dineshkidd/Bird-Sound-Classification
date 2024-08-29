#from pydub import AudioSegment
#sound = AudioSegment.from_mp3("/path/to/file.mp3")
#sound.export("/output/path/file.wav", format="wav")
import os
import glob

# files
lst = glob.glob("*.mp3")
print(lst)
for file in lst:
    # convert wav to mp3
    os.system(f"""ffmpeg -i {file} -acodec pcm_u8 -ar 22050 {file[:-4]}.wav""")