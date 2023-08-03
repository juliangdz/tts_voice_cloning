from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark
#from scipy.io.writable import write as write_wav
import scipy 
import numpy as np 

config = BarkConfig()
model = Bark.init_from_config(config)
model.load_checkpoint(config,checkpoint_dir="bark/",eval=True)
model.to("cpu")

text = ["- At Docsun we have invested in a 5 year research effort ","- We have build a A.I model capable of predicting the physilogical vital signs of a user from a simple facial video scan","- Our system has undergone an in-house clinical test and is on par with the medical testing devices"]
data_array=[]
for idx in range(len(text)):
    output_dict = model.synthesize(
        text[idx],
        config,
        speaker_id="speaker",
        voice_dirs="bark_voices",
        temperature=0.95
    )
    data_array.append(output_dict["wav"])
#    scipy.io.wavfile.write(f"cloned_{idx}.wav",rate=24000,data=output_dict["wav"])

combined_data = np.concatenate(data_array)
scipy.io.wavfile.write("cloned_combined.wav",rate=24000,data=combined_data)

#write_wav("cloned.wav",24000,output_dict["wav"])
