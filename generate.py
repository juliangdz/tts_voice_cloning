from transformers import AutoProcessor, BarkModel
import scipy

processor = AutoProcessor.from_pretrained('suno/bark')
model = BarkModel.from_pretrained('suno/bark')
model.to("cpu")

def generate_audio(text,preset,output):
    inputs=  processor(text,voice_preset=preset)
    #for k,v in inputs.items():
     #   inputs[k] = v.to("cuda")
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(output,rate=sample_rate,data=audio_array)

generate_audio(text="Hi , welcome to Julians Chat Bot",preset="v2/en_speaker_9",output="output.wav")
