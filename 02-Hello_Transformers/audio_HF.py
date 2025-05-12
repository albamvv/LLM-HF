import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import torch, transformers, torchaudio,librosa
from transformers import AutoFeatureExtractor, ASTForAudioClassification

audio_path ='quedarte.mp3'

# sr -sample rate
y, sr = librosa.load(audio_path, sr=None)
#print(type(y))
#print("y->",y.shape)
#print('sr-> ',sr) # Hz  48000 Hz --> 48 kHz --> 1/sec

plt.figure(figsize=(14,5),dpi=150)
plt.plot(y)
plt.xlabel('Time - Samples')
plt.ylabel('Amplitude')
#plt.show()
Audio(data=y,rate=sr)

feature_extractor =AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
#print("feature extractor -> ",feature_extractor)
result=feature_extractor(y, return_tensors='pt')
print(result['input_values'])

model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
print(result['input_values'])
prediction_logits=model(result['input_values']).logits
print(prediction_logits)