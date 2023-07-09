### import all the packages that is needed for the project
import huggingface_hub
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
from mutagen.mp3 import MP3
from PIL import Image
from pathlib import Path
from moviepy import editor
import os 
from os import listdir
import openai 
import re
import pyttsx3
import time
import numpy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

### I have givent the script as a string so for what text i should generate video through model, I have put "" around the text so the below 
# function is to find the string between " " which will then pass through the string
def extract_strings(sentence):
    pattern = r'"([^"]*)"'
    matches = re.findall(pattern, sentence)
    return matches

### this function is to convert the text into audio file and save it in audio.mp3
def sayLine(line):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', 100) # set the rate as you like
    # engine.say(line)
    engine.save_to_file(line, 'audio.mp3')
    engine.runAndWait()
    time.sleep(5) 
    engine.stop()

### there are different GIF that will be created, so this function appends all the gif and return the appended gif
def append_gifs(gif_files, output_file):
    images = []
    
    for gif_file in gif_files:
        gif = imageio.imread(gif_file)
        images.append(gif)
    
    imageio.mimsave(output_file, images)

## this fucntion generates the GIF
def generate_gif(prompt,folder_path,idx):
    audio = MP3("/DATA/bhumika1/Documents/Nikhil/Assignment/audio.mp3")
    ### take the audio length and divide it by the total words which will be passed through model.
    ### here I have taken consant time for all the GIF to generate the number of frames but actually this will not be the case.
    ### there should be script which  will read the string and as the "" occurs save the time in the list, so finally we wil be having the list 
    ## in which there will be different duration but i couldn't try due to resources constraint as for this is have train number of times so that 
    ### it matches the audio and video perfectly.
    audio_length = round(audio.info.length)
    video_duration = int(audio_length / len(strings)) 
    # print(video_duration)
    num_frames = video_duration * 10
    video_frames = pipe(prompt, negative_prompt="low quality", num_inference_steps=25, num_frames=num_frames).frames
    video_path = export_to_video(video_frames)
    video = imageio.mimread(video_path)
    return video

if __name__=="__main__":
    huggingface_hub.login(token='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')
    ### i have imported hugging face model which converts the text to gif
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    pipe=pipe.to('cuda:4')

    full_audio_path = "/DATA/bhumika1/Documents/Nikhil/Assignment/audio.mp3"
    full_video_path = "/DATA/bhumika1/Documents/Nikhil/Assignment/result.mp4"
    folder_path = "/DATA/bhumika1/Documents/Nikhil/Assignment"

    ### read the file and generate video that are highlighted
    with open('data.txt') as f:
        lines =f.readlines()
        strings = extract_strings(lines[0])
        # print(strings)    
        f.close()
    ### read the string and pass the words that are between " " and generate the video corresponding to it
    for idx,words in enumerate(strings):
        video = generate_gif(words,folder_path,idx)
        temp_path = folder_path+"/gif"+str(idx)+".gif"
        imageio.mimwrite(temp_path,video)

    path_gif = Path(folder_path)
    gif_files = list(path_gif.glob('*.gif'))
    output_file = "/DATA/bhumika1/Documents/Nikhil/Assignment/result.gif"
    append_gifs(gif_files, output_file)

    ### convert gif to video
    video = editor.VideoFileClip("/DATA/bhumika1/Documents/Nikhil/Assignment/appended.gif")
    ### generate audio
    with open('data.txt') as f:
        lines =f.readlines()
        sayLine(lines)
        f.close()
     
    audio = editor.AudioFileClip(full_audio_path)
    final_video = video.set_audio(audio)
    final_video.set_fps(60)
    final_video.write_videofile(full_video_path)### final video will be saved

    ### i could perform and try many things but couldn't perform due to lack of resources as there is limit in google colab but yeah it was good
    ## experience to learn new stuff. I have attached the final video along with the code. Lastly, I would like to work with you team and work aggressively on new stuff,
    ## i hope i'll get this opportunity.

