import os
import whisper
from flask import Flask, render_template, request
from chains import get_chat_chain, get_search_agent, get_qa_chain
import math
import time

# get path for static files
static_dir = os.path.join(os.path.dirname(__file__), 'static')  
if not os.path.exists(static_dir): 
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# audio file
audio_file = 'recording.webm'

# load whisper model
model = whisper.load_model('large-v2')

chat_chain = get_chat_chain()
search_agent = get_search_agent()
qa_chain = get_qa_chain()

# start server
server = Flask(__name__, static_folder=static_dir, template_folder=static_dir)

@server.route('/')
def landing():
    return render_template('index.html')

@server.route('/record', methods=['POST'])
def record():
    # get file from request and save it
    file = request.files['audio']
    file.save(audio_file)
    
    stt_start = time.time()

    # transcribe the audio file using Whisper and extract the text
    audio = whisper.load_audio(audio_file) 
    result = model.transcribe(audio, fp16=False)
    text = result["text"]
    
    math.factorial(100000)
    stt_end = time.time()
    print(text)
    print("걸린 시간" + " " + f"{stt_end - stt_start:.5f} sec")

    # remove the temp audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)

    # predict the response to get the output
    # output = chat_chain.predict(human_input=text)
    # output = search_agent.run(input=text)
    output = qa_chain.run(text)

    # remove the temp audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)
        
    return {"input": text, "output": output.replace("\n", "<br />")}