from flask import Flask, request, jsonify

from src.stt import transcribe

app = Flask(__name__)

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    print(request.files)
    audio_file = request.files['file']
    text = transcribe(audio_file)
    return jsonify({'text': text})
