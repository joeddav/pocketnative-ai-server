from flask import Flask, request, jsonify, Response
import os

from src.stt import transcribe

app = Flask(__name__)

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    from pprint import pprint
    print("\n\n\n")
    print("Headers:")
    pprint(dict(request.headers))
    print("Form data:")
    pprint(request.form)
    print("Files:")
    pprint(request.files)
    print("\n\n\n")

    audio_file = request.files['file']
    # audio_file.save("/Users/joeddav/Downloads/" + audio_file.name)
    # save_path = os.path.join('.', audio_file.filename)
    # audio_file.save(save_path)
    # print("File saved to:", save_path)
    
    temperature = request.args.get('temperature', default=0.2, type=float)
    prompt = request.args.get('prompt', default=None, type=str)
    user_examples = request.args.getlist('user_examples')
    if len(user_examples) == 0:
        user_examples = None
    text = transcribe(audio_file, temperature, prompt, user_examples)
    
    print("\n\n\n\nTEXT:\n", text, "\n\n\n\n")

    return jsonify({'text': text})




from src.chat import StatefulAssistant

assistant = StatefulAssistant(
    name="Andrea",
    instructions=[
        "Always keep messages concise",
        "The user is named Joe",
    ],
)

@app.route('/chat/completions', methods=['POST'])
def chat_completion():
    if 'model' not in request.json:
        return jsonify({'error': 'No model provided'}), 400

    if 'messages' not in request.json:
        return jsonify({'error': 'No messages provided'}), 400

    if 'stream' not in request.json or request.json['stream'] is not True:
        return jsonify({'error': 'Only streaming responses are implemented at the moment'}), 400

    model = request.json['model']
    if model == "gpt-3.5-turbo":
        model = "gpt-35-turbo"
    messages = request.json['messages']
    stream = request.json['stream']
    api_version = request.args.get('api_version', default='2023-05-15', type=str)

    assistant.model = model

    stream = assistant.chat(messages, like_api=True)

    def generate_bytes():
        for chunk in stream:
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate_bytes(), mimetype='text/event-stream')


