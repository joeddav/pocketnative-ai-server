import os
from pathlib import Path

import openai
from pydub import AudioSegment

def to_chunks(text, max_chars=4000, delim="\n"):
    paragraphs = text.split(delim)

    chunk_paragraphs = []
    char_count = 0
    chunks = []
    for i, p in enumerate(paragraphs):
        if char_count + len(p) >= max_chars or i == len(paragraphs) - 1:
            chunks.append(delim.join(chunk_paragraphs))
            char_count = 0
            chunk_paragraphs = []

        chunk_paragraphs.append(p)
        char_count += len(p)

    return chunks

def vocalize_text(text, output_path, model="tts-1-hd", voice="nova", speed=1.0, response_format="mp3"):
    response = openai.audio.speech.create(
        input=text,
        model=model,
        voice=voice,
        speed=speed,
        response_format=response_format
    )
    response.stream_to_file(output_path)

def concat_files(audio_file_paths: list[Path | str], output_path: str | Path, rm_existing=True):
    """Reads in the audio files and concatenates them, saves to output_path """
    # Load the first audio file
    combined = AudioSegment.from_file(audio_file_paths[0], format="mp3")

    # Concatenate with the rest of the files
    for file_path in audio_file_paths[1:]:
        audio = AudioSegment.from_file(file_path, format="mp3")
        combined += audio

    # Export the combined audio
    combined.export(output_path, format="mp3")

    for file in audio_file_paths:
        os.remove(file)

def vocalize_text_long(text, output_path, model="tts-1-hd", voice="nova", speed=1.0):
    fpath = Path(output_path)
    ftemplate = fpath.stem + "-chunk_{}" + fpath.suffix
    response_format = fpath.suffix[1:]

    chunks = to_chunks(text)
    chunk_fpaths = [fpath.parent / ftemplate.format(i + 1) for i in range(len(chunks))]
    
    for path, chunk in zip(chunk_fpaths, chunks):
        vocalize_text(chunk, path, model=model, speed=speed, voice=voice, response_format=response_format)

    concat_files(chunk_fpaths, output_path)


if __name__ == "__main__":
    inputs_dir = Path("text")

    for f in inputs_dir.glob("*.txt"):
        if "p5" in f.as_posix():
            continue
        out_path = Path("outputs") / f.with_suffix(".mp3").name
        text = f.read_text()
        vocalize_text_long(text, out_path)