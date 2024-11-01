import subprocess
import requests


def stream_audio(
    text: str,
    voice_id: str,
    stability: float,
    similarity_boost: float,
    use_speaker_boost: bool,
    style: float,
) -> None:

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    headers = {
        "accept": "*/*",
        "xi-api-key": "feb1c562ba80143343da54d46d2c2297",
        "Content-Type": "application/json",
    }

    data = {
        "seed": 42,
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "use_speaker_boost": use_speaker_boost,
            "style": style,
        },
    }

    response = requests.post(
        url=url, headers=headers, json=data, stream=True, timeout=30
    )
    ffplay_cmd = ["ffplay", "-nodisp", "-autoexit", "-"]
    ffplay_proc = subprocess.Popen(
        ffplay_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for chunk in response.iter_content(chunk_size=4096):
        if ffplay_proc.stdin:
            ffplay_proc.stdin.write(chunk)
            # print("downloading...")

    # if ffplay_proc.stdin:
    #     ffplay_proc.stdin.close()
    #     ffplay_proc.wait()


if __name__ == "__main__":
    pass
