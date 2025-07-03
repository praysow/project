import os
import pyttsx3
import speech_recognition as sr
from pydub import AudioSegment

def convert_mp3_to_wav(mp3_path, wav_path):
    # MP3를 WAV로 변환
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    print(f"✅ MP3 -> WAV 변환 완료: {wav_path}")

def transcribe_audio(audio_path):
    # 오디오 -> 텍스트
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="ko-KR")
        print(f"📝 인식된 텍스트: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ 음성을 이해하지 못했습니다.")
        return None
    except sr.RequestError as e:
        print(f"❌ Google STT API 요청 실패: {e}")
        return None

def speak_text(text):
    # 텍스트 -> 음성 출력
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def process_and_speak(input_path):
    # MP3 → WAV → 텍스트 → TTS
    if input_path.endswith('.mp3'):
        wav_path = input_path.replace('.mp3', '.wav')
        convert_mp3_to_wav(input_path, wav_path)
        text = transcribe_audio(wav_path)
    elif input_path.endswith('.wav'):
        text = transcribe_audio(input_path)
    else:
        print("❌ 지원하지 않는 오디오 형식입니다.")
        return

    # 텍스트 → 음성 출력
    if text:
        speak_text(text)

# 🧪 예시 사용
input_audio_path = '/home/RXO/ts/940.wav'  # 또는 .mp3 파일도 가능
process_and_speak(input_audio_path)
