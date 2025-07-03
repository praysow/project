import os
import pyttsx3
import speech_recognition as sr
# from pydub import AudioSegment # 이 모듈은 파일 변환에만 필요하므로, 실시간 마이크 입력에서는 직접 사용되지 않습니다.

def transcribe_live_audio():
    """
    마이크에서 실시간으로 음성을 입력받아 텍스트로 변환합니다.
    3초 동안 음성 입력이 없으면 현재까지의 음성을 처리합니다.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 말해주세요! (마이크 입력 대기 중...)")
        recognizer.adjust_for_ambient_noise(source) # 주변 소음 수준을 자동으로 조정합니다.
        try:
            # 음성 입력이 시작된 후 3초 동안 음성이 없으면 녹음을 중단하고 처리합니다.
            # timeout=None으로 설정하여 음성 입력이 시작될 때까지 무한정 대기합니다.
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=3)
            print("✨ 음성 인식 중...")
            text = recognizer.recognize_google(audio, language="ko-KR")
            print(f"📝 인식된 텍스트: {text}")
            return text
        except sr.UnknownValueError:
            print("❌ 음성을 이해하지 못했습니다. 다시 시도합니다.")
            return None
        except sr.RequestError as e:
            print(f"❌ Google STT API 요청 실패: {e}. 다시 시도합니다.")
            return None
        # timeout=None으로 설정했기 때문에 WaitTimeoutError는 발생하지 않습니다.
        # 하지만 혹시 모를 상황을 위해 남겨둘 수 있습니다.
        # except sr.WaitTimeoutError:
        #     print("❌ 음성 입력이 없었습니다. 다시 시도합니다.")
        #     return None


def speak_text(text):
    """
    주어진 텍스트를 음성으로 출력합니다.
    """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def process_continuous_speech():
    """
    마이크에서 연속적으로 음성을 입력받아 텍스트로 변환하고,
    그 텍스트를 다시 음성으로 출력하는 전체 프로세스입니다.
    사용자가 프로그램을 종료하기 전까지 계속 작동합니다.
    """
    print("--- 실시간 연속 음성 처리 시작 (종료하려면 Ctrl+C를 누르세요) ---")
    while True:
        recognized_text = transcribe_live_audio()

        if recognized_text:
            print("\n--- 인식된 텍스트를 다시 말합니다 ---")
            speak_text(recognized_text)
        else:
            # 음성 인식이 실패했거나 텍스트가 없는 경우에도 계속 대기합니다.
            print("처리할 텍스트가 없거나 오류가 발생했습니다. 다음 음성 입력을 대기합니다.")

# 🧪 예시 사용
if __name__ == "__main__":
    # 이 스크립트를 실행하면 마이크 입력이 계속 대기 상태가 됩니다.
    # 프로그램을 종료하려면 터미널에서 Ctrl+C를 누르세요.
    process_continuous_speech()
