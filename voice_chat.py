import os
import google.generativeai as genai
from google.cloud import speech
from google.cloud import texttospeech
# import pyttsx3 # 移除，因為不再使用
import pyaudio
import queue
import time
from duckduckgo_search import DDGS
import configparser
import tempfile
from playsound import playsound # 改回 playsound
# import simpleaudio as sa # 移除 simpleaudio
from google.oauth2 import service_account
import traceback # 新增 for more detailed error printing

# 全域變數 (部分會從 config 載入)
CONFIG_FILE = 'pro_config.ini' # 指定設定檔名稱
SERVICE_ACCOUNT_PATH = None
GEMINI_API_KEY = None
GEMINI_MODEL_NAME = 'models/gemini-1.5-pro-latest' # 預設值，會被 config 覆寫
STT_LANGUAGE_CODE = 'zh-TW'
TTS_LANGUAGE_CODE = 'zh-TW'
TTS_VOICE_NAME = None
TTS_SPEAKING_RATE = 1.0
TTS_PITCH = 0.0
# TTS_AUDIO_ENCODING 改回可以從 config 讀取，預設 MP3
TTS_AUDIO_ENCODING = texttospeech.AudioEncoding.MP3 

STREAMING_LIMIT = 240000
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10) 

speech_client = None
tts_client = None
gemini_model = None

def load_and_configure():
    global SERVICE_ACCOUNT_PATH, GEMINI_API_KEY, GEMINI_MODEL_NAME, STT_LANGUAGE_CODE
    global TTS_LANGUAGE_CODE, TTS_VOICE_NAME, TTS_SPEAKING_RATE, TTS_PITCH, TTS_AUDIO_ENCODING # 加回 TTS_AUDIO_ENCODING
    global speech_client, tts_client, gemini_model, SAMPLE_RATE, CHUNK_SIZE # CHUNK_SIZE 也可能在config裡

    config = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        print(f"錯誤：設定檔 '{CONFIG_FILE}' 找不到。請確認該檔案存在且與 voice_chat.py 在同一目錄下，或提供正確路徑。")
        return False
    config.read(CONFIG_FILE, encoding='utf-8')

    try:
        SERVICE_ACCOUNT_PATH = config.get('GoogleCloud', 'service_account_key_path', fallback=None)
        if not SERVICE_ACCOUNT_PATH or not os.path.exists(SERVICE_ACCOUNT_PATH):
            print(f"錯誤：Google Cloud 服務帳戶金鑰路徑未在 '{CONFIG_FILE}' ([GoogleCloud] -> service_account_key_path) 中正確設定或檔案 '{SERVICE_ACCOUNT_PATH}' 不存在。")
            return False
        
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)
        speech_client = speech.SpeechClient(credentials=credentials)
        tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
        print("Google Cloud Speech-to-Text 和 Text-to-Speech 用戶端初始化成功。")

        GEMINI_API_KEY = config.get('Gemini', 'api_key', fallback=None)
        if not GEMINI_API_KEY or GEMINI_API_KEY == 'YOUR_API_KEY_HERE': # pro_config.ini 中的預設值
            print(f"錯誤：Gemini API 金鑰未在 '{CONFIG_FILE}' ([Gemini] -> api_key) 中設定。")
            return False
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL_NAME = config.get('Gemini', 'model_name', fallback='models/gemini-1.5-pro-latest')
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"Gemini 模型 '{GEMINI_MODEL_NAME}' 初始化成功。")

        STT_LANGUAGE_CODE = config.get('SpeechRecognition', 'language', fallback='zh-TW')
        SAMPLE_RATE = config.getint('Audio', 'sample_rate', fallback=16000)
        CHUNK_SIZE = config.getint('Audio', 'chunk_size', fallback=int(SAMPLE_RATE / 10))
        
        TTS_LANGUAGE_CODE = config.get('TTS', 'tts_language_code', fallback=STT_LANGUAGE_CODE) # pro_config.ini 沒有 language_code for TTS, 沿用S2T
        TTS_VOICE_NAME = config.get('TTS', 'tts_voice_name', fallback=None)
        TTS_SPEAKING_RATE = config.getfloat('TTS', 'speaking_rate', fallback=1.0)
        TTS_PITCH = config.getfloat('TTS', 'pitch', fallback=0.0)
        # 從 config 讀取 TTS audio_encoding，預設 MP3
        tts_audio_encoding_str = config.get('TTS', 'audio_encoding', fallback='MP3').upper()
        if hasattr(texttospeech.AudioEncoding, tts_audio_encoding_str):
            TTS_AUDIO_ENCODING = getattr(texttospeech.AudioEncoding, tts_audio_encoding_str)
            print(f"TTS 音訊編碼從設定檔載入為: {tts_audio_encoding_str}")
        else:
            print(f"警告：'{CONFIG_FILE}' 中的 TTS audio_encoding '{tts_audio_encoding_str}' 無效，將使用 MP3。")
            TTS_AUDIO_ENCODING = texttospeech.AudioEncoding.MP3
        return True

    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"設定檔 '{CONFIG_FILE}' 錯誤: {e}")
        return False
    except Exception as e:
        print(f"載入設定或初始化用戶端時發生未預期錯誤: {e}")
        traceback.print_exc() # 印出詳細錯誤
        return False

class MicrophoneStream:
    """開啟麥克風錄音串流並將音訊區塊放入佇列。"""
    def __init__(self, rate, chunk_size):
        self._rate = rate
        self._chunk_size = chunk_size
        self._buff = queue.Queue()
        self.closed = True
        self._audio_interface = None # 初始化
        self._audio_stream = None # 初始化

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        if self._audio_stream and not self.closed:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            except Exception as e:
                print(f"關閉麥克風串流時發生錯誤: {e}")
            finally:
                self.closed = True
        
        # 清空佇列並發送結束信號
        if hasattr(self, '_buff'):
            with self._buff.mutex:
                self._buff.queue.clear()
            self._buff.put(None)

        if self._audio_interface:
            try:
                self._audio_interface.terminate()
            except Exception as e:
                print(f"終止 PyAudio 介面時發生錯誤: {e}")
        self._audio_interface = None
        self._audio_stream = None

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """持續從音訊串流收集資料到佇列中。"""
        if not self.closed:
            self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """從佇列產生音訊區塊。"""
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        self._buff.put(None)
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

def speak_google_tts(text_to_speak):
    global tts_client, TTS_LANGUAGE_CODE, TTS_VOICE_NAME, TTS_AUDIO_ENCODING, TTS_SPEAKING_RATE, TTS_PITCH, SAMPLE_RATE
    if not tts_client:
        print("錯誤：TextToSpeech 用戶端未初始化。")
        return
    print(f"AI: {text_to_speak}")
    audio_file_path = None # 初始化
    try:
        print("DEBUG: speak_google_tts - Step 1: Preparing synthesis input (using playsound)")
        synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)
        voice_params = texttospeech.VoiceSelectionParams(language_code=TTS_LANGUAGE_CODE)
        if TTS_VOICE_NAME and TTS_VOICE_NAME.strip():
            voice_params.name = TTS_VOICE_NAME
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=TTS_AUDIO_ENCODING, # 使用 config 載入的或預設 MP3
            speaking_rate=TTS_SPEAKING_RATE,
            pitch=TTS_PITCH
            # sample_rate_hertz is not typically specified for MP3 encoding with playsound
        )
        if TTS_AUDIO_ENCODING != texttospeech.AudioEncoding.MP3: # 如果不是MP3，例如LINEAR16，則指定sample_rate
            audio_config.sample_rate_hertz = SAMPLE_RATE

        print("DEBUG: speak_google_tts - Step 2: Calling synthesize_speech API")
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice_params, audio_config=audio_config
        )
        print("DEBUG: speak_google_tts - Step 3: TTS API response received")
        
        suffix = ".mp3" if TTS_AUDIO_ENCODING == texttospeech.AudioEncoding.MP3 else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmpfile:
            tmpfile.write(response.audio_content)
            audio_file_path = tmpfile.name
        print(f"DEBUG: speak_google_tts - Step 4: Audio content written to temporary file: {audio_file_path}")
        
        print("DEBUG: speak_google_tts - Step 5: Attempting to play audio with playsound")
        playsound(audio_file_path)
        print("DEBUG: speak_google_tts - Step 6: playsound() completed")

    except Exception as e:
        print(f"Google Cloud T2S 或 playsound 處理時發生錯誤: {e}")
        traceback.print_exc() # 印出完整的錯誤追蹤訊息
    finally:
        print(f"DEBUG: speak_google_tts - Entering finally block.")
        if audio_file_path and os.path.exists(audio_file_path):
            print(f"DEBUG: speak_google_tts - Attempting to remove {audio_file_path}")
            try:
                os.remove(audio_file_path)
                print(f"DEBUG: speak_google_tts - Successfully removed {audio_file_path}")
            except Exception as e_remove:
                print(f"DEBUG: speak_google_tts - Error removing {audio_file_path}: {e_remove}")
        elif audio_file_path:
            print(f"DEBUG: speak_google_tts - File {audio_file_path} not found for removal (or was never created properly).")
        else:
            print("DEBUG: speak_google_tts - audio_file_path was None, no file to remove.")
    print("DEBUG: speak_google_tts - Exiting function normally.") # 確保函式正常結束時也會印出

def listen_print_loop(responses, stream):
    """迭代 Speech API 回應並處理結果，更流暢地顯示中間結果。"""
    num_chars_printed = 0
    final_transcript_parts = []

    for response in responses:
        if stream.closed: # 檢查串流是否在外部被關閉
            print("\n串流已關閉，停止辨識。")
            break
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        if not result.is_final:
            print(f"\r{' ' * (num_chars_printed + 20)}\r{transcript}", end="", flush=True)
            num_chars_printed = len(transcript)
        else:
            print(f"\r{' ' * (num_chars_printed + 20)}\r", end="", flush=True)
            print(f"您說: {transcript}")
            final_transcript_parts.append(transcript)
            break
    
    return "".join(final_transcript_parts)

def listen_for_speech():
    """使用 Google Cloud Speech-to-Text 進行串流語音辨識。"""
    global speech_client, STT_LANGUAGE_CODE, SAMPLE_RATE, CHUNK_SIZE
    if not speech_client:
        print("錯誤：SpeechClient 未初始化。")
        return None
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE, language_code=STT_LANGUAGE_CODE,
        max_alternatives=1, enable_automatic_punctuation=True,
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    print("\nAI: 請說話 (您可以隨時開始，說 '結束' 來結束程式)...", end="", flush=True)
    
    recognized_text = None
    mic_stream = None  # 初始化 mic_stream
    try:
        mic_stream = MicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
        with mic_stream as stream:
            audio_generator = stream.generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )
            print("\rAI: 正在聆聽... 您說: ", end="", flush=True) 
            responses = speech_client.streaming_recognize(streaming_config, requests)
            recognized_text = listen_print_loop(responses, stream)
    except KeyboardInterrupt:
        print("\n偵測到使用者中斷，正在關閉程式...")
        if mic_stream: # 確保在關閉前 mic_stream 已被賦值
            mic_stream.closed = True # 手動關閉串流
        recognized_text = "結束" # 觸發程式結束
    except Exception as e:
        print(f"\n語音辨識時發生錯誤: {e}")
    finally:
        # print() # 移到 listen_print_loop 或 main 迴圈後，避免過多空行
        if mic_stream and not mic_stream.closed: # 確保串流被關閉
            mic_stream.closed = True
        return recognized_text

def needs_web_search(user_query):
    """詢問 LLM 是否需要網路搜尋。"""
    global gemini_model
    if not gemini_model:
        print("錯誤: Gemini 模型未初始化。")
        return False
    try:
        # 使用 generate_content 進行簡單的 YES/NO 判斷，避免 chat session 的複雜性
        prompt = f"針對以下問題：「{user_query}」 是否需要透過網路搜尋才能獲得更完整或即時的答案？請只回答 'YES' 或 'NO'。"
        response = gemini_model.generate_content(prompt)
        decision = response.text.strip().upper()
        print(f"AI (判斷是否搜尋): {decision}")
        return decision == "YES"
    except Exception as e:
        print(f"判斷是否需要網路搜尋時發生錯誤: {e}")
        return False # 預設為不需要搜尋以避免錯誤

def main():
    global gemini_model
    chat_history_for_gemini = []
    
    print("DEBUG: main() 進入點") # 新增除錯
    speak_google_tts("您好！我是您的 AI 助手。")
    print("DEBUG: speak_google_tts() 完成") # 新增除錯
    
    keep_listening = True
    print(f"DEBUG: keep_listening 初始化為: {keep_listening}") # 新增除錯

    while keep_listening:
        print("DEBUG: 進入 while keep_listening 迴圈頂部") # 新增除錯
        print("-" * 30)
        user_input = listen_for_speech()
        print(f"DEBUG: listen_for_speech() 回傳了: '{user_input}'") 
        if user_input is None: continue
        if not user_input.strip(): continue
        print(f"使用者輸入 (S2T 結果): {user_input}")
        if user_input.lower() in ['再見', '拜拜', '結束']:
            speak_google_tts("謝謝您的使用，再見！")
            keep_listening = False # 跳出迴圈
            break
            
        # 將使用者輸入加入 Gemini 的對話歷史
        # chat_history_for_gemini.append({"role": "user", "parts": [{"text": user_input}]})
        # ^ 暫時移除，因為下面的邏輯每次都重新構建提問

        if needs_web_search(user_input):
            speak_google_tts(f"好的，我將為您搜尋關於「{user_input}」的資訊。")
            search_results_text = ""
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(user_input, max_results=3, region='wt-wt'))
                    if results:
                        search_results_text = "根據我找到的網路資料：\n"
                        for i, res in enumerate(results):
                            title = res.get('title', '')
                            body = res.get('body', '')
                            # 嘗試提取更精簡的內容，避免過長
                            if len(body) > 150:
                                body = body[:150] + "..."
                            search_results_text += f"{i+1}. 標題: {title}\n內容摘要: {body}\n"
                        print(f"--- 搜尋到的內容 ---\n{search_results_text}--------------------")
                    else:
                        search_results_text = "抱歉，我沒有在網路上找到相關的明確資訊。"
            except Exception as e:
                print(f"網路搜尋時發生錯誤: {e}")
                search_results_text = "抱歉，網路搜尋時發生了錯誤。"
            
            # speak(search_results_text) # 可以選擇是否唸出搜尋到的摘要
            
            # 修改提示，更強調基於提供的文本
            prompt_with_search = (
                f"請你扮演一個樂於助人的AI助理。根據以下提供的文本資訊來回答問題。請盡可能直接從文本中提取、整合或總結答案。"
                f"如果文本中沒有直接答案，但能推斷，請嘗試回答。如果文本完全無法回答，請誠實告知。"
                f"不要引導使用者去其他地方查看，除非文本本身就是一個網址列表。\n\n"
                f"提供的文本：\n---開始---{search_results_text}---結束---"
                f"問題是：「{user_input}」\n"
                f"你的回答："
            )
            current_question_for_gemini = prompt_with_search
        else:
            speak_google_tts("好的，我來思考一下。")
            current_question_for_gemini = f"請回答以下問題：「{user_input}」"

        try:
            # 為了簡化，這裡每次都基於最新的問題（可能包含搜尋結果）進行一次性提問
            # 這樣做會丟失多輪對話的上下文，如果需要多輪對話，需要修改 chat_history_for_gemini 的管理
            contents_for_gemini = []
            # 建構 Gemini 的 contents 列表，包含歷史紀錄和目前問題
            for entry in chat_history_for_gemini:
                contents_for_gemini.append(entry) # 假設 history 中的 entry 格式已符合要求
            contents_for_gemini.append({'role': 'user', 'parts': [{'text': current_question_for_gemini}]})
            
            print(f"--- 正在發送給 Gemini ({len(contents_for_gemini)} 條內容) ---")
            # for i, content_part in enumerate(contents_for_gemini):
            #     print(f"  Part {i} Role: {content_part['role']}, Text: {content_part['parts'][0]['text'][:100]}...")
            # print("--------------------------------------")

            response_object = gemini_model.generate_content(
                contents=contents_for_gemini,
                generation_config=genai.types.GenerationConfig(temperature=0.7)
            )
            ai_response_text = response_object.text
            speak_google_tts(ai_response_text)
            
            # 更新對話歷史
            chat_history_for_gemini.append({'role': 'user', 'parts': [{'text': user_input}]}) # 先加使用者這次的輸入
            chat_history_for_gemini.append({'role': 'model', 'parts': [{'text': ai_response_text}]})
        except Exception as e:
            print(f"\n與 Gemini 通訊時發生錯誤: {e}")
            speak_google_tts("抱歉，我遇到了一些問題，請再試一次。")

if __name__ == "__main__":
    if load_and_configure():
        main()
    else:
        print("程式因設定錯誤而終止。") 