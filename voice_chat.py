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
from google.oauth2 import service_account
import traceback # 新增 for more detailed error printing
import webrtcvad # 匯入 VAD 函式庫
import threading # 用於 TTS 背景播放
import re # 新增 re 模塊

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
# CHUNK_SIZE = int(SAMPLE_RATE / 10) # 原始 CHUNK_SIZE
# VAD 支援的幀時長 (毫秒)
VAD_FRAME_MS = 30 
# VAD 處理的音訊塊大小 (frames)，對應 VAD_FRAME_MS
# PyAudio 資料是 int16 (2 bytes per frame)
VAD_CHUNK_SIZE = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)

# 原始的 CHUNK_SIZE，現在改名以避免混淆，主要用於 PyAudio 的 buffer
# 建議其大小為 VAD_CHUNK_SIZE 的整數倍，方便處理
PYAUDIO_CHUNK_SIZE = VAD_CHUNK_SIZE * 10 # 例如，一次讀取 300ms 的音訊

STT_IDLE_TIMEOUT_SECONDS = 180 # STT 聆聽時的空閒超時預設值

# TTS 播放狀態旗標
tts_playback_thread = None
tts_audio_playing = False
tts_playback_should_stop = False

speech_client = None
tts_client = None
gemini_model = None

# 文本分割成句子的分隔符號
SENTENCE_DELIMITERS = re.compile(r'([.。！？!?\\n])') # 保留分隔符

def split_text_into_sentences(text):
    """將文本分割成句子列表，保留標點符號在句子末尾。"""
    if not text or not text.strip():
        return []
    
    # 先替換多個換行符為單個，並確保標點後有換行以便分割
    text = re.sub(r'\n+', '\n', text).strip()
    # text = text.replace('。', '。\n').replace('.', '.\n').replace('！', '！\n').replace('!', '!\n').replace('？', '？\n').replace('?', '?\n')

    sentences_with_delimiters = SENTENCE_DELIMITERS.split(text)
    
    result_sentences = []
    current_sentence = ""
    for part in sentences_with_delimiters:
        if not part:
            continue
        current_sentence += part
        if SENTENCE_DELIMITERS.match(part):
            if current_sentence.strip():
                result_sentences.append(current_sentence.strip())
            current_sentence = ""
            
    if current_sentence.strip(): # 加入最後一部分 (如果有的話)
        result_sentences.append(current_sentence.strip())
    
    # 過濾掉可能的空字串
    result_sentences = [s for s in result_sentences if s.strip()]
    
    print(f"DEBUG: [split_text_into_sentences] Original: '{text}' -> Split: {result_sentences}")
    return result_sentences

def load_and_configure():
    global SERVICE_ACCOUNT_PATH, GEMINI_API_KEY, GEMINI_MODEL_NAME, STT_LANGUAGE_CODE
    global TTS_LANGUAGE_CODE, TTS_VOICE_NAME, TTS_SPEAKING_RATE, TTS_PITCH, TTS_AUDIO_ENCODING
    global speech_client, tts_client, gemini_model, SAMPLE_RATE, CHUNK_SIZE # CHUNK_SIZE 也可能在config裡
    global VAD_AGGRESSIVENESS, VAD_ACTIVATION_FRAMES, SILENCE_DURATION_END_SPEECH
    global PYAUDIO_CHUNK_SIZE, VAD_CHUNK_SIZE, VAD_FRAME_MS # VAD 專用 chunk 設定
    global STT_IDLE_TIMEOUT_SECONDS # 新增 STT 空閒超時設定

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
        # CHUNK_SIZE = config.getint('Audio', 'chunk_size', fallback=int(SAMPLE_RATE / 10))
        # 更新 CHUNK_SIZE 的讀取和設定，以適應 VAD
        # PYAUDIO_CHUNK_SIZE 優先從 config 讀取 'chunk_size'
        # VAD_CHUNK_SIZE 根據 SAMPLE_RATE 和 VAD_FRAME_MS 計算得到
        global PYAUDIO_CHUNK_SIZE, VAD_CHUNK_SIZE, VAD_FRAME_MS
        PYAUDIO_CHUNK_SIZE = config.getint('Audio', 'chunk_size', fallback=int(SAMPLE_RATE * VAD_FRAME_MS / 1000 * 10)) # 預設 300ms
        VAD_FRAME_MS = config.getint('Audio', 'vad_frame_duration_ms', fallback=30) # VAD 幀時長
        VAD_CHUNK_SIZE = int(SAMPLE_RATE * VAD_FRAME_MS / 1000) # VAD 處理的塊大小
        # 確保 PYAUDIO_CHUNK_SIZE 是 VAD_CHUNK_SIZE 的整數倍
        if PYAUDIO_CHUNK_SIZE % VAD_CHUNK_SIZE != 0:
            print(f"警告：Audio chunk_size ({PYAUDIO_CHUNK_SIZE}) 不是 VAD frame ({VAD_CHUNK_SIZE}) 的整數倍。可能導致 VAD 處理不精確。自動調整 PYAUDIO_CHUNK_SIZE。")
            PYAUDIO_CHUNK_SIZE = (PYAUDIO_CHUNK_SIZE // VAD_CHUNK_SIZE) * VAD_CHUNK_SIZE
            if PYAUDIO_CHUNK_SIZE == 0: # 避免除以零或過小
                PYAUDIO_CHUNK_SIZE = VAD_CHUNK_SIZE * 10 # 如果調整後為0，設回預設的10倍VAD幀長
            print(f"調整後的 PYAUDIO_CHUNK_SIZE: {PYAUDIO_CHUNK_SIZE}")

        # 讀取 VAD 設定
        VAD_AGGRESSIVENESS = config.getint('Audio', 'vad_aggressiveness', fallback=1)
        VAD_ACTIVATION_FRAMES = config.getint('Audio', 'vad_activation_frames', fallback=3)
        SILENCE_DURATION_END_SPEECH = config.getfloat('Audio', 'silence_duration_end_speech', fallback=1.2)

        # 從 [SpeechRecognition] 讀取 STT 空閒超時
        STT_IDLE_TIMEOUT_SECONDS = config.getint('SpeechRecognition', 'timeout_idle_seconds', fallback=180)
        
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
        # self._chunk_size = chunk_size # 改用全域 PYAUDIO_CHUNK_SIZE
        self._chunk_size = PYAUDIO_CHUNK_SIZE
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

# TTS 播放的背景工作函式
def _tts_playback_worker(audio_file_path, original_encoding):
    global tts_audio_playing, tts_playback_should_stop
    tts_audio_playing = True
    print(f"DEBUG: TTS playback worker started for {audio_file_path}. tts_playback_should_stop is {tts_playback_should_stop}")
    try:
        # 確保使用絕對路徑
        abs_path = os.path.abspath(audio_file_path)
        
        if not tts_playback_should_stop:
            try:
                # 嘗試使用 winsound 播放 WAV 檔案
                import winsound
                print(f"DEBUG: 嘗試使用 winsound 播放: {abs_path}")
                # winsound.SND_FILENAME 標誌表示第一個參數是檔案名
                # winsound.SND_SYNC 標誌表示同步播放 (阻塞直到播放完成)
                # 移除 SND_SYNC，依賴預設的同步行為
                winsound.PlaySound(abs_path, winsound.SND_FILENAME)
                print(f"DEBUG: winsound 播放成功: {abs_path}")
            except Exception as ws_error:
                print(f"DEBUG: winsound 播放失敗: {ws_error}")
                # 如果 winsound 失敗，這裡不再嘗試 playsound，因為 playsound 之前也失敗了
                # 如果需要，可以在這裡加入其他備用方案，例如 subprocess 調用外部播放器

        else:
            print("DEBUG: TTS playback was stopped before playback call.")

    except Exception as e:
        print(f"TTS 播放時發生錯誤: {e}")
        traceback.print_exc()
    finally:
        print(f"DEBUG: TTS playback worker finishing. tts_playback_should_stop: {tts_playback_should_stop}")
        tts_audio_playing = False
        try:
            # 等待一小段時間確保檔案不再被使用
            time.sleep(0.5)
            if audio_file_path and os.path.exists(audio_file_path):
                os.remove(audio_file_path)
                print(f"DEBUG: Temporary TTS file {audio_file_path} removed by worker.")
        except Exception as e_remove:
            print(f"DEBUG: Error removing temporary TTS file {audio_file_path} by worker: {e_remove}")

def speak_google_tts(text_to_speak):
    # 首先宣告所有全域變數
    global tts_client, TTS_LANGUAGE_CODE, TTS_VOICE_NAME, TTS_AUDIO_ENCODING
    global TTS_SPEAKING_RATE, TTS_PITCH, SAMPLE_RATE
    global tts_playback_thread, tts_audio_playing, tts_playback_should_stop

    if not tts_client:
        print("錯誤：TextToSpeech 用戶端未初始化。")
        return
    
    print(f"AI (原文): {text_to_speak}") # 印出 AI 準備說的完整原文

    sentences = split_text_into_sentences(text_to_speak)
    if not sentences:
        print("DEBUG: 沒有句子需要播放。")
        tts_audio_playing = False # 確保旗標正確
        tts_playback_should_stop = False
        return

    # 在開始播放句子序列前，重設停止旗標，確保新的播放序列不受舊旗標影響
    tts_playback_should_stop = False 
    # tts_audio_playing 會在第一個句子開始播放時由 worker thread 設為 True

    for i, sentence in enumerate(sentences):
        if tts_playback_should_stop: # 在播放每個句子前檢查是否被要求停止
            print(f"DEBUG: TTS playback stopped by flag before playing sentence: '{sentence}'")
            break # 跳出句子播放迴圈

        print(f"AI (播放第 {i+1}/{len(sentences)} 句): {sentence}")
        audio_file_path_local = None
        try:
            synthesis_input = texttospeech.SynthesisInput(text=sentence)
            voice_params = texttospeech.VoiceSelectionParams(language_code=TTS_LANGUAGE_CODE)
            if TTS_VOICE_NAME and TTS_VOICE_NAME.strip():
                voice_params.name = TTS_VOICE_NAME
            
            # 強制設定為 WAV 編碼
            current_tts_audio_encoding = texttospeech.AudioEncoding.LINEAR16
            audio_config = texttospeech.AudioConfig(
                audio_encoding=current_tts_audio_encoding, 
                speaking_rate=TTS_SPEAKING_RATE,
                pitch=TTS_PITCH,
                sample_rate_hertz=SAMPLE_RATE # WAV 需要設定採樣率
            )

            response = tts_client.synthesize_speech(
                input=synthesis_input, voice=voice_params, audio_config=audio_config
            )
            
            # 副檔名改為 .wav
            suffix = ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmpfile:
                tmpfile.write(response.audio_content)
                audio_file_path_local = tmpfile.name
                # print(f"DEBUG: TTS audio for sentence written to: {audio_file_path_local}")

                # 等待上一個句子的播放執行緒結束 (如果有的話)
                if tts_playback_thread is not None and tts_playback_thread.is_alive():
                    print("DEBUG: Waiting for previous sentence TTS thread to complete...")
                    tts_playback_thread.join() # 等待播放完成
                    print("DEBUG: Previous sentence TTS thread completed.")
                
                # 即使上一個被要求停止，也要確保 tts_audio_playing 更新了
                # 但 _tts_playback_worker 的 finally 會處理 tts_audio_playing = False

                if tts_playback_should_stop: # 再次檢查，因為 join 可能耗時
                    print(f"DEBUG: TTS playback stopped by flag after waiting for previous thread, before new sentence: '{sentence}'")
                    if audio_file_path_local and os.path.exists(audio_file_path_local):
                        try: os.remove(audio_file_path_local) # 清理未播放的檔案
                        except: pass
                    break

                # tts_audio_playing = True # 將由 worker thread 設定
                tts_playback_thread = threading.Thread(target=_tts_playback_worker, args=(audio_file_path_local, current_tts_audio_encoding)) # 傳入 encoding 備用
                tts_playback_thread.daemon = True
                tts_playback_thread.start()
                # print(f"DEBUG: TTS playback thread started for sentence: {audio_file_path_local}")

                # 在這裡需要等待當前句子播放完畢，或者能被中斷
                # 由於 winsound.PlaySound(..., winsound.SND_FILENAME | winsound.SND_SYNC) 是阻塞的，thread.join() 會等待它播完
                # 我們需要在 join 的同時能監聽打斷，這比較困難直接用 join 实现
                # 所以，打斷邏輯主要依賴 main loop 中的 tts_audio_playing 和 tts_playback_should_stop
                # speak_google_tts 內部迴圈則依賴 tts_playback_should_stop

                # 為了讓主迴圈的打斷檢測有機會執行，這裡不能完全阻塞等待每個句子播放完畢
                # 而是啟動播放後，主迴圈會檢測 tts_audio_playing
                # 但我們也需要確保在這個迴圈內，如果一個句子播放完了，能繼續下一個
                # 因此，我們需要在啟動一個句子的播放後，等待它自然結束或被外部停止
                # 這個等待由 tts_playback_thread.join() (在下一輪迴圈開始時) 間接完成
                # 也可以在這裡加入一個小的 sleep 或者更精細的檢查 tts_audio_playing
                while tts_audio_playing and not tts_playback_should_stop:
                    time.sleep(0.05) # 短暫等待，讓打斷檢測迴圈有機會執行，也讓播放執行緒執行
                
                if tts_playback_should_stop: # 如果在播放當前句子時被要求停止
                    print(f"DEBUG: TTS playback of sentence '{sentence}' was interrupted.")
                    # _tts_playback_worker 中的 winsound.PlaySound 是阻塞的，這裡無法直接停止
                    # 但下次迴圈會 break
                    if tts_playback_thread is not None and tts_playback_thread.is_alive():
                        print("DEBUG: Signaling current sentence TTS thread to stop (though winsound might not immediately)...")
                        # winsound 沒有直接停止的 API，依賴 thread 自然結束或被 OS 中斷
                    break

        except Exception as e:
            print(f"TTS 處理句子 '{sentence}' 時發生錯誤: {e}")
            traceback.print_exc()
            tts_audio_playing = False # 確保出錯時旗標是 False
            if audio_file_path_local and os.path.exists(audio_file_path_local):
                try: os.remove(audio_file_path_local)
                except Exception as er: print(f"Error removing temp TTS file for sentence: {er}")
            if i < len(sentences) - 1: # 如果不是最後一句出錯，詢問是否繼續
                print("DEBUG: TTS 處理句子時發生錯誤，將嘗試跳過此句並繼續（如果有的話）。")
            else:
                break # 最後一句出錯，直接結束
    
    # 等待最後一個句子的播放執行緒結束
    if tts_playback_thread is not None and tts_playback_thread.is_alive():
        print("DEBUG: Waiting for the final TTS playback thread to complete...")
        tts_playback_thread.join()
    print("DEBUG: All sentences processed or playback stopped.")
    tts_audio_playing = False # 確保在所有句子處理完畢或中斷後，此旗標為 False
    # tts_playback_should_stop = False # 此旗標應由打斷邏輯重設，或在新的 TTS 序列開始時重設

STT_IDLE_TIMEOUT_FLAG = "__STT_IDLE_TIMEOUT__"

# LLM 分析使用者輸入的分類標籤
ANALYSIS_END_CONVERSATION = "END_CONVERSATION"
ANALYSIS_IGNORE = "IGNORE_NOISE_OR_UNDIRECTED"
ANALYSIS_VALID = "VALID_FOR_ASSISTANT"
ANALYSIS_UNKNOWN = "UNKNOWN_ANALYSIS"

# LLM 分析 TTS 打斷意圖的分類標籤
TTS_INTERRUPT_YES = "INTERRUPT_TTS"
TTS_INTERRUPT_NO = "CONTINUE_TTS"
TTS_INTERRUPT_UNKNOWN = "UNKNOWN_TTS_INTERRUPT_ANALYSIS"

def listen_print_loop(responses, stream, idle_timeout_seconds):
    """迭代 Speech API 回應並處理結果，更流暢地顯示中間結果，並加入空閒超時。"""
    num_chars_printed = 0
    final_transcript_parts = []
    last_activity_time = time.time() # 追蹤上次活動時間

    try:
        for response in responses:
            if stream.closed: # 檢查串流是否在外部被關閉
                print("\n串流已關閉，停止辨識。")
                break
                
            current_time = time.time()
            if current_time - last_activity_time > idle_timeout_seconds:
                print(f"\nSTT 閒置超過 {idle_timeout_seconds} 秒，將返回 VAD 模式。")
                if not stream.closed:
                    stream.closed = True # 關閉麥克風串流
                return STT_IDLE_TIMEOUT_FLAG # 返回超時標誌

            if not response.results:
                # 即使沒有 results，也算是一種活動 (API 仍在回應)
                # 但如果長時間都是這樣，上面的 idle_timeout 會處理
                # 我們可以考慮如果連續多次都是 no results，也視為一種 idle，但目前先依賴 timeout
                last_activity_time = current_time # 更新活動時間
                continue

            result = response.results[0]
            if not result.alternatives:
                last_activity_time = current_time # 更新活動時間
                continue

            last_activity_time = current_time # 偵測到有效結果，更新活動時間
            transcript = result.alternatives[0].transcript

            if not result.is_final:
                print(f"\r{' ' * (num_chars_printed + 20)}\r{transcript}", end="", flush=True)
                num_chars_printed = len(transcript)
            else:
                print(f"\r{' ' * (num_chars_printed + 20)}\r", end="", flush=True)
                print(f"您說: {transcript}")
                final_transcript_parts.append(transcript)
                # Google STT 的 is_final 通常意味著一句話結束或一個較長的靜音期
                # 如果需要一句話結束就立即返回 VAD，可以在這裡直接 break
                # 目前行為是等待 listen_for_speech 中的 streaming_recognize 自然結束或 KeyboardInterrupt
                # 或者由 idle_timeout_seconds 控制
                # 若要實現說完一句就回 VAD，則取消下面的 break 註解
                # break 
                # 為了符合「沒有最終辨識結果則超時」的邏輯，這裡獲得 is_final 後應該算是成功，不再觸發 idle timeout
                # 若要一句話結束就返回，應該在 listen_for_speech 層級做，或這裡直接 return
                # 目前保持原樣，讓 Google STT 的 streaming_recognize 本身的 utterance detection 或 deadline 控制主要流程
                # 我們的 idle_timeout 主要處理「長時間完全沒聲音或沒辨識結果」的情況
                return "".join(final_transcript_parts) # 收到 final result 就返回，避免後續的 idle timeout

    except Exception as e:
        # responses 迭代器可能會因為串流關閉等原因拋出例外，例如 google.api_core.exceptions.Cancelled
        if "Deadline Exceeded" in str(e) or ("cancelled" in str(e).lower() and stream.closed):
            print(f"\nSTT 串流結束 (可能因超時或手動關閉): {e}")
        elif " الروم" in str(e): # 特殊字元錯誤，可能由 playsound 或 Unicode 問題引起，這裡忽略以避免崩潰
            print(f"\nSTT 處理中遇到非預期字元錯誤，忽略: {e}")
        else:
            print(f"\nlisten_print_loop 中發生錯誤: {e}")
            # traceback.print_exc() # 需要時可以打開來除錯
        if not stream.closed:
            stream.closed = True # 確保串流關閉
        # 如果是因為 iterator 結束或錯誤，且沒有 final_transcript，則返回 None
        # 如果已有部分 transcript，則返回已有的部分，或根據需求返回 None/Timeout
        if not final_transcript_parts: # 如果出錯且沒有任何最終結果
            return STT_IDLE_TIMEOUT_FLAG # 也可視為一種超時或失敗
    
    # 如果迴圈正常結束 (例如 streaming_recognize 結束) 且有結果
    return "".join(final_transcript_parts)

def analyze_user_input_with_llm(user_input_text):
    """使用 LLM 分析使用者輸入的意圖。"""
    global gemini_model
    if not gemini_model:
        print("錯誤: Gemini 模型未初始化，無法分析使用者輸入。")
        return ANALYSIS_UNKNOWN
    if not user_input_text or not user_input_text.strip():
        return ANALYSIS_IGNORE # 空輸入直接視為可忽略

    prompt = f"""你是一個對話分析助手。請分析以下使用者語音輸入：
使用者輸入：'{user_input_text}'

請根據以下分類判斷此輸入最符合哪一項：
1. END_CONVERSATION：使用者明確表示想結束當前這一輪對話，但不是要關閉整個程式。(例如："沒事了"、"我先這樣"、"好了，謝謝")
2. IGNORE_NOISE_OR_UNDIRECTED：這段語音可能是背景噪音、無意義的聲音片段、使用者並非在與你 (AI助手) 說話，或者語句非常不完整以至於無法理解意圖。
3. VALID_FOR_ASSISTANT：這段語音是使用者針對你 (AI助手) 發出的、有意義的請求、陳述或問題，並且不是上述兩種情況。

請只專注於判斷意圖，不要嘗試回答使用者的問題。
你必須只返回以下三個標籤之一：END_CONVERSATION、IGNORE_NOISE_OR_UNDIRECTED、VALID_FOR_ASSISTANT。
你的判斷是："""

    try:
        print(f"DEBUG: [analyze_user_input_with_llm] 正在向 Gemini 發送分析請求: '{user_input_text}'")
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1) # 低溫以求穩定分類
        )
        analysis_text = response.text.strip().upper()
        print(f"DEBUG: [analyze_user_input_with_llm] Gemini 分析結果原文: '{analysis_text}'")

        if analysis_text == ANALYSIS_END_CONVERSATION:
            return ANALYSIS_END_CONVERSATION
        elif analysis_text == ANALYSIS_IGNORE:
            return ANALYSIS_IGNORE
        elif analysis_text == ANALYSIS_VALID:
            return ANALYSIS_VALID
        else:
            print(f"警告: LLM 分析返回了未知的標籤: '{analysis_text}'")
            return ANALYSIS_UNKNOWN
    except Exception as e:
        print(f"LLM 分析使用者輸入時發生錯誤: {e}")
        traceback.print_exc()
        return ANALYSIS_UNKNOWN

def is_tts_interruption_intent_llm(text_input):
    """使用 LLM 判斷使用者是否想打斷 TTS 播放。"""
    global gemini_model
    if not gemini_model:
        print("錯誤: Gemini 模型未初始化。")
        return TTS_INTERRUPT_UNKNOWN
    if not text_input or not text_input.strip():
        return TTS_INTERRUPT_NO # 空輸入不視為打斷

    # 提示工程：設計一個精確的提示來識別打斷意圖
    prompt = f"""你是一個對話分析助手。目前 AI 助手正在說話，使用者同時也說了話。
AI 正在說話時，使用者說了：'{text_input}'

請判斷使用者的這句話是否意圖是要求 AI 助手「停止當前正在播放的語音」？
可能表達停止的詞語包括但不限於："停止"、"停下來"、"別說了"、"夠了"、"stop"、"shut up"、"安靜" 等等。
如果使用者只是提出一個新的問題，或者評論 AI 說的內容，但不包含明確的停止指令，則不應視為打斷意圖。

你必須只返回以下兩個標籤之一：
- INTERRUPT_TTS (使用者想讓 AI 停止說話)
- CONTINUE_TTS (使用者並非想讓 AI 停止說話)

你的判斷是："""

    try:
        print(f"DEBUG: [is_tts_interruption_intent_llm] 正在向 Gemini 發送 TTS 打斷分析請求: '{text_input}'")
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.05) # 極低溫以求最精確分類
        )
        analysis_text = response.text.strip().upper()
        print(f"DEBUG: [is_tts_interruption_intent_llm] Gemini TTS 打斷分析結果原文: '{analysis_text}'")

        if analysis_text == TTS_INTERRUPT_YES:
            return TTS_INTERRUPT_YES
        elif analysis_text == TTS_INTERRUPT_NO:
            return TTS_INTERRUPT_NO
        else:
            print(f"警告: LLM TTS 打斷分析返回了未知的標籤: '{analysis_text}'")
            return TTS_INTERRUPT_UNKNOWN
    except Exception as e:
        print(f"LLM TTS 打斷分析時發生錯誤: {e}")
        traceback.print_exc()
        return TTS_INTERRUPT_UNKNOWN

def quick_stt_for_barge_in(audio_data_chunk):
    """為打斷偵測執行一次快速的 STT。"""
    global speech_client, STT_LANGUAGE_CODE, SAMPLE_RATE
    if not speech_client:
        print("錯誤：SpeechClient 未初始化，無法執行快速 STT。")
        return None
    if not audio_data_chunk:
        return None

    print("DEBUG: [quick_stt_for_barge_in] 執行快速 STT...")
    try:
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=STT_LANGUAGE_CODE,
            max_alternatives=1,
            enable_automatic_punctuation=False, # 打斷指令通常不需要標點
        )
        # 使用 non-streaming recognize，因為我們只有一個小音訊塊
        # 這裡不使用 StreamingRecognitionConfig，而是 RecognizeRequest
        request = speech.RecognizeRequest(config=config, audio=speech.RecognitionAudio(content=audio_data_chunk))
        
        response = speech_client.recognize(request=request)

        if response and response.results:
            transcript = response.results[0].alternatives[0].transcript.strip()
            print(f"DEBUG: [quick_stt_for_barge_in] 快速 STT 結果: '{transcript}'")
            return transcript
        else:
            print("DEBUG: [quick_stt_for_barge_in] 快速 STT 未返回結果。")
            return None
    except Exception as e:
        print(f"快速 STT 辨識時發生錯誤: {e}")
        # traceback.print_exc()
        return None

def listen_for_speech():
    """使用 Google Cloud Speech-to-Text 進行串流語音辨識。"""
    global speech_client, STT_LANGUAGE_CODE, SAMPLE_RATE, STT_IDLE_TIMEOUT_SECONDS # 加入 STT_IDLE_TIMEOUT_SECONDS
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
        mic_stream = MicrophoneStream(SAMPLE_RATE, PYAUDIO_CHUNK_SIZE) # 使用 PYAUDIO_CHUNK_SIZE
        with mic_stream as stream:
            audio_generator = stream.generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )
            print("\rAI: 正在聆聽... 您說: ", end="", flush=True) 
            responses = speech_client.streaming_recognize(streaming_config, requests)
            recognized_text = listen_print_loop(responses, stream, STT_IDLE_TIMEOUT_SECONDS) # 傳入超時設定
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
    # 首先宣告所有全域變數
    global gemini_model, vad
    global VAD_AGGRESSIVENESS, VAD_ACTIVATION_FRAMES, SILENCE_DURATION_END_SPEECH
    global SAMPLE_RATE, PYAUDIO_CHUNK_SIZE, VAD_CHUNK_SIZE, VAD_FRAME_MS
    global STT_IDLE_TIMEOUT_SECONDS
    global tts_audio_playing, tts_playback_should_stop, tts_playback_thread

    chat_history_for_gemini = []
    
    print("DEBUG: main() 進入點") # 新增除錯
    speak_google_tts("您好！我是您的 AI 助手。請開始說話，我會聆聽。") # 修改提示語
    print("DEBUG: speak_google_tts() 完成") # 新增除錯
    
    # 初始化 VAD
    vad = webrtcvad.Vad()
    if not 0 <= VAD_AGGRESSIVENESS <= 3:
        print(f"警告：無效的 VAD_AGGRESSIVENESS 值 ({VAD_AGGRESSIVENESS})，將使用預設值 1。")
        VAD_AGGRESSIVENESS = 1
    vad.set_mode(VAD_AGGRESSIVENESS)
    
    keep_listening = True
    print(f"DEBUG: keep_listening 初始化為: {keep_listening}") # 新增除錯
    
    listening_for_activation = True # 新增狀態：是否正在等待語音啟動
    consecutive_speech_frames = 0
    silence_start_time = None
    accumulated_barge_in_audio = bytearray() # 用於累積打斷時的音訊
    BARGE_IN_AUDIO_DURATION_MS = 1500 # 嘗試收集約 1.5 秒的音訊用於打斷 STT
    MAX_BARGE_IN_AUDIO_CHUNKS = int(BARGE_IN_AUDIO_DURATION_MS / VAD_FRAME_MS)

    mic_stream_vad = None # 用於 VAD 偵測的麥克風串流

    while keep_listening:
        # print("DEBUG: 進入 while keep_listening 迴圈頂部") # 新增除錯
        # print("-" * 30)

        # 主要狀態機：
        # 1. TTS 正在播放且未被要求停止：進入打斷偵測模式
        # 2. 非 TTS 播放，或 TTS 已被要求停止：進入 VAD -> STT -> LLM 正常對話流程

        if tts_audio_playing and not tts_playback_should_stop:
            # --- 插入處理模式 (Barge-in Detection) ---
            # print("AI: (TTS 播放中，監聽打斷...)", end="\r", flush=True)
            if mic_stream_vad is None or mic_stream_vad.closed:
                try:
                    mic_stream_vad = MicrophoneStream(SAMPLE_RATE, PYAUDIO_CHUNK_SIZE)
                    mic_stream_vad_context = mic_stream_vad.__enter__()
                    audio_iterator_vad = mic_stream_vad_context.generator()
                    print("DEBUG: Barge-in VAD stream opened.")
                    accumulated_barge_in_audio.clear()
                    consecutive_speech_frames = 0 # 重置 VAD 連續語音幀
                except Exception as e_vad_open:
                    print(f"開啟 VAD 麥克風串流時發生錯誤 (barge-in): {e_vad_open}")
                    time.sleep(0.1) # 稍後重試
                    continue
            
            try:
                audio_chunk = next(audio_iterator_vad)
                if audio_chunk is None: 
                    time.sleep(0.01)
                    continue

                is_speech_in_chunk_barge_in = False
                for i in range(0, len(audio_chunk), VAD_CHUNK_SIZE * 2):
                    frame = audio_chunk[i : i + VAD_CHUNK_SIZE * 2]
                    if len(frame) < VAD_CHUNK_SIZE * 2: continue
                    try:
                        if vad.is_speech(frame, SAMPLE_RATE):
                            is_speech_in_chunk_barge_in = True
                            accumulated_barge_in_audio.extend(frame) # 累積偵測到的語音部分
                            # print(f"DEBUG: Barge-in VAD speech frame detected. Total accumulated: {len(accumulated_barge_in_audio)} bytes")
                            break 
                    except Exception as e_vad_barge:
                        print(f"Barge-in VAD 處理時發生錯誤: {e_vad_barge}")
                        is_speech_in_chunk_barge_in = False
                        break
                
                if is_speech_in_chunk_barge_in:
                    consecutive_speech_frames += (len(audio_chunk) // (VAD_CHUNK_SIZE * 2))
                    # print(f"DEBUG: Barge-in consecutive speech frames: {consecutive_speech_frames}")
                    # 累積足夠的音訊 (或 VAD 連續幀數達到某個閾值) 後才進行 STT
                    # 這裡用累積的音訊長度判斷，或者用連續 VAD 幀數
                    if len(accumulated_barge_in_audio) >= (VAD_CHUNK_SIZE * 2 * (MAX_BARGE_IN_AUDIO_CHUNKS // 2)) or consecutive_speech_frames >= (VAD_ACTIVATION_FRAMES // 2 +1 ) :
                        print("\nDEBUG: 偵測到足夠的插入語音，嘗試進行快速 STT 和意圖分析...")
                        barge_in_text = quick_stt_for_barge_in(bytes(accumulated_barge_in_audio))
                        accumulated_barge_in_audio.clear() # 清空已處理的音訊
                        consecutive_speech_frames = 0 # 重置

                        if barge_in_text:
                            interrupt_analysis = is_tts_interruption_intent_llm(barge_in_text)
                            if interrupt_analysis == TTS_INTERRUPT_YES:
                                print("AI: 偵測到停止 TTS 指令！")
                                tts_playback_should_stop = True # 設定停止旗標
                                # speak_google_tts 內部的 worker thread 目前無法被 winsound.PlaySound 中斷，這是已知限制
                                # 但 tts_audio_playing 會在 worker 結束後變 False
                                # 並且主迴圈接下來會因為 tts_playback_should_stop=True 而跳出此分支
                                speak_google_tts("好的，已停止。") # 可以加一個簡短的回應
                                listening_for_activation = True # 重設回 VAD 等待模式
                                if mic_stream_vad is not None and not mic_stream_vad.closed:
                                    mic_stream_vad.__exit__(None, None, None)
                                    mic_stream_vad = None
                                continue # 跳到主迴圈頂部，會進入 VAD 模式
                            else:
                                print(f"AI: (使用者在 TTS 播放時說了：'{barge_in_text}'，但非停止指令)", flush=True)
                    else:
                            print("AI: (插入語音 STT 未能辨識)", flush=True)
                else: # audio_chunk 中沒有語音
                    if len(accumulated_barge_in_audio) > 0 and consecutive_speech_frames == 0:
                        # 如果之前有累積的語音，但現在這一個chunk沒有語音了，說明可能是一小段話結束
                        # 也可以考慮在這裡處理 accumulated_barge_in_audio
                        print("DEBUG: Barge-in VAD speech ended, clearing small accumulated audio.")
                        accumulated_barge_in_audio.clear()
                    consecutive_speech_frames = 0 # 重置連續計數

            except StopIteration:
                print("DEBUG: Barge-in VAD audio stream ended unexpectedly.")
                if mic_stream_vad is not None and not mic_stream_vad.closed:
                    mic_stream_vad.__exit__(None, None, None)
                mic_stream_vad = None 
                time.sleep(0.1)
            except Exception as e_barge_loop:
                print(f"Barge-in 監聽迴圈發生錯誤: {e_barge_loop}")
                traceback.print_exc()
                if mic_stream_vad is not None and not mic_stream_vad.closed:
                    mic_stream_vad.__exit__(None, None, None)
                mic_stream_vad = None
                time.sleep(0.1)
            
            time.sleep(VAD_FRAME_MS / 2000) # 在 TTS 播放時，VAD 偵測可以更頻繁一些
            continue # 繼續 TTS 打斷偵測迴圈

        # --- 非 TTS 播放期間的正常 VAD -> STT -> LLM 流程 ---
        if listening_for_activation:
            # print("AI: (等待人聲啟動...)", end="\r", flush=True)
            if mic_stream_vad is None or mic_stream_vad.closed: # 確保 VAD 麥克風是開啟的
                try:
                    mic_stream_vad = MicrophoneStream(SAMPLE_RATE, PYAUDIO_CHUNK_SIZE) 
                    mic_stream_vad_context = mic_stream_vad.__enter__() 
                    audio_iterator_vad = mic_stream_vad_context.generator()
                    print("DEBUG: Standard VAD stream opened.")
                    consecutive_speech_frames = 0 # 重置 VAD 連續語音幀
                except Exception as e_vad_open_std:
                    print(f"開啟 VAD 麥克風串流時發生錯誤 (standard): {e_vad_open_std}")
                    time.sleep(0.1) # 稍後重試
                    continue
            
            try:
                audio_chunk = next(audio_iterator_vad)
                if audio_chunk is None: 
                    time.sleep(0.01)
                    continue

                is_speech_in_chunk_std = False
                for i in range(0, len(audio_chunk), VAD_CHUNK_SIZE * 2):
                    frame = audio_chunk[i : i + VAD_CHUNK_SIZE * 2]
                    if len(frame) < VAD_CHUNK_SIZE * 2: continue
                    try:
                        if vad.is_speech(frame, SAMPLE_RATE):
                            is_speech_in_chunk_std = True
                            break 
                    except Exception as e_vad_std:
                        print(f"Standard VAD 處理時發生錯誤: {e_vad_std}")
                        is_speech_in_chunk_std = False
                        break
                
                if is_speech_in_chunk_std:
                    consecutive_speech_frames += (len(audio_chunk) // (VAD_CHUNK_SIZE * 2))
                    # print(f"DEBUG: Standard consecutive speech frames: {consecutive_speech_frames}")
                    if consecutive_speech_frames >= VAD_ACTIVATION_FRAMES:
                        print("\nAI: 偵測到足夠語音，啟動完整聆聽模式。")
                        listening_for_activation = False
                        consecutive_speech_frames = 0 # 重置
                        silence_start_time = None # 重置靜音計時
                        # 關閉僅用於 VAD 的串流
                        if mic_stream_vad is not None and not mic_stream_vad.closed:
                            mic_stream_vad.__exit__(None, None, None)
                            mic_stream_vad = None
                        # 接下來的迴圈會進入 STT 模式
                else:
                    consecutive_speech_frames = 0 # 沒有偵測到語音，重置連續計數
                    print("AI: (等待人聲啟動... 清除)", end="\r", flush=True)

            except StopIteration:
                print("DEBUG: Standard VAD audio stream ended unexpectedly.")
                if mic_stream_vad is not None and not mic_stream_vad.closed:
                    mic_stream_vad.__exit__(None, None, None)
                mic_stream_vad = None 
                time.sleep(0.1)
            except Exception as e_std_loop:
                print(f"Standard 監聽迴圈發生錯誤: {e_std_loop}")
                traceback.print_exc()
                if mic_stream_vad is not None and not mic_stream_vad.closed:
                    mic_stream_vad.__exit__(None, None, None)
                mic_stream_vad = None
                time.sleep(0.1)
            
            time.sleep(VAD_FRAME_MS / 2000) # 在 TTS 播放時，VAD 偵測可以更頻繁一些
            continue # 繼續 VAD 打斷偵測迴圈

        # 如果 tts_playback_should_stop 為 True，表示 TTS 播放被要求停止，進入 VAD -> STT -> LLM 流程
        if tts_playback_should_stop:
            print("AI: TTS 播放被要求停止，進入 VAD -> STT -> LLM 流程")
            listening_for_activation = True
            silence_start_time = None
            consecutive_speech_frames = 0
            continue # 返回迴圈頂部，進入 VAD 模式

    # 迴圈結束，清理 VAD 麥克風串流
    if mic_stream_vad is not None and not mic_stream_vad.closed:
        print("程式結束，關閉 VAD 麥克風串流...")
        mic_stream_vad.__exit__(None, None, None)
        mic_stream_vad = None
    
    # 確保 TTS 播放執行緒也被妥善處理 (如果仍在執行)
    if tts_playback_thread is not None and tts_playback_thread.is_alive():
        print("程式結束，通知 TTS 播放執行緒停止...")
        tts_playback_should_stop = True
        tts_playback_thread.join(timeout=1.0) # 等待最多1秒
        if tts_playback_thread.is_alive():
            print("警告: TTS 播放執行緒未能及時停止。")

if __name__ == "__main__":
    if load_and_configure():
        main()
    else:
        print("程式因設定錯誤而終止。") 