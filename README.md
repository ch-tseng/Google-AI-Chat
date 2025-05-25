# Gemini 語音對話助手

這是一個使用 Google Gemini AI 模型的語音對話程式，可以透過麥克風與 AI 進行對話。

## 前置需求

- Python 3.7 或更新版本
- 麥克風
- 喇叭或耳機
- Google Gemini API 金鑰

## 安裝步驟

1. 安裝所需的套件：
```bash
pip install -r requirements.txt
```

2. 在 `voice_chat.py` 中設定您的 Google Gemini API 金鑰：
   - 將 `YOUR_API_KEY` 替換為您的實際 API 金鑰

## 使用方法

1. 執行程式：
```bash
python voice_chat.py
```

2. 當看到「請說話...」的提示時，開始說話
3. 程式會自動辨識您的語音，並透過 Gemini AI 產生回應
4. 要結束對話，請說「再見」、「拜拜」或「結束」

## 注意事項

- 確保您的麥克風已正確連接並設定
- 需要穩定的網路連線
- 語音辨識支援中文（繁體） 