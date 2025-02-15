'''
起動コマンド
streamlit run test_LlamaCpp.py
'''

# 各ライブラリのインポート
import streamlit as st
import concurrent.futures
from langchain_community.llms import LlamaCpp

# セッション状態を管理するためのキーチェーン設定
if "messages" not in st.session_state:
    st.session_state.messages = [] # 初期化
    
# モデルのパスを定義
MODEL_NAME = "お気に入りのモデルのパス"
print(f"Using model from: {MODEL_NAME}")

# モデルの準備
llm = LlamaCpp(model_path=MODEL_NAME)

# モデルを実行し回答を生成
def use_model(text):     
    # プロンプトを定義 (例)
    prompt = (
        "あなたは誠実で優秀な日本人のアシスタントです。"
        "指示がない限り常に日本語で回答、かつ詳細に回答してください。"
        "具体的な例や詳細を挙げて、豊かに説明してください"
        "どんなに短い回答でも何度も繰り返さないようにしてください"
        "設定したトークン数を750を超えないようにしてください:"
        " {}。".format(text)
    )
    
    
    # トークン数を設定 （）
    max_tokens = 750 # 最大トークン数
    
    # モデルの実行
    try:
        response = llm.invoke(prompt, max_tokens=max_tokens)
    except Exception as e:
        response = "エラーが発生しました: {}".format(str(e))
        
    return response

# メッセージを処理する関数
def process_message(text):
    response = use_model(text)
    
    return response

# UI(チャットのやり取りの部分)
def main():
    # タイトル
    st.title("CHAT - ELYZA (試作版・調整中)")
    st.caption("・このアプリで使用しているモデルは、[ Llama-3-ELYZA-JP-8B-q4_k_m ]です。")
    st.caption("・AIは妄想・暴走することがあります。")
    st.caption("・AIの回答精度を過信せず、必ずファクトチェックを行ってください。")
    st.caption("・回答までの時間は、質問内容や本体のスペックに左右されます。")
    
    #会話をリセット
    if st.button("会話をリセット"):
        st.session_state.messages = [] # 保存された会話をリセット
        
    # 入力フォームと送信ボタンのUI
    st.chat_message("assistant").markdown("何か聞きたいことはありませんか？")
    text = st.chat_input("ここにメッセージを入力してください")

    # チャットのUI
    if text: 
        st.session_state.messages.append({"role": "user", "content": text})       
        st.chat_message("user").markdown(text)
        
        # 非同期で応答を取得"
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(process_message, text) 
            answer = future.result() # 応答を待つ
            
        st.session_state.messages.append({"role": "assistant", "content": answer})  
        st.chat_message("assistant").markdown(answer)

        
    # 会話履歴を表示
    st.subheader("過去の会話履歴")
    
    if st.session_state.messages:
        for message in st.session_state.messages:    
            st.chat_message(message["role"]).markdown(message["content"])

if __name__ == "__main__":
    main()