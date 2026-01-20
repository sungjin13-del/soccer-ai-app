import streamlit as st
import pandas as pd
import io
import json
import re
import os
import time
from datetime import datetime
from PIL import Image
import google.generativeai as genai
from duckduckgo_search import DDGS

# ==========================================
# 1. 설정 및 메모리 시스템
# ==========================================
st.set_page_config(page_title="AI 축구 분석기 V22 (Model Auto-Discovery)", layout="wide")
HISTORY_FILE = "match_history.csv"

def init_session():
    if 'history' not in st.session_state:
        if os.path.exists(HISTORY_FILE):
            st.session_state.history = pd.read_csv(HISTORY_FILE)
        else:
            st.session_state.history = pd.DataFrame(columns=["Date", "Home", "Away", "AI_Pick", "Result", "Correct"])
    
    if 'last_analysis' not in st.session_state: st.session_state.last_analysis = None
    if 'ref_data' not in st.session_state: st.session_state.ref_data = {}
    if 'available_models' not in st.session_state: st.session_state.available_models = []

init_session()

def save_result(home, away, ai_pick, actual_result):
    new_data = {
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Home": home, "Away": away, "AI_Pick": ai_pick,
        "Result": actual_result, "Correct": (ai_pick == actual_result)
    }
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_data])], ignore_index=True)
    st.session_state.history.to_csv(HISTORY_FILE, index=False)

def get_learning_context():
    df = st.session_state.history
    if df.empty: return "학습 기록 없음 (첫 분석)"
    total = len(df)
    acc = (len(df[df['Correct']==True])/total)*100
    return f"총 분석: {total}회 | 적중률: {acc:.1f}%"

# ==========================================
# 2. [핵심] 실제 사용 가능한 모델 조회 (404 방지)
# ==========================================
def fetch_available_models(api_key):
    """
    사용자 키로 실제 접속 가능한 모델 목록을 가져옵니다.
    """
    try:
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            # 텍스트 생성이 가능한 모델만 필터링
            if 'generateContent' in m.supported_generation_methods:
                # models/gemini-pro -> gemini-pro 로 변환
                clean_name = m.name.replace("models/", "")
                models.append(clean_name)
        return models
    except Exception as e:
        return []

def call_gemini_safe(api_key, model_name, content):
    """
    429(사용량 초과) 오류 발생 시 대기 후 재시도
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(content)
            return response.text
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower():
                wait = 60
                st.warning(f"⚠️ 사용량 제한(429) 감지. {wait}초 대기 후 재시도 ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            elif "404" in err_msg:
                return f"ERROR: 모델을 찾을 수 없습니다 ({model_name}). 모델 설정을 변경하세요."
            else:
                return f"ERROR: {err_msg}"
    return "ERROR: 재시도 횟수 초과"

# ==========================================
# 3. 기능 함수
# ==========================================
def search_web(home, away):
    ddgs = DDGS()
    q = f"{home} vs {away} match prediction stats injuries {datetime.now().year}"
    txt = ""
    try:
        results = ddgs.text(q, max_results=3)
        if results:
            for r in results: txt += f"- {r['body']}\n"
        else:
            txt = "검색 결과 없음."
    except Exception as e:
        txt = f"웹 검색 오류: {str(e)}"
    return txt

def analyze_match_final(api_key, model_name, home_in, away_in, search_txt, img_objs, learning_ctx):
    prompt = f"""
    Act as a professional football analyst.
    
    **Task 1: Translation & Identity**
    Input: "{home_in}" vs "{away_in}"
    Identify standard English team names.
    
    **Task 2: Analysis**
    - Match: {home_in} vs {away_in}
    - Web Info: {search_txt}
    - Memory: {learning_ctx}
    
    Analyze winner, score, and reasons.
    
    **Output JSON ONLY:**
    {{
        "teams_en": "Home(En) vs Away(En)",
        "winner": "{home_in}" or "{away_in}" or "Draw",
        "confidence": 0-100,
        "score": "2-1",
        "reason": "Detailed analysis in Korean",
        "learning_note": "Feedback in Korean"
    }}
    """
    
    content = [prompt] + img_objs if img_objs else [prompt]
    
    # 안전 호출
    raw_text = call_gemini_safe(api_key, model_name, content)
    
    if "ERROR:" in raw_text:
        return {"error": raw_text}
        
    try:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match: return json.loads(match.group(0))
        return {"error": "JSON 파싱 실패", "raw": raw_text}
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 4. UI 구성
# ==========================================
st.title("🛡️ AI 축구 분석기 V22 (모델 자동 검색)")
st.caption("당신의 API 키로 사용 가능한 모델만 찾아내어 오류를 방지합니다.")

# 사이드바
st.sidebar.header("설정")
api_key = st.sidebar.text_input("API Key", type="password")

# [핵심] 모델 목록 불러오기 버튼
if api_key:
    if st.sidebar.button("🔄 사용 가능 모델 조회"):
        with st.sidebar.spinner("구글 서버에 모델 목록 요청 중..."):
            found_models = fetch_available_models(api_key)
            if found_models:
                st.session_state.available_models = found_models
                st.sidebar.success(f"{len(found_models)}개 모델 발견!")
            else:
                st.sidebar.error("사용 가능한 모델이 없습니다. API 키 권한을 확인하세요.")

# 모델 선택 박스
if st.session_state.available_models:
    model_name = st.sidebar.selectbox("사용할 모델 선택", st.session_state.available_models)
else:
    # 목록을 아직 못 불러왔을 때 기본값 (하지만 이게 404 원인이 될 수 있으므로 조회 권장)
    model_name = st.sidebar.text_input("모델명 직접 입력 (예: gemini-pro)", "gemini-pro")
    st.sidebar.info("👆 위 '모델 조회' 버튼을 누르면 정확한 목록이 뜹니다.")

# 학습 현황
if not st.session_state.history.empty:
    acc = (len(st.session_state.history[st.session_state.history['Correct']==True])/len(st.session_state.history))*100
    st.sidebar.metric("적중률", f"{acc:.1f}%")

# 입력창
c1, c2 = st.columns(2)
home_in = c1.text_input("🏠 홈팀 (한글)", "카이라트")
away_in = c2.text_input("✈️ 원정팀 (한글)", "클뤼브뤼")

st.divider()
col_s, col_u = st.columns(2)
with col_s: 
    use_search = st.checkbox("🌐 웹 검색 사용", value=True)
with col_u: 
    files = st.file_uploader("📸 분석 자료", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)

if st.button("🚀 분석 시작", type="primary"):
    if not api_key:
        st.error("❌ API 키를 입력해주세요.")
    else:
        with st.status("AI 가동 중...", expanded=True) as status:
            try:
                # 1. 검색
                search_res = "검색 안함"
                if use_search:
                    status.write("🌍 해외 정보 검색 중...")
                    search_res = search_web(home_in, away_in)
                
                # 2. 분석
                status.write(f"🧠 선택된 모델({model_name})로 분석 중...")
                ctx = get_learning_context()
                imgs = [Image.open(io.BytesIO(f.getvalue())) for f in files] if files else []
                
                result = analyze_match_final(api_key, model_name, home_in, away_in, search_res, imgs, ctx)
                
                if result and 'winner' in result:
                    st.session_state.last_analysis = result
                    st.session_state.ref_data = {
                        "teams": result.get('teams_en', 'N/A'),
                        "search": search_res,
                        "memory": ctx
                    }
                    status.update(label="분석 완료!", state="complete", expanded=False)
                elif result and 'error' in result:
                    status.update(label="오류 발생", state="error")
                    st.error(f"❌ {result['error']}")
                else:
                    st.error("❌ 분석 실패")
            except Exception as e:
                st.error(f"시스템 오류: {str(e)}")

# 결과 화면
if st.session_state.last_analysis:
    res = st.session_state.last_analysis
    st.divider()
    st.subheader(f"🎯 예측: {res.get('winner')} 승리")
    st.caption(f"신뢰도: {res.get('confidence')}% | 스코어: {res.get('score')}")
    st.info(f"📝 **분석:** {res.get('reason')}")
    st.warning(f"🎓 **학습 노트:** {res.get('learning_note')}")
    
    with st.expander("📚 원본 데이터 확인"):
        st.write(f"**팀명(영문):** {st.session_state.ref_data['teams']}")
        st.code(st.session_state.ref_data['search'])

    st.divider()
    b1, b2, b3 = st.columns(3)
    if b1.button(f"{home_in} 승"): save_result(home_in, away_in, res['winner'], home_in); st.toast("저장!"); st.rerun()
    if b2.button("무승부"): save_result(home_in, away_in, res['winner'], "무승부"); st.toast("저장!"); st.rerun()
    if b3.button(f"{away_in} 승"): save_result(home_in, away_in, res['winner'], away_in); st.toast("저장!"); st.rerun()
