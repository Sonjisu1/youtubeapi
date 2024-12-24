from googleapiclient.discovery import build
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  # LSA 요약 알고리즘
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from krwordrank.sentence import summarize_with_sentences


from konlpy.tag import Okt
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer



from googleapiclient.discovery import build

# 호스트
API_KEY ='apikey'
YOUTUBE_API_VERSION='v3' #이용할 api 버전 
YOUTUBE_API_SERVICE='youtube' #이용할 api 


def get_video_links():
    youtube = build(YOUTUBE_API_SERVICE, YOUTUBE_API_VERSION, developerKey=API_KEY)
    
    # 
    search_request = youtube.search().list(
        q='대통령령', #검색할 키워드명
        order='relevance',
        part='snippet',
        maxResults='3',
        type='video'
    )
    search_response = search_request.execute()
    
    video_ids = []
    video_links = []
    
    # 검색 결과에서 동영상 ID 수집
    for item in search_response['items']:
        video_id = item['id']['videoId']
        video_ids.append(video_id)
        video_links.append({
            "title": item['snippet']['title'],
            "link": f'https://www.youtube.com/watch?v={video_id}',
            "channel": item['snippet']['channelTitle'],
            "thumbnail": item['snippet']['thumbnails']['high']['url'],
            "published_at": item['snippet']['publishedAt']
        })
    
     # 동영상 상세 정보 요청 
    if video_ids:
        video_request = youtube.videos().list(
            part='snippet,statistics',
            id=','.join(video_ids) # 여러 ID를 콤마로 구분
        )
        video_response = video_request.execute()
        
        # 상세 정보 매칭
        for idx, video in enumerate(video_response['items']):
            video_links[idx]['description'] = video['snippet']['description']
            video_links[idx]['views'] = video['statistics'].get('viewCount', 'N/A')
            video_links[idx]['likes'] = video['statistics'].get('likeCount', 'N/A')
            video_links[idx]['comments'] = video['statistics'].get('commentCount', 'N/A')

    
    return video_links

# 텍스트 전처리 함수
def preprocess_text(text):
    # 불필요한 공백 제거
    cleaned_text = " ".join(text.split())
    return cleaned_text


    #기사요약
def summarize_text(text):
    
    # cleaned_text = re.sub(r"[\r\n]+", " ", text)
    
    korean_text =clean_text(text)

    preprocessed_text = preprocess_text_konlpy(korean_text)
    # cleaned_text = preprocess_text(cleaned_text)
    parser = PlaintextParser.from_string(preprocessed_text, Tokenizer("korean"))

# 요약 알고리즘 및 설정
    summarizer = LsaSummarizer()  # LSA 기반 요약
    summarizer.stop_words = get_stop_words("korean")  # 불용어 설정

# 요약 생성 (2개의 문장으로 요약)
    summary = summarizer(parser.document, 2)

# 결과 출력
    print("요약 결과:")
    for sentence in summary:
        print(sentence)


okt = Okt()

def preprocess_text_konlpy(text):
   
    #특수문자 제거 
    text = re.sub(r'[^가-힣\s]', '', text)
    text = text.strip()
    
    #명사만 출력력
    nouns = okt.nouns(text)
    
    # 불용어 제거 
    stopwords = ['에서', '으로', '그리고', '입니다', '를', '은', '는', '이', '가','해서']
    filtered_nouns = [word for word in nouns if word not in stopwords and len(word) > 1]
    
    return " ".join(filtered_nouns)


def summarize_with_transformer(text, model_name="gogamza/kobart-base-v2"):
   
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True) 

def preprocess_text(text):
   
    text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
    text = re.sub(r'\s+', ' ', text)  # 공백 압축
    return text.strip()

# 요약문에 포함할 문장의 개수를 지정 (기본값: 3개).
# TextRank 알고리즘을 사용하여 텍스트에서 중요한 문장을 선택.
def summarize_with_textrank(text, num_sentences=3):
  
    text = preprocess_text(text)
    # 텍스트를 한글에 맞는 토큰으로 분리
    parser = PlaintextParser.from_string(text, Tokenizer("korean"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    # 문장들을 개행 문자(\n)로 연결하여 최종 요약문 생성.
    return "\n".join(str(sentence) for sentence in summary)

def clean_text(text):
  
    text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
    text = re.sub(r'\s+', ' ', text)  # 공백 여러 개를 하나로 압축
    return text.strip()
if __name__ == "__main__":
    video_info = get_video_links()
    for video in video_info:
        print(video)
