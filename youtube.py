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
if __name__ == "__main__":
    video_info = get_video_links()
    for video in video_info:
        print(video)
