# 프로젝트 설명(서강대학교 빅데이터기반마케팅 2차 과제)
-----
## 개요
----
2022년 1분기 기준, 10개 외식업종 중 개업률이 가장 높은 분야가 커피·음료다. 
타업종에 비해 진입장벽이 낮은 카페 창업에 많은 사람들이 관심을 보이고 있다. 
이미 레드오션이 된 카페시장에서 신규진입자들은 면밀한 조사를 통한 전략수립이 필요해진 상황이다. 
## 분석 목적
-----
어디에, 어떻게 카페를 창업할지 고민하고 있는, 예비창업자들을 위하여 서비스를 준비하고자 한다.
카카오맵 게시글과 댓글을 크롤링해 온라인 상의 VoC(Voice of Customer) 데이터를 수집하고,
이를 바탕으로 지역구별 카페 트렌드를 비교분석한다. 
이러한 분석결과를 바탕으로, 카페 창업을 위한 지역구 추천을 해주고, 
지역구별 카페 트렌드를 시각화하여 보여주어, 창업자들에게 유용한 정보를 제공한다. 
## 분석 과정
----
![image](https://user-images.githubusercontent.com/89527573/209966604-5970367f-df1b-4dff-9fe6-27bf10773f13.png)

#### 클러스터링 예시
----
![image](https://user-images.githubusercontent.com/89527573/209966831-1f5f04c8-0fe6-4781-875c-5e517f7044fc.png)
![image](https://user-images.githubusercontent.com/89527573/209966914-1f77e358-9db9-4afc-a005-2cb0783215af.png)


#코드설명
-----
### cafe.ipnyb
-----
카카오맵 상의 서울에 위치한 카페 링크 및 데이터 수집
### cafe_review.ipnyb
-----
각 카페에 해당하는 링크에 들어가서 블로그 및 카카오 리뷰 수집
### clean.py
연관성 없는 데이터, 너무 짧은 데이터 삭제(리눅스 환경에서 작동)
cpu 개수에 따른 병렬 처리
### std.py
-----
customized okt를 이용, 토큰화 진행 전 단어 정규화
cpu 개수에 따른 병렬 처리
### tok.py 
-----
품사 태깅 및 토큰화
cpu 개수에 따른 병렬 처리
final_sample.csv 
### freq.py
-----
단어 빈도 

### Tableu
![image](https://user-images.githubusercontent.com/89527573/209967295-d090f2be-52b4-418c-846b-4a1e32882a52.png)
https://public.tableau.com/app/profile/.75641703/viz/_16708103803200/1
