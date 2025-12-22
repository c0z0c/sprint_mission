import random


def random_name(text: str = "") -> str:
    """랜덤 한글 이름 생성"""
    last_names = "김이박최정강조윤장임한오서신권황안송홍전"
    first_names = "민준서연지후서준하준도윤시우주원지안서우예준수아은우지우연우건우현우준서민서하은유진지훈지민소율지아시윤정우은지승현수빈예은서현민재다은채원윤서시은아인준혁지율"

    last_name = random.choice(last_names)
    # 이름은 2글자로 (랜덤하게 2글자 선택)
    idx = random.randrange(0, len(first_names) - 1, 2)
    first_name = first_names[idx : idx + 2]

    name = last_name + first_name
    return text + name if text else name


def random_review_text(text: str = "") -> str:
    """랜덤 리뷰 텍스트 생성 (짧은글, 중간글, 장문 포함)"""

    # 랜덤 단어 풀
    emotions = [
        "감동적",
        "놀라운",
        "최고의",
        "멋진",
        "환상적",
        "훌륭한",
        "완벽한",
        "대단한",
        "끝내주는",
        "감탄스러운",
    ]
    negative_emotions = [
        "실망스러운",
        "지루한",
        "평범한",
        "아쉬운",
        "답답한",
        "어설픈",
        "억지스러운",
        "졸린",
        "따분한",
        "뻔한",
    ]
    story_words = [
        "스토리",
        "이야기",
        "전개",
        "플롯",
        "구성",
        "서사",
        "줄거리",
        "내러티브",
    ]
    acting_words = ["연기", "연출", "표현", "감정선", "몰입감", "캐릭터", "캐스팅"]
    visual_words = [
        "영상미",
        "촬영",
        "CG",
        "특수효과",
        "미장센",
        "색감",
        "화면",
        "비주얼",
    ]
    music_words = ["음악", "OST", "사운드", "배경음악", "삽입곡", "스코어"]

    # 짧은 리뷰 (1-2문장)
    short_reviews = [
        f"{random.choice(emotions)}이에요! 강추합니다.",
        f"진짜 재밌네요 ㅋㅋㅋ",
        f"{random.choice(negative_emotions)}..ㅠㅠ",
        f"최고의 영화! 꼭 보세요!",
        f"시간 아까워요 비추",
        f"기대 이상이었습니다",
        f"별로예요",
        f"역대급이네요!!",
        f"그냥 그래요",
        f"완전 최악ㅠ",
    ]

    # 중간 리뷰 (3-5문장)
    medium_positive = [
        f"{random.choice(emotions)} 영화였습니다. {random.choice(story_words)}가 탄탄하고 {random.choice(acting_words)}도 {random.choice(emotions)}었어요. {random.choice(visual_words)}까지 완벽했습니다. 올해 본 영화 중 최고네요!",
        f"정말 재미있게 봤어요! {random.choice(acting_words)}가 인상적이었고, {random.choice(music_words)}도 좋았습니다. 몰입감이 대단해서 시간 가는 줄 몰랐어요. 추천합니다!",
        f"{random.choice(visual_words)}가 정말 아름다웠습니다. {random.choice(story_words)}도 탄탄했고, 배우들의 {random.choice(acting_words)}도 뛰어났어요. 웃다가 울다가 감정의 롤러코스터를 탔네요.",
        f"명작이에요. {random.choice(story_words)}의 완성도가 높고, {random.choice(acting_words)}가 자연스러웠습니다. {random.choice(music_words)}도 영화와 찰떡이었어요. 여운이 오래 남을 것 같습니다.",
        f"기대 이상이었어요! {random.choice(visual_words)}가 압도적이고, {random.choice(story_words)}도 흥미진진했습니다. 끝까지 긴장감을 놓을 수 없었네요. 강력 추천!",
    ]

    medium_negative = [
        f"{random.choice(negative_emotions)} 영화였어요. {random.choice(story_words)}가 산만하고 {random.choice(acting_words)}도 어색했습니다. 기대했는데 실망스럽네요. 비추천합니다.",
        f"너무 지루했어요. {random.choice(story_words)}가 뻔하고 예측 가능했습니다. {random.choice(acting_words)}는 괜찮았지만 각본이 문제예요. 중간에 졸 뻔 했네요.",
        f"{random.choice(negative_emotions)}했습니다. {random.choice(visual_words)}만 좋고 {random.choice(story_words)}는 엉망이에요. 과대평가된 영화 같아요. 기대치를 낮추고 보세요.",
        f"러닝타임이 너무 길어서 지쳤어요. {random.choice(story_words)}가 복잡하고 이해하기 어려웠습니다. {random.choice(acting_words)}도 과했어요. 추천하지 않습니다.",
        f"예고편이 더 재밌었어요. 본편은 {random.choice(negative_emotions)}하고 {random.choice(story_words)}도 {random.choice(negative_emotions)}했습니다. 억지 감동이 느껴졌어요.",
    ]

    medium_mixed = [
        f"호불호가 갈릴 것 같은 영화네요. {random.choice(visual_words)}는 좋았지만 {random.choice(story_words)}는 평범했어요. {random.choice(acting_words)}는 준수했습니다. 볼만은 해요.",
        f"전반부는 좋았는데 후반부가 아쉬웠어요. {random.choice(story_words)}의 완급조절이 좀 부족했습니다. 하지만 {random.choice(music_words)}는 인상적이었어요.",
        f"그럭저럭 봤습니다. {random.choice(acting_words)}는 훌륭했지만 {random.choice(story_words)}가 식상했어요. 기대만큼은 아니었지만 나쁘지는 않았습니다.",
    ]

    # 장문 리뷰 (1000자 이상)
    long_reviews = [
        f"""주말에 {random.choice(emotions)} 영화를 봤습니다. 처음 예고편을 봤을 때부터 기대가 컸는데, 역시 기대를 저버리지 않았어요.

{random.choice(story_words)}부터 이야기하자면, 정말 탄탄하게 짜여있었습니다. 기승전결이 명확하면서도 중간중간 반전이 있어서 지루할 틈이 없었어요. 특히 2막에서의 전환이 인상적이었는데, 관객들이 예상하지 못한 방향으로 {random.choice(story_words)}가 흘러가면서도 억지스럽지 않고 자연스러웠습니다. 복선도 영리하게 깔려 있어서 영화를 다 보고 나서 "아, 그때 그게 이거였구나" 하는 순간들이 많았어요.

{random.choice(acting_words)} 측면에서도 배우들의 열연이 돋보였습니다. 주연 배우의 {random.choice(acting_words)}는 정말 완벽했어요. 감정의 폭이 넓은 캐릭터를 소화하면서도 과하지 않게, 절제된 {random.choice(acting_words)}로 깊은 여운을 남겼습니다. 조연 배우들도 하나같이 훌륭했고, 특히 중반부에 나오는 감정씬에서는 같이 울컥했네요. 배우들 간의 케미스트리도 좋아서 관계의 변화가 설득력 있게 다가왔습니다.

{random.choice(visual_words)}와 {random.choice(music_words)}도 영화의 완성도를 한층 높였습니다. {random.choice(visual_words)}가 정말 압도적이었어요. 특히 클라이맥스 장면의 {random.choice(visual_words)}는 입이 떡 벌어질 정도였습니다. CG와 실사의 조화도 완벽했고, 색감 처리도 영화의 분위기와 잘 맞았어요. {random.choice(music_words)}는 장면마다 딱 맞아떨어져서 감정을 증폭시키는 역할을 톡톡히 했습니다. 엔딩곡도 여운이 남아서 영화관을 나온 후에도 계속 귓가에 맴돌았어요.

감독의 {random.choice(acting_words)}력도 빛을 발했습니다. 복잡한 {random.choice(story_words)}를 명확하게 전달하면서도 관객에게 생각할 여지를 남겨둔 것이 인상적이었어요. 카메라 워크도 훌륭해서 한 장면 한 장면이 그림 같았습니다.

전체적으로 올해 본 영화 중 최고였다고 자신 있게 말할 수 있습니다. 웃음과 감동, 긴장감을 모두 선사하는 {random.choice(emotions)} 영화예요. 아직 안 보신 분들은 꼭 극장에서 보시길 추천드립니다. 큰 화면으로 봐야 {random.choice(visual_words)}의 진가를 느낄 수 있어요. 다시 봐도 재밌을 것 같아서 재관람 의사 100%입니다!""",
        f"""솔직히 이 영화는 많이 {random.choice(negative_emotions)}했습니다. 평점이 높고 입소문도 많이 나서 큰 기대를 하고 봤는데, 기대치에 한참 못 미쳤어요.

가장 큰 문제는 {random.choice(story_words)}입니다. 전개가 너무 느리고 산만했어요. 1시간이 넘도록 본격적인 이야기가 시작되지 않고 불필요한 장면들로 시간을 끌었습니다. 감독이 뭔가 예술적인 걸 보여주려고 한 것 같은데, 저한테는 그냥 {random.choice(negative_emotions)}하고 {random.choice(negative_emotions)}하게만 느껴졌어요. 중간중간 {random.choice(story_words)}의 개연성도 부족했습니다. "왜 저 캐릭터가 저런 선택을 하지?" 하는 의문이 계속 들었는데, 영화 끝까지 봐도 명확한 답을 주지 않았어요.

{random.choice(acting_words)} 면에서는 호불호가 갈릴 것 같습니다. 주연 배우는 나름 열심히 했지만, 캐릭터 자체가 매력이 없어서 감정이입이 안 됐어요. 대사도 어색한 부분이 많았고, 감정 표현이 과장되거나 어색한 순간들이 있었습니다. 조연들은 더 심각했어요. 특히 악역 캐릭터는 너무 전형적이고 일차원적이어서 몰입을 방해했습니다.

{random.choice(visual_words)}만 좋았다고 할까요? 화면은 예쁘게 찍혔습니다. 색감도 좋고 구도도 나쁘지 않았어요. 하지만 {random.choice(visual_words)}만으로는 영화를 떠받칠 수 없죠. 오히려 {random.choice(story_words)}와 {random.choice(acting_words)}가 부실한 걸 {random.choice(visual_words)}로 가리려고 한 게 아닌가 하는 생각마저 들었습니다. {random.choice(music_words)}는 그냥 평범했어요. 특별히 기억에 남는 곡도 없었고, 장면과의 조화도 그저 그랬습니다.

러닝타임도 문제였습니다. 2시간 30분이 넘는데, 진짜 1시간 30분이면 충분한 내용을 억지로 늘린 느낌이었어요. 중간에 시계를 여러 번 봤고, 언제 끝나나 기다리게 됐습니다. 영화를 보면서 지루함을 느낀 건 정말 오랜만이에요.

감독의 전작들이 좋았어서 이번에도 기대했는데, 이번 작품은 실패작인 것 같습니다. 너무 욕심을 부린 건지, 아니면 상업성과 예술성 사이에서 균형을 잃은 건지 모르겠지만, 아무튼 만족스럽지 못했어요.

평점 테러를 하려는 건 아니지만, 솔직하게 제 의견을 말씀드리자면 비추천입니다. 과대평가된 영화라고 생각해요. 호평 일색인 리뷰들을 보면서 "내가 뭘 놓친 건가?" 싶기도 했는데, 두 번 볼 생각은 전혀 없네요. 차라리 그 시간에 다른 영화를 보는 게 나을 것 같습니다. 굳이 극장에서 볼 필요도 없고, OTT로 나와도 안 볼 것 같아요.""",
        f"""이 영화는 정말 평가하기 애매한 작품이네요. 좋은 점도 분명히 있고, 아쉬운 점도 많아서 호불호가 확실히 갈릴 것 같습니다.

먼저 좋았던 점부터 말씀드리면, {random.choice(visual_words)}가 정말 뛰어났습니다. 모든 장면이 정성스럽게 찍혔고, 특히 야외 촬영 장면들은 숨이 멎을 정도로 아름다웠어요. 색감 보정도 영화의 분위기와 잘 맞았고, 카메라 무브먼트도 세련됐습니다. {random.choice(music_words)}도 인상적이었어요. OST가 정말 좋아서 영화 보고 나서 바로 찾아 들었습니다. 장면과의 싱크로율도 높았고, 감정을 증폭시키는 역할을 잘 해냈어요.

배우들의 {random.choice(acting_words)}도 대체로 좋았습니다. 특히 주연 배우의 {random.choice(acting_words)}는 정말 훌륭했어요. 복잡한 감정을 표현하는 장면에서는 소름이 돋을 정도였습니다. 조연 중 몇몇도 강렬한 인상을 남겼고, 앙상블 {random.choice(acting_words)}도 나쁘지 않았어요.

하지만 가장 중요한 {random.choice(story_words)}에서 아쉬움이 많이 남습니다. 전반부는 흥미진진하게 시작했는데, 중반부터 힘이 빠지기 시작했어요. 전개가 예측 가능해지고, 클리셰를 답습하는 느낌이 강했습니다. 특히 후반부의 반전은 너무 뻔해서 오히려 실망스러웠어요. 복선이라고 깔아놓은 것들도 대부분 예상 범위 안이었고, 놀라운 순간이 별로 없었습니다.

캐릭터 묘사도 불균형했습니다. 주인공은 입체적으로 잘 그려졌지만, 다른 캐릭터들은 평면적이었어요. 특히 악역은 너무 전형적이어서 공감하기 어려웠습니다. 캐릭터 간의 관계 발전도 급격해서 설득력이 떨어지는 부분들이 있었어요.

러닝타임 배분도 문제였습니다. 어떤 장면은 너무 길게 끌고, 중요한 장면은 너무 빨리 지나가버렸어요. 특히 클라이맥스가 너무 허무하게 끝나서 카타르시스가 부족했습니다.

그럼에도 불구하고 이 영화를 완전히 실패작이라고 말하기는 어렵습니다. 감독의 연출력은 분명히 있고, {random.choice(visual_words)}와 {random.choice(music_words)} 같은 기술적 완성도는 높으니까요. 다만 {random.choice(story_words)}가 좀 더 탄탄했다면 명작이 될 수 있었을 텐데 하는 아쉬움이 크네요.

결론적으로, 이 영화는 기대치를 적절히 조절하고 보면 괜찮은 작품입니다. {random.choice(visual_words)}나 {random.choice(music_words)}를 중시하는 분들은 만족하실 것 같고, {random.choice(story_words)} 중심으로 영화를 보시는 분들은 실망하실 수도 있어요. 저는 중간 정도의 평가를 주고 싶네요.""",
    ]

    # 랜덤하게 짧은글, 중간글, 장문 중 선택 (비율: 3:6:1)
    review_type = random.choices(["short", "medium", "long"], weights=[3, 6, 1], k=1)[0]

    if review_type == "short":
        selected_review = random.choice(short_reviews)
    elif review_type == "medium":
        medium_reviews = medium_positive + medium_negative + medium_mixed
        selected_review = random.choice(medium_reviews)
    else:  # long
        selected_review = random.choice(long_reviews)

    return text + " " + selected_review if text else selected_review
