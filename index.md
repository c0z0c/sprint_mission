---
layout: default
title: 스프린트 미션 보관함
description: 코드잇 AI 4기 스프린트 미션 결과물 보관함
cache-control: no-cache
expires: 0
pragma: no-cache
---

<script>

// 폴더 정보 가져오기 함수
function getFolderInfo(folderName) {
  // 폴더명에 따른 아이콘과 설명 (중복 정리됨)
  const folderMappings = {
    '멘토': { icon: '👨‍🏫', desc: '멘토 관련 자료' },
    '스프린트미션_완료': { icon: '✅', desc: '완료된 스프린트 미션들' },
    '스프린트미션_작업중': { icon: '🚧', desc: '진행 중인 미션들' },
    '위클리페이퍼': { icon: '📰', desc: '주간 학습 리포트' },
    '스터디': { icon: '📒', desc: '학습 자료' },
    '실습': { icon: '🔬', desc: '실습 자료' },
    '백업': { icon: '💾', desc: '백업 파일들' },
    '셈플': { icon: '📂', desc: '샘플 파일들' },
    '테스트': { icon: '🧪', desc: '테스트 파일들' },
    'image': { icon: '🖼️', desc: '이미지 파일들' },
    'Learning': { icon: '📚', desc: '학습 자료' },
    'Learning Daily': { icon: '📅', desc: '일일 학습 기록' },
    'md': { icon: '📝', desc: 'Markdown 문서' },
    '회의록': { icon: '📋', desc: '팀 회의록' },
    'assets': { icon: '🎨', desc: '정적 자원' },
    '경구약제이미지데이터': { icon: '💊', desc: '약물 데이터' },
    'AI 모델 환경 설치가이드': { icon: '⚙️', desc: '설치 가이드' },
    '경구약제 이미지 데이터(데이터 설명서, 경구약제 리스트)': { icon: '📊', desc: '데이터 설명서' },
    '발표자료': { icon: '📊', desc: '발표 자료' },
    '협업일지': { icon: '📓', desc: '협업 일지' }
  };

  return folderMappings[folderName] || { icon: '📁', desc: '폴더' };
}

{% assign cur_dir = "/" %}
{% include cur_files.liquid %}
{% include page_values.html %}
{% include page_folders_tree.html %}

</script>

# 📁 폴더별 탐색

<div class="folder-grid">
  <!-- 폴더 목록이 JavaScript로 동적 생성됩니다 -->
</div>

---

<div class="section-card">
  <h2>🔗 빠른 링크</h2>
  <div class="quick-links">
    <a href="https://c0z0c.github.io/" target="_blank">
      <span class="link-icon">🌐</span> 메인
    </a>
    <a href="https://github.com/c0z0c/sprint_mission" target="_blank">
      <span class="link-icon">📱</span> GitHub 저장소
    </a>
    <a href="{{ site.baseurl }}/스프린트미션_완료/info">
      <span class="link-icon">📖</span> Info
    </a>
    <a href="https://c0z0c.github.io/codeit_ai_health_eat" target="_blank">
      <span class="link-icon">📱</span> 초급 프로젝트
    </a>      
  </div>
</div>

---

<div class="footer-info">
<small>
<strong>코드잇 AI 4기</strong> | 5팀 김명환<br>
마지막 업데이트: {{ site.time | date: "%Y년 %m월 %d일" }}
</small>
</div>
