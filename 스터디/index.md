---
layout: default
title: 스터디 - 스터디 보관함
description: 스터디 자료들
cache-control: no-cache
expires: 0
pragma: no-cache
---

# ✅ 스터디

<script>


  // 폴더 정보 가져오기 함수
  function getFolderInfo(folderPath) {
    const folderName = folderPath.split("/").filter(s => s).pop() || "root";
    
    // 폴더명에 따른 아이콘과 설명
    const folderMappings = {
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

// // 폴더 정보 가져오기 함수
// function getFolderInfo(folderName) {
//   // 폴더명에 따른 아이콘과 설명 (index.md와 유사하게)
//   const folderMappings = {
//     '멘토': { icon: '👨‍🏫', desc: '멘토 관련 자료' },
//     '스프린트미션_완료': { icon: '✅', desc: '완료된 스프린트 미션들' },
//     '스프린트미션_작업중': { icon: '🚧', desc: '진행 중인 미션들' },
//     '위클리페이퍼': { icon: '📰', desc: '주간 학습 리포트' },
//     '스터디': { icon: '📒', desc: '학습 자료' },
//     '실습': { icon: '🔬', desc: '실습 자료' },
//     '백업': { icon: '💾', desc: '백업 파일들' },
//     '셈플': { icon: '📂', desc: '샘플 파일들' },
//     '테스트': { icon: '🧪', desc: '테스트 파일들' },
//     'image': { icon: '🖼️', desc: '이미지 파일들' },
//     'Learning': { icon: '📚', desc: '학습 자료' },
//     'Learning Daily': { icon: '📅', desc: '일일 학습 기록' },
//     'md': { icon: '📝', desc: 'Markdown 문서' },
//     '회의록': { icon: '📋', desc: '팀 회의록' },
//     'assets': { icon: '🎨', desc: '정적 자원' },
//     '경구약제이미지데이터': { icon: '💊', desc: '약물 데이터' },
//     'AI 모델 환경 설치가이드': { icon: '⚙️', desc: '설치 가이드' },
//     '경구약제 이미지 데이터(데이터 설명서, 경구약제 리스트)': { icon: '📊', desc: '데이터 설명서' },
//     '발표자료': { icon: '📊', desc: '발표 자료' },
//     '협업일지': { icon: '📓', desc: '협업 일지' }
//   };
  
//   return folderMappings[folderName] || { icon: '📁', desc: '폴더' };
// }


{% assign cur_dir = "/스터디/" %}
{% include cur_files.liquid %}
{% include page_values.html %}
{% include page_files_table.html %}
{% include page_folders.html %}

</script>

<div class="file-grid">
  <!-- 파일 목록이 JavaScript로 동적 생성됩니다 -->
</div>

---

## 폴더목록

<div class="folder-grid">
  <!-- 폴더 목록이 JavaScript로 동적 생성됩니다 -->
</div>


---

<div class="navigation-footer">
  <a href="{{- site.baseurl -}}/" class="nav-button home">
    <span class="nav-icon">🏠</span> 홈으로
  </a>
</div>