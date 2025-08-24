---
layout: default
title: 스프린트 미션 보관함
description: 코드잇 AI 4기 스프린트 미션 결과물 보관함
cache-control: no-cache
expires: 0
pragma: no-cache
---

<script>

{%- assign cur_dir = "/스프린트미션_완료/" -%}
{%- include cur_files.liquid -%}

  var curDir = '{{- cur_file_dir -}}';
  var curFiles = {{- cur_files_json -}};
  var curPages = {{- cur_pages_json -}};
  
  console.log('curDir:', curDir);
  console.log('curFiles:', curFiles);
  console.log('curPages:', curPages);

  // 폴더 목록 렌더링 함수
  function renderFolderList() {
    const folderListContainer = document.getElementById('folder-list');
    if (!folderListContainer) return;

    // 폴더 아이콘과 설명 매핑
    const folderIcons = {
      '멘토': '👨‍🏫',
      '스프린트미션_완료': '✅',
      '스프린트미션_작업중': '🚧',
      '위클리페이퍼': '📰',
      '스터디': '📒',
      '실습': '🔬',
      '백업': '💾',
      'Learning': '📚',
      'Learning Daily': '📅'
    };

    const folderDescs = {
      '멘토': '멘토 관련 자료',
      '스프린트미션_완료': '완료된 스프린트 미션들',
      '스프린트미션_작업중': '진행 중인 미션들',
      '위클리페이퍼': '주간 학습 리포트',
      '스터디': '학습 자료',
      '실습': '실습 자료',
      '백업': '백업 파일들',
      'Learning': '학습 자료',
      'Learning Daily': '일일 학습 기록'
    };

    // curFiles에서 폴더 추출
    const folderSet = new Set();
    curFiles.forEach(file => {
      if (file.path) {
        const pathParts = file.path.split('/').filter(part => part !== '');
        if (pathParts.length > 1) {
          const folder = pathParts[0];
          // 필터링: 빈 문자열, 점이 포함된 폴더, assets, _layouts 제외
          if (folder && !folder.includes('.') && folder !== 'assets' && folder !== '_layouts') {
            folderSet.add(folder);
          }
        }
      }
    });

    // 폴더 정렬
    const sortedFolders = Array.from(folderSet).sort((a, b) => 
      a.localeCompare(b, 'ko-KR', { numeric: true })
    );

    // HTML 생성
    folderListContainer.innerHTML = '';
    sortedFolders.forEach(folder => {
      const folderIcon = folderIcons[folder] || '📁';
      const folderDesc = folderDescs[folder] || '';

      const folderLink = document.createElement('a');
      folderLink.href = `{{ site.baseurl }}/${folder}/`;
      folderLink.className = 'folder-link';

      folderLink.innerHTML = `
        <span class="folder-icon">${folderIcon}</span>
        <span class="folder-name">${folder}</span>
        ${folderDesc ? `<span class="folder-desc">${folderDesc}</span>` : ''}
      `;

      folderListContainer.appendChild(folderLink);
    });
  }

  // 이벤트 리스너 등록
  document.addEventListener('DOMContentLoaded', function() {
    renderFolderList();
  });

</script>

<div class="nav-sections">
  <div class="section-card">
    <h2>📂 폴더별 탐색</h2>
    <div class="folder-links" id="folder-list">
      <!-- JavaScript로 동적 생성 -->
    </div>
  </div>

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
    </div>
  </div>
</div>

---

<div class="footer-info">
<small>
<strong>코드잇 AI 4기</strong> | 5팀 김명환<br>
마지막 업데이트: {{ site.time | date: "%Y년 %m월 %d일" }}
</small>
</div>
