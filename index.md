---
layout: default
title: 스프린트 미션 보관함
description: 코드잇 AI 4기 스프린트 미션 결과물 보관함
cache-control: no-cache
expires: 0
pragma: no-cache
---

<script>

{%- assign cur_dir = "/" -%}
{%- include cur_files.liquid -%}

  var curDir = '{{- cur_file_dir -}}';
  var curFiles = {{- cur_files_json -}};
  var curPages = {{- cur_pages_json -}};
  
  console.log('curDir:', curDir);
  console.log('curFiles:', curFiles);
  console.log('curPages:', curPages);

  curPages.forEach(page => {
  // curFiles에 같은 name과 path가 있는지 확인
  const exists = curFiles.some(file => file.name === page.name && file.path === page.path);

  if (!exists) {
    // 확장자 추출
    let extname = '';
    if (page.name && page.name.includes('.')) {
      extname = '.' + page.name.split('.').pop();
    }

    // basename 추출
    let basename = page.name ? page.name.replace(/\.[^/.]+$/, '') : '';

    // modified_time 처리 (page.date가 없으면 빈 문자열)
    let modified_time = page.date || '';

    // curFiles 포맷에 맞게 변환해서 추가
    curFiles.push({
      name: page.name || '',
      path: page.path || '',
      extname: extname,
      modified_time: modified_time,
      basename: basename,
      url: page.url || ''
    });
  }
});

// curFiles.sort((a, b) => {
//   // 날짜가 ISO 형식이 아니면 Date 파싱이 안 될 수 있으니, 우선 문자열 비교
//   // 최신 날짜가 앞으로 오도록 내림차순
//   if (!a.modified_time) return 1;
//   if (!b.modified_time) return -1;
//   return b.modified_time.localeCompare(a.modified_time);
// });

curFiles.sort((a, b) => {
  // 파일명으로 한글/영문 구분하여 정렬
  if (!a.name) return 1;
  if (!b.name) return -1;
  return a.name.localeCompare(b.name, 'ko-KR', { numeric: true, caseFirst: 'lower' });
});

// // 정렬 후 출력
// curFiles.forEach(f => {
// /*
//       "name": "Grad-CAM_정상.png",
//       "path": "/스프린트미션_완료/image/06_4팀_김명환/Grad-CAM_정상.png",
//       "extname": ".png",
//       "modified_time": "2025-08-24 12:11:59 +0900",
//       "basename": "Grad-CAM_정상",
// */  
//   console.log('curfiles:', JSON.stringify(f, null, 2));
// });

  console.log('총 파일 수:', curFiles.length);
  console.log('파일 목록:', curFiles);

  var project_path = site.baseurl
  var site_url = `https://c0z0c.github.io${project_path}${curDir}`
  var raw_url = `https://raw.githubusercontent.com/c0z0c${project_path}/master${curDir}`;
  var git_url = `https://github.com/c0z0c${project_path}/blob/master${curDir}`
  var colab_url = `https://colab.research.google.com/github/c0z0c${project_path}/blob/master${curDir}`;
  
  console.log('site_url:', site_url);
  console.log('raw_url:', raw_url);
  console.log('colab_url:', colab_url);

  // 파일 목록 렌더링 함수
  function renderFileList(files = curFiles) {
    const fileListContainer = document.getElementById('file-list');
    if (!fileListContainer) return;

    fileListContainer.innerHTML = '';

    files.forEach(file => {
      const fileItem = document.createElement('div');
      fileItem.className = 'file-item';
      
      // 파일 타입에 따른 아이콘
      const getFileIcon = (extname) => {
        const iconMap = {
          '.md': '📝', '.txt': '📄', '.pdf': '📕',
          '.jpg': '🖼️', '.jpeg': '🖼️', '.png': '🖼️', '.gif': '🖼️',
          '.mp4': '🎬', '.avi': '🎬', '.mov': '🎬',
          '.py': '🐍', '.js': '📜', '.html': '🌐', '.css': '🎨',
          '.ipynb': '📓', '.json': '📋', '.xml': '📋',
          '.zip': '📦', '.rar': '📦', '.7z': '📦'
        };
        return iconMap[extname.toLowerCase()] || '📄';
      };

      const icon = getFileIcon(file.extname);
      const fileName = file.basename || file.name;
      const fileExt = file.extname;
      const modifiedTime = file.modified_time ? new Date(file.modified_time).toLocaleDateString('ko-KR') : '';

      // URL 생성
      let viewUrl = '';
      if (file.url) {
        viewUrl = `{{ site.baseurl }}${file.url}`;
      } else {
        viewUrl = `${site_url}${file.path}`;
      }

      fileItem.innerHTML = `
        <div class="file-icon">${icon}</div>
        <div class="file-info">
          <div class="file-name">
            <a href="${viewUrl}" target="_blank">${fileName}</a>
            <span class="file-ext">${fileExt}</span>
          </div>
          ${modifiedTime ? `<div class="file-date">${modifiedTime}</div>` : ''}
        </div>
        <div class="file-actions">
          ${file.extname === '.ipynb' ? `<a href="${colab_url}${file.path}" target="_blank" class="colab-btn">Colab</a>` : ''}
          <a href="${raw_url}${file.path}" target="_blank" class="raw-btn">Raw</a>
        </div>
      `;

      fileListContainer.appendChild(fileItem);
    });
  }

  // 정렬 함수
  function sortFiles(criteria) {
    let sortedFiles = [...curFiles];
    
    switch(criteria) {
      case 'name':
        sortedFiles.sort((a, b) => a.name.localeCompare(b.name, 'ko-KR', { numeric: true }));
        break;
      case 'date':
        sortedFiles.sort((a, b) => {
          if (!a.modified_time) return 1;
          if (!b.modified_time) return -1;
          return new Date(b.modified_time) - new Date(a.modified_time);
        });
        break;
      case 'type':
        sortedFiles.sort((a, b) => {
          const extA = a.extname || '';
          const extB = b.extname || '';
          return extA.localeCompare(extB);
        });
        break;
    }
    
    renderFileList(sortedFiles);
  }

  // 검색 함수
  function searchFiles(searchTerm) {
    const filteredFiles = curFiles.filter(file => 
      file.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      file.basename.toLowerCase().includes(searchTerm.toLowerCase())
    );
    renderFileList(filteredFiles);
  }

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
    renderFileList();
    renderFolderList();

    const sortSelect = document.getElementById('sort-select');
    if (sortSelect) {
      sortSelect.addEventListener('change', (e) => sortFiles(e.target.value));
    }

    const searchInput = document.getElementById('search-input');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => searchFiles(e.target.value));
    }
  });

</script>

<div class="nav-sections">
  <div class="section-card">
    <h2>� 현재 디렉토리 파일 목록</h2>
    <div id="file-list-container">
      <div class="file-controls">
        <select id="sort-select">
          <option value="name">이름순</option>
          <option value="date">날짜순</option>
          <option value="type">타입순</option>
        </select>
        <input type="text" id="search-input" placeholder="파일 검색...">
      </div>
      <div id="file-list" class="file-grid"></div>
    </div>
  </div>

  <div class="section-card">
    <h2>�📂 폴더별 탐색</h2>
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
