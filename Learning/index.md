---
layout: default
title: Learning - 보관함
description: Learning 자료들
date: 2025-08-24
cache-control: no-cache
expires: 0
pragma: no-cache
---

# ✅ Learning

<script>

{%- assign cur_dir = "/Learning/" -%}
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


  // 파일 아이콘 및 타입 결정 함수
  function getFileInfo(extname) {
    switch(extname.toLowerCase()) {
      case '.ipynb':
        return { icon: '📓', type: 'Jupyter Notebook' };
      case '.py':
        return { icon: '🐍', type: 'Python 파일' };
      case '.md':
        return { icon: '📝', type: 'Markdown 문서' };
      case '.json':
        return { icon: '⚙️', type: 'JSON 설정' };
      case '.zip':
        return { icon: '📦', type: '압축 파일' };
      case '.png':
      case '.jpg':
      case '.jpeg':
        return { icon: '🖼️', type: '이미지 파일' };
      case '.csv':
        return { icon: '📊', type: '데이터 파일' };
      case '.pdf':
        return { icon: '📄', type: 'PDF 문서' };
      case '.docx':
        return { icon: '📊', type: 'Word 문서' };
      default:
        return { icon: '📄', type: '파일' };
    }
  }

  // 파일 액션 버튼 생성 함수
  function getFileActions(file) {
    const fileName = file.name;
    const fileExt = file.extname.toLowerCase();
    
    let actions = '';
    
    if (fileExt === '.md' && fileName !== 'index.md') {
      const mdName = fileName.replace('.md', '');
      actions += `<a href="${site_url}${mdName}" class="file-action" title="렌더링된 페이지 보기" target="_blank">🌐</a>`;
      actions += `<a href="${git_url}${fileName}" class="file-action" title="GitHub에서 원본 보기" target="_blank">📖</a>`;
    } else if (fileExt === '.ipynb') {
      actions += `<a href="${git_url}${fileName}" class="file-action" title="GitHub에서 보기" target="_blank">📖</a>`;
      actions += `<a href="${colab_url}${fileName}" class="file-action" title="Colab에서 열기" target="_blank">🚀</a>`;
    } else if (fileExt === '.pdf') {
      actions += `<a href="${git_url}${fileName}" class="file-action" title="GitHub에서 보기" target="_blank">📖</a>`;
      actions += `<a href="https://docs.google.com/viewer?url=${raw_url}${fileName}" class="file-action" title="PDF 뷰어로 열기" target="_blank">📄</a>`;
    } else if (fileExt === '.docx') {
      actions += `<a href="${git_url}${fileName}" class="file-action" title="GitHub에서 보기" target="_blank">📖</a>`;
      actions += `<a href="https://docs.google.com/viewer?url=${raw_url}${fileName}" class="file-action" title="Google에서 열기" target="_blank">📊</a>`;
    } else if (fileExt === '.html') {
      actions += `<a href="${site_url}${fileName}" class="file-action" title="웹페이지로 보기" target="_blank">🌐</a>`;
      actions += `<a href="${git_url}${fileName}" class="file-action" title="GitHub에서 원본 보기" target="_blank">📖</a>`;
    } else {
      actions += `<a href="${file.path}" class="file-action" title="파일 열기">📖</a>`;
    }
    
    return actions;
  }

  // DOM이 로드된 후 파일 목록 렌더링
  document.addEventListener('DOMContentLoaded', function() {
    const fileGrid = document.querySelector('.file-grid');
    
    if (curFiles.length === 0) {
      fileGrid.innerHTML = `
        <div class="empty-message">
          <span class="empty-icon">📄</span>
          <h3>파일이 없습니다</h3>
          <p>현재 이 위치에는 완료된 미션 파일이 없습니다.</p>
        </div>
      `;
      return;
    }

    let html = '';
    curFiles.forEach(file => {
      if (file.name === 'index.md' || file.name === 'info.md') return;

      const fileInfo = getFileInfo(file.extname);
      const fileDate = file.modified_time ? new Date(file.modified_time).toLocaleDateString('ko-KR') : '';
      const actions = getFileActions(file);
      
      html += `
        <div class="file-item">
          <div class="file-icon">${fileInfo.icon}</div>
          <div class="file-info">
            <h4 class="file-name">${file.name}</h4>
            <p class="file-type">${fileInfo.type}</p>
            <p class="file-size">${fileDate}</p>
          </div>
          <div class="file-actions">
            ${actions}
          </div>
        </div>
      `;
    });
    
    fileGrid.innerHTML = html;
  });
</script>

<div class="file-grid">
  <!-- 파일 목록이 JavaScript로 동적 생성됩니다 -->
</div>

---

<div class="navigation-footer">
  <a href="{{- site.baseurl -}}/" class="nav-button home">
    <span class="nav-icon">🏠</span> 홈으로
  </a>
</div>