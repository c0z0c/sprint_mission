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

{%- assign cur_dir = "/스터디/" -%}
{%- include cur_files.liquid -%}

  var curDir = '{{- cur_file_dir -}}';
  var curFiles = {{- cur_files_json -}};
  var curPages = {{- cur_pages_json -}};
  
  console.log('curDir:', curDir);
  console.log('curFiles:', curFiles);
  console.log('curPages:', curPages);

  // 기본 타이틀 추가
  curFiles.forEach(file => {
    if (!file.title) {
      file.title = file.name;
    }
  });

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
      url: page.url || '',
      title: page.title ? page.title : page.name || ''
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
  if (!a.title) return 1;
  if (!b.title) return -1;
  return a.title.localeCompare(b.title, 'ko-KR', { numeric: true, caseFirst: 'lower' });
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
        return { icon: '📓', type: 'Colab' };
      case '.py':
        return { icon: '🐍', type: 'Python' };
      case '.md':
        return { icon: '📝', type: 'Markdown' };
      case '.json':
        return { icon: '⚙️', type: 'JSON' };
      case '.zip':
        return { icon: '📦', type: '압축' };
      case '.png':
      case '.jpg':
      case '.jpeg':
        return { icon: '🖼️', type: '이미지' };
      case '.csv':
        return { icon: '📊', type: '데이터' };
      case '.pdf':
        return { icon: '📄', type: 'PDF' };
      case '.docx':
        return { icon: '📊', type: 'Word' };
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
      actions += `<a href="${git_url}${fileName}" class="file-action" title="파일 열기" target="_blank">📖</a>`;
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

    let html = `
      <table class="file-table">
        <thead>
          <tr>
            <th onclick="sortTable(0)" style="cursor: pointer; width:110px;">날짜 ⬍</th>
            <th onclick="sortTable(1)" style="cursor: pointer;">제목 ⬍</th>
            <th onclick="sortTable(2)" style="cursor: pointer;">파일명 ⬍</th>
            <th onclick="sortTable(3)" style="cursor: pointer;">타입 ⬍</th>
            <th onclick="sortTable(4)" style="cursor: pointer;">View ⬍</th>
            <th onclick="sortTable(5)" style="cursor: pointer;">Git⬍</th>
          </tr>
        </thead>
        <tbody>
    `;
    
    curFiles.forEach(file => {
      if (file.name === 'index.md' || file.name === 'info.md') return;

      const fileInfo = getFileInfo(file.extname);
      const fileDate = file.modified_time ? new Date(file.modified_time).toLocaleDateString('ko-KR') : '';
      const fileName = file.name;
      const fileExt = file.extname.toLowerCase();
      
      // 렌더링페이지 링크 생성
      let renderLink = '';
      if (fileExt === '.md' && fileName !== 'index.md') {
        const mdName = fileName.replace('.md', '');
        renderLink = `<a href="${site_url}${mdName}" title="렌더링된 페이지 보기" target="_blank">🌐</a>`;
      } else if (fileExt === '.ipynb') {
        renderLink = `<a href="${colab_url}${fileName}" title="Colab에서 열기" target="_blank">🚀</a>`;
      } else if (fileExt === '.pdf') {
        renderLink = `<a href="https://docs.google.com/viewer?url=${raw_url}${fileName}" title="PDF 뷰어로 열기" target="_blank">📄</a>`;
      } else if (fileExt === '.docx') {
        renderLink = `<a href="https://docs.google.com/viewer?url=${raw_url}${fileName}" title="Google에서 열기" target="_blank">📊</a>`;
      } else if (fileExt === '.html') {
        renderLink = `<a href="${site_url}${fileName}" title="웹페이지로 보기" target="_blank">🌐</a>`;
      } else {
        renderLink = '-';
      }
      
      // Git 직접 링크
      const gitLink = `<a href="${git_url}${fileName}" title="GitHub에서 원본 보기" target="_blank">📖</a>`;
      
      // 제목 클릭 시 렌더링 페이지 링크 생성
      let titleClickable = `<span class="file-icon">${fileInfo.icon}</span> ${file.title}`;
      if (fileExt === '.md' && fileName !== 'index.md') {
        const mdName = fileName.replace('.md', '');
        titleClickable = `<span class="file-icon">${fileInfo.icon}</span> <a href="${site_url}${mdName}" title="렌더링된 페이지 보기" target="_blank" style="text-decoration: none; color: inherit;">${file.title}</a>`;
      } else if (fileExt === '.ipynb') {
        titleClickable = `<span class="file-icon">${fileInfo.icon}</span> <a href="${colab_url}${fileName}" title="Colab에서 열기" target="_blank" style="text-decoration: none; color: inherit;">${file.title}</a>`;
      } else if (fileExt === '.pdf') {
        titleClickable = `<span class="file-icon">${fileInfo.icon}</span> <a href="https://docs.google.com/viewer?url=${raw_url}${fileName}" title="PDF 뷰어로 열기" target="_blank" style="text-decoration: none; color: inherit;">${file.title}</a>`;
      } else if (fileExt === '.docx') {
        titleClickable = `<span class="file-icon">${fileInfo.icon}</span> <a href="https://docs.google.com/viewer?url=${raw_url}${fileName}" title="Google에서 열기" target="_blank" style="text-decoration: none; color: inherit;">${file.title}</a>`;
      } else if (fileExt === '.html') {
        titleClickable = `<span class="file-icon">${fileInfo.icon}</span> <a href="${site_url}${fileName}" title="웹페이지로 보기" target="_blank" style="text-decoration: none; color: inherit;">${file.title}</a>`;
      }
      
      // 파일명 클릭 시 Git 직접 연결
      const fileNameClickable = `<a href="${git_url}${fileName}" title="GitHub에서 원본 보기" target="_blank" style="text-decoration: none; color: inherit;">${fileName}</a>`;
      
      html += `
        <tr>
          <td>${fileDate}</td>
          <td>${titleClickable}</td>
          <td>${fileNameClickable}</td>
          <td>${fileInfo.type}</td>
          <td>${renderLink}</td>
          <td>${gitLink}</td>
        </tr>
      `;
    });
    
    html += `
        </tbody>
      </table>
    `;
    
    fileGrid.innerHTML = html;
  });

  // 테이블 정렬 기능
  let sortDirection = {}; // 각 컬럼의 정렬 방향을 저장

  function sortTable(columnIndex) {
    const table = document.querySelector('.file-table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    // 현재 정렬 방향 확인 (기본값: 오름차순)
    const isAscending = sortDirection[columnIndex] !== 'asc';
    sortDirection[columnIndex] = isAscending ? 'asc' : 'desc';
    
    // 헤더 화살표 업데이트
    const headers = table.querySelectorAll('th');
    headers.forEach((header, index) => {
      if (index === columnIndex) {
        const arrow = isAscending ? ' ⬆' : ' ⬇';
        header.innerHTML = header.innerHTML.replace(/ [⬆⬇⬍]/g, '') + arrow;
      } else {
        header.innerHTML = header.innerHTML.replace(/ [⬆⬇⬍]/g, '') + ' ⬍';
      }
    });
    
    // 행 정렬
    rows.sort((a, b) => {
      let aValue = a.cells[columnIndex].textContent || a.cells[columnIndex].innerText;
      let bValue = b.cells[columnIndex].textContent || b.cells[columnIndex].innerText;
      
      // 날짜 컬럼인 경우 날짜로 파싱
      if (columnIndex === 0) {
        aValue = aValue ? new Date(aValue).getTime() : 0;
        bValue = bValue ? new Date(bValue).getTime() : 0;
      }
      // 숫자가 포함된 문자열의 경우 자연 정렬
      else {
        // 아이콘 제거 (제목 컬럼의 경우)
        aValue = aValue.replace(/[📓🐍📝⚙️📦🖼️📊📄]/g, '').trim();
        bValue = bValue.replace(/[📓🐍📝⚙️📦🖼️📊📄]/g, '').trim();
      }
      
      let comparison = 0;
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        comparison = aValue - bValue;
      } else {
        comparison = aValue.toString().localeCompare(bValue.toString(), 'ko-KR', { 
          numeric: true, 
          caseFirst: 'lower' 
        });
      }
      
      return isAscending ? comparison : -comparison;
    });
    
    // 정렬된 행들을 다시 tbody에 추가
    rows.forEach(row => tbody.appendChild(row));
  }
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