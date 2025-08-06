---
layout: default
title: CensusIncome - 멘토 자료
description: Census Income 데이터 분석 프로젝트
cache-control: no-cache
expires: 0
pragma: no-cache
---

# 📊 CensusIncome 프로젝트

Census Income 데이터 분석 프로젝트입니다.

## 📁 폴더 목록

<div class="file-list">
  <div class="folder-item">
    <div class="item-link folder-display">
      <span class="item-icon">📂</span>
      <span class="item-name">.commit_pandas</span>
      <span class="item-desc">Pandas 커밋 데이터</span>
    </div>
  </div>
  
  <div class="folder-item">
    <div class="item-link folder-display">
      <span class="item-icon">📂</span>
      <span class="item-name">fonts</span>
      <span class="item-desc">폰트 파일들</span>
    </div>
  </div>
  
  <div class="folder-item">
    <div class="item-link folder-display">
      <span class="item-icon">📂</span>
      <span class="item-name">__pycache__</span>
      <span class="item-desc">Python 캐시 파일들</span>
    </div>
  </div>
</div>

## 📄 파일 목록

<div class="file-list">
  <div class="file-item">
    <a href="{{ site.baseurl }}/멘토/CensusIncome/CensusIncome.ipynb" class="item-link file" target="_blank">
      <span class="item-icon">📓</span>
      <span class="item-name">CensusIncome.ipynb</span>
      <span class="item-desc">메인 분석 노트북</span>
    </a>
  </div>
  
  <div class="file-item">
    <a href="{{ site.baseurl }}/멘토/CensusIncome/helper_c0z0c_dev.py" class="item-link file" target="_blank">
      <span class="item-icon">🐍</span>
      <span class="item-name">helper_c0z0c_dev.py</span>
      <span class="item-desc">헬퍼 유틸리티 모듈</span>
    </a>
  </div>
  
  <div class="file-item">
    <a href="{{ site.baseurl }}/멘토/CensusIncome/readme.md" class="item-link file" target="_blank">
      <span class="item-icon">📖</span>
      <span class="item-name">readme.md</span>
      <span class="item-desc">프로젝트 설명 문서</span>
    </a>
  </div>
  
  <div class="file-item">
    <div class="item-link file-display">
      <span class="item-icon">⚙️</span>
      <span class="item-name">.gitignore</span>
      <span class="item-desc">Git 무시 파일 목록</span>
    </div>
  </div>
</div>

## 📋 프로젝트 정보

- **주제**: Census Income 데이터를 활용한 수입 예측 분석
- **파일 형식**: Jupyter Notebook (`.ipynb`)
- **언어**: Python
- **주요 도구**: pandas, matplotlib, helper 모듈

---

<div class="navigation-footer">
  <a href="{{ site.baseurl }}/멘토/" class="nav-button back">
    <span class="nav-icon">⬅️</span> 멘토 폴더로
  </a>
  <a href="{{ site.baseurl }}/" class="nav-button home">
    <span class="nav-icon">🏠</span> 홈으로
  </a>
</div>

<style>
.file-list {
  margin: 20px 0;
}

.folder-item, .file-item {
  margin-bottom: 10px;
}

.item-link {
  display: flex;
  align-items: center;
  padding: 15px;
  background: white;
  border-radius: 8px;
  text-decoration: none;
  border: 1px solid #dee2e6;
  transition: all 0.3s ease;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.item-link:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  text-decoration: none;
}

.item-link.folder:hover, .folder-display:hover {
  background: #fff3e0;
  border-color: #ff9800;
}

.item-link.file:hover {
  background: #e8f5e8;
  border-color: #4caf50;
}

.folder-display, .file-display {
  cursor: default;
}

.item-icon {
  font-size: 24px;
  margin-right: 15px;
  width: 30px;
  text-align: center;
}

.item-name {
  font-weight: bold;
  color: #2c3e50;
  margin-right: 15px;
  flex: 1;
}

.item-desc {
  color: #666;
  font-size: 0.9em;
  font-style: italic;
}

.navigation-footer {
  margin-top: 40px;
  padding-top: 20px;
  border-top: 1px solid #eee;
  text-align: center;
}

.nav-button {
  display: inline-flex;
  align-items: center;
  padding: 12px 24px;
  background: #3498db;
  color: white;
  border-radius: 6px;
  text-decoration: none;
  transition: all 0.3s ease;
  margin: 0 10px;
}

.nav-button:hover {
  background: #2980b9;
  transform: translateY(-2px);
  text-decoration: none;
  color: white;
}

.nav-button.home {
  background: #27ae60;
}

.nav-button.home:hover {
  background: #219a52;
}

.nav-button.back {
  background: #95a5a6;
}

.nav-button.back:hover {
  background: #7f8c8d;
}

.nav-icon {
  margin-right: 8px;
  font-size: 16px;
}
</style>
