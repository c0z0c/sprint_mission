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

<details>
<ul>
{% for file in site.static_files %}
  {% if file.path contains 'CensusIncome' %}
    <li>Static File: {{ file.path }} ({{ file.name }})</li>
  {% endif %}
{% endfor %}
{% for page in site.pages %}
  {% if page.path contains 'CensusIncome' %}
    <li>Page: {{ page.path }} ({{ page.name }})</li>
  {% endif %}
{% endfor %}
</ul>
</details>

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

<div class="file-grid">
  {% assign current_path = "/sprint_mission/멘토/CensusIncome/" %}
  {% assign static_files = site.static_files | where_exp: "item", "item.path contains current_path" %}
  {% assign markdown_pages = site.pages | where_exp: "page", "page.path contains 'CensusIncome'" %}
  
  {% assign all_files = "" | split: "" %}
  
  <!-- Add static files -->
  {% for file in static_files %}
    {% assign relative_path = file.path | remove: current_path %}
    {% unless relative_path contains "/" or file.name == "index.md" %}
      {% assign all_files = all_files | push: file %}
    {% endunless %}
  {% endfor %}
  
  <!-- Add markdown pages -->
  {% for page in markdown_pages %}
    {% assign relative_path = page.path | remove_first: "CensusIncome" | remove_first: "/" %}
    {% unless relative_path contains "/" or page.name == "index.md" %}
      {% assign all_files = all_files | push: page %}
    {% endunless %}
  {% endfor %}
  
  {% if all_files.size > 0 %}
    {% for file in all_files %}
      {% assign file_ext = file.extname | downcase %}
      {% if file_ext == "" and file.path %}
        {% assign file_name = file.path | split: "/" | last %}
        {% assign file_ext = file_name | split: "." | last | downcase %}
        {% assign file_ext = "." | append: file_ext %}
      {% endif %}
      {% assign file_icon = "📄" %}
      {% assign file_type = "파일" %}
      
      {% if file_ext == ".ipynb" %}
        {% assign file_icon = "📓" %}
        {% assign file_type = "Jupyter Notebook" %}
      {% elsif file_ext == ".py" %}
        {% assign file_icon = "🐍" %}
        {% assign file_type = "Python 파일" %}
      {% elsif file_ext == ".md" %}
        {% assign file_icon = "📝" %}
        {% assign file_type = "Markdown 문서" %}
      {% elsif file_ext == ".json" %}
        {% assign file_icon = "⚙️" %}
        {% assign file_type = "JSON 설정" %}
      {% elsif file_ext == ".csv" %}
        {% assign file_icon = "📊" %}
        {% assign file_type = "데이터 파일" %}
      {% endif %}
      
      <div class="file-item">
        <div class="file-icon">{{ file_icon }}</div>
        <div class="file-info">
          <h4 class="file-name">{% if file.name %}{{ file.name }}{% else %}{{ file.path | split: "/" | last }}{% endif %}</h4>
          <p class="file-type">{{ file_type }}</p>
          <p class="file-size">{% if file.modified_time %}{{ file.modified_time | date: "%Y-%m-%d" }}{% else %}{{ file.date | date: "%Y-%m-%d" }}{% endif %}</p>
        </div>
        <div class="file-actions">
          {% if file_ext == ".md" and file.name != "index.md" %}
            {% assign file_name_clean = file.name %}
            {% if file_name_clean == nil %}
              {% assign file_name_clean = file.path | split: "/" | last %}
            {% endif %}
            {% assign md_name_clean = file_name_clean | remove: '.md' %}
            <a href="https://c0z0c.github.io/sprint_mission/멘토/CensusIncome/{{ md_name_clean }}" class="file-action" title="렌더링된 페이지 보기" target="_blank">🌐</a>
            <a href="https://github.com/c0z0c/sprint_mission/blob/master/멘토/CensusIncome/{{ file_name_clean }}" class="file-action" title="GitHub에서 원본 보기" target="_blank">📖</a>
          {% elsif file_ext == ".ipynb" %}
            {% assign file_name_clean = file.name %}
            {% if file_name_clean == nil %}
              {% assign file_name_clean = file.path | split: "/" | last %}
            {% endif %}
            <a href="https://github.com/c0z0c/sprint_mission/blob/master/멘토/CensusIncome/{{ file_name_clean }}" class="file-action" title="GitHub에서 보기" target="_blank">📖</a>
            <a href="https://colab.research.google.com/github/c0z0c/sprint_mission/blob/master/멘토/CensusIncome/{{ file_name_clean }}" class="file-action" title="Colab에서 열기" target="_blank">🚀</a>
          {% else %}
            <a href="{{ file.path | relative_url }}" class="file-action" title="파일 열기">📖</a>
          {% endif %}
        </div>
      </div>
    {% endfor %}
  {% else %}
    <div class="empty-message">
      <span class="empty-icon">📄</span>
      <h3>파일이 없습니다</h3>
      <p>현재 이 위치에는 파일이 없습니다.</p>
    </div>
  {% endif %}
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

/* File Grid Styles */
.file-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.file-item {
  background: white;
  border: 1px solid #e1e8ed;
  border-radius: 12px;
  padding: 20px;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  position: relative;
}

.file-item:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 25px rgba(0,0,0,0.15);
  border-color: #007acc;
}

.file-icon {
  font-size: 48px;
  text-align: center;
  margin-bottom: 15px;
}

.file-info {
  text-align: center;
  margin-bottom: 15px;
}

.file-name {
  margin: 0 0 8px 0;
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
  word-break: break-word;
}

.file-type {
  margin: 0 0 5px 0;
  color: #666;
  font-size: 14px;
}

.file-size {
  margin: 0;
  color: #999;
  font-size: 12px;
}

.file-actions {
  display: flex;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
}

.file-action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 8px;
  text-decoration: none;
  font-size: 16px;
  transition: all 0.2s ease;
  color: #495057;
}

.file-action:hover {
  background: #007acc;
  color: white;
  border-color: #007acc;
  transform: scale(1.1);
  text-decoration: none;
}

.empty-message {
  text-align: center;
  padding: 60px 20px;
  color: #666;
  grid-column: 1 / -1;
}

.empty-icon {
  font-size: 64px;
  margin-bottom: 20px;
  opacity: 0.5;
}

.empty-message h3 {
  margin: 0 0 10px 0;
  color: #999;
}

.empty-message p {
  margin: 0;
  color: #bbb;
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
