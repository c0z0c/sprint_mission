---
layout: default
title: 스프린트미션_완료 - 완료된 미션 보관함
description: 완료된 스프린트 미션 자료들
cache-control: no-cache
expires: 0
pragma: no-cache
-<!-- Debugging Section -->
<details style="margin: 20px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
<summary style="cursor: pointer; font-weight: bold;">🔍 디버깅 정보 (파일 감지 상태)</summary>
<h4>Static Files in 스프린트미션_완료/:</h4>
<ul>
{% for file in site.static_files %}
  {% if file.path contains '스프린트미션_완료/' %}
    <li>{{ file.path }} ({{ file.name }}) - {{ file.extname }}</li>
  {% endif %}
{% endfor %}
</ul>
<h4>Pages containing '스프린트미션_완료':</h4>
<ul>
{% for page in site.pages %}
  {% if page.path contains '스프린트미션_완료' %}
    <li>{{ page.path }} ({{ page.name }}) - {{ page.url }}</li>
  {% endif %}
{% endfor %}
</ul>
</details>

완료된 스프린트 미션 자료들을 모아둔 폴더입니다.

## 📄 파일 목록

<div class="file-grid">
  {% assign current_folder = "스프린트미션_완료/" %}
  {% assign static_files = site.static_files | where_exp: "item", "item.path contains current_folder" %}
  {% assign markdown_pages = site.pages | where_exp: "page", "page.path contains '스프린트미션_완료'" %}
  
  {% assign all_files = "" | split: "" %}
  
  <!-- Add static files -->
  {% for file in static_files %}
    {% assign relative_path = file.path | remove: current_folder %}
    {% unless relative_path contains "/" or file.name == "index.md" %}
      {% assign all_files = all_files | push: file %}
    {% endunless %}
  {% endfor %}
  
  <!-- Add markdown pages -->
  {% for page in markdown_pages %}
    {% assign relative_path = page.path | remove_first: "스프린트미션_완료" | remove_first: "/" %}
    {% unless relative_path contains "/" or page.name == "index.md" %}
      {% assign all_files = all_files | push: page %}
    {% endunless %}
  {% endfor %}
  
  {% assign sorted_files = all_files | sort: "name" %}
  
  {% if sorted_files.size > 0 %}
    {% for file in sorted_files %}
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
      {% elsif file_ext == ".docx" %}
        {% assign file_icon = "📄" %}
        {% assign file_type = "Word 문서" %}
      {% elsif file_ext == ".pdf" %}
        {% assign file_icon = "📄" %}
        {% assign file_type = "PDF 문서" %}
      {% elsif file_ext == ".html" %}
        {% assign file_icon = "🌐" %}
        {% assign file_type = "HTML 문서" %}
      {% elsif file_ext == ".md" %}
        {% assign file_icon = "📝" %}
        {% assign file_type = "Markdown 문서" %}
      {% elsif file_ext == ".py" %}
        {% assign file_icon = "🐍" %}
        {% assign file_type = "Python 파일" %}
      {% elsif file_ext == ".json" %}
        {% assign file_icon = "⚙️" %}
        {% assign file_type = "JSON 설정" %}
      {% elsif file_ext == ".zip" %}
        {% assign file_icon = "📦" %}
        {% assign file_type = "압축 파일" %}
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
          {% if file_ext == ".ipynb" %}
            {% assign file_name_clean = file.name %}
            {% if file_name_clean == nil %}
              {% assign file_name_clean = file.path | split: "/" | last %}
            {% endif %}
            <a href="https://github.com/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="GitHub에서 보기" target="_blank">📖</a>
            <a href="https://colab.research.google.com/github/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="Colab에서 열기" target="_blank">🚀</a>
          {% elsif file_ext == ".pdf" %}
            {% assign file_name_clean = file.name %}
            {% if file_name_clean == nil %}
              {% assign file_name_clean = file.path | split: "/" | last %}
            {% endif %}
            <a href="https://github.com/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="GitHub에서 보기" target="_blank">📖</a>
            <a href="https://docs.google.com/viewer?url=https://github.com/c0z0c/sprint_mission/raw/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="Google Docs Viewer에서 열기" target="_blank">👁️</a>
          {% elsif file_ext == ".docx" %}
            {% assign file_name_clean = file.name %}
            {% if file_name_clean == nil %}
              {% assign file_name_clean = file.path | split: "/" | last %}
            {% endif %}
            <a href="https://github.com/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="GitHub에서 보기" target="_blank">📖</a>
            <a href="https://docs.google.com/viewer?url=https://github.com/c0z0c/sprint_mission/raw/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="Google Docs에서 열기" target="_blank">�</a>
          {% elsif file_ext == ".html" %}
            {% assign file_name_clean = file.name %}
            {% if file_name_clean == nil %}
              {% assign file_name_clean = file.path | split: "/" | last %}
            {% endif %}
            <a href="https://c0z0c.github.io/sprint_mission/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="렌더링된 페이지 보기" target="_blank">🌐</a>
            <a href="https://github.com/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="GitHub에서 원본 보기" target="_blank">📖</a>
          {% elsif file_ext == ".md" and file.name != "index.md" %}
            {% assign file_name_clean = file.name %}
            {% if file_name_clean == nil %}
              {% assign file_name_clean = file.path | split: "/" | last %}
            {% endif %}
            {% assign md_name_clean = file_name_clean | remove: '.md' %}
            <a href="https://c0z0c.github.io/sprint_mission/스프린트미션_완료/{{ md_name_clean }}" class="file-action" title="렌더링된 페이지 보기" target="_blank">🌐</a>
            <a href="https://github.com/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="GitHub에서 원본 보기" target="_blank">📖</a>
          {% else %}
            <a href="{{ file.path | relative_url }}" class="file-action" title="파일 열기">📖</a>
          {% endif %}
        </div>
      </div>
    {% endfor %}
  {% else %}
    <div class="empty-message">
      <span class="empty-icon">�</span>
      <h3>파일이 없습니다</h3>
      <p>현재 이 위치에는 파일이 없습니다.</p>
    </div>
  {% endif %}
</div>

<!-- Debugging Section -->
<details style="margin: 20px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
<summary style="cursor: pointer; font-weight: bold;">� 디버깅 정보 (파일 감지 상태)</summary>
<h4>Static Files in /sprint_mission/스프린트미션_완료/:</h4>
<ul>
{% for file in site.static_files %}
  {% if file.path contains '/sprint_mission/스프린트미션_완료/' %}
    <li>{{ file.path }} ({{ file.name }}) - {{ file.extname }}</li>
  {% endif %}
{% endfor %}
</ul>
<h4>Pages containing '스프린트미션_완료':</h4>
<ul>
{% for page in site.pages %}
  {% if page.path contains '스프린트미션_완료' %}
    <li>{{ page.path }} ({{ page.name }}) - {{ page.url }}</li>
  {% endif %}
{% endfor %}
</ul>
</details>

## 📊 완료 현황

{% assign current_folder = "스프린트미션_완료/" %}
{% assign completed_files = site.static_files | where_exp: "file", "file.path contains current_folder" %}
{% assign mission_files = completed_files | where_exp: "file", "file.name contains '미션'" %}
{% assign exclude_files = "index.md,info.md,info.html" | split: "," %}
{% assign filtered_files = "" | split: "" %}

{% for file in completed_files %}
  {% assign relative_path = file.path | remove: current_folder %}
  {% unless relative_path contains "/" or exclude_files contains file.name %}
    {% assign filtered_files = filtered_files | push: file %}
  {% endunless %}
{% endfor %}

{% assign total_files = filtered_files | size %}
{% assign unique_missions = "" | split: "" %}

{% for file in mission_files %}
  {% assign mission_number = file.name | split: '_' | first %}
  {% unless unique_missions contains mission_number %}
    {% assign unique_missions = unique_missions | push: mission_number %}
  {% endunless %}
{% endfor %}

<div class="completion-stats">
  <div class="stat-card">
    <div class="stat-number">{{ unique_missions.size }}</div>
    <div class="stat-label">완료된 미션</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">{{ total_files }}</div>
    <div class="stat-label">총 파일 수</div>
  </div>
<!--
  <div class="stat-card">
    <div class="stat-number">100%</div>
    <div class="stat-label">진행률</div>
  </div>
-->
</div>

---

<div class="navigation-footer">
  <a href="{{ site.baseurl }}/" class="nav-button home">
    <span class="nav-icon">🏠</span> 홈으로
  </a>
</div>

<style>
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

.completion-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.stat-card {
  background: white;
  border-radius: 10px;
  padding: 20px;
  text-align: center;
  border: 2px solid #3498db;
  box-shadow: 0 2px 8px rgba(52, 152, 219, 0.1);
}

.stat-number {
  font-size: 2.5em;
  font-weight: bold;
  color: #3498db;
  margin-bottom: 5px;
}

.stat-label {
  color: #666;
  font-size: 0.9em;
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
  background: #27ae60;
  color: white;
  border-radius: 6px;
  text-decoration: none;
  transition: all 0.3s ease;
  margin: 0 10px;
}

.nav-button:hover {
  background: #219a52;
  transform: translateY(-2px);
  text-decoration: none;
  color: white;
}

.nav-icon {
  margin-right: 8px;
  font-size: 16px;
}
</style>
