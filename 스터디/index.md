---
layout: default
title: 스터디 - 스터디 자료 보관함
description: 스터디 관련 자료들
cache-control: no-cache
expires: 0
pragma: no-cache
---

# � 스터디

스터디 관련 자료들을 모아둔 폴더입니다.

## 📄 파일 목록

<!-- 디버깅: 모든 파일 출력 -->
<details>
<summary>🔍 디버깅: 감지된 모든 파일들</summary>
<ul>
{% for file in site.static_files %}
  {% if file.path contains '스터디' %}
    <li>Static File: {{ file.path }} ({{ file.name }})</li>
  {% endif %}
{% endfor %}
{% for page in site.pages %}
  {% if page.path contains '스터디' %}
    <li>Page: {{ page.path }} ({{ page.name }})</li>
  {% endif %}
{% endfor %}
</ul>
</details>

<table>
  <thead>
    <tr>
      <th>파일명</th>
      <th>타입</th>
    </tr>
  </thead>
  <tbody>
    {% assign folder_path = 'sprint_mission/스터디/' %}
    {% assign exclude_files = "index.md,info.md" | split: "," %}
    {% assign files = site.static_files | where_exp: "file", "file.path contains '스터디/'" %}
    {% assign sorted_files = files | sort: 'name' | reverse %}
    
    {% for file in sorted_files %}
      {% unless exclude_files contains file.name %}
        <tr>
          <td>
            {% if file.extname == '.ipynb' %}
              <div>
                <a href="https://github.com/c0z0c/sprint_mission/blob/master/스터디/{{ file.name }}" target="_blank">{{ file.name }}</a>
                <br>
                <small>
                  <a href="https://colab.research.google.com/github/c0z0c/sprint_mission/blob/master/스터디/{{ file.name }}" target="_blank" style="color: #f57c00;">🔗 Colab에서 열기</a>
                </small>
              </div>
            {% elsif file.extname == '.docx' %}
              <div>
                <a href="https://github.com/c0z0c/sprint_mission/blob/master/스터디/{{ file.name }}" target="_blank">{{ file.name }}</a>
                <br>
                <small>
                  <a href="https://docs.google.com/viewer?url=https://raw.githubusercontent.com/c0z0c/sprint_mission/master/스터디/{{ file.name }}" target="_blank" style="color: #4285f4;">🔗 Google에서 열기</a>
                </small>
              </div>
            {% elsif file.extname == '.pdf' %}
              <div>
                <a href="https://github.com/c0z0c/sprint_mission/blob/master/스터디/{{ file.name }}" target="_blank">{{ file.name }}</a>
                <br>
                <small>
                  <a href="https://docs.google.com/viewer?url=https://raw.githubusercontent.com/c0z0c/sprint_mission/master/스터디/{{ file.name }}" target="_blank" style="color: #dc3545;">🔗 PDF 뷰어로 열기</a>
                </small>
              </div>
            {% elsif file.extname == '.html' %}
              <a href="https://c0z0c.github.io/sprint_mission/스터디/{{ file.name }}" target="_blank">{{ file.name }}</a>
            {% elsif file.extname == '.md' and file.name != 'index.md' %}
              {% assign md_name = file.name | remove: '.md' %}
              <a href="https://c0z0c.github.io/sprint_mission/스터디/{{ md_name }}" target="_blank">{{ file.name }}</a>
            {% else %}
              <a href="https://github.com/c0z0c/sprint_mission/blob/master/스터디/{{ file.name }}" target="_blank">{{ file.name }}</a>
            {% endif %}
          </td>
          <td>
            {% if file.extname == '.ipynb' %}
              <span style="background: #ff9800; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">📓 Notebook</span>
            {% elsif file.extname == '.docx' %}
              <span style="background: #2196f3; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">📄 Word</span>
            {% elsif file.extname == '.pdf' %}
              <span style="background: #f44336; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">📋 PDF</span>
            {% elsif file.extname == '.html' %}
              <span style="background: #4caf50; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">🌐 HTML</span>
            {% elsif file.extname == '.md' %}
              <span style="background: #9c27b0; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">📝 Markdown</span>
            {% else %}
              <span style="background: #757575; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">📎 기타</span>
            {% endif %}
          </td>
        </tr>
      {% endunless %}
    {% endfor %}
  </tbody>
</table>

## 📊 스터디 현황

<div class="study-stats">
  <div class="stat-card">
    <div class="stat-number">0</div>
    <div class="stat-label">완료된 스터디</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">0</div>
    <div class="stat-label">총 파일 수</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">0%</div>
    <div class="stat-label">진행률</div>
  </div>
</div>

---

<div class="navigation-footer">
  <a href="{{ site.baseurl }}/" class="nav-button home">
    <span class="nav-icon">🏠</span> 홈으로
  </a>
</div>

<style>
.study-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 15px;
  margin: 30px 0;
}

.stat-card {
  background: white;
  border-radius: 8px;
  padding: 15px;
  text-align: center;
  border: 2px solid #3498db;
  box-shadow: 0 2px 6px rgba(52, 152, 219, 0.1);
}

.stat-number {
  font-size: 2em;
  font-weight: bold;
  color: #3498db;
  margin-bottom: 5px;
}

.stat-label {
  color: #666;
  font-size: 0.85em;
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
