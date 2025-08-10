---
layout: default
title: 위클리페이퍼 - 스프린트 미션 보관함
description: 주간 학습 리포트
cache-control: no-cache
expires: 0
pragma: no-cache
---

# 📰 위클리페이퍼

주간 학습 리포트들을 모아둔 폴더입니다.

## 📄 파일 목록

<details>
<ul>
{% for file in site.static_files %}
  {% if file.path contains '위클리페이퍼' %}
    <li>Static File: {{ file.path }} ({{ file.name }})</li>
  {% endif %}
{% endfor %}
{% for page in site.pages %}
  {% if page.path contains '위클리페이퍼' %}
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
    {% assign folder_path = '/sprint_mission/위클리페이퍼/' %}
    {% assign exclude_files = "index.md,info.md" | split: "," %}
    {% assign files = site.static_files | where_exp: "file", "file.path contains '/sprint_mission/위클리페이퍼/'" %}
    {% assign sorted_files = files | sort: 'name' | reverse %}
    
    {% for file in sorted_files %}
      {% unless exclude_files contains file.name %}
        <tr>
          <td>
            {% if file.extname == '.ipynb' %}
              <div>
                <a href="https://github.com/c0z0c/sprint_mission/blob/master/위클리페이퍼/{{ file.name }}" target="_blank">{{ file.name }}</a>
                <br>
                <small>
                  <a href="https://colab.research.google.com/github/c0z0c/sprint_mission/blob/master/위클리페이퍼/{{ file.name }}" target="_blank" style="color: #f57c00;">🔗 Colab에서 열기</a>
                </small>
              </div>
            {% elsif file.extname == '.docx' %}
              <div>
                <a href="https://github.com/c0z0c/sprint_mission/blob/master/위클리페이퍼/{{ file.name }}" target="_blank">{{ file.name }}</a>
                <br>
                <small>
                  <a href="https://docs.google.com/viewer?url=https://raw.githubusercontent.com/c0z0c/sprint_mission/master/위클리페이퍼/{{ file.name }}" target="_blank" style="color: #4285f4;">🔗 Google에서 열기</a>
                </small>
              </div>
            {% elsif file.extname == '.pdf' %}
              <div>
                <a href="https://github.com/c0z0c/sprint_mission/blob/master/위클리페이퍼/{{ file.name }}" target="_blank">{{ file.name }}</a>
                <br>
                <small>
                  <a href="https://docs.google.com/viewer?url=https://raw.githubusercontent.com/c0z0c/sprint_mission/master/위클리페이퍼/{{ file.name }}" target="_blank" style="color: #dc3545;">🔗 PDF 뷰어로 열기</a>
                </small>
              </div>
            {% elsif file.extname == '.html' %}
              <a href="https://c0z0c.github.io/sprint_mission/위클리페이퍼/{{ file.name }}" target="_blank">{{ file.name }}</a>
            {% elsif file.extname == '.md' and file.name != 'index.md' %}
              {% assign md_name = file.name | remove: '.md' %}
              <div>
                <a href="https://c0z0c.github.io/sprint_mission/위클리페이퍼/{{ md_name }}" target="_blank">{{ file.name }}</a>
                <br>
                <small>
                  <a href="https://github.com/c0z0c/sprint_mission/blob/master/위클리페이퍼/{{ file.name }}" target="_blank" style="color: #6c757d;">🔗 GitHub에서 원본 보기</a>
                </small>
              </div>
            {% else %}
              <a href="https://github.com/c0z0c/sprint_mission/blob/master/위클리페이퍼/{{ file.name }}" target="_blank">{{ file.name }}</a>
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

## 📊 위클리페이퍼 현황

{% assign weekly_files = site.static_files | where_exp: "file", "file.path contains '/sprint_mission/위클리페이퍼/'" %}
{% assign weekly_papers = weekly_files | where_exp: "file", "file.name contains '위클리_페이퍼_'" %}
{% assign total_files = weekly_files | where_exp: "file", "file.name != 'index.md'" | size %}
{% assign completed_papers = weekly_papers | size %}

<div class="weekly-stats">
  <div class="stat-card">
    <div class="stat-number">{{ completed_papers }}</div>
    <div class="stat-label">작성 완료</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">{{ total_files }}</div>
    <div class="stat-label">총 파일 수</div>
  </div>
  <div class="stat-card">
    <div class="stat-number">주간</div>
    <div class="stat-label">학습 리포트</div>
  </div>
</div>

## 📈 학습 진행사항

{% assign weekly_files = site.static_files | where_exp: "file", "file.path contains '/sprint_mission/위클리페이퍼/'" %}
{% assign weekly_papers = weekly_files | where_exp: "file", "file.name contains '위클리_페이퍼_'" %}
{% assign sorted_papers = weekly_papers | sort: 'name' %}

<div class="progress-timeline">
  {% for paper in sorted_papers %}
    {% assign paper_number = paper.name | remove: '위클리_페이퍼_' | remove: '_AI4기_김명환.ipynb' | remove: '_AI4기_김명환.md' | remove: '_AI4기_김명환.html' %}
    {% assign is_last = forloop.last %}
    
    <div class="timeline-item completed{% if is_last %} current{% endif %}">
      <div class="timeline-marker">{% if is_last %}🔥{% else %}✅{% endif %}</div>
      <div class="timeline-content">
        <h4>위클리 페이퍼 #{{ paper_number }}</h4>
        <p>{% if is_last %}최신 주간 학습 정리 (현재){% else %}{{ paper_number }}번째 주간 학습 정리{% endif %}</p>
      </div>
    </div>
  {% endfor %}
</div>

---

<div class="navigation-footer">
  <a href="{{ site.baseurl }}/" class="nav-button home">
    <span class="nav-icon">🏠</span> 홈으로
  </a>
</div>

<style>
.file-list {
  margin: 20px 0;
}

.file-item {
  margin-bottom: 8px;
}

.weekly-group, .utility-group {
  margin: 25px 0;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 10px;
  border-left: 4px solid #17a2b8;
}

.weekly-group.latest {
  border-left-color: #28a745;
  background: linear-gradient(135deg, #f8fff9 0%, #f8f9fa 100%);
}

.weekly-group h3, .utility-group h3 {
  margin: 0 0 15px 0;
  color: #2c3e50;
  font-size: 1.1em;
  display: flex;
  align-items: center;
}

.latest-badge {
  background: #28a745;
  color: white;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.7em;
  font-weight: bold;
  margin-left: 10px;
}

.item-link {
  display: flex;
  align-items: center;
  padding: 12px 15px;
  background: white;
  border-radius: 6px;
  text-decoration: none;
  border: 1px solid #dee2e6;
  transition: all 0.3s ease;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.item-link:hover {
  transform: translateY(-1px);
  box-shadow: 0 3px 6px rgba(0,0,0,0.1);
  text-decoration: none;
}

.item-link.notebook:hover {
  background: #fff3e0;
  border-color: #ff9800;
}

.item-link.markdown:hover {
  background: #e8f5e8;
  border-color: #4caf50;
}

.file-display {
  cursor: default;
}

.file-display:hover {
  background: #f5f5f5;
  border-color: #ccc;
}

.item-icon {
  font-size: 20px;
  margin-right: 12px;
  width: 25px;
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
  font-size: 0.85em;
  font-style: italic;
}

.weekly-stats {
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
  border: 2px solid #17a2b8;
  box-shadow: 0 2px 6px rgba(23, 162, 184, 0.1);
}

.stat-number {
  font-size: 2em;
  font-weight: bold;
  color: #17a2b8;
  margin-bottom: 5px;
}

.stat-label {
  color: #666;
  font-size: 0.85em;
}

.progress-timeline {
  margin: 30px 0;
  position: relative;
}

.timeline-item {
  display: flex;
  align-items: flex-start;
  margin-bottom: 20px;
  position: relative;
}

.timeline-item:not(:last-child)::after {
  content: '';
  position: absolute;
  left: 15px;
  top: 35px;
  width: 2px;
  height: 40px;
  background: #dee2e6;
}

.timeline-item.current::after {
  background: #28a745;
}

.timeline-marker {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  margin-right: 15px;
  flex-shrink: 0;
  background: white;
  border: 2px solid #28a745;
}

.timeline-content {
  flex: 1;
  padding-top: 2px;
}

.timeline-content h4 {
  margin: 0 0 5px 0;
  color: #2c3e50;
  font-size: 1em;
}

.timeline-content p {
  margin: 0;
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
