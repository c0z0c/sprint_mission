---
layout: default
title: 멘토 - 스프린트 미션 보관함
description: 멘토 관련 자료들
cache-control: no-cache
expires: 0
pragma: no-cache
---

# 👨‍🏫 멘토

멘토 관련 자료들을 모아둔 폴더입니다.

## 📁 폴더 목록

<div class="file-list">
  <div class="folder-item">
    <a href="{{ site.baseurl }}/멘토/CensusIncome/" class="item-link folder">
      <span class="item-icon">📂</span>
      <span class="item-name">CensusIncome</span>
      <span class="item-desc">Census Income 데이터 분석 프로젝트</span>
    </a>
  </div>
</div>

## 📄 파일 목록

<table>
  <thead>
    <tr>
      <th>파일명</th>
    </tr>
  </thead>
  <tbody>
    {% assign folder = '/멘토/' %}
    {% assign exclude_files = "index.md,info.html,info.md" | split: "," %}
    {% for file in site.static_files %}
      {% if file.path contains folder and exclude_files contains file.name == false %}
        <tr>
          <td>
            {% if file.extname == '.ipynb' %}
              <a href="https://github.com/c0z0c/sprint_mission/blob/master/멘토/{{ file.name }}" target="_blank">{{ file.name }}</a>
              &nbsp;&nbsp;
              <a href="https://colab.research.google.com/github/c0z0c/sprint_mission/blob/master/멘토/{{ file.name }}" target="_blank">(Colab에서 열기)</a>
            {% elsif file.extname == '.docx' %}
              <a href="https://github.com/c0z0c/sprint_mission/blob/master/멘토/{{ file.name }}" target="_blank">{{ file.name }}</a>
              &nbsp;&nbsp;
              <a href="https://docs.google.com/viewer?url=https://c0z0c.github.io/sprint_mission/blob/master/멘토/{{ file.name }}" target="_blank">(Google에서 열기)</a>
            {% elsif file.extname == '.html' or file.extname == '.md' %}
              <a href="https://c0z0c.github.io/sprint_mission/멘토/{{ file.name }}" target="_blank">{{ file.name }}</a>
            {% else %}
              <span>{{ file.name }}</span>
            {% endif %}
          </td>
        </tr>
      {% endif %}
    {% endfor %}
  </tbody>
</table>

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

.item-link.folder:hover {
  background: #fff3e0;
  border-color: #ff9800;
}

.item-link.file:hover {
  background: #e8f5e8;
  border-color: #4caf50;
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

.empty-message {
  text-align: center;
  padding: 40px 20px;
  color: #666;
}

.empty-icon {
  font-size: 48px;
  display: block;
  margin-bottom: 15px;
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

.nav-icon {
  margin-right: 8px;
  font-size: 16px;
}
</style>
