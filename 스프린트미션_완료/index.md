---
layout: default
title: 스터디 - 스터디 자료 보관함
description: 스터디 관련 자료들
cache-control: no-cache
expires: 0
pragma: no-cache
---

# 📚 스터디

스터디 관련 자료들을 모아둔 폴더입니다.

## 📄 파일 목록


<table>
  <thead>
    <tr>
      <th>파일명</th>
    </tr>
  </thead>
  <tbody>
    {% assign folder = '/스프린트미션_완료/' %}
    {% assign exclude_files = "index.md,info.html,info.md" | split: "," %}
    {% for file in site.static_files %}
      {% if file.path contains folder and exclude_files contains file.name == false %}
        <tr>
          <td>
            {% if file.extname == '.ipynb' %}
              <a href="https://github.com/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file.name }}" target="_blank">{{ file.name }}</a>
              &nbsp;&nbsp;
              <a href="https://colab.research.google.com/github/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file.name }}" target="_blank">(Colab에서 열기)</a>
            {% elsif file.extname == '.docx' %}
              <a href="https://github.com/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file.name }}" target="_blank">{{ file.name }}</a>
              &nbsp;&nbsp;
              <a href="https://docs.google.com/viewer?url=https://c0z0c.github.io/sprint_mission/blob/master/스프린트미션_완료/{{ file.name }}" target="_blank">(Google에서 열기)</a>
            {% else %}
              <a href="https://c0z0c.github.io/sprint_mission/스프린트미션_완료/{{ file.name }}" target="_blank">{{ file.name }}</a>
            {% endif %}
          </td>
        </tr>
      {% endif %}
    {% endfor %}
  </tbody>
</table>

## 📊 완료 현황

<div class="completion-stats">
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
.file-list {
  margin: 20px 0;
}

.file-item {
  margin-bottom: 8px;
}

.file-item.featured {
  margin-bottom: 20px;
}

.mission-group {
  margin: 30px 0;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 10px;
  border-left: 4px solid #3498db;
}

.mission-group h3 {
  margin: 0 0 15px 0;
  color: #2c3e50;
  font-size: 1.2em;
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
  position: relative;
}

.item-link:hover {
  transform: translateY(-1px);
  box-shadow: 0 3px 6px rgba(0,0,0,0.1);
  text-decoration: none;
}

.item-link.readme:hover {
  background: #e3f2fd;
  border-color: #2196f3;
}

.item-link.notebook:hover {
  background: #fff3e0;
  border-color: #ff9800;
}

.item-link.document:hover {
  background: #e8f5e8;
  border-color: #4caf50;
}

.item-link.pdf:hover {
  background: #ffebee;
  border-color: #f44336;
}

.item-link.python:hover {
  background: #f3e5f5;
  border-color: #9c27b0;
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

.item-badge {
  background: #e74c3c;
  color: white;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.75em;
  font-weight: bold;
  margin-left: 10px;
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

.file-actions {
  margin-top: 8px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.action-link {
  display: inline-flex;
  align-items: center;
  padding: 6px 12px;
  text-decoration: none;
  border-radius: 4px;
  font-size: 0.85em;
  transition: all 0.3s ease;
  border: 1px solid;
}

.action-link.github {
  background: #f6f8fa;
  color: #24292e;
  border-color: #d0d7de;
}

.action-link.github:hover {
  background: #24292e;
  color: white;
  text-decoration: none;
}

.action-link.nbviewer {
  background: #fff8e1;
  color: #e65100;
  border-color: #ffb74d;
}

.action-link.nbviewer:hover {
  background: #e65100;
  color: white;
  text-decoration: none;
}

.action-link.colab {
  background: #fff3e0;
  color: #f57c00;
  border-color: #ffb74d;
}

.action-link.colab:hover {
  background: #f57c00;
  color: white;
  text-decoration: none;
}

.action-icon {
  margin-right: 6px;
  font-size: 14px;
}

.action-text {
  font-size: 0.8em;
}
</style>
