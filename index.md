---
layout: default
title: 스프린트 미션 보관함
description: 코드잇 AI 4기 스프린트 미션 결과물 보관함
cache-control: no-cache
expires: 0
pragma: no-cache
---

<div class="nav-sections">
  <div class="section-card">
    <h2>� 폴더별 탐색</h2>
    <div class="folder-links">
      {% assign folder_set = "" | split: "" %}
      {% assign folder_icons = "멘토:👨‍🏫,스프린트미션_완료:✅,스프린트미션_작업중:🚧,위클리페이퍼:📰,스터디:📒,실습:🔬,백업:💾" | split: "," %}
      {% assign folder_descs = "멘토:멘토 관련 자료,스프린트미션_완료:완료된 스프린트 미션들,스프린트미션_작업중:진행 중인 미션들,위클리페이퍼:주간 학습 리포트,스터디:학습,실습:실습 자료,백업:백업 파일들" | split: "," %}
      
      <!-- 정적 파일에서 폴더 추출 -->
      {% for file in site.static_files %}
        {% assign path_parts = file.path | split: '/' %}
        {% if path_parts.size > 1 %}
          {% assign folder = path_parts[0] %}
          {% unless folder_set contains folder or folder == '' or folder contains '.' or folder == 'assets' or folder == '_layouts' %}
            {% assign folder_set = folder_set | push: folder %}
          {% endunless %}
        {% endif %}
      {% endfor %}
      
      <!-- 페이지에서 폴더 추출 -->
      {% for page in site.pages %}
        {% assign path_parts = page.path | split: '/' %}
        {% if path_parts.size > 1 %}
          {% assign folder = path_parts[0] %}
          {% unless folder_set contains folder or folder == '' or folder contains '.' or folder == 'assets' or folder == '_layouts' %}
            {% assign folder_set = folder_set | push: folder %}
          {% endunless %}
        {% endif %}
      {% endfor %}
      
      <!-- 폴더 목록 출력 -->
      {% assign sorted_folders = folder_set | sort %}
      {% for folder in sorted_folders %}
        {% assign folder_icon = "📁" %}
        {% assign folder_desc = "" %}
        
        <!-- 아이콘 찾기 -->
        {% for icon_pair in folder_icons %}
          {% assign icon_parts = icon_pair | split: ":" %}
          {% if icon_parts[0] == folder %}
            {% assign folder_icon = icon_parts[1] %}
            {% break %}
          {% endif %}
        {% endfor %}
        
        <!-- 설명 찾기 -->
        {% for desc_pair in folder_descs %}
          {% assign desc_parts = desc_pair | split: ":" %}
          {% if desc_parts[0] == folder %}
            {% assign folder_desc = desc_parts[1] %}
            {% break %}
          {% endif %}
        {% endfor %}
        
        <a href="{{ site.baseurl }}/{{ folder }}/" class="folder-link">
          <span class="folder-icon">{{ folder_icon }}</span>
          <span class="folder-name">{{ folder }}</span>
          {% if folder_desc != "" %}
            <span class="folder-desc">{{ folder_desc }}</span>
          {% endif %}
        </a>
      {% endfor %}
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
