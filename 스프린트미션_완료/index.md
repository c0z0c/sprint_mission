---
layout: default
title: 스프린트미션_완료 - 완료된 미션 보관함
description: 완료된 스프린트 미션 자료들
cache-control: no-cache
expires: 0
pragma: no-cache
---

# ✅ 스프린트미션_완료

{% assign current_folder = "스프린트미션_완료/" %}
{% assign folders = site.static_files | where_exp: "item", "item.path contains current_folder" | where_exp: "item", "item.path != item.name" | map: "path" | join: "|" | split: "|" %}
{% assign unique_folders = "" | split: "" %}

<div class="file-grid">
  <!-- Static files (non-markdown) -->
  {% assign current_folder = "스프린트미션_완료/" %}
  {% assign static_files = site.static_files | where_exp: "item", "item.path contains current_folder" %}
  {% assign markdown_pages = site.pages | where_exp: "page", "page.path contains '스프린트미션_완료'" %}
  
  {% assign all_files = "" | split: "" %}
  {% assign all_file_names = "" | split: "" %}

  <!-- Add static files -->
  {% for file in static_files %}
    {% unless file.name == "index.md" or all_file_names contains file.name %}
      {% assign all_files = all_files | push: file %}
      {% assign all_file_names = all_file_names | push: file.name %}
    {% endunless %}
  {% endfor %}

  <!-- Add markdown pages -->
  {% for page in markdown_pages %}
    {% unless page.name == "index.md" or all_file_names contains page.name %}
      {% assign all_files = all_files | push: page %}
      {% assign all_file_names = all_file_names | push: page.name %}
    {% endunless %}
  {% endfor %}
  
  <!-- Debug: Show what files are being processed -->
  <!-- Total files found: {{ all_files.size }} -->
  {% if all_files.size > 0 %}
    <!-- Sort files by date (newest first) -->
    {% assign sorted_files = all_files | sort: 'modified_time' | reverse %}
    {% if sorted_files.size == 0 or sorted_files[0].modified_time == nil %}
      {% assign sorted_files = all_files | sort: 'date' | reverse %}
    {% endif %}
    {% for file in sorted_files %}
      <!-- file {{ file.name }} -->
      {% assign file_ext = file.extname | downcase %}
      {% if file_ext == "" and file.path %}
        {% assign file_name = file.path | split: "/" | last %}
        {% assign file_ext = file_name | split: "." | last | downcase %}
        {% assign file_ext = "." | append: file_ext %}
      {% endif %}
      
      <!-- Handle page objects differently from static files -->
      {% assign is_page = false %}
      {% if file.url %}
        {% assign is_page = true %}
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
      {% elsif file_ext == ".zip" %}
        {% assign file_icon = "📦" %}
        {% assign file_type = "압축 파일" %}
      {% elsif file_ext == ".png" or file_ext == ".jpg" or file_ext == ".jpeg" %}
        {% assign file_icon = "🖼️" %}
        {% assign file_type = "이미지 파일" %}
      {% elsif file_ext == ".csv" %}
        {% assign file_icon = "📊" %}
        {% assign file_type = "데이터 파일" %}
      {% endif %}
      
      <div class="file-item">
        <div class="file-icon">{{ file_icon }}</div>
        <div class="file-info">
          <h4 class="file-name">
            {% if is_page %}
              {% assign display_name = file.name | default: file.path | split: "/" | last %}
            {% else %}
              {% assign display_name = file.name | default: file.path | split: "/" | last %}
            {% endif %}
            {{ display_name }}
          </h4>
          <p class="file-type">{{ file_type }}</p>
          <p class="file-size">
            {% if is_page %}
              {% if file.date %}{{ file.date | date: "%Y-%m-%d" }}{% else %}Page{% endif %}
            {% else %}
              {% if file.modified_time %}{{ file.modified_time | date: "%Y-%m-%d" }}{% else %}{{ file.date | date: "%Y-%m-%d" }}{% endif %}
            {% endif %}
          </p>
        </div>
        <div class="file-actions">
        <!-- file_ext {{ file_ext }} -->
        <!-- display_name {{ display_name }} -->
          {% if file_ext == ".md" and display_name != "index.md" %}
            {% assign file_name_clean = display_name %}
            {% assign md_name_clean = file_name_clean | remove: '.md' %}
            <a href="https://c0z0c.github.io/sprint_mission/스프린트미션_완료/{{ md_name_clean }}" class="file-action" title="렌더링된 페이지 보기" target="_blank">🌐</a>
            <a href="https://github.com/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="GitHub에서 원본 보기" target="_blank">📖</a>
          {% elsif file_ext == ".ipynb" %}
            {% assign file_name_clean = display_name %}
            <a href="https://github.com/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="GitHub에서 보기" target="_blank">📖</a>
            <a href="https://colab.research.google.com/github/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="Colab에서 열기" target="_blank">🚀</a>
          {% elsif file_ext == ".pdf" %}
            {% assign file_name_clean = display_name %}
            <a href="https://github.com/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="GitHub에서 보기" target="_blank">📖</a>
            <a href="https://docs.google.com/viewer?url=https://raw.githubusercontent.com/c0z0c/sprint_mission/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="PDF 뷰어로 열기" target="_blank">📄</a>
          {% elsif file_ext == ".docx" %}
            {% assign file_name_clean = display_name %}
            <a href="https://github.com/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="GitHub에서 보기" target="_blank">📖</a>
            <a href="https://docs.google.com/viewer?url=https://raw.githubusercontent.com/c0z0c/sprint_mission/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="Google에서 열기" target="_blank">📊</a>
          {% elsif file_ext == ".html" %}
            {% assign file_name_clean = display_name %}
            <a href="https://c0z0c.github.io/sprint_mission/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="웹페이지로 보기" target="_blank">🌐</a>
            <a href="https://github.com/c0z0c/sprint_mission/blob/master/스프린트미션_완료/{{ file_name_clean }}" class="file-action" title="GitHub에서 원본 보기" target="_blank">📖</a>
          {% else %}
            {% if is_page %}
              <a href="{{ file.url | relative_url }}" class="file-action" title="페이지 열기">🌐</a>
            {% else %}
              <a href="{{ file.path | relative_url }}" class="file-action" title="파일 열기">📖</a>
            {% endif %}
          {% endif %}
        </div>
      </div>
    {% endfor %}
  {% else %}
    <div class="empty-message">
      <span class="empty-icon">📄</span>
      <h3>파일이 없습니다</h3>
      <p>현재 이 위치에는 완료된 미션 파일이 없습니다.</p>
    </div>
  {% endif %}
</div>

---

<div class="navigation-footer">
  <a href="{{ site.baseurl }}/" class="nav-button home">
    <span class="nav-icon">🏠</span> 홈으로
  </a>
</div>
