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
  
  <!-- 파일을 최근 수정/생성일 순으로 내림차순 정렬 -->
  {% assign sorted_files = all_files | sort: "modified_time" %}

  {% if sorted_files.size > 0 %}
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

<style>
.file-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 15px;
  margin: 20px 0;
}

.file-item {
  display: flex;
  align-items: center;
  padding: 15px;
  background: white;
  border-radius: 8px;
  border: 1px solid #dee2e6;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  transition: all 0.3s ease;
}

.file-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  border-color: #3498db;
}

.folder-item {
  border-left: 4px solid #f39c12;
}

.file-item:not(.folder-item) {
  border-left: 4px solid #3498db;
}

.file-icon {
  font-size: 24px;
  margin-right: 15px;
  width: 40px;
  text-align: center;
}

.file-info {
  flex: 1;
}

.file-name {
  margin: 0 0 4px 0;
  font-size: 1em;
  color: #2c3e50;
  font-weight: 600;
}

.file-type {
  margin: 0 0 2px 0;
  font-size: 0.85em;
  color: #666;
}

.file-size {
  margin: 0;
  font-size: 0.8em;
  color: #999;
}

.file-actions {
  display: flex;
  gap: 8px;
}

.file-action {
  padding: 6px 8px;
  background: #f8f9fa;
  border-radius: 4px;
  text-decoration: none;
  font-size: 16px;
  transition: background 0.3s ease;
}

.file-action:hover {
  background: #e9ecef;
  text-decoration: none;
}

.empty-message {
  grid-column: 1 / -1;
  text-align: center;
  padding: 60px 20px;
  background: #f8f9fa;
  border-radius: 12px;
  border: 2px dashed #dee2e6;
}

.empty-folder {
  margin: 30px 0;
}

.empty-icon {
  font-size: 64px;
  display: block;
  margin-bottom: 20px;
  opacity: 0.6;
}

.empty-message h3 {
  color: #6c757d;
  margin-bottom: 10px;
}

.empty-message p {
  color: #6c757d;
  margin: 0;
  font-style: italic;
}

.preparation-section {
  margin: 40px 0;
  padding: 30px;
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border-radius: 12px;
  border: 1px solid #dee2e6;
}

.preparation-section h3 {
  margin-top: 0;
  color: #2c3e50;
  text-align: center;
  margin-bottom: 25px;
}

.prep-card {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
  padding: 15px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  border-left: 4px solid #17a2b8;
}

.prep-card:last-child {
  margin-bottom: 0;
}

.prep-icon {
  font-size: 24px;
  margin-right: 15px;
  width: 40px;
  text-align: center;
  color: #17a2b8;
}

.prep-content h4 {
  margin: 0 0 5px 0;
  color: #2c3e50;
  font-size: 1em;
}

.prep-content p {
  margin: 0;
  color: #666;
  font-size: 0.9em;
}

.progress-overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.progress-card {
  background: white;
  border-radius: 10px;
  padding: 20px;
  text-align: center;
  border: 2px solid #28a745;
  box-shadow: 0 2px 8px rgba(40, 167, 69, 0.1);
}

.progress-card.waiting {
  border-color: #ffc107;
  box-shadow: 0 2px 8px rgba(255, 193, 7, 0.1);
}

.progress-card.working {
  border-color: #17a2b8;
  box-shadow: 0 2px 8px rgba(23, 162, 184, 0.1);
}

.progress-number {
  font-size: 2.5em;
  font-weight: bold;
  color: #28a745;
  margin-bottom: 5px;
}

.progress-card.waiting .progress-number {
  color: #ffc107;
}

.progress-card.working .progress-number {
  color: #17a2b8;
}

.progress-label {
  color: #666;
  font-size: 0.9em;
  margin-bottom: 10px;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background: #e9ecef;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #28a745;
  transition: width 0.3s ease;
}

.waiting-fill {
  background: linear-gradient(90deg, #ffc107, #fd7e14);
  animation: pulse 2s infinite;
}

.working-fill {
  background: linear-gradient(90deg, #17a2b8, #20c997);
  animation: progress 3s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 0.7; }
}

@keyframes progress {
  0%, 100% { opacity: 0.7; }
  50% { opacity: 1; }
}

.related-links {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin: 30px 0;
}

.related-link {
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

.related-link:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  text-decoration: none;
  background: #f8f9fa;
  border-color: #3498db;
}

.link-icon {
  font-size: 20px;
  margin-right: 12px;
  color: #3498db;
}

.link-text {
  color: #2c3e50;
  font-weight: 500;
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
