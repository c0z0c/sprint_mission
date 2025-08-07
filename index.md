---
layout: default
title: 스프린트 미션 보관함
description: 코드잇 AI 4기 스프린트 미션 결과물 보관함
cache-control: no-cache
expires: 0
pragma: no-cache
---

# 📚 스프린트 미션 보관함

코드잇 AI 4기 과정의 스프린트 미션 결과물들을 보관하는 공간입니다.

<div class="nav-sections">
  <div class="section-card">
    <h2>📂 폴더별 탐색</h2>
    <div class="folder-links">
      <a href="{{ site.baseurl }}/멘토/" class="folder-link">
        <span class="folder-icon">👨‍🏫</span>
        <span class="folder-name">멘토</span>
        <span class="folder-desc">멘토 관련 자료</span>
      </a>
      
      <a href="{{ site.baseurl }}/스프린트미션_완료/" class="folder-link">
        <span class="folder-icon">✅</span>
        <span class="folder-name">스프린트미션_완료</span>
        <span class="folder-desc">완료된 스프린트 미션들</span>
      </a>
      
      <a href="{{ site.baseurl }}/스프린트미션_작업중/" class="folder-link">
        <span class="folder-icon">🚧</span>
        <span class="folder-name">스프린트미션_작업중</span>
        <span class="folder-desc">진행 중인 미션들</span>
      </a>
      
      <a href="{{ site.baseurl }}/위클리페이퍼/" class="folder-link">
        <span class="folder-icon">📰</span>
        <span class="folder-name">위클리페이퍼</span>
        <span class="folder-desc">주간 학습 리포트</span>
      </a>
    </div>
  </div>

  <div class="section-card">
    <h2>🔗 빠른 링크</h2>
    <div class="quick-links">
      <a href="https://github.com/c0z0c/sprint_mission" target="_blank">
        <span class="link-icon">📱</span> GitHub 저장소
      </a>
      <a href="{{ site.baseurl }}/스프린트미션_완료/info">
        <span class="link-icon">📖</span> Info
      </a>
    </div>
  </div>
</div>

## 📋 최근 업데이트

- **2025년 8월**: GitHub Pages 웹호스팅 설정 완료
- **미션 4**: 완료된 미션 결과물 업로드
- **위클리페이퍼 #4**: 최신 학습 리포트 작성 완료

---

<div class="footer-info">
<small>
<strong>코드잇 AI 4기</strong> | 5팀 김명환<br>
마지막 업데이트: {{ site.time | date: "%Y년 %m월 %d일" }}
</small>
</div>

<style>
.nav-sections {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 30px;
  margin: 30px 0;
}

.section-card {
  background: #f8f9fa;
  border-radius: 12px;
  padding: 25px;
  border: 2px solid #e9ecef;
}

.section-card h2 {
  margin-top: 0;
  color: #2c3e50;
  border-bottom: 2px solid #3498db;
  padding-bottom: 10px;
}

.folder-links {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.folder-link {
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

.folder-link:hover {
  background: #e3f2fd;
  border-color: #3498db;
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
  text-decoration: none;
}

.folder-icon {
  font-size: 24px;
  margin-right: 15px;
  width: 30px;
  text-align: center;
}

.folder-name {
  font-weight: bold;
  color: #2c3e50;
  margin-right: 10px;
  flex: 1;
}

.folder-desc {
  color: #666;
  font-size: 0.9em;
  font-style: italic;
}

.quick-links {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.quick-links a {
  display: flex;
  align-items: center;
  padding: 12px;
  background: white;
  border-radius: 6px;
  text-decoration: none;
  border: 1px solid #dee2e6;
  transition: all 0.3s ease;
}

.quick-links a:hover {
  background: #e8f5e8;
  border-color: #27ae60;
  text-decoration: none;
}

.link-icon {
  margin-right: 10px;
  font-size: 16px;
}

.footer-info {
  text-align: center;
  margin-top: 40px;
  padding-top: 20px;
  border-top: 1px solid #eee;
  color: #666;
}

/* 반응형 디자인 */
@media (max-width: 768px) {
  .nav-sections {
    grid-template-columns: 1fr;
  }
  
  .folder-link {
    flex-direction: column;
    text-align: center;
    gap: 5px;
  }
  
  .folder-name, .folder-desc {
    margin: 0;
  }
}
</style>
