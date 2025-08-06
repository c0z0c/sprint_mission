---
layout: default
title: 스프린트미션_작업중 - 스프린트 미션 보관함
description: 진행 중인 미션들
---

# 🚧 스프린트미션_작업중

현재 진행 중인 스프린트 미션들을 모아둔 폴더입니다.

## 📁 폴더 목록

<div class="empty-folder">
  <div class="empty-message">
    <span class="empty-icon">📭</span>
    <h3>폴더가 비어있습니다</h3>
    <p>현재 진행 중인 미션이 없습니다.</p>
  </div>
</div>

## 📄 파일 목록

<div class="empty-folder">
  <div class="empty-message">
    <span class="empty-icon">📄</span>
    <h3>파일이 없습니다</h3>
    <p>새로운 미션이 시작되면 여기에 작업 파일들이 추가됩니다.</p>
  </div>
</div>

## 🎯 다음 미션 준비

<div class="preparation-section">
  <h3>📋 준비 사항</h3>
  <div class="prep-card">
    <div class="prep-icon">⚡</div>
    <div class="prep-content">
      <h4>개발 환경 설정</h4>
      <p>Jupyter Notebook과 필요한 라이브러리들이 준비되어 있습니다.</p>
    </div>
  </div>
  
  <div class="prep-card">
    <div class="prep-icon">📚</div>
    <div class="prep-content">
      <h4>학습 자료</h4>
      <p>이전 미션들의 경험과 위클리페이퍼를 통한 학습 정리가 완료되었습니다.</p>
    </div>
  </div>
  
  <div class="prep-card">
    <div class="prep-icon">🔧</div>
    <div class="prep-content">
      <h4>헬퍼 모듈</h4>
      <p>helper_c0z0c_dev.py 모듈을 통한 효율적인 개발 환경이 구축되어 있습니다.</p>
    </div>
  </div>
</div>

## 📈 진행률

<div class="progress-overview">
  <div class="progress-card">
    <div class="progress-number">4</div>
    <div class="progress-label">완료된 미션</div>
    <div class="progress-bar">
      <div class="progress-fill" style="width: 100%"></div>
    </div>
  </div>
  
  <div class="progress-card waiting">
    <div class="progress-number">?</div>
    <div class="progress-label">다음 미션</div>
    <div class="progress-bar">
      <div class="progress-fill waiting-fill" style="width: 0%"></div>
    </div>
  </div>
</div>

## 🔗 관련 링크

<div class="related-links">
  <a href="{{ site.baseurl }}/스프린트미션_완료/" class="related-link">
    <span class="link-icon">✅</span>
    <span class="link-text">완료된 미션들 보기</span>
  </a>
  
  <a href="{{ site.baseurl }}/위클리페이퍼/" class="related-link">
    <span class="link-icon">📰</span>
    <span class="link-text">위클리페이퍼 확인</span>
  </a>
  
  <a href="{{ site.baseurl }}/멘토/" class="related-link">
    <span class="link-icon">👨‍🏫</span>
    <span class="link-text">멘토 자료 참고</span>
  </a>
</div>

---

<div class="navigation-footer">
  <a href="{{ site.baseurl }}/" class="nav-button home">
    <span class="nav-icon">🏠</span> 홈으로
  </a>
</div>

<style>
.empty-folder {
  margin: 30px 0;
}

.empty-message {
  text-align: center;
  padding: 60px 20px;
  background: #f8f9fa;
  border-radius: 12px;
  border: 2px dashed #dee2e6;
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

.progress-number {
  font-size: 2.5em;
  font-weight: bold;
  color: #28a745;
  margin-bottom: 5px;
}

.progress-card.waiting .progress-number {
  color: #ffc107;
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

@keyframes pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 0.7; }
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
