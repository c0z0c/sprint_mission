---
layout: default
title: 스터디 - 스터디 보관함
description: 스터디 자료들
cache-control: no-cache
expires: 0
pragma: no-cache
---

# ✅ 스터디

<script>

{% assign cur_dir = "/스터디/" %}
{% include cur_files.liquid %}
{% include page_values.html %}
{% include page_files.html %}

</script>

<div class="file-grid">
  <!-- 파일 목록이 JavaScript로 동적 생성됩니다 -->
</div>

---

<div class="navigation-footer">
  <a href="{{- site.baseurl -}}/" class="nav-button home">
    <span class="nav-icon">🏠</span> 홈으로
  </a>
</div>