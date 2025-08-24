---
layout: default
title: 스프린트미션_완료 - 완료된 미션 보관함
description: 완료된 스프린트 미션 자료들
cache-control: no-cache
expires: 0
pragma: no-cache
---

# ✅ 스프린트미션_완료

<script>

{%- assign cur_dir = "/스프린트미션_완료/" -%}
{%- include cur_files.liquid -%}

  var curDir = '{{- cur_file_dir -}}';
  var curFiles = {{- cur_files_json -}};
  var curPages = {{- cur_pages_json -}};
  
  console.log('curDir:', curDir);
  console.log('curFiles:', curFiles);
  console.log('curPages:', curPages);

</script>
