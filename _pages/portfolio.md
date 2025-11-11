---
layout: default
title: Portfolio
permalink: /portfolio/
---

<p>Welcome to my portfolio! Here you'll find a collection of my projects, skills, and experiences.</p>


<div class="posts">
  {% for post in site.posts %}
  {% if post.is_work == true %}
    <article class="post">
      <a href="{{ site.baseurl }}{{ post.url }}">
        <h1>{{ post.title }}</h1>
        <div>
          <img src="/{{ site.baseurl }}images{{ post.url }}preview.png" alt="{{ post.title }}" width="50%" height="50%"/>
        </div>
      </a>
      <div class="entry">
        {{ post.excerpt }}
      </div>

      <a href="{{ site.baseurl }}{{ post.url }}" class="read-more">Read More</a>
    </article>
    
  {% endif %}
  {% endfor %}

</div>
