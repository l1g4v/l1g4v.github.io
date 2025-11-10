---
layout: default
title: Portfolio
---

<p>Welcome to my portfolio! Here you'll find a collection of my projects, skills, and experiences.</p>


<div class="posts">
  {% for post in site.posts %}
  {% if post.is_work == true %}
    <article class="post">
      <a href="{{ site.baseurl }}{{ post.url }}">
        <h1>{{ post.title }}</h1>
        <div>
          <p class="post_date">{{ post.date | date: "%B %e, %Y" }}</p>
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
