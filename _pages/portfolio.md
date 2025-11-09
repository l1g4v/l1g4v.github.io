---
layout: page
title: Portfolio
permalink: /portfolio/
---

<h1>Portfolio</h1>
<p>Welcome to my portfolio! Here you'll find a collection of my projects, skills, and experiences.</p>


<div class="portfolio-posts">
  {% for portfolio in paginator.portfolios %}
    <article class="portfolio">
      <a href="{{ site.baseurl }}{{ portfolio.url }}">
        <h1>{{ portfolio.title }}</h1>

        <div>
          <p class="portfolio_date">{{ portfolio.date | date: "%B %e, %Y" }}</p>
        </div>
      </a>
      <div class="entry">
        {{ portfolio.excerpt }}
      </div>

      <a href="{{ site.baseurl }}{{ portfolio.url }}" class="read-more">Read More</a>
    </article>
  {% endfor %}

  <!-- pagination -->
  {% if paginator.total_pages > 1 %}
  <div class="pagination">
    {% if paginator.previous_page %}
      <a href="{{ paginator.previous_page_path | prepend: site.baseurl | replace: '//', '/' }}">&laquo; Prev</a>
    {% else %}
      <span>&laquo; Prev</span>
    {% endif %}

    {% for page in (1..paginator.total_pages) %}
      {% if page == paginator.page %}
        <span class="webjeda">{{ page }}</span>
      {% elsif page == 1 %}
        <a href="{{ '/' | prepend: site.baseurl | replace: '//', '/' }}">{{ page }}</a>
      {% else %}
        <a href="{{ site.paginate_path | prepend: site.baseurl | replace: '//', '/' | replace: ':num', page }}">{{ page }}</a>
      {% endif %}
    {% endfor %}

    {% if paginator.next_page %}
      <a href="{{ paginator.next_page_path | prepend: site.baseurl | replace: '//', '/' }}">Next &raquo;</a>
    {% else %}
      <span>Next &raquo;</span>
    {% endif %}
  </div>
  {% endif %}
</div>
