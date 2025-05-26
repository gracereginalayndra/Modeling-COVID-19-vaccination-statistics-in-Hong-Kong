# Modeling-COVID-19-vaccination-statistics-in-Hong-Kong
This project models COVID-19 vaccination statistics in Hong Kong using Pandas and Matplotlib. The goal is to recreate figures similar to those published on the official HKSAR coronavirus website (https://www.coronavirus.gov.hk/eng/index.html). The vaccination data is obtained directly from the website and processed to visualize trends and patterns in a format that mirrors the original government-provided charts.

# Interpreting the figures
Before diving into the implementation, it's important to understand how the figures on the official site are structured. The website displays daily counts of administered vaccine doses, broken down by dose number and age group. The percentage values shown in the charts are calculated by dividing the cumulative dose counts by the total population of the respective age groups. This means that each bar in the chart reflects the vaccination rate per population segment, providing insight into vaccine uptake across different age brackets.

# Data Sources:
1. Daily count of vaccination by age groups: https://data.gov.hk/en-data/dataset/hk-hhb-hhbcovid19-vaccination-rates-over-time-by-age
2. Table 110-01002 : Population by Sex and Age: https://www.censtatd.gov.hk/en/web_table.html?id=110-01002
