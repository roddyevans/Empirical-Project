A python model that uses publicly available data of strokes gained statisitcs to predict major golf winners.

For both the 2025 and 2026 seasons,
1. Uses strokes gained data for the top 20 players of that season.
2. Applies course relevant strokes gained weightings for each major.
3. Calculates probabilties depeding on course setup and previous successors.
4. Outputs a table of favourties to win.

Manually import the data from the public dataset, therefore do not have to use the whole data which we are not looking at and would skew the data.

Each major is assigned weights accoring to their key course factors.

The win probability is calculated by the product of the course weight and the specific strokes gained category (weight X SG category).

Use the exact same code for each year, but with the data of the year required.

Run both sets of code independently and both will produce their own tables in decreasing order.

The regression model after each set of code looks at the importance of each strokes gained statistic in order to win a major.

https://hackmd.io/@PbXgXf4uT8ioGHapnw3IWg/BJWN8WG0bg   -   URL to blog.

https://datagolf.com/stats/tour-lists   -   URL to data used.
