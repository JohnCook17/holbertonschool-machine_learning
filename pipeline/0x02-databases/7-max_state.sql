-- max temp by state name

SELECT state, MAX(value) FROM temperatures GROUP BY state ORDER BY state;