# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/lightkurve/lkdata/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                             |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------- | -------: | -------: | ------: | --------: |
| src/lkdata/\_\_init\_\_.py       |       12 |        0 |    100% |           |
| src/lkdata/datacube.py           |      352 |       21 |     94% |196, 206, 210, 236, 241-243, 353, 376, 380-386, 418, 463, 529, 534, 688, 745 |
| src/lkdata/dataseries.py         |      123 |        5 |     96% |53, 118, 223, 250, 253 |
| src/lkdata/dataset.py            |      507 |       34 |     93% |77, 82, 87, 127, 136-162, 173-174, 241-242, 303, 332, 334, 578, 663, 677, 722, 736, 738, 838, 952-955 |
| src/lkdata/mixins.py             |      680 |       67 |     90% |101, 152-158, 294, 321-329, 421, 426, 429-436, 452, 455-462, 639, 695-697, 779-780, 793, 795-798, 870, 901-906, 988, 1310-1311, 1328-1337, 1376, 1640, 1656-1658, 1673-1676, 1696, 1728, 1744-1788, 1817, 1906 |
| src/lkdata/seriescollection.py   |      134 |        9 |     93% |115, 217, 285, 320, 323-328 |
| src/lkdata/utils/\_\_init\_\_.py |        1 |        0 |    100% |           |
| src/lkdata/utils/bitset.py       |      141 |        3 |     98% |148, 155, 162 |
| src/lkdata/utils/exceptions.py   |        7 |        0 |    100% |           |
| src/lkdata/utils/uncertainty.py  |       96 |        2 |     98% |  556, 587 |
| src/lkdata/version.py            |        1 |        0 |    100% |           |
| **TOTAL**                        | **2054** |  **141** | **93%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/lightkurve/lkdata/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/lightkurve/lkdata/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lightkurve/lkdata/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/lightkurve/lkdata/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Flightkurve%2Flkdata%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/lightkurve/lkdata/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.