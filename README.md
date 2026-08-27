# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/lightkurve/lkdata/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                             |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------- | -------: | -------: | ------: | --------: |
| src/lkdata/\_\_init\_\_.py       |       12 |        0 |    100% |           |
| src/lkdata/datacube.py           |      305 |       72 |     76% |197, 204, 209-211, 222-229, 239, 255, 267-269, 272, 316, 329, 333-353, 366, 374-379, 396, 409, 429, 445, 450, 539-552, 556, 559-572, 576, 597, 601, 604, 655-682, 993, 1003 |
| src/lkdata/dataseries.py         |      123 |       14 |     89% |53, 109-111, 118, 168-172, 221-225, 250, 253 |
| src/lkdata/dataset.py            |      507 |      115 |     77% |77, 82, 87, 112, 127, 136-162, 169-196, 238-242, 303, 307, 314-321, 332, 334, 358, 374-375, 379-383, 386-387, 410, 578, 586, 595-615, 652, 663, 677, 686, 706, 722, 736, 738, 752, 754, 756, 838, 951-957, 1010, 1017, 1021, 1031, 1054-1072 |
| src/lkdata/mixins.py             |      680 |      134 |     80% |101, 138-166, 198, 200, 294, 312, 321-329, 421, 426, 429-436, 452, 455-462, 636, 654, 686-687, 692-694, 703-704, 776-777, 790, 792-795, 867, 880, 884-890, 898-903, 957-959, 985, 1000, 1043-1045, 1112-1130, 1149-1155, 1158, 1161, 1208-1209, 1240, 1294, 1298-1309, 1326-1335, 1374, 1476-1478, 1517, 1638, 1654-1656, 1671-1674, 1694, 1726, 1742-1786, 1815, 1904 |
| src/lkdata/seriescollection.py   |      134 |       10 |     93% |115, 217, 221, 285, 320, 323-328 |
| src/lkdata/utils/\_\_init\_\_.py |        1 |        0 |    100% |           |
| src/lkdata/utils/bitset.py       |      141 |       16 |     89% |89-91, 110, 135, 147-150, 154-157, 161-164, 182, 206 |
| src/lkdata/utils/exceptions.py   |        7 |        0 |    100% |           |
| src/lkdata/utils/uncertainty.py  |       96 |       10 |     90% |61, 415-416, 479, 485, 493-494, 548, 556, 587 |
| src/lkdata/version.py            |        1 |        0 |    100% |           |
| **TOTAL**                        | **2007** |  **371** | **82%** |           |


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