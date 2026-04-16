# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/lightkurve/lkdata/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                             |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------- | -------: | -------: | ------: | --------: |
| src/lkdata/\_\_init\_\_.py       |       12 |        0 |    100% |           |
| src/lkdata/datacube.py           |      296 |       72 |     76% |197, 204, 209-211, 222-229, 239, 255, 267-269, 272, 316, 327, 331-351, 360, 455-468, 472, 475-488, 492, 513, 517, 520, 529, 564, 572-577, 610-637, 673, 678, 682, 961, 971 |
| src/lkdata/dataseries.py         |      117 |       12 |     90% |51, 106-108, 115, 165-169, 210, 235, 238 |
| src/lkdata/dataset.py            |      509 |      115 |     77% |77, 82, 87, 112, 127, 136-162, 169-196, 238-242, 305, 309, 316-323, 333, 335, 359, 375-376, 380-384, 387-388, 411, 579, 587, 597-617, 654, 665, 679, 688, 708, 724, 738, 740, 754, 756, 758, 840, 950-956, 1009, 1016, 1020, 1030, 1053-1071 |
| src/lkdata/mixins.py             |      669 |      132 |     80% |101, 138-166, 198, 200, 294, 312, 321-329, 421, 426, 429-436, 452, 455-462, 636, 654, 685-686, 696-697, 769-770, 783, 785-788, 860, 873, 877-883, 891-896, 950-952, 978, 993, 1027-1029, 1096-1114, 1133-1139, 1142, 1145, 1192-1193, 1224, 1278, 1282-1293, 1310-1319, 1358, 1460-1462, 1483, 1501, 1622, 1638-1640, 1655-1658, 1678, 1710, 1726-1770, 1799, 1888 |
| src/lkdata/seriescollection.py   |      133 |       10 |     92% |115, 217, 221, 286, 321, 324-329 |
| src/lkdata/utils/\_\_init\_\_.py |        1 |        0 |    100% |           |
| src/lkdata/utils/bitset.py       |      141 |       16 |     89% |89-91, 110, 135, 147-150, 154-157, 161-164, 182, 206 |
| src/lkdata/utils/exceptions.py   |        7 |        0 |    100% |           |
| src/lkdata/utils/uncertainty.py  |       96 |       10 |     90% |61, 415-416, 479, 485, 493-494, 548, 556, 587 |
| src/lkdata/version.py            |        1 |        0 |    100% |           |
| **TOTAL**                        | **1982** |  **367** | **81%** |           |


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