# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/lightkurve/lkdata/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                             |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------- | -------: | -------: | ------: | --------: |
| src/lkdata/\_\_init\_\_.py       |       12 |        0 |    100% |           |
| src/lkdata/datacube.py           |      352 |      100 |     72% |180, 187-189, 194-206, 210, 224, 236, 241-243, 246, 305-314, 345, 353, 361, 366, 368, 373-376, 380-388, 400, 413, 417-437, 450, 458-463, 480, 493, 513, 529, 534, 623-636, 640, 643-656, 660, 681, 685, 688, 739-766, 1077, 1087 |
| src/lkdata/dataseries.py         |      123 |       14 |     89% |53, 109-111, 118, 168-172, 221-225, 250, 253 |
| src/lkdata/dataset.py            |      507 |      115 |     77% |77, 82, 87, 112, 127, 136-162, 169-196, 238-242, 303, 307, 314-321, 332, 334, 358, 374-375, 379-383, 386-387, 410, 578, 586, 595-615, 652, 663, 677, 686, 706, 722, 736, 738, 752, 754, 756, 838, 951-957, 1010, 1017, 1021, 1031, 1054-1072 |
| src/lkdata/mixins.py             |      681 |      134 |     80% |101, 138-166, 198, 200, 294, 312, 321-329, 421, 426, 429-436, 452, 455-462, 639, 657, 689-690, 695-697, 706-707, 779-780, 793, 795-798, 870, 883, 887-893, 901-906, 960-962, 988, 1003, 1046-1048, 1115-1133, 1152-1158, 1161, 1164, 1211-1212, 1243, 1297, 1301-1312, 1329-1338, 1377, 1479-1481, 1520, 1641, 1657-1659, 1674-1677, 1697, 1729, 1745-1789, 1818, 1907 |
| src/lkdata/seriescollection.py   |      134 |       10 |     93% |115, 217, 221, 285, 320, 323-328 |
| src/lkdata/utils/\_\_init\_\_.py |        1 |        0 |    100% |           |
| src/lkdata/utils/bitset.py       |      141 |       16 |     89% |89-91, 110, 135, 147-150, 154-157, 161-164, 182, 206 |
| src/lkdata/utils/exceptions.py   |        7 |        0 |    100% |           |
| src/lkdata/utils/uncertainty.py  |       96 |       10 |     90% |61, 415-416, 479, 485, 493-494, 548, 556, 587 |
| src/lkdata/version.py            |        1 |        0 |    100% |           |
| **TOTAL**                        | **2055** |  **399** | **81%** |           |


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