# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/lightkurve/lkdata/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                             |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------- | -------: | -------: | ------: | --------: |
| src/lkdata/\_\_init\_\_.py       |       12 |        0 |    100% |           |
| src/lkdata/datacube.py           |      296 |       71 |     76% |197, 204, 209-211, 222-229, 239, 255, 267-269, 272, 316, 327, 331-351, 360, 455-468, 472, 475-488, 492, 517, 520, 529, 564, 572-577, 610-637, 673, 678, 682, 961, 971 |
| src/lkdata/dataseries.py         |      117 |       12 |     90% |51, 106-108, 115, 165-169, 210, 235, 238 |
| src/lkdata/dataset.py            |      507 |      115 |     77% |77, 82, 87, 112, 127, 136-162, 169-196, 238-242, 304, 308, 315-322, 332, 334, 358, 374-375, 379-383, 386-387, 410, 578, 586, 596-616, 653, 664, 678, 687, 707, 723, 737, 739, 753, 755, 757, 839, 949-955, 1008, 1015, 1019, 1029, 1052-1070 |
| src/lkdata/mixins.py             |      661 |      122 |     82% |101, 130-136, 143-151, 191, 193, 287, 305, 314-322, 414, 419, 422-429, 445, 448-455, 629, 647, 678-679, 689-690, 762-763, 776, 778-781, 853, 866, 870-876, 884-889, 943-945, 971, 986, 1033-1035, 1102-1120, 1139-1145, 1148, 1151, 1198-1199, 1230, 1284, 1288-1299, 1316-1325, 1364, 1456-1457, 1476, 1494, 1609, 1625-1627, 1642-1645, 1665, 1697, 1713-1757, 1786, 1875 |
| src/lkdata/seriescollection.py   |      133 |       10 |     92% |115, 217, 221, 286, 321, 324-329 |
| src/lkdata/utils/\_\_init\_\_.py |        1 |        0 |    100% |           |
| src/lkdata/utils/bitset.py       |      141 |       16 |     89% |89-91, 110, 135, 147-150, 154-157, 161-164, 182, 206 |
| src/lkdata/utils/exceptions.py   |        7 |        0 |    100% |           |
| src/lkdata/utils/uncertainty.py  |       96 |       10 |     90% |61, 415-416, 479, 485, 493-494, 548, 556, 587 |
| src/lkdata/version.py            |        1 |        0 |    100% |           |
|                        **TOTAL** | **1972** |  **356** | **82%** |           |


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