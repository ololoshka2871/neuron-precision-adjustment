#!/usr/bin/env python

import sys
import pstats


# открываем файл с результатами профилирования
stats = pstats.Stats(sys.argv[1])

count = len(sys.argv) > 2 and int(sys.argv[2]) or 10

# выводим топ функций по времени выполнения
stats.sort_stats('tottime').print_stats(count)