# example.py ---
# 
# Filename: example.py
# Description: 
# 
# Original author:    Yu Lu
# Email:     yulu@utexas.edu
# Github:    https://github.com/SuperYuLu 
# 
# Created: Thu Nov 15 00:03:18 2018 (-0600)
# Version: 
# Last-Updated: Thu Nov 15 16:32:53 2018 (-0600)
#           By: yulu
#     Update #: 28

# Modified by Yang Li
# Email:     22210200008@m.fudan.edu.cn


import sys
import os

pkgPath = os.path.dirname(os.path.realpath(__file__ + '/../'))


try:
    from opRb_87 import Simulator
except ModuleNotFoundError:
    sys.path.insert(0, pkgPath)
    from opRb_87  import Simulator


if __name__ == '__main__':
    """
    test single run 
    currently this script can only be run 
    under /example folder.
    """
    s = Simulator(config = os.path.join(pkgPath, 'examples/config.in'))
    print(s)
    s.run()


