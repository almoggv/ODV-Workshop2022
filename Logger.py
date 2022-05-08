# Import packages
import os
import sys
import time

# Simple Logger
# Log Levels:
# 0 - Off
# 1 - Warnings + Errors

class Logger:
    def __init__(self, level):
        self.level = level
    
    def error(self, output):
        if (self.level == 1):
            print("Log(ERROR): " + output)

    def warn(self, output):
       if (self.level == 1):
            print("Log(WARRNING): " + output)
