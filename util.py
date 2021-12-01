import time

class SmartTimer():
    def __init__(self, verb = True) -> None:
        self.last = time.time()
        self.verb = verb

    def record(self):
        self.last = time.time()
    
    def cal_and_update(self, name):
        now = time.time()
        if self.verb:
            print(name,now-self.last)
        self.record()
