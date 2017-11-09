import sys,time
class ProgressBar:
    def __init__(self, count = 0, total = 0, width = 50):
        self.count = count
        self.total = total
        self.width = width
    def move(self):
        self.count += 1
    def log(self, s):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        #print s
        progress =0
        progress = self.width * self.count / self.total
        percent = self.count/float(self.total) *100
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('['+'>' * progress + '-' * (self.width - progress)+'] '+'{:>5,.2f}'.format(percent)+'%' + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()

