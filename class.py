class Template(object):

    def __init__(self):
        self.value = "hello"

    def init(self):
        pass
    
    def info(self):
        print(self.value)

class Deep(Template):
    def init(self):
        self.value = "howdy"
        
def main():
    A = Deep()
    A.info()
      
if __name__ == '__main__':
    main()

