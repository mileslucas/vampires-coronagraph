import os

def rootdir(*args):
    return os.path.join("/", "Users", "miles", "dev", "research", "vampires-coronagraph", *args)

def datadir(*args):
    return rootdir("data", *args)