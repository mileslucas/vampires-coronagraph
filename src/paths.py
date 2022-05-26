import os


def rootdir(*args):
    return os.path.join(
        "/", "Users", "mileslucas", "dev", "research", "vampires-coronagraph", *args
    )


def datadir(*args):
    return rootdir("data", *args)


def figuredir(*args):
    return rootdir("figures", *args)


# make sure all paths exist
for path in [datadir(), figuredir()]:
    os.makedirs(path, exist_ok=True)
