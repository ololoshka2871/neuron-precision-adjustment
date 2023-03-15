
from learn_v1 import learn_main

if __name__ == "__main__":
    import cProfile
    cProfile.run('learn_main({}, {})'.format(5, 2),
                 sort='cumtime', filename='profile.out')