import engine
import os


def get_music():
    path = engine.get_random_music()
    new_path = "".join(path.split(".mid")[:-1]) + ".mp3"
    to_run = "timidity " + path + " -Ow -o - | lame - -b 64 " + new_path
    os.system(to_run)
    return new_path
