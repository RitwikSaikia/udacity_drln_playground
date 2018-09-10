from collections import namedtuple

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])