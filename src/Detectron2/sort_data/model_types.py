from enum import Enum, auto

# using enums to test for cases across models 
class Type(Enum):
    RGB = auto() 
    DEPTH = auto()
    RGD = auto()
