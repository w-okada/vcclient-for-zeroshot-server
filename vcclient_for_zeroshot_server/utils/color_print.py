

from typing import Literal, TypeAlias


Color: TypeAlias = Literal["BLACK","RED", "GREEN", "YELLOW", "BLUE", "PURPLE", "MAGENTA", "CYAN",  "WHITE",
                           "BRIGHT_BLACK", "BRIGHT_RED", "BRIGHT_GREEN", "BRIGHT_YELLOW", "BRIGHT_BLUE", "BRIGHT_MAGENTA", "BRIGHT_CYAN", "BRIGHT_WHITE",
                           "BG_BLACK", "BG_RED", "BG_GREEN", "BG_YELLOW", "BG_BLUE", "BG_MAGENTA", "BG_CYAN", "BG_WHITE",
                           "BG_BRIGHT_BLACK", "BG_BRIGHT_RED", "BG_BRIGHT_GREEN", "BG_BRIGHT_YELLOW", "BG_BRIGHT_BLUE", "BG_BRIGHT_MAGENTA", "BG_BRIGHT_CYAN", "BG_BRIGHT_WHITE",
                           "DEFAULT", 
                           ]


Colors: dict[Color,int]={
    "BLACK": 30,
    "RED": 31,
    "GREEN": 32,
    "YELLOW": 33,
    "BLUE": 34,
    "PURPLE": 35,
    "MAGENTA": 35,
    "CYAN": 36,
    "WHITE": 37,
    "BRIGHT_BLACK": 90,
    "BRIGHT_RED": 91,
    "BRIGHT_GREEN": 92,
    "BRIGHT_YELLOW": 93,
    "BRIGHT_BLUE": 94,
    "BRIGHT_MAGENTA": 95,
    "BRIGHT_CYAN": 96,
    "BRIGHT_WHITE": 97,
    "BG_BLACK": 40,
    "BG_RED": 41,
    "BG_GREEN": 42,
    "BG_YELLOW": 43,
    "BG_BLUE": 44,
    "BG_MAGENTA": 45,
    "BG_CYAN": 46,
    "BG_WHITE": 47,
    "BG_BRIGHT_BLACK": 100,
    "BG_BRIGHT_RED": 101,
    "BG_BRIGHT_GREEN": 102,
    "BG_BRIGHT_YELLOW": 103,
    "BG_BRIGHT_BLUE": 104,
    "BG_BRIGHT_MAGENTA": 105,
    "BG_BRIGHT_CYAN": 106,
    "BG_BRIGHT_WHITE": 107,
    "DEFAULT": -1,
}

Format = Literal["NORMAL","BOLD", "FAINT", "ITALIC", "UNDERLINE", "BLINK", "FAST_BLINK", "REVERSE", "CONCEAL", "STRIKE"]
Formats: dict[Format,int]={
    "NORMAL": 0,
    "BOLD": 1,
    "FAINT": 2,
    "ITALIC": 3,
    "UNDERLINE": 4,
    "BLINK": 5,
    "FAST_BLINK": 6,
    "REVERSE": 7,
    "CONCEAL": 8,
    "STRIKE": 9,
} 

def color_text(text:str,color:Color="DEFAULT", format:Format="NORMAL")->str:
    color_code = Colors[color]
    format_code = Formats[format]
    if color_code == -1:
        start = "\033[" + str(format_code) + "m"
        reset = "\033[0m"
    else:
        start = "\033[" + str(format_code) + ";" + str(color_code) + "m"
        reset = "\033[0m"

    return start + text + reset
