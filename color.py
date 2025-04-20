class Style:
    bold = "\033[1m"
    italic = "\033[3m"
    underline = "\033[4m"
    strikethrough = "\033[9m"
    reset = "\033[0m"


class Back:
    red = "\033[101m"
    green = "\033[102m"
    yellow = "\033[103m"
    blue = "\033[104m"
    magenta = "\033[105m"
    cyan = "\033[106m"
    white = "\033[107m"
    reset = "\033[0m"


class Fore:
    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    cyan = "\033[96m"
    white = "\033[97m"
    reset = "\033[0m"


Fore = Fore()
Back = Back()
Style = Style()
