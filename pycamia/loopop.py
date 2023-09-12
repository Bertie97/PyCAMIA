
from .manager import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "<main>",
    author = "Yuncheng Zhou",
    create = "2023-08",
    fileinfo = "File of loop operations. "
)

__all__ = """
    Collector
    iterate
""".split()

import time
from .inout import StrIO
from .strop import tokenize
from .listop import median
from .exception import avouch, touch
from .decorators import alias
from datetime import datetime, timedelta
from threading import Thread

with __info__:
    use_nput = False
    try:
        from pynput import keyboard as kb
        use_nput = True
    except ImportError: ...

class Collector:
    def __init__(self):
        self._collection = {}
        self._format = {}
    def register(self, name, value, format='.4f/-.4f'):
        if len(name.split(':')) == 2:
            name, format = name.split(':')
        if format != '':
            if '/' not in format: format = format + '/' + format
            avouch(len(format.split('/')) == 2, f"Invalid format: {format}, should be like '.4f' or '2d/.4f'.")
            format = format.split('/')
        if name not in self._collection:
            self._collection[name] = []
            avouch(name not in self._format, "Collector having format for values never collected, Please contact the developer for more information. (Error Code: C388)")
            self._format[name] = format
        elif name not in self._format or self._format[name] == '':
            self._format[name] = format
        else:
            avouch(self._format[name] == format or format == '', "Cannot assign different format for the same collection in class 'Collector'. ")
        self._collection[name].append(value)
    def __contains__(self, name):
        return name in self._collection
    def __getitem__(self, name):
        if name in self._collection: return self._collection[name]
        raise NameError(f"No element registered as {name}. ")
    @alias("as_array")
    def as_numpy(self, name):
        import numpy as np
        return np.array(self[name])
    def as_str_tuple(self, name, ignore_if_not_collected=False):
        if ignore_if_not_collected and name not in self._collection: return ('', '')
        a = self.as_numpy(name)
        format = self._format.get(name, '')
        if format == '': format = ['', '']
        ret = []
        for f, v in zip(format, (a.mean(), a.std())):
            if f.startswith('-'):
                ret.append(f"{{var:{f[1:]}}}".format(var=v).lstrip('0'))
            else:
                ret.append(f"{{var:{f}}}".format(var=v))
        return tuple(ret)
    def as_pm(self, name, ignore_if_not_collected=False):
        if ignore_if_not_collected and name not in self._collection: return ''
        m, s = self.as_str_tuple(name)
        return m + '±' + s
    def as_latex(self, *names, ignore_if_not_collected=False):
        output_cells = []
        for name in names:
            if name not in self._collection:
                if ignore_if_not_collected: continue
                output_cells.append('-'); continue
            m, s = self.as_str_tuple(name)
            output_cells.append(m + '$\pm$' + s)
        return ' & '.join(output_cells)

class HotKeyListener:
    keyboard = "asdfhgzxcv bqweryt123465=97-80]ou[ip<enter>lj'k;\\,/nm.<tab> `<backspace> <esc><cmd-r><cmd><shift><caps-lock><alt><ctrl><shift-r> <ctrl-r>  . * + \x1B   /\x03 -   01234567 89   <f5><f6><f7><f3><f8><f9> <f11> <f13> <f14> <f10>\x10<f12> <f15>\x05<home><page-up><delete><f4><end><f2><page-down><f1><left><right><down><up>"
    enumKeys = {k: v for k, v in enumerate(tokenize(keyboard, sep=[''], by='<>')[1:])}
    def __init__(self, hotkey, callback=lambda:None):
        avouch(use_nput, "HotKeyListener cannot be used without package 'pynput'.")
        self.hotkey = set()
        for k in hotkey.split('+'):
            if len(k) > 1: self.hotkey.add(getattr(kb.Key, k, None))
            else: self.hotkey.add(k)
        self.key_record = set()
        self.detected = False
        self.callback = callback
        self.listener = kb.Listener(
            on_press = lambda x: self.on_keyboard(x, 'press'),
            on_release = lambda x: self.on_keyboard(x, 'release'))
    def on_keyboard(self, key, act):
        if isinstance(key, kb.KeyCode): key = key.char
        elif isinstance(key, kb._darwin.KeyCode): key = self.enumKeys[key.vk]
        elif isinstance(key, kb.Key):
            key_index = str(key.value).strip('<>')
            key_value_char = touch(lambda: self.enumKeys[int(key_index)], key_index)
            key = getattr(kb.Key, key.name, key_value_char)
        else: raise TypeError(f"Unrecognized type {type(key)} for a key {key}. ")
        if act == 'release':
            if key in self.key_record: self.key_record.remove(key)
        elif act == 'press':
            self.key_record.add(key)
            if self.key_record == self.hotkey:
                self.detected = True
                self.callback()
    def start(self): self.listener.start()
    def stop(self): self.listener.stop()

if not use_nput:
    latest_line = ''
    terminate = True
    def on_input():
        while True:
            l = input()
            if terminate: break
            if l == '': continue
            global latest_line
            latest_line = l

class InputKeywordListener:
    def __init__(self, keyword, callback=lambda:None):
        self.keyword = keyword.lower()
        self.detected = False
        self.callback = callback
        self.listener = Thread(target=self.on_keyword)
        self.terminate = False
    def on_keyword(self):
        while True:
            global latest_line
            if latest_line.strip().lower() == self.keyword:
                self.detected = True
                self.callback()
                latest_line = ''
            if self.terminate: break
            time.sleep(1)
    def start(self):
        global terminate
        if terminate:
            terminate = False
            Thread(target=on_input).start()
        self.listener.start()
    def stop(self):
        self.terminate = True
        global terminate
        terminate = True

def iterate(list_:(list, int), breakable=True, break_key=None):
    """
    iterate the input list_ with a visualized progress bar and a key break function 
        that can terminate the loop whenever a break key is entered.
        If we have package 'pynput' installed, we listen to the keyboard for hot keys. The break key is 'cmd+b' by default. 
        (If Darwin(mac) systems notify with security problem, please add your IDLE in the accessibility (辅助功能) list in system settings.)
        If we donot have access to the package, one have to enter the break_key string and press enter to send it, the break key is 'b' by default. 
        This might cause problem in formatting when a new line of output is printed during entering the break key, 
        hence it is recommended to use short break_keys and reduce the number of iterations for the loop using 'iterate'.
        P.S. One may find the program hard to exit in the end due to the 'input' function, feel free to press enter or close it the hard way. 
    Note: The input should be either an integer for a range or a list or tuple with fixed length. Generators should
        be cast into lists first. This ristriction is applied to avoid endless loading here when there are to many
        elements to be generated.
        
    Arguments:
    ----------
    list_ [list, int]: iterative object to iterate by a 'for' loop. Using 'range' object for integer.
    breakable [bool]: whether the user can interrupt the loop by 'break_key' or not.
    break_key [NoneType, str]: the break key. e.g. 'ctrl+b'.

    Example:
    ----------
    >>> for i in iterate(20):
    ...     for _ in range(10000): ...
    ...     print(f'iteration {i}')
    ...
    
    """
    if breakable:
        if use_nput:
            if break_key is None: break_key = 'ctrl+b'
            listener = HotKeyListener(break_key)
        else:
            if break_key is None: break_key = 'b'
            listener = InputKeywordListener(break_key)
            print("Warning: 'iterate' cannot listen break keys without package 'pynput', ")
            print("builtin function 'input' will be used which demonds the user press enter after the default keyword 'b'. ")
            print("Note that if one is using darwin(mac) systems, 'pynput' may not be trusted, ")
            print("please add the IDLE in the accessibility (辅助功能) list of the system settings first. ")
    progress_chars = " ▏▎▍▌▋▊▉█"
    progress_len = 5
    if isinstance(list_, int): list_ = list(range(list_))
    use_progress_bar = True
    if not isinstance(list_, list):
        use_progress_bar = False
        print("Warning: Function 'iterate' cannot predict progress with non-list iterative objects, consider casting the argument first.")
    if breakable: listener.start()
    n_ele = len(list_)
    n_timespan_store = 20
    iter_timespans = []
    print(f"Loop starting at {datetime.now().strftime('%m/%d %H:%M:%S')}...")
    for i, x in enumerate(list_):
        iter_begin = datetime.now()
        if use_progress_bar:
            progress_pos = int(i * 40 / n_ele)
            progress_bar = '%02d%%'%(i * 100 // n_ele)
            progress_bar += progress_chars[-1] * (progress_pos // 8)
            progress_bar += progress_chars[progress_pos % 8]
            progress_bar += ' ' * (5 - progress_pos // 8 - 1)
            if i > 0:
                t_iter = median(iter_timespans)
                remaining_time = int(t_iter * (n_ele - i))
                secs = remaining_time % 60
                mins = (remaining_time // 60) % 60
                hours = remaining_time // 3600
                print_time = (iter_begin + timedelta(seconds=t_iter)).strftime("%H:%M:%S")
                progress_bar += f"R{hours:2d}:{mins:02d}:{secs:02d}({t_iter:.2f}s/it) |[{print_time}]"
            else:
                print_time = (iter_begin + timedelta(seconds=2)).strftime("%H:%M:%S")
                progress_bar += f"R--:--:--( -  s/it) |[{print_time}]"
            print(progress_bar, end=" ")
        yield x
        iter_timespans.append((datetime.now() - iter_begin).total_seconds())
        if len(iter_timespans) > n_timespan_store: iter_timespans.pop(0)
        if breakable and listener.detected: break
    else:
        if breakable: listener.stop()
        return
    print("-- manual termination of the loop --")
    if breakable: listener.stop()
