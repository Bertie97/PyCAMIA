
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

import time, math, builtins
from .inout import SPrint
from .strop import tokenize, format as str_format
from .listop import median
from .exception import avouch, touch, Warn
from .decorators import alias
from datetime import datetime, timedelta
from threading import Thread
from typing import Iterable

with __info__:
    use_nput = False
    try:
        from pynput import keyboard as kb
        use_nput = True
    except ImportError: ...

class Collector:
    def __init__(self, max_storage=None):
        self._collection = {}
        self._format = {}
        self._indices = {} # min, max, count, sum, sqsum
        assert max_storage is None or isinstance(max_storage, int)
        self._max_storage = max_storage
    def register(self, name, value, format='.4f/-.4f'):
        """format: mean format / std format. '-' for omission of '0'."""
        if len(name.split(':')) == 2:
            name, format = name.split(':')
        name = name.strip()
        format = format.strip()
        if any(format.endswith(t) for t in 'df'):
            type = "number"
            assert touch(lambda: '%d'%value) or touch(lambda: '%f'%value), "Python number required for number format in class Collector. "
        elif format == 'c':
            type = "category"
            assert touch(lambda: hash(value)), "Hashable objects required for category format in class Collector. "
        if format != '':
            if '/' not in format: format = format + '/' + format
            avouch(len(format.split('/')) == 2, f"Invalid format: {format}, should be like '.4f' or '2d/.4f'.")
            format = format.split('/')
        if name not in self._collection:
            self._collection[name] = []
            avouch(name not in self._format, "Collector having format for values never collected, Please contact the developer for more information. (Error Code: C388)")
            self._format[name] = format
            if type == 'number':
                self._indices[name] = [float('inf'), -float('inf'), 0., 0., 0.]
            elif type == 'category':
                self._indices[name] = {}
        else:
            if name not in self._format or self._format[name] == '': self._format[name] = format
            else: avouch(self._format[name] == format or format == '', "Cannot assign different format for the same collection in class 'Collector'. ")
            assert name in self._indices
        value = getattr(value, 'item', lambda: value)()
        if format != '':
            if type == 'number':
                prev_min, prev_max, prev_count, prev_sum, prev_sqsum = self._indices[name]
                self._indices[name] = [min(prev_min, value), max(prev_max, value), prev_count + 1, prev_sum + value, prev_sqsum + value * value]
            elif type == 'category':
                self._indices[name][value] = self._indices[name].get(value, 0) + 1
        self._collection[name].append(value)
        if self._max_storage is not None and n_count >= self._max_storage: self._collection[name].pop(0)
    def __contains__(self, name):
        return name in self._collection
    def __getitem__(self, name):
        if name in self._collection: return self._collection[name]
        raise NameError(f"No element registered as {name}. ")
    def __str__(self):
        sprint = SPrint()
        sprint(f"{self.__class__.__module__}.{self.__class__.__name__}<0x{id(self):x}> (")
        for name in sorted(list(self._collection.keys())):
            l = len(self[name])
            sprint(f"\t{name} ({l} item{'s' if l > 1 else ''}): [{self[name][0]}, ..., {self[name][-1]}]")
        sprint(')')
        return sprint.text
    def wilcoxon(self, other, name):
        from scipy.stats import wilcoxon
        if isinstance(other, Collector): other = other.as_numpy(name)
        return wilcoxon(self.as_numpy(name), other)[1]
    @alias("as_array")
    def as_numpy(self, name):
        import numpy as np
        return np.array(self[name])
    def as_mean_std_tuple(self, name, ignore_if_not_collected=False):
        if ignore_if_not_collected and name not in self._collection: return ('', '')
        _, _, _count, _sum, _sqsum = self._indices[name]
        format = self._format.get(name, '')
        if format == '': format = ['', '']
        ret = []
        for f, v in zip(format, (_sum / _count, math.sqrt((_sqsum - _sum * _sum / _count) / _count))):
            ret.append(str_format(f"{{var:{f}}}", var=v))
        return tuple(ret)
    def as_min_max_tuple(self, name, ignore_if_not_collected=False):
        if ignore_if_not_collected and name not in self._collection: return ('', '')
        _min, _max, _, _, _ = self._indices[name]
        format = self._format.get(name, '')
        if format != '': format = format[0]
        return str_format(f"{{var:{format}}}", var=_min), str_format(f"{{var:{format}}}", var=_max)
    def as_pm(self, name, ignore_if_not_collected=False):
        if ignore_if_not_collected and name not in self._collection: return ''
        m, s = self.as_mean_std_tuple(name)
        return m + '±' + s
    def as_range(self, name, ignore_if_not_collected=False):
        if ignore_if_not_collected and name not in self._collection: return ''
        mn, mx = self.as_min_max_tuple(name)
        if mn == mx: return mn
        return mn + '~' + mx
    def as_categories(self, name, ignore_if_not_collected=False):
        if ignore_if_not_collected and name not in self._collection: return ''
        dic = self._indices[name]
        return '{' + ', '.join(f"{k}({c})" for k, c in dic.items()) + '}'
    def as_latex(self, *names, ignore_if_not_collected=False):
        output_cells = []
        if len(names) == 0: names = self._collection.keys()
        for name in names:
            if name not in self._collection:
                if ignore_if_not_collected: continue
                output_cells.append('-'); continue
            m, s = self.as_mean_std_tuple(name)
            output_cells.append(m + r'$\pm$' + s)
        return ' & '.join(output_cells)

class HotKeyListener:
    keyboard = "asdfhgzxcv bqweryt123465=97-80]ou[ip<enter>lj'k;\\,/nm.<tab> `<backspace> <esc><cmd-r><cmd><shift><caps-lock><alt><ctrl><shift-r> <ctrl-r>  . * + \x1B   /\x03 -   01234567 89   <f5><f6><f7><f3><f8><f9> <f11> <f13> <f14> <f10>\x10<f12> <f15>\x05<home><page-up><delete><f4><end><f2><page-down><f1><left><right><down><up>"
    enumKeys = {k: v for k, v in enumerate(tokenize(keyboard, sep=[''], by='<>')[1:])}
    def __init__(self, *hotkeys, callback=lambda x:None):
        avouch(use_nput, "HotKeyListener cannot be used without package 'pynput'.")
        self.hotkeys = []
        self.str_reprs = []
        for ht in hotkeys: self.add_listen(ht)
        self.key_record = set()
        self.started = False
        self.detected = None
        self.callback = callback
        self.listener = None
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
            if self.key_record in self.hotkeys:
                self.detected = self.str_reprs[self.hotkeys.index(self.key_record)]
                self.callback(self)
    def add_listen(self, hotkey):
        ht = set()
        for k in hotkey.split('+'):
            k = k.lower()
            if len(k) > 1: ht.add(getattr(kb.Key, k, None))
            else: ht.add(k)
        self.hotkeys.append(ht)
        self.str_reprs.append(hotkey)
    def is_listening(self, hotkey):
        ht = set()
        for k in hotkey.split('+'):
            k = k.lower()
            if len(k) > 1: ht.add(getattr(kb.Key, k, None))
            else: ht.add(k)
        return ht in self.hotkeys
    def start(self):
        self.started = True
        if self.listener is None:
            self.listener = kb.Listener(
                on_press = lambda x: self.on_keyboard(x, 'press'),
                on_release = lambda x: self.on_keyboard(x, 'release'))
        self.listener.start()
    def stop(self):
        self.started = False
        self.listener.stop()
        self.listener = None
        self.detected = None

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
    def __init__(self, *keywords, callback=lambda x:None):
        self.keywords = [x.lower() for x in keywords]
        self.detected = None
        self.callback = callback
        self.listener = None
        self.terminate = False
    def on_keyword(self):
        while True:
            global latest_line
            if latest_line.strip().lower() in self.keywords:
                self.detected = latest_line.strip().lower()
                self.callback(self)
                latest_line = ''
            if self.terminate: break
            time.sleep(1)
        self.started = False
        self.listener = None
    def add_listen(self, keyword): self.keywords.append(keyword)
    def is_listening(self, keyword): return keyword in self.keywords
    def start(self):
        global terminate
        if terminate:
            terminate = False
            Thread(target=on_input).start()
        self.started = True
        if self.listener is None:
            self.listener = Thread(target=self.on_keyword)
        self.listener.start()
    def stop(self):
        self.terminate = True
        global terminate
        terminate = True
        self.detected = None

def iterate(list_:(list, int), inner_loop=None, breakable=True, break_key=None):
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
        
    Note2: Make sure to use `print` at the end of each iteration for correct progress bar. 
        Feel free to use nested loops with an inner loop without output.
        
    Args:
        list_ (list, int): iterative object to iterate by a 'for' loop. Using 'range' object for integer.
        inner_loop (int, iterable, callable):
            (int): the number of iteration in inner loop;
            (iterable): an iterable to loop through; or
            (callable): a function to create previous alternatives. 
                the function should accept arguments i and x (as in `enumerate(list_)`; x = i if `list_` is an integer):
                    i is the index of outer loop and x is the item.
                the function should return either the number of iterations or the data list to loop.
            Defaults to None.
            Using inner_loop will result in a two-element tuple for each iteration.
        breakable (bool): whether the user can interrupt the loop by 'break_key' or not.
        break_key (NoneType, str): the break key. e.g. 'ctrl+b'.

    Examples::
        >>> for i in iterate(20):
        ...     for _ in range(10000): ...
        ...     print(f'iteration {i}')
        ...
        00%     R--:--:--( -  s/it) |[15:46:48] iteration 0
            [...omitted...]
        99%████▉R 0:00:01(0.32s/it) |[15:46:52] iteration 19
        >>> for i, j in iterate(20, 10000):
        ...     print(f'epoch {i}, iteration {j}')
        ...
    """
    if breakable:
        if use_nput:
            if break_key is None: break_key = 'ctrl+b'
            listener = HotKeyListener(break_key)
        else:
            if break_key is None: break_key = 'b'
            listener = InputKeywordListener(break_key)
            Warn("'iterate' cannot listen break keys without package 'pynput', ")
            print("builtin function 'input' will be used which demonds the user press enter after the default keyword 'b'. ")
            print("Note that if one is using darwin(mac) systems, 'pynput' may not be trusted, ")
            print("please add the IDLE in the accessibility (辅助功能) list of the system settings first. ")
    progress_chars = " ▏▎▍▌▋▊▉█"
    n_block_share = len(progress_chars) - 1
    progress_len = 10
    n_timespan_store = 20
    use_progress_bar = True
    if isinstance(list_, int):
        n_out = list_
        list_ = range(list_)
    else:
        if not hasattr(list_, '__len__'):
            use_progress_bar = False
            Warn(f"Function 'iterate' cannot predict progress with non-list iterative object {list_}, consider casting the argument first.")
        n_out = len(list_)
    if breakable: listener.start()
    iter_timespans = []
    inner_loop_timespans = []
    print(f"Loop starting at {datetime.now().strftime('%m/%d %H:%M:%S')}...")
    for i, x in enumerate(list_):
        iter_begin = datetime.now()
        if inner_loop is not None:
            if not isinstance(inner_loop, (int, Iterable)) and callable(inner_loop):
                inner_loop = inner_loop(i, x)
            use_inner_progress_bar = True
            if isinstance(inner_loop, int):
                n_in = inner_loop
                inner_loop = range(inner_loop)
            else:
                if not hasattr(inner_loop, '__len__'):
                    use_inner_progress_bar = False
                    Warn(f"Function 'iterate' cannot predict progress with non-list iterative object {inner_loop}, consider casting the argument first.")
                n_in = len(inner_loop)
        if inner_loop is None:
            if use_progress_bar:
                progress_pos = int(i * progress_len * n_block_share / n_out)
                progress_bar = '%02d%%'%(i * 100 // n_out)
                progress_bar += progress_chars[-1] * (progress_pos // n_block_share)
                progress_bar += progress_chars[progress_pos % n_block_share]
                progress_bar += ' ' * (progress_len - progress_pos // n_block_share - 1)
                if i > 0:
                    t_iter = median(iter_timespans)
                    remaining_time = int(t_iter * (n_out - i))
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
            iter_timespans = []
            for j, y in enumerate(inner_loop):
                if use_progress_bar or use_inner_progress_bar:
                    progress_pos_out = int(i * progress_len / n_out)
                    progress_pos_in = int(j * progress_len * n_block_share / n_in)
                    progress_bar = '%02d%%'%(((i * n_in + j) * 100) / n_in / n_out)
                    if progress_pos_out * n_block_share < progress_pos_in:
                        progress_bar += progress_chars[-1] * (progress_pos_out - 1)
                        progress_bar += progress_chars[-2] * (progress_pos_out > 0)
                        progress_bar += progress_chars[-1] * (progress_pos_in // n_block_share - progress_pos_out)
                        progress_bar += progress_chars[progress_pos_in % n_block_share]
                        progress_bar += progress_chars[0] * (progress_len - progress_pos_in // n_block_share - 1)
                    else:
                        progress_bar += progress_chars[-1] * (progress_pos_in // n_block_share)
                        if progress_pos_out * n_block_share == progress_pos_in:
                            progress_bar += progress_chars[1]
                        else:
                            progress_bar += progress_chars[progress_pos_in % n_block_share]
                            progress_bar += progress_chars[0] * (progress_pos_out - progress_pos_in // n_block_share - 1)
                            progress_bar += progress_chars[1]
                        progress_bar += progress_chars[0] * (progress_len - progress_pos_out - 1)
                    if i > 0 or j > 0:
                        if i > 0: t_outer = median(inner_loop_timespans)
                        if j > 0: t_iter = median(iter_timespans)
                        if i == 0: t_outer = t_iter * n_in
                        if j == 0: t_iter = t_outer / n_in
                        remaining_time = int(t_iter * (n_in - j) + t_outer * (n_out - i - 1))
                        secs = remaining_time % 60
                        mins = (remaining_time // 60) % 60
                        hours = remaining_time // 3600
                        print_time = (iter_begin + timedelta(seconds=t_iter)).strftime("%H:%M:%S")
                        progress_bar += f"R{hours:2d}:{mins:02d}:{secs:02d}({t_iter:.2f}s/it) |[{print_time}]"
                    else:
                        print_time = (iter_begin + timedelta(seconds=2)).strftime("%H:%M:%S")
                        progress_bar += f"R--:--:--( -  s/it) |[{print_time}]"
                    print(progress_bar, end=" ")
                    yield x, y
                    iter_timespans.append((datetime.now() - iter_begin).total_seconds())
                    if len(iter_timespans) > n_timespan_store: iter_timespans.pop(0)
                    if breakable and listener.detected: break
            else:
                inner_loop_timespans.append(n_in * median(iter_timespans))
                if len(inner_loop_timespans) > n_timespan_store: inner_loop_timespans.pop(0)
                continue
            break
    else:
        if breakable: listener.stop()
        return
    print("-- manual termination of the loop --")
    if breakable: listener.stop()
    
builtin_print = builtins.print

class TaskProgress:
    
    def __init__(self, progress_len=10, n_timespan_store=20, show_progress_bar=True, breakable=None, break_key=None):
        self.progress_chars = " ▏▎▍▌▋▊▉█"
        self.n_block_share = len(self.progress_chars) - 1
        self.progress_len = progress_len
        self.n_timespan_store = n_timespan_store
        self.show_progress_bar = show_progress_bar
        self.breakable = use_nput if breakable is None else breakable
        self.sizes = []
        self.progresses = []
        self.show_progresses = []
        self.iter_estimations = []

        if break_key is None: break_key = 'ctrl+b' if use_nput else 'b'
        self.break_key = break_key
        self.break_keys = []
        self.should_break = []
        if breakable:
            self.listener = (HotKeyListener if use_nput else InputKeywordListener)(break_key)
            if not use_nput:
                Warn("'TaskProgress' cannot listen break keys without package 'pynput', ", by_func=builtin_print)
                builtin_print("builtin function 'input' will be used which demonds the user press enter after the default keyword 'b'. ")
                builtin_print("Note that if one is using darwin(mac) systems, 'pynput' may not be trusted, ")
                builtin_print("please add the IDLE in the accessibility (辅助功能) list of the system settings first. ")

    def builtin_print(self, *values, sep=' ', end='\n', **kwargs):
        self.parent_print(self.prefix, *values, sep=sep, end=end, **kwargs)

    @property
    def prefix(self):
        total_progress = 0
        unit = 1
        for i, n in zip(self.progresses, self.sizes):
            if n < 0: break
            total_progress += i * 100 / n / unit
            unit *= n
        progress_bar = '%02d%%'%total_progress
        if total_progress >= 100: progress_bar = "1oo"
        if sum(self.show_progresses) >= 2 and sum(n > 1 for s, n in zip(self.show_progresses, self.sizes) if s) >= 2:
            progress_pos_out = int(total_progress * self.progress_len / 100)
        else: progress_pos_out = 0
        if sum(self.show_progresses) >= 1:
            last_act_loop = [i for i, x in enumerate(self.show_progresses) if x][-1]
            progress_pos_in = int(self.progresses[last_act_loop] * self.progress_len * self.n_block_share / self.sizes[last_act_loop])
        else: return "[not in loop]"
        if progress_pos_out * self.n_block_share < progress_pos_in:
            progress_bar += self.progress_chars[-1] * (progress_pos_out - 1)
            progress_bar += self.progress_chars[-2] * (progress_pos_out > 0)
            progress_bar += self.progress_chars[-1] * (progress_pos_in // self.n_block_share - progress_pos_out)
            if progress_pos_in < self.progress_len * self.n_block_share:
                progress_bar += self.progress_chars[progress_pos_in % self.n_block_share]
            progress_bar += self.progress_chars[0] * (self.progress_len - progress_pos_in // self.n_block_share - 1)
        else:
            progress_bar += self.progress_chars[-1] * (progress_pos_in // self.n_block_share)
            if progress_pos_out * self.n_block_share == progress_pos_in:
                if progress_pos_in < self.progress_len * self.n_block_share:
                    progress_bar += self.progress_chars[1]
            else:
                progress_bar += self.progress_chars[progress_pos_in % self.n_block_share]
                progress_bar += self.progress_chars[0] * (progress_pos_out - progress_pos_in // self.n_block_share - 1)
                progress_bar += self.progress_chars[1]
            progress_bar += self.progress_chars[0] * (self.progress_len - progress_pos_out - 1)
        first_not_act_loop = [i for i, x in enumerate(self.show_progresses + [None]) if not x][0]
        if sum(self.progresses[:first_not_act_loop]) == 0:
            print_time = datetime.now().strftime("%H:%M:%S")
            prefix = progress_bar + f"R--:--:--( -  s/it) |[{print_time}]"
        else:
            iter_estimations = []
            for i, t in enumerate(self.iter_estimations[first_not_act_loop-1::-1]):
                if t is not None: iter_estimations.append(t); continue
                if i == 0: iter_estimations.append(1); continue
                iter_estimations.append(iter_estimations[i - 1] * self.sizes[first_not_act_loop-i])
            iter_estimations = list(reversed(iter_estimations))
            t_iter = iter_estimations[-1]
            remaining_time = max(0, int(sum(t * (n - i - 1) for t, i, n, _ in zip(iter_estimations, self.progresses, self.sizes, range(first_not_act_loop))) + t_iter))
            secs = remaining_time % 60
            mins = (remaining_time // 60) % 60
            hours = remaining_time // 3600
            print_time = datetime.now().strftime("%H:%M:%S")
            prefix = progress_bar + f"R{hours:2d}:{mins:02d}:{secs:02d}({t_iter:.2f}s/it) |[{print_time}]"
        return prefix
    
    def __call__(self, arg:(list, int), n_timespan_store: int=None, show_progress_bar: bool=None, breakable: bool=None, break_key: str=None):
        if n_timespan_store is None: n_timespan_store = self.n_timespan_store
        if show_progress_bar is None: show_progress_bar = self.show_progress_bar
        if breakable is None: breakable = self.breakable
        if break_key is None: break_key = self.break_key
        i_loop = len(self.sizes)
        if isinstance(arg, int):
            self.sizes.append(arg)
            arg = range(arg)
        else:
            if not hasattr(arg, '__len__'):
                show_progress_bar = False
                self.sizes.append(-1)
                builtin_Warn(f"'TaskProcess' cannot predict progress with non-list iterative object {arg}, consider casting the argument first or use show_progress_bar=False.")
            else: self.sizes.append(len(arg))
        self.show_progresses.append(show_progress_bar)
        self.progresses.append(0.)
        self.break_keys.append(break_key if breakable else None)
        self.should_break.append(False)

        start_in_this_loop = False
        if breakable:
            if not self.listener.is_listening(break_key):
                self.listener.add_listen(break_key)
            if not self.listener.started:
                start_in_this_loop = True
                self.listener.start()

        succeeded = False
        iter_timespans = []
        if len(self.iter_estimations) < i_loop + 1: self.iter_estimations.append(None)
        print(f"Loop starting at {datetime.now().strftime('%m/%d %H:%M:%S')}...")
        for i, x in enumerate(arg):
            iter_begin = datetime.now()
            self.progresses[i_loop] = i
            yield x
            iter_timespans.append((datetime.now() - iter_begin).total_seconds())
            if len(iter_timespans) > n_timespan_store: iter_timespans.pop(0)
            self.iter_estimations[i_loop] = median(iter_timespans)
            if breakable and self.listener.detected is not None and self.listener.detected in self.break_keys:
                b_loop = self.break_keys.index(self.listener.detected)
                self.listener.detected = None
                self.should_break = self.should_break[:b_loop] + [True] * (len(self.break_keys) - b_loop)
            if self.should_break[i_loop]: self.should_break[i_loop] = False; break
        else: succeeded = True
        if not succeeded: print("-- manual termination of the loop --")
        if start_in_this_loop: self.listener.stop()
        avouch(len(self.sizes) == i_loop + 1)
        self.sizes.pop(-1)
        self.progresses.pop(-1)
        self.show_progresses.pop(-1)
        if i_loop > 0: self.progresses[i_loop - 1] += 1

    def __enter__(self):
        self.parent_print = builtins.print
        builtins.print = self.print

    def __exit__(self):
        builtins.print = self.parent_print
    
iterate = TaskProgress()
        