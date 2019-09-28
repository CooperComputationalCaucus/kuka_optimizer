from __future__ import print_function
import os
import json

from .observer import _Tracker
from .event import Events
from .util import Colours


def _get_default_logger(verbose):
    return ScreenLogger(verbose=verbose)


def _get_discrete_logger(verbose):
    return DiscreteLogger(verbose=verbose)


class ScreenLogger(_Tracker):
    _default_cell_size = 12
    _default_precision = 4

    def __init__(self, verbose=2):
        self._verbose = verbose
        self._header_length = None
        super(ScreenLogger, self).__init__()

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, v):
        self._verbose = v

    def _format_number(self, x):
        if isinstance(x, int):
            s = "{x:< {s}d}".format(
                x=x,
                s=self._default_cell_size,
            )
        else:
            s = "{x:^ {s}.{p}f}".format(
                x=x,
                s=self._default_cell_size,
                p=self._default_precision,
            )

        if len(s) > self._default_cell_size:
            if "." in s:
                return s[:self._default_cell_size]
            else:
                return s[:self._default_cell_size - 3] + "..."
        return s

    def _format_key(self, key):
        s = "{key:^{s}}".format(
            key=key,
            s=self._default_cell_size
        )
        if len(s) > self._default_cell_size:
            return s[:self._default_cell_size - 3] + "..."
        return s

    def _step(self, instance, colour=Colours.black):
        res = instance.res[-1]
        cells = []

        cells.append(self._format_number(self._iterations + 1))
        cells.append(self._format_number(res["target"]))

        for key in instance.space.keys:
            cells.append(self._format_number(res["params"][key]))

        return "| " + " | ".join(map(colour, cells)) + " |"

    def _header(self, instance):
        cells = []
        cells.append(self._format_key("iter"))
        cells.append(self._format_key("target"))
        for key in instance.space.keys:
            cells.append(self._format_key(key))

        line = "| " + " | ".join(cells) + " |"
        self._header_length = len(line)
        return line + "\n" + ("-" * self._header_length)

    def _is_new_max(self, instance):
        if self._previous_max is None:
            self._previous_max = instance.max["target"]
        return instance.max["target"] > self._previous_max

    def update(self, event, instance):
        if event == Events.OPTMIZATION_START:
            line = self._header(instance) + "\n"
        elif event == Events.OPTMIZATION_STEP:
            is_new_max = self._is_new_max(instance)
            if self._verbose == 1 and not is_new_max:
                line = ""
            else:
                colour = Colours.purple if is_new_max else Colours.black
                line = self._step(instance, colour=colour) + "\n"
        elif event == Events.OPTMIZATION_END:
            line = "=" * self._header_length + "\n"
        elif event == Events.BATCH_END:
            line = "END BATCH" + "-" * (self._header_length - 9) + "\n"

        if self._verbose:
            print(line, end="")
        self._update_tracker(event, instance)


class DiscreteLogger(ScreenLogger):
    '''Adds changes to printout for discrete set-up'''
    _default_cell_size = 12
    _default_precision = 4

    def __init__(self, verbose=2):
        self._verbose = verbose
        self._header_length = None
        super(ScreenLogger, self).__init__()

    def _step(self, instance, colour=Colours.black):
        res = instance.res[-1]
        cells = []
        disc_cells = []

        cells.append(self._format_number(self._iterations + 1))
        disc_cells.append(self._format_number(self._iterations + 1))
        cells.append(self._format_number(res["target"]))
        disc_cells.append(self._format_number(res["target"]))

        for key in instance.space.keys:
            cells.append(self._format_number(res["params"][key]))
        x = list(instance._space._bin([res["params"][key] for key in res["params"]]))
        for c in x:
            disc_cells.append(self._format_number(c))
        line = "| " + " | ".join(map(colour, cells)) + " |\n"
        line += "| " + " | ".join(map(colour, disc_cells)) + " |"
        return line


class JSONLogger(_Tracker):
    def __init__(self, path):
        self._path = path if path[-5:] == ".json" else path + ".json"
        try:
            os.remove(self._path)
        except OSError:
            pass
        super(JSONLogger, self).__init__()

    def update(self, event, instance):
        if event == Events.OPTMIZATION_STEP:
            data = dict(instance.res[-1])

            now, time_elapsed, time_delta = self._time_metrics()
            data["datetime"] = {
                "datetime": now,
                "elapsed": time_elapsed,
                "delta": time_delta,
            }

            with open(self._path, "a") as f:
                f.write(json.dumps(data) + "\n")

        self._update_tracker(event, instance)
