import copy
import threading

from ._find import Find
from ._gen import Gen
from ._select import Select

# Thread-local storage for block tracking so concurrent threads don't conflict.
_thread_local = threading.local()


class PathFinder:
    token_in = 0
    token_out = 0

    @staticmethod
    def _get_open_block():
        return getattr(_thread_local, "open_block", None)

    @staticmethod
    def _set_open_block(value):
        _thread_local.open_block = value

    @staticmethod
    def _get_empty_block():
        return getattr(_thread_local, "empty_block", True)

    @staticmethod
    def _set_empty_block(value):
        _thread_local.empty_block = value


    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self._variables = {}
        self.chat = []

        self.prefix_text = ""
        self.text_to_consume = ""

    def _current_prompt(self):
        raise NotImplementedError

    def _format_chat_entry_as_html(self, entry):
        # Bold the role tag
        role_tag = f'<strong>{entry["role"].upper()}</strong>'
        return f'<div>{role_tag}: {entry["content"]}</div>'

    def html(self):
        if isinstance(self.chat, list):
            # Process each chat entry and format it as HTML
            html_entries = [
                self._format_chat_entry_as_html(entry) for entry in self.chat
            ]
            prompt_render = "".join(html_entries)
        else:
            # Format a single chat entry as HTML
            prompt_render = self._format_chat_entry_as_html(self.chat)
        return prompt_render

    def copy(self):
        """Create a shallow copy of the model object."""

        # start with a shallow copy
        new_lm = copy.copy(self)
        new_lm._variables = self._variables.copy()
        if isinstance(self.chat, list):
            new_lm.chat = self.chat.copy()
        else:
            new_lm.chat = self.chat

        return new_lm

    def _consume_assistant_text(self, value):
        pass

    def __add__(self, value):
        lm = self.copy()

        # Track whether we flipped init_tag so we can restore it if the
        # operation below raises.  init_tag lives on the shared ContextBlock
        # while the assistant chat entry is created on the local *copy*.  If
        # the copy is discarded (exception), init_tag must be restored so the
        # next __add__ call re-creates the entry.
        _restore_init_tag = False

        if (
            len(lm.chat) == 0
            and PathFinder._get_empty_block()
            and PathFinder._get_open_block() is None
        ):
            # We are not in a chat block, so we simply add the string
            lm.chat = ""
        elif PathFinder._get_open_block() is not None and PathFinder._get_open_block().init_tag:
            _restore_init_tag = True
            PathFinder._get_open_block().init_tag = False
            lm.chat.append(
                {
                    "role": PathFinder._get_open_block().role,
                    "content": "",
                }
            )
            if PathFinder._get_open_block().role == "assistant":
                lm.prefix_text = ""

        try:
            if isinstance(value, str):
                if isinstance(lm.chat, list):
                    lm.chat[-1]["content"] += value
                else:
                    lm.chat += value

                if lm.chat[-1]["role"] == "assistant":
                    lm._consume_assistant_text(value)
            else:
                if isinstance(lm.chat, list):
                    if lm.chat[-1]["role"] != "assistant":
                        raise Exception(
                            f"{value} can be used only in assistant block, not in"
                            f" {lm.chat[-1]['role']} block!"
                        )
                if isinstance(value, Gen):
                    res = lm._get_gen(value)
                    original_res = res
                elif isinstance(value, Find):
                    res, original_res = lm._get_find(value)
                elif isinstance(value, Select):
                    res = lm._get_select(value)
                    original_res = res
                else:
                    raise Exception(f"Unknown type {type(value)}")

                if isinstance(lm.chat, list):
                    lm.chat[-1]["content"] += original_res
                else:
                    lm.chat += original_res
                lm._variables[value.name] = res

            return lm
        except Exception:
            # Restore init_tag so the next __add__ in the same block
            # context will re-create the chat entry that was lost.
            if _restore_init_tag and PathFinder._get_open_block() is not None:
                PathFinder._get_open_block().init_tag = True
            raise

    def __getitem__(self, key):
        if key in self._variables:
            return self._variables[key]

    def set(self, key, value):
        lm = self.copy()
        lm._variables[key] = value
        return lm

    def _get_gen(self, value: Gen):
        raise NotImplementedError

    def _get_find(self, value: Find):
        raise NotImplementedError

    def _get_select(self, value: Select):
        raise NotImplementedError

