import random

import modules.scripts as scripts
import gradio as gr
import os
import csv
from itertools import chain
from io import StringIO
from copy import copy
import contextlib
import math

from altair import value
from modules import images, script_callbacks
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

from modules.ui_components import ToolButton

from modules import script_callbacks, extra_networks, prompt_parser
from modules.sd_hijack import model_hijack
from functools import partial, reduce

from typing import overload

print(
    f"[-] Prompt Tools initialized"
)

fill_values_symbol = "\U0001f4d2"


def log(data: object, text: object = None, prefix: str = 'PT', new_line: str = '\n', tab: str = '\t') -> None:
    log_text = f'{new_line}[{prefix}]'
    if type(data) is not list:
        log_text = log_text + f'{new_line}{tab}{data}: {text}'
    else:
        list_text = new_line.join([f'{tab}{label}: {value}' for label, value in data if label and value])
        log_text = f'{log_text}{new_line}{list_text}'

    print(log_text)


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


def suffix(n: int, c: str = " ") -> str:
    return "" if n == 0 else c + ordinal(n + 1)


def elem_id(item_id: str, n: int, is_img2img: bool) -> str:
    tap = "img2img" if is_img2img else "txt2img"
    suf = suffix(n, "_")
    return f"script_{tap}_prompttools_{item_id}{suf}"


def get_list_of_tokens(prompts):
    prompt_flatten = prompts
    if type(prompts) == list:
        prompt_flatten = ','.join(prompts)
    tokens = [x.strip() for x in chain.from_iterable(csv.reader(StringIO(prompt_flatten))) if x]
    return tokens


def get_prompt_tokens(prompt_text, steps, with_extra_networks=True):
    prompts, res = get_prompts(prompt_text, steps, with_extra_networks)
    tokens = get_list_of_tokens(prompts)
    return tokens


def get_prompts(text, steps, with_extra_networks=True):
    # adapted from modules.ui.py
    try:
        prompt, res = extra_networks.parse_prompt(text)

        _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text if with_extra_networks else prompt])
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]
        res = []

    flat_prompts = reduce(lambda list1, list2: list1 + list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    token_count, max_length = max([model_hijack.get_prompt_lengths(prompt) for prompt in prompts],
                                  key=lambda args: args[0])

    log('Prompts', prompts)
    log_list = []
    for key, values in res.items():
        l = [f'{value.items}, {value.named}' for value in res[key]]
        log_list.append((key, l))
    log(log_list)
    return prompts, res


def get_shuffled_prompt(prompt, tokens):
    shuffled = random.sample(tokens, len(tokens))
    token_dict = {}
    new_prompt = None
    for token, replacement in zip(tokens, shuffled):
        token_dict[token] = replacement
    list_of_tokens = get_list_of_tokens(prompt)
    log("List of tokens", list_of_tokens)
    for token in get_list_of_tokens(prompt):
        new_token = token if token not in token_dict.keys() else token_dict[token]
        new_prompt = f'{new_prompt}, {new_token}' if new_prompt is not None else f'{new_token}'

    return new_prompt


class PromptTools(scripts.Script):
    # Extension title in menu UI
    def __init__(self):
        super().__init__()
        self.i2i_steps = 0
        self.t2i_steps = 0
        self.i2i_prompt = None
        self.t2i_prompt = None

    def title(self):
        return "Prompt Tools"

    # Decide to show menu in txt2img or img2img
    # - in "txt2img" -> is_img2img is `False`
    # - in "img2img" -> is_img2img is `True`
    #
    # below code always show extension menu
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # Setup menu ui detail
    def ui(self, is_img2img):
        eid = partial(elem_id, n=0, is_img2img=is_img2img)

        with gr.Accordion('Prompt Tools', open=False):
            with gr.Column(scale=6):
                pt_enable = gr.Checkbox(
                    label="Enable Prompt Tools",
                    value=False,
                    visible=True,
                    elem_id=eid("pt_enable"),
                )

            with gr.Tabs():
                with gr.Tab("Prompt Shuffler"):
                    with gr.Row():
                        n_prompts = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            step=1,
                            value=1,
                            label="# of prompts to generate",
                            elem_id=eid("pt_prompt_generate"),
                        )

                        with gr.Column():
                            include_extra = gr.Checkbox(
                                label="Include Extra Networks",
                                value=True,
                                visible=True,
                                elem_id=eid("include_extra"),
                            )

                            with gr.Row():
                                tokens = gr.Dropdown(
                                    label="Tokens",
                                    visible=True,
                                    multiselect=True,
                                    interactive=True,
                                    elem_id=eid("pt_tokens"),
                                    show_label=False
                                )
                                fill_button = ToolButton(
                                    value=fill_values_symbol,
                                    elem_id=eid("pt_fill_tokens"),
                                    visible=True,
                                    width=40
                                )
        # TODO: add more UI components (cf. https://gradio.app/docs/#components)

        self.tokens = tokens

        with contextlib.suppress(AttributeError):  # Ignore the error if the attribute is not present
            if is_img2img:
                # Bind the click event of the button to the send_text_to_prompt function
                # Inputs: text_to_be_sent (textbox), self.boxxIMG (textbox)
                # Outputs: self.boxxIMG (textbox)
                fill_button.click(fn=get_prompt_tokens, inputs=[self.i2i_prompt, self.i2i_steps, include_extra], outputs=[tokens])
            else:
                # Bind the click event of the button to the send_text_to_prompt function
                # Inputs: text_to_be_sent (textbox), self.boxx (textbox)
                # Outputs: self.boxx (textbox)
                fill_button.click(fn=get_prompt_tokens, inputs=[self.t2i_prompt, self.t2i_steps, include_extra], outputs=[tokens])
        return [pt_enable, n_prompts, include_extra, tokens, fill_button]

    def after_component(self, component, **kwargs):
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/7456#issuecomment-1414465888 helpfull link
        # Find the text2img textbox component
        if kwargs.get("elem_id") == "txt2img_prompt":
            self.t2i_prompt = component
        # Find the img2img textbox component
        if kwargs.get("elem_id") == "img2img_prompt":
            self.i2i_prompt = component
        if kwargs.get("elem_id") == "txt2img_steps":
            self.t2i_steps = component
        if kwargs.get("elem_id") == "img2img_steps":
            self.i2i_steps = component

    # Extension main process
    # Type: (StableDiffusionProcessing, List<UI>) -> (Processed)
    # args is [StableDiffusionProcessing, UI1, UI2, ...]
    def process(self, p, pt_enable, n_prompts, include_extra, tokens, fill_button):
        print('Running!...')
        if not pt_enable:
            return

        # print(f'Original Prompt: {p.prompt}')
        # print(f'Tokens: {tokens}')
        p.prompt = [get_shuffled_prompt(p.prompt, tokens) for _ in range(int(n_prompts))]
        p.n_iter = math.ceil(len(p.prompt) / p.batch_size)
        p.do_not_save_grid = True

        print(f"Prompt Tools will create {len(p.prompt)} images using a total of {p.n_iter} batches.")

        p.seed = [p.seed + i for i in range(len(p.prompt))]

        # print(f'Shuffled Prompts: ')
        # _ = [print(p) for p in p.prompt]

        return
