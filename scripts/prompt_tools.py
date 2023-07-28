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
from modules.processing import fix_seed
from functools import partial, reduce

from typing import overload

print(
    f"[-] Prompt Tools initialized"
)

fill_values_symbol = "\U0001f4d2"

do_logging = False


def log(log_type: str, data: object, text: object = None, prefix: str = 'PT', new_line: str = '\n',
        tab: str = '\t') -> None:
    if not do_logging and log_type is not "Info":
        return
    log_text = f'{new_line}[{prefix}]'
    if type(data) is not list:
        log_text = log_text + f'{new_line}{tab}{data}: {text}'
    else:
        list_text = new_line.join([f'{tab}{label}: {value}' for label, value in data if label and value])
        log_text = f'{log_text}{new_line}{list_text}'

    print(log_text)


def get_list_of_tokens(prompts):
    log('Debug','Get List', prompts)
    prompt_flatten = prompts if type(prompts) is not tuple else prompts[0]
    if type(prompt_flatten) == list:
        prompt_flatten = ','.join(prompt_flatten)
    log('Debug','Flatten', prompt_flatten)
    tokens = [x.strip() for x in chain.from_iterable(csv.reader(StringIO(prompt_flatten))) if x]
    return tokens


def get_prompt_tokens(prompt_text, steps, with_extra_networks=True):
    prompts, res = get_prompts(prompt_text, steps, with_extra_networks) if prompt_text else [], []
    tokens = get_list_of_tokens(prompts)
    return tokens


def update_choices(prompt_text, steps, with_extra_networks=True):
    return gr.update(choices=get_prompt_tokens(prompt_text, steps, with_extra_networks))


def get_prompts(text, steps, with_extra_networks=True):
    # adapted from modules.ui.py
    try:
        prompt, res = extra_networks.parse_prompt(text)

        _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text if with_extra_networks else prompt])
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        log('Error', 'Error', 'Unable to parse prompt')
        prompt_schedules = [[[steps, text]]]
        res = []

    flat_prompts = reduce(lambda list1, list2: list1 + list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    token_count, max_length = max([model_hijack.get_prompt_lengths(prompt) for prompt in prompts],
                                  key=lambda args: args[0])

    log('Debug', 'Prompts', prompts)
    log_list = []
    for key, values in res.items():
        l = [f'{value.items}, {value.named}' for value in res[key]]
        log_list.append((key, l))
    log('Debug', log_list)
    return prompts, res


# adapted from dynamic prompt extension
def get_seeds(
        p,
        num_seeds,
        use_fixed_seed,
):
    if p.subseed_strength != 0:
        seed = int(p.all_seeds[0])
        subseed = int(p.all_subseeds[0])
    else:
        seed = int(p.seed)
        subseed = int(p.subseed)

    if use_fixed_seed:
        all_seeds = [seed] * num_seeds
        all_subseeds = [subseed] * num_seeds
    else:
        if p.subseed_strength == 0:
            all_seeds = [seed + i for i in range(num_seeds)]
        else:
            all_seeds = [seed] * num_seeds

        all_subseeds = [subseed + i for i in range(num_seeds)]

    return all_seeds, all_subseeds


def get_shuffled_prompt(prompt, tokens, seed):
    random.seed(seed)
    shuffled = random.sample(tokens, len(tokens))
    token_dict = {}
    new_prompt = None
    for token, replacement in zip(tokens, shuffled):
        token_dict[token] = replacement
    for token in get_list_of_tokens(prompt):
        new_token = token if token not in token_dict.keys() else token_dict[token]
        new_prompt = f'{new_prompt}, {new_token}' if new_prompt is not None else f'{new_token}'

    return new_prompt


class PromptTools(scripts.Script):
    # Extension title in menu UI
    def __init__(self):
        super().__init__()
        self.tokens = None
        self.include_extras = None
        self.i2i_steps = None
        self.t2i_steps = None
        self.i2i_prompt = None
        self.t2i_prompt = None

        self.t2i_prompt_connected = False
        self.i2i_prompt_connected = False

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
        with gr.Accordion('Prompt Tools', open=False):
            with gr.Column(scale=6):
                pt_enable = gr.Checkbox(
                    label="Enable Prompt Tools",
                    value=False,
                    visible=True,
                    elem_id=self.elem_id("pt_enable"),
                )

            with gr.Tabs():
                with gr.Tab("Prompt Shuffler"):
                    with gr.Row():
                        with gr.Column():
                            n_prompts = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                step=1,
                                value=1,
                                label="# of prompts to generate",
                                elem_id=self.elem_id("pt_prompt_generate"),
                            )
                            fixed_seed = gr.Checkbox(
                                label="Fix seed",
                                value=True,
                                visible=True,
                                elem_id=self.elem_id("fix_seed"),
                            )
                            include_extra = gr.Checkbox(
                                label="Include Extra Networks",
                                value=True,
                                visible=True,
                                elem_id=self.elem_id("include_extra"),
                            )

                        with gr.Column():
                            with gr.Row(scale=0):
                                tokens = gr.Dropdown(
                                    label="Tokens",
                                    visible=True,
                                    multiselect=True,
                                    interactive=True,
                                    elem_id=self.elem_id("pt_tokens"),
                                    show_label=True
                                )
                                fill_button = ToolButton(
                                    value=fill_values_symbol,
                                    elem_id=self.elem_id("pt_fill_tokens"),
                                    visible=True,
                                    width=20
                                )
        # TODO: add more UI components (cf. https://gradio.app/docs/#components)

        self.tokens = tokens
        self.include_extras = include_extra

        with contextlib.suppress(AttributeError):  # Ignore the error if the attribute is not present
            if is_img2img:
                # Bind the click event of the button to the send_text_to_prompt function
                # Inputs: text_to_be_sent (textbox), self.boxxIMG (textbox)
                # Outputs: self.boxxIMG (textbox)
                fill_button.click(
                    fn=get_prompt_tokens,
                    inputs=[self.i2i_prompt, self.i2i_steps, include_extra],
                    outputs=[tokens]
                )
            else:
                # Bind the click event of the button to the send_text_to_prompt function
                # Inputs: text_to_be_sent (textbox), self.boxx (textbox)
                # Outputs: self.boxx (textbox)
                fill_button.click(
                    fn=get_prompt_tokens,
                    inputs=[self.t2i_prompt, self.t2i_steps, include_extra],
                    outputs=[tokens]
                )
        return [pt_enable, fixed_seed, n_prompts, include_extra, tokens, fill_button]

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

        if not self.t2i_prompt_connected:
            if self.t2i_prompt is not None and self.t2i_steps is not None and self.include_extras is not None:
                self.t2i_prompt.blur(
                    fn=update_choices,
                    inputs=[self.t2i_prompt, self.t2i_steps, self.include_extras],
                    outputs=[self.tokens]
                )
                self.t2i_prompt_connected = True

        if not self.i2i_prompt_connected:
            if self.i2i_prompt is not None and self.i2i_steps is not None and self.include_extras is not None:
                self.i2i_prompt.blur(
                    fn=update_choices,
                    inputs=[self.i2i_prompt, self.i2i_steps, self.include_extras],
                    outputs=[self.tokens]
                )
                self.i2i_prompt_connected = True

    # Extension main process
    # Type: (StableDiffusionProcessing, List<UI>) -> (Processed)
    # args is [StableDiffusionProcessing, UI1, UI2, ...]
    def process(self, p, pt_enable, fixed_seed, n_prompts, include_extra, tokens, fill_button):
        if not pt_enable:
            return

        fix_seed(p)

        original_prompt = p.all_prompts[0] if p.all_prompts else p.prompt
        original_negative_prompt = p.all_negative_prompts[0] if p.all_negative_prompts else p.negative_prompt

        num_images = p.n_iter * p.batch_size * n_prompts

        log('Debug', f'Number of images: {num_images}, Number of iterations: {p.n_iter}')

        p.all_seeds, p.all_subseeds = get_seeds(
            p,
            num_images,
            fixed_seed,
        )

        prompt_seeds, _ = get_seeds(p, num_images, False)
        all_prompts = [get_shuffled_prompt(original_prompt, tokens, seed) for seed in prompt_seeds]

        p.n_iter = math.ceil(num_images // p.batch_size)
        if num_images > 1:
            log(
                'Info',
                f"Prompt Tools will create {num_images} images in a total of {p.n_iter} batches.",
            )
        else:
            p.do_not_save_grid = True

        p.all_prompts = all_prompts
        p.all_negative_prompts = [original_negative_prompt] * num_images

        p.prompt = original_prompt

        log('Debug', 'All prompts', p.all_prompts)
        log('Debug', 'All negative prompts', p.all_negative_prompts)

        return
