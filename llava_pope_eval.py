#!/usr/bin/env python3
"""
Evaluate a LLaVA model (base or DPO-trained) on the POPE benchmark.

This script:
- Loads either the base LLaVA model or a DPO-adapted model
- Loads POPE benchmark questions
- Runs inference with images + questions
- Saves predictions to JSONL
- Evaluates simple accuracy metrics
"""

import argparse
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from metrics.pope_calculator import PopeCalculator
from args.validate_pope import is_pope, validate_pope
from evaluators.pope_evaluator import PopeEvaluator

def load_model(model_name, dpo_checkpoint=None, device="cuda"):
    """
    Load either base model or DPO adapter on top of base model.
    """
    print(f"Loading base model: {model_name}")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if dpo_checkpoint:
        print("==============================================================")
        print(f"Loading DPO adapter from: {dpo_checkpoint}")
        print("==============================================================")
        model = PeftModel.from_pretrained(model, dpo_checkpoint)

    model.eval()
    return model.to(device)

def main(args, eval, calc):

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name, do_image_splitting=False)

    model = load_model(args.model_name, args.dpo_checkpoint)

    questions, answers = eval.eval(model, processor)
    calc.parse(answers, questions)
    print(calc.calculate_results())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaVA model on POPE benchmark")
    parser.add_argument("--model_type", type=str, help="Which model to evaluate", choices=["base", "dpo"], default="base")
    parser.add_argument("--model_name", type=str, required=True, help="Base model name or path")
    parser.add_argument("--dpo_checkpoint", type=str, default=None, help="Path to DPO adapter checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--benchmark", type=str, help="How to evaluate the model", choices=["pope"], default="pope")
    parser.add_argument("--pope_path", type=str, help="Path to POPE dataset folder")
    parser.add_argument("--coco_path", type=str, help="Path to COCO images")
    parser.add_argument("--set_name", type=str, default="random", choices=["random", "popular", "adv"],
                        help="Which POPE split to evaluate")
    args = parser.parse_args()

    # Could split this out to another function
    if is_pope(args):
        if validate_pope(args):
            evaluator = PopeEvaluator(args)
            calculator = PopeCalculator()
        else:
            print("POPE requires the coco_path, pope_path, and set_name arguments set!")
            exit()
    else:
        print(f"Unsupported benchmark: {args.benchmark}")
        exit()
    main(args, evaluator, calculator)

    