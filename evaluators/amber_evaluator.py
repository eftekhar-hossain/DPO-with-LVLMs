from .evaluator import Evaluator
import os
from PIL import Image
from tqdm import tqdm
import json
import torch

class AmberEvaluator(Evaluator):
    def __init__(self, args):
        self.data_path = os.path.join(args.amber_path, "data")
        self.image_path = os.path.join(self.data_path, "image")
        self.query_path = os.path.join(self.data_path, "query")
        self.output_dir=args.output_dir

    def eval(self, model, processor):
        # For our purposes, run entire benchmark
        query_file = os.path.join(self.query_path, "query_all.json")
        inference_file = "amber_inf.jsonl"

        with open(query_file, "r") as q:
            data = json.load(q)

        inferences = []
        print("==============================================================")
        print("=  BEGIN AMBER BENCHMARKING                                  =")
        print("==============================================================")
        for obj in tqdm(data, desc="Generating on AMBER"):
            id = obj["id"]
            img = obj["image"]
            query = obj["query"]

            image_path = os.path.join(self.image_path, img)
            image = Image.open(image_path).convert("RGB")

            # Prepare prompt
            chat_data = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query}
                    ]
                }
            ]
            prompt = processor.apply_chat_template(chat_data, add_generation_prompt=True)

            # Process inputs
            inputs = processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            ).to("cuda")

            # Generate answer
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            inferences.append({
                "id": id,
                "response": output_text
            })

        # Save results
        os.makedirs(self.output_dir, exist_ok=True)
        out_file = os.path.join(self.output_dir, inference_file)
        with open(out_file, "w") as f:
            for inf in inferences:
                f.write(json.dumps(inf) + "\n")
        print(f"Saved inferences to {out_file}")
        print("Benchmark: AMBER done.")
        # Could do this in memory, but use artifacts on disk for now
        return (query_file, out_file)



