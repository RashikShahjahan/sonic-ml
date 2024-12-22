from flow import inference_workflow

# Test inference with Wikipedia-style prompts
test_prompts = [
    "বাংলাদেশের ইতিহাস",     # History of Bangladesh
    "রবীন্দ্রনাথ ঠাকুর হলেন", # Rabindranath Tagore is
    "ঢাকা শহর হল",           # Dhaka city is
    "সুন্দরবন হল একটি",      # The Sundarbans is a
]

# Define the models to compare
models = [
    {
        "name": "Model 1",
        "path": "checkpoints/wikipedia_bn_model_15M/checkpoint_step_1999.pth",
        "tokenizer": "wikipedia_bn_tokenizer"
    },
    {
        "name": "Model 2",
        "path": "checkpoints/wikipedia_bn_model/checkpoint_step_5999.pth",  # Adjust path as needed
        "tokenizer": "wikipedia_bn_tokenizer"
    },
    {
        "name": "Model 3",
        "path": "checkpoints/wikipedia_bn_model_15M_4096/checkpoint_step_2999.pth",
        "tokenizer": "wikipedia_bn_tokenizer_4096"
    }
]

print("\nComparing models with Wikipedia-style prompts:")
print("="*70)
for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    print("-"*70)
    
    for model in models:
        print(f"\n{model['name']} output:")
        inference_workflow(
            model_path=model["path"],
            prompt=prompt,
            tokenizer_prefix=model["tokenizer"]
        )
        print("-"*35)
    
    print("="*70)


