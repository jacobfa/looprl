def register():
    from vllm import ModelRegistry

    if "DoryForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("DoryForCausalLM", "looprl.vllm_dory:DoryForCausalLM")
